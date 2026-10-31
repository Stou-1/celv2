// Portions of this file are derived from the AVM v1.0.0 decoder (av2/decoder),
// Copyright (c) 2021, Alliance for Open Media, under the BSD 3-Clause Clear License
// and the Alliance for Open Media Patent License 1.0: see third_party/avm/.

#include "core/encoder.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <immintrin.h>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "bitstream/frame_header.h"
#include "bitstream/obu.h"
#include "core/quant.h"
#include "core/global_motion.h"
#include "core/threads.h"
#include "core/transform.h"
#include "gpu/intra_common.h"

extern "C" {
#include "av2/common/idct.h"
#include "av2/common/quant_common.h"
#include "avm_dsp/bitwriter_buffer.h"
#include "av2/common/reconinter.h"
#include "av2/common/reconintra.h"
}

namespace {

// Estimated bits of coding a luma transform unit as split (any partition) instead of whole.
constexpr double kSplitBits = 4;

// One 4x4 Hadamard stage across the four 16-bit lanes of each half: (c0+c1, c0-c1, c2+c3, c2-c3)
// then (b0+b2, b1+b3, b0-b2, b1-b3), as celv2::hadamard4x4 does along a row.
__m128i hadamard4_lanes(__m128i x) {
  const __m128i pair = _mm_setr_epi16(1, -1, 1, -1, 1, -1, 1, -1), half = _mm_setr_epi16(1, 1, -1, -1, 1, 1, -1, -1);
  const __m128i b = _mm_add_epi16(_mm_shufflehi_epi16(_mm_shufflelo_epi16(x, 0xB1), 0xB1), _mm_sign_epi16(x, pair));
  return _mm_add_epi16(_mm_shufflehi_epi16(_mm_shufflelo_epi16(b, 0x4E), 0x4E), _mm_sign_epi16(b, half));
}

// Sum of |4x4 Hadamard(src - dst)| over the w x h block (both multiples of 4), the same value as
// celv2::hadamard4x4 summed per 4x4 tile: two tiles per pass, the transform down the rows, then
// along them (|H D H^T| does not depend on the order). Sources of 8 or 16 bits (up to 10-bit
// samples: a 4x4 transform of their differences stays within 16 bits).
template <typename S>
int64_t satd_src(const S* s, ptrdiff_t ss, const uint16_t* d, ptrdiff_t ds, int w, int h) {
  const __m128i ones = _mm_set1_epi16(1);
  __m128i acc = _mm_setzero_si128();
  int64_t total = 0;
  for (int y = 0; y < h; y += 4) {
    for (int x = 0; x < w; x += 8) {
      __m128i r[4];
      for (int i = 0; i < 4; i++) {
        const S* sp = s + (y + i) * ss + x;
        const uint16_t* dp = d + (y + i) * ds + x;
        const bool two = x + 8 <= w;
        __m128i sv;
        if constexpr (sizeof(S) == 1)
          sv = _mm_cvtepu8_epi16(two ? _mm_loadl_epi64(reinterpret_cast<const __m128i*>(sp))
                                     : _mm_cvtsi32_si128(*reinterpret_cast<const int*>(sp)));
        else
          sv = two ? _mm_loadu_si128(reinterpret_cast<const __m128i*>(sp)) : _mm_loadl_epi64(reinterpret_cast<const __m128i*>(sp));
        const __m128i dv = two ? _mm_loadu_si128(reinterpret_cast<const __m128i*>(dp))
                               : _mm_loadl_epi64(reinterpret_cast<const __m128i*>(dp));
        r[i] = _mm_sub_epi16(sv, dv);
      }
      const __m128i a0 = _mm_add_epi16(r[0], r[1]), a1 = _mm_sub_epi16(r[0], r[1]);
      const __m128i a2 = _mm_add_epi16(r[2], r[3]), a3 = _mm_sub_epi16(r[2], r[3]);
      const __m128i v[4] = {_mm_add_epi16(a0, a2), _mm_add_epi16(a1, a3), _mm_sub_epi16(a0, a2), _mm_sub_epi16(a1, a3)};
      for (int k = 0; k < 4; k++) acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_abs_epi16(hadamard4_lanes(v[k])), ones));
    }
    if ((y & 60) == 60 || y + 4 >= h) {  // drain before the 32-bit lanes could overflow
      alignas(16) int32_t lanes[4];
      _mm_store_si128(reinterpret_cast<__m128i*>(lanes), acc);
      total += static_cast<int64_t>(lanes[0]) + lanes[1] + lanes[2] + lanes[3];
      acc = _mm_setzero_si128();
    }
  }
  return total;
}

PARTITION_TYPE to_avm(Part p) {
  switch (p) {
    case Part::None: return PARTITION_NONE;
    case Part::Horz: return PARTITION_HORZ;
    case Part::Vert: return PARTITION_VERT;
    case Part::Split: return PARTITION_SPLIT;
  }
  return PARTITION_NONE;
}

int num_children(Part p) { return p == Part::Split ? 4 : p == Part::None ? 0 : 2; }

// decodeframe.c set_offsets()
void set_offsets(AV2_COMMON* cm, MACROBLOCKD* xd, BLOCK_SIZE bsize, int mi_row, int mi_col,
                 int bw, int bh, int x_mis, int y_mis, PARTITION_TREE* parent, int index) {
  const int num_planes = av2_num_planes(cm);
  const CommonModeInfoParams* const mi_params = &cm->mi_params;
  set_mi_offsets(mi_params, xd, mi_row, mi_col, x_mis, y_mis);
  xd->mi[0]->sb_type[xd->tree_type == CHROMA_PART] = bsize;
  if (xd->tree_type != CHROMA_PART) {
    xd->mi[0]->mi_row_start = mi_row;
    xd->mi[0]->mi_col_start = mi_col;
  }
  if (xd->tree_type == SHARED_PART) {
    for (int x = 1; x < x_mis; ++x) xd->mi[x] = xd->mi[0];
    int idx = mi_params->mi_stride;
    for (int y = 1; y < y_mis; ++y) {
      std::memcpy(&xd->mi[idx], &xd->mi[0], x_mis * sizeof(xd->mi[0]));
      idx += mi_params->mi_stride;
    }
  }
  CHROMA_REF_INFO* cri = &xd->mi[0]->chroma_ref_info;
  set_chroma_ref_info(xd->tree_type, mi_row, mi_col, index, bsize, cri,
                      parent ? &parent->chroma_ref_info : nullptr,
                      parent ? parent->bsize : BLOCK_INVALID,
                      parent ? parent->partition : PARTITION_NONE, xd->plane[1].subsampling_x,
                      xd->plane[1].subsampling_y);
  set_plane_n4(xd, bw, bh, num_planes, cri);
  set_entropy_context(xd, mi_row, mi_col, num_planes, cri);
  set_mi_row_col(cm, xd, &xd->tile, mi_row, bh, mi_col, bw, mi_params->mi_rows,
                 mi_params->mi_cols, cri);
  xd->mi[0]->chroma_mi_row_start = mi_row;
  xd->mi[0]->chroma_mi_col_start = mi_col;
  av2_setup_dst_planes(xd->plane, &cm->cur_frame->buf, mi_row, mi_col, 0, num_planes, cri);
}

// decodeframe.c extend_mc_border()
void extend_mc_border(const struct scale_factors* sf, struct buf_2d* pre_buf, MV32 scaled_mv,
                      PadBlock block, int subpel_x_mv, int subpel_y_mv, int do_warp,
                      int is_intrabc, uint16_t* mc_buf, uint16_t** pre, int* src_stride) {
  int x_pad = 0, y_pad = 0;
  if (update_extend_mc_border_params(sf, pre_buf, scaled_mv, &block, subpel_x_mv, subpel_y_mv,
                                     do_warp, is_intrabc, &x_pad, &y_pad, nullptr)) {
    const uint16_t* const buf_ptr = pre_buf->buf0 + block.y0 * pre_buf->stride + block.x0;
    const int b_w = block.x1 - block.x0;
    const int b_h = block.y1 - block.y0;
    highbd_build_mc_border(buf_ptr, pre_buf->stride, mc_buf, b_w, block.x0, block.y0, b_w, b_h,
                           pre_buf->width, pre_buf->height);
    *src_stride = b_w;
    *pre = mc_buf + y_pad * (AVM_INTERP_EXTEND - 1) * b_w + x_pad * (AVM_INTERP_EXTEND - 1);
  }
}

// decodeframe.c dec_calc_subpel_params_and_extend(): the decoder's motion-compensation callback.
void calc_subpel_params_and_extend(const MV* const src_mv, InterPredParams* const ipp,
                                   MACROBLOCKD* xd, int mi_x, int mi_y, int ref,
                                   int use_optflow_refinement, uint16_t** mc_buf, uint16_t** pre,
                                   SubpelParams* subpel_params, int* src_stride) {
  if (ipp->use_ref_padding) {
    common_calc_subpel_params_and_extend(src_mv, ipp, xd, mi_x, mi_y, ref, use_optflow_refinement,
                                         mc_buf, pre, subpel_params, src_stride);
    return;
  }
  PadBlock block;
  MV32 scaled_mv;
  int subpel_x_mv, subpel_y_mv;
  dec_calc_subpel_params(src_mv, ipp, xd, mi_x, mi_y, pre, subpel_params, src_stride, &block,
                         use_optflow_refinement, &scaled_mv, &subpel_x_mv, &subpel_y_mv);
  extend_mc_border(ipp->scale_factors, &ipp->ref_frame_buf, scaled_mv, block, subpel_x_mv,
                   subpel_y_mv, ipp->mode == WARP_PRED, ipp->is_intrabc, mc_buf[ref], pre,
                   src_stride);
}

}  // namespace


// Codes one tile: the decoder's per-tile state (CodecState::xd(t), its CDFs, its MC buffers) and
// this walk's own state, so tiles can be coded on different threads.
// The superblock rows of one tile coded at once: row r codes superblock c once row r - 1 has
// finished c + 1 (the above-right superblock its MV candidates and intra edges read), starting from
// the CDFs row r - 1 had after its second superblock (as HEVC's WPP).
class Wavefront {
 public:
  Wavefront(CodecState& st, int t, int rows, int cols) : st_(st), t_(t), cols_(cols), done_(rows) {}
  // Before superblock c of row r.
  void wait(int r, int c) {
    if (r == 0) return;
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [&] { return failed_ || done_[r - 1] >= std::min(c + 2, cols_); });
    if (failed_) throw std::runtime_error("an earlier superblock row failed");
  }
  // After a superblock of row r, coded with the CDFs fc.
  void finish(int r, const FRAME_CONTEXT& fc) {
    // done_[r] is row r's alone to change, so it may be read before the lock.
    if (done_[r] + 1 == std::min(2, cols_)) *st_.handoff(t_, r & 1) = fc;
    {
      std::lock_guard<std::mutex> lock(mu_);
      done_[r]++;
    }
    cv_.notify_all();
  }
  void fail() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      failed_ = true;
    }
    cv_.notify_all();
  }
  // The CDFs row r > 0 starts from.
  const FRAME_CONTEXT& start(int r) const { return *st_.handoff(t_, (r - 1) & 1); }

 private:
  CodecState& st_;
  const int t_, cols_;
  std::vector<int> done_;  // per row: superblocks finished
  std::mutex mu_;
  std::condition_variable cv_;
  bool failed_ = false;
};

class TileCoder {
 public:
  // xd, mc_buf: the block state and scratch to code with (a row worker's, set up for tile t).
  TileCoder(Encoder& enc, CodecState& st, MACROBLOCKD* xd, uint16_t** mc_buf, const Frame420& src,
            const DecisionMap& dm, const RowGate* gate)
      : enc_(enc), cm_(st.cm()), xd_(xd), mc_buf_(mc_buf), src_(&src), dm_(dm), gate_(gate) {}
  // decode_tile() for the tile's superblock row r (at mi_row) through bw, in the wavefront wf.
  void code_row(BlockWriter& bw, int mi_row, int r, Wavefront& wf);

 private:
  // A DecisionMap tree node: children indices (-1 if none) and, for a leaf, its block index.
  struct Node {
    Part part;
    int child[4] = {-1, -1, -1, -1};
    int leaf = -1;
    int row = 0, col = 0, high = 0, wide = 0;  // the node's block, in mi units
  };
  // The subtree at pos for the block at (row, col), high x wide mi (square SPLIT from 128x128 up).
  int parse_tree(size_t& pos, int& leaf, int row, int col, int high, int wide);
  // The deepest node whose block holds the given one (0, the superblock's, at least).
  int node_holding(int row, int col, int high, int wide) const;
  void code_partition(int mi_row, int mi_col, BLOCK_SIZE bsize, PARTITION_TREE* ptree);
  void code_block(int mi_row, int mi_col, PARTITION_TYPE partition, BLOCK_SIZE bsize,
                  PARTITION_TREE* ptree, const BlockDecision& d);
  void code_intra_txb(int plane, int blk_row, int blk_col, TX_SIZE tx_size);
  // BlockWriter::IntraCost: the current block's luma prediction with joint mode `joint` (its
  // largest transform block, predicted into dst as code_intra_txb does), SATD against the source.
  int64_t intra_satd(int joint);
  // Predicts the current block's luma as mbmi says (one reference or two averaged, into dst as
  // code_inter_block does).
  void predict_luma();
  // BlockWriter::RdCost: predict_luma(), then the luma residual at its largest transform size by DCT
  // quantize_cost() (squared error in steps plus weighted level bits), stopping past bound.
  double pred_rd(double bound);
  // The source minus dst (stride dst_stride) over a tx_size block of plane, edge pixels repeated.
  void residual_of(int plane, TX_SIZE tx_size, const uint16_t* dst, int dst_stride, int16_t* res) const;
  void code_inter_block(BLOCK_SIZE bsize);
  // The luma partition of the inter block's max-size transform unit at (blk_row, blk_col) whose
  // parts' DCT quantize() costs (on the prediction in dst) plus partition bits are least.
  TX_PARTITION_TYPE choose_tx_partition(BLOCK_SIZE bsize, TX_SIZE max_tx, int blk_row, int blk_col);
  // About quantize()'s cost of the luma DCT of residual tx (quantize_cost).
  double luma_tx_cost(const int16_t* res, int stride, TX_SIZE tx);
  // Residual of the prediction already in dst: transform, quantize, write, reconstruct into dst.
  void residual_code(int plane, TX_SIZE tx_size, uint16_t* dst, int dst_stride, int blk_row,
                     int blk_col);

  Encoder& enc_;
  AV2_COMMON* const cm_;
  MACROBLOCKD* const xd_;
  uint16_t** const mc_buf_;
  const Frame420* src_;
  const DecisionMap& dm_;
  const RowGate* const gate_;
  BlockWriter* bw_ = nullptr;
  const SbDecision* sb_ = nullptr;
  SbDecision composed_;  // a 128x128 superblock built from 64x64 decisions (compose_sb())
  int q_ = 0;  // the coded qindex of the superblock being coded (CodecState::sb_qindex())
  std::vector<Node> nodes_;
};

// decodeframe.c predict_inter_block() + the inter branch of decode_token_recon_block().
void TileCoder::code_inter_block(BLOCK_SIZE bsize) {
  AV2_COMMON* const cm = cm_;
  MACROBLOCKD* const xd = xd_;
  MB_MODE_INFO* const mbmi = xd->mi[0];
  const int num_planes = av2_num_planes(cm);
  const int plane_start = get_partition_plane_start(xd->tree_type);
  const int plane_end = get_partition_plane_end(xd->tree_type, num_planes);
  av2_init_txk_skip_array(cm, xd->mi_row, xd->mi_col, bsize, 0, xd->tree_type,
                          &mbmi->chroma_ref_info, plane_start, plane_end);
  // predict_inter_block(): one reference, or two averaged.
  for (int ref = 0; ref < 1 + has_second_ref(mbmi); ++ref) {
    const RefCntBuffer* ref_buf = get_ref_frame_buf(cm, mbmi->ref_frame[ref]);
    const struct scale_factors* sf = get_ref_scale_factors_const(cm, mbmi->ref_frame[ref]);
    xd->block_ref_scale_factors[ref] = sf;
    av2_setup_pre_planes(xd, ref, &ref_buf->buf, xd->mi_row, xd->mi_col, sf, num_planes,
                         &mbmi->chroma_ref_info);
  }
  if (mbmi->refinemv_flag || is_interintra_pred(mbmi) || mbmi->morph_pred)
    throw std::runtime_error("unsupported inter prediction tool");
  for (int plane = 0; plane < num_planes; ++plane) {
    if (plane && !xd->is_chroma_ref) break;
    av2_build_inter_predictors(cm, xd, plane, mbmi, nullptr, 0, 1, xd->plane[plane].width,
                               xd->plane[plane].height, xd->mi_col * MI_SIZE,
                               xd->mi_row * MI_SIZE, mc_buf_, calc_subpel_params_and_extend);
  }
  // parse_decode_block(): each max-size luma transform unit's partition, by the cost of its parts.
  if (cm->features.tx_mode == TX_MODE_SELECT && block_signals_txsize(bsize)) {
    const TX_SIZE max_tx = max_txsize_rect_lookup[bsize];
    for (int idy = 0; idy < mi_size_high[bsize]; idy += tx_size_high_unit[max_tx])
      for (int idx = 0; idx < mi_size_wide[bsize]; idx += tx_size_wide_unit[max_tx])
        bw_->tx_partition(max_tx, idy, idx, choose_tx_partition(bsize, max_tx, idy, idx));
  }
  // Residual (skip_txfm is always 0 in celv2).
  const int max_blocks_wide = max_block_wide(xd, bsize, 0);
  const int max_blocks_high = max_block_high(xd, bsize, 0);
  const int mu_blocks_wide = AVMMIN(max_blocks_wide, mi_size_wide[BLOCK_64X64]);
  const int mu_blocks_high = AVMMIN(max_blocks_high, mi_size_high[BLOCK_64X64]);
  const int mu128_wide = mi_size_wide[BLOCK_128X128];
  const int mu128_high = mi_size_high[BLOCK_128X128];
  for (int row128 = 0; row128 < max_blocks_high; row128 += mu128_high)
    for (int col128 = 0; col128 < max_blocks_wide; col128 += mu128_wide)
      for (int row = row128; row < AVMMIN(row128 + mu128_high, max_blocks_high);
           row += mu_blocks_high)
        for (int col = col128; col < AVMMIN(col128 + mu128_wide, max_blocks_wide);
             col += mu_blocks_wide)
          for (int plane = plane_start; plane < plane_end; ++plane) {
            if (plane && !xd->is_chroma_ref) break;
            struct macroblockd_plane* const pd = &xd->plane[plane];
            const int ss_x = pd->subsampling_x, ss_y = pd->subsampling_y;
            const BLOCK_SIZE plane_bsize = get_mb_plane_block_size(xd, mbmi, plane, ss_x, ss_y);
            const TX_SIZE max_tx_size = get_vartx_max_txsize(xd, plane_bsize, plane);
            const int bh_var_tx = tx_size_high_unit[max_tx_size];
            const int bw_var_tx = tx_size_wide_unit[max_tx_size];
            const int plane_unit_height = get_plane_tx_unit_height(xd, plane_bsize, plane, row, ss_y);
            const int plane_unit_width = get_plane_tx_unit_width(xd, plane_bsize, plane, col, ss_x);
            if (skip_parsing_recon(row, col, ss_x, ss_y, false)) continue;
            for (int blk_row = row >> ss_y; blk_row < plane_unit_height; blk_row += bh_var_tx)
              for (int blk_col = col >> ss_x; blk_col < plane_unit_width; blk_col += bw_var_tx) {
                // decode_reconstruct_tx()
                const int txp_index = av2_get_txb_size_index(plane_bsize, blk_row, blk_col);
                if (blk_row >= max_block_high(xd, plane_bsize, plane) ||
                    blk_col >= max_block_wide(xd, plane_bsize, plane))
                  continue;
                if (mbmi->tx_partition_type[txp_index] == TX_PARTITION_NONE || plane) {
                  uint16_t* dst = &pd->dst.buf[(blk_row * pd->dst.stride + blk_col) << MI_SIZE_LOG2];
                  residual_code(plane, max_tx_size, dst, pd->dst.stride, blk_row, blk_col);
                  continue;
                }
                get_tx_partition_sizes(mbmi->tx_partition_type[txp_index], max_tx_size, &mbmi->txb_pos,
                                       mbmi->sub_txs, xd->error_info);
                for (int i = 0; i < mbmi->txb_pos.n_partitions; ++i) {
                  mbmi->txb_idx = 0;
                  const int r = blk_row + mbmi->txb_pos.row_offset[i], c = blk_col + mbmi->txb_pos.col_offset[i];
                  if (r >= max_block_high(xd, plane_bsize, plane) || c >= max_block_wide(xd, plane_bsize, plane)) continue;
                  uint16_t* dst = &pd->dst.buf[(r * pd->dst.stride + c) << MI_SIZE_LOG2];
                  residual_code(plane, mbmi->sub_txs[i], dst, pd->dst.stride, r, c);
                }
              }
          }
}

double TileCoder::luma_tx_cost(const int16_t* res, int stride, TX_SIZE tx) {
  tran_low_t coeff[1024];
  const double scale = enc_.coef_scale(0, tx, xd_);
  fwd_dct(res, stride, tx, scale, coeff);
  return quantize_cost(coeff, tx, q_, DCT_DCT, nullptr,
                       tcq_enable(cm_->features.tcq_mode, 0, AVM_PLANE_Y, TX_CLASS_2D)) +
         dropped_cost(res, stride, tx, scale, coeff, q_);
}

TX_PARTITION_TYPE TileCoder::choose_tx_partition(BLOCK_SIZE bsize, TX_SIZE max_tx, int blk_row, int blk_col) {
  MACROBLOCKD* const xd = xd_;
  if (blk_row >= max_block_high(xd, bsize, 0) || blk_col >= max_block_wide(xd, bsize, 0)) return TX_PARTITION_NONE;
  const bool horz = allow_tx_horz_split(bsize, max_tx), vert = allow_tx_vert_split(bsize, max_tx);
  if (!horz && !vert) return TX_PARTITION_NONE;
  // The unit's residual (source - prediction), once for every partition.
  const struct macroblockd_plane* const pd = &xd->plane[AVM_PLANE_Y];
  const uint16_t* const dst = &pd->dst.buf[(blk_row * pd->dst.stride + blk_col) << MI_SIZE_LOG2];
  const ptrdiff_t off = dst - cm_->cur_frame->buf.y_buffer;
  const int y0 = static_cast<int>(off / pd->dst.stride), x0 = static_cast<int>(off % pd->dst.stride);
  const int uw = tx_size_wide[max_tx], uh = tx_size_high[max_tx];
  int16_t res[64 * 64];
  auto fill = [&](const auto& sp) {
    for (int r = 0; r < uh; r++) {
      const auto* const s = &sp[static_cast<size_t>(std::min(y0 + r, src_->ph - 1)) * src_->stride_y()];
      for (int c = 0; c < uw; c++) res[r * uw + c] = static_cast<int16_t>(s[std::min(x0 + c, src_->pw - 1)] - dst[r * pd->dst.stride + c]);
    }
  };
  if (src_->bit_depth > 8) fill(src_->hy);
  else fill(src_->y);
  // Rough symbol bits: "no partition" is the usual case.
  const struct {
    TX_PARTITION_TYPE p;
    bool ok;
    double bits;
  } cands[] = {{TX_PARTITION_NONE, true, 1},
               {TX_PARTITION_SPLIT, horz && vert, kSplitBits},
               {TX_PARTITION_HORZ, horz, kSplitBits},
               {TX_PARTITION_VERT, vert, kSplitBits}};
  TX_PARTITION_TYPE best = TX_PARTITION_NONE;
  double least = 1e300;
  for (const auto& k : cands) {
    if (!k.ok) continue;
    TXB_POS_INFO pos;
    TX_SIZE subs[MAX_TX_PARTITIONS];
    const int n = get_tx_partition_sizes(k.p, max_tx, &pos, subs, xd->error_info);
    double cost = 0.2 * k.bits;  // quantize()'s weight per bit
    for (int i = 0; i < n && cost < least; i++) {
      const int r = blk_row + pos.row_offset[i], c = blk_col + pos.col_offset[i];
      if (r < max_block_high(xd, bsize, 0) && c < max_block_wide(xd, bsize, 0))
        cost += luma_tx_cost(&res[(pos.row_offset[i] * uw + pos.col_offset[i]) * MI_SIZE], uw, subs[i]);
    }
    if (cost < least) least = cost, best = k.p;
  }
  return best;
}
double Encoder::coef_scale(int plane, TX_SIZE tx, MACROBLOCKD* xd) {
  std::atomic<double>& cached = scale_[plane > 0][tx];
  if (const double s = cached.load(std::memory_order_acquire); s > 0) return s;  // every call but the first
  std::lock_guard<std::mutex> lock(scale_mu_);
  if (const double s = cached.load(std::memory_order_relaxed); s > 0) return s;
  // Measure the DC gain of AVM's inverse transform and check the coefficient layout
  // (pos 1 = first horizontal frequency).
  AV2_COMMON* const cm = st_->cm();
  const int w = tx_size_wide[tx], h = tx_size_high[tx];
  std::vector<uint16_t> buf(static_cast<size_t>(w) * h);
  std::vector<tran_low_t> dq(MAX_TX_SQUARE);  // the inverse copies tx_size_2d[tx] values
  auto run = [&](int pos, int val) {
    std::fill(buf.begin(), buf.end(), uint16_t(128));
    std::fill(dq.begin(), dq.end(), 0);
    dq[pos] = val;
    av2_inverse_transform_block(xd, dq.data(), plane, DCT_DCT, tx, buf.data(), w,
                                av2_get_max_eob(tx), 0,
                                is_reduced_tx_set_used(cm, get_plane_type(plane)));
  };
  int val = 16;
  double gain = 0;
  for (;; val *= 2) {
    run(0, val);
    double sum = 0;
    int peak = 0;
    for (uint16_t v : buf) {
      sum += v - 128;
      peak = std::max(peak, std::abs(v - 128));
    }
    if (peak >= 64 || val >= (1 << 15)) {
      gain = sum / buf.size() / val;
      break;
    }
  }
  if (gain <= 0) throw std::runtime_error("inverse transform calibration failed");
  run(1, val);
  long horiz = 0, vert = 0;
  for (int x = 1; x < w; x++) horiz += std::abs(buf[x] - buf[x - 1]);
  for (int y = 1; y < h; y++) vert += std::abs(buf[y * w] - buf[(y - 1) * w]);
  if (horiz <= vert) throw std::runtime_error("unexpected coefficient layout");
  const double k = 1.0 / (gain * std::sqrt(static_cast<double>(w) * h));
  cached.store(k, std::memory_order_release);
  return k;
}

// decodeframe.c read_coeffs_tx_intra_block() + predict_and_reconstruct_intra_block(), with the
// coefficients chosen here: predict, transform and quantize the residual, write, reconstruct.
void TileCoder::code_intra_txb(int plane, int blk_row, int blk_col, TX_SIZE tx_size) {
  AV2_COMMON* const cm = cm_;
  MACROBLOCKD* const xd = xd_;
  bw_->pre_txb(plane, tx_size);
  av2_predict_intra_block_facade(cm, xd, plane, blk_col, blk_row, tx_size);
  struct macroblockd_plane* const pd = &xd->plane[plane];
  uint16_t* dst = &pd->dst.buf[(blk_row * pd->dst.stride + blk_col) << MI_SIZE_LOG2];
  residual_code(plane, tx_size, dst, pd->dst.stride, blk_row, blk_col);
}

int64_t TileCoder::intra_satd(int joint) {
  MACROBLOCKD* const xd = xd_;
  MB_MODE_INFO* const mbmi = xd->mi[0];
  const TX_SIZE tx = max_txsize_rect_lookup[mbmi->sb_type[PLANE_TYPE_Y]];
  av2_set_y_mode_and_delta_angle(joint, mbmi);
  std::memset(mbmi->tx_partition_type, TX_PARTITION_NONE, sizeof(mbmi->tx_partition_type));
  mbmi->tx_size = tx;
  mbmi->txb_idx = 0;
  bw_->pre_txb(AVM_PLANE_Y, tx);
  av2_predict_intra_block_facade(cm_, xd, AVM_PLANE_Y, 0, 0, tx);
  const struct macroblockd_plane* const pd = &xd->plane[AVM_PLANE_Y];
  const uint16_t* dst = pd->dst.buf;
  const ptrdiff_t off = dst - cm_->cur_frame->buf.y_buffer;
  const int y0 = static_cast<int>(off / pd->dst.stride), x0 = static_cast<int>(off % pd->dst.stride);
  const int w = std::min<int>(tx_size_wide[tx], src_->pw - x0), h = std::min<int>(tx_size_high[tx], src_->ph - y0);
  const size_t at = static_cast<size_t>(y0) * src_->stride_y() + x0;
  if (src_->bit_depth > 8) return satd_src(&src_->hy[at], src_->stride_y(), dst, pd->dst.stride, w & ~3, h & ~3);
  return satd_src(&src_->y[at], src_->stride_y(), dst, pd->dst.stride, w & ~3, h & ~3);
}

void TileCoder::predict_luma() {
  AV2_COMMON* const cm = cm_;
  MACROBLOCKD* const xd = xd_;
  MB_MODE_INFO* const mbmi = xd->mi[0];
  for (int ref = 0; ref < 1 + has_second_ref(mbmi); ++ref) {
    const RefCntBuffer* ref_buf = get_ref_frame_buf(cm, mbmi->ref_frame[ref]);
    const struct scale_factors* sf = get_ref_scale_factors_const(cm, mbmi->ref_frame[ref]);
    xd->block_ref_scale_factors[ref] = sf;
    av2_setup_pre_planes(xd, ref, &ref_buf->buf, xd->mi_row, xd->mi_col, sf, 1, &mbmi->chroma_ref_info);
  }
  av2_build_inter_predictors(cm, xd, 0, mbmi, nullptr, 0, 1, xd->plane[0].width, xd->plane[0].height,
                             xd->mi_col * MI_SIZE, xd->mi_row * MI_SIZE, mc_buf_, calc_subpel_params_and_extend);
}

void TileCoder::residual_of(int plane, TX_SIZE tx_size, const uint16_t* dst, int dst_stride, int16_t* res) const {
  const YV12_BUFFER_CONFIG& fb = cm_->cur_frame->buf;
  const uint16_t* base = plane == 0 ? fb.y_buffer : plane == 1 ? fb.u_buffer : fb.v_buffer;
  const ptrdiff_t off = dst - base;
  const int y0 = static_cast<int>(off / dst_stride), x0 = static_cast<int>(off % dst_stride);
  const int ss = plane ? src_->stride_uv() : src_->stride_y();
  const int sw = plane ? src_->pw / 2 : src_->pw, sh = plane ? src_->ph / 2 : src_->ph;
  const int w = tx_size_wide[tx_size], h = tx_size_high[tx_size];
  auto fill = [&](const auto& sp) {
    if (y0 + h <= sh && x0 + w <= sw) {  // inside the padded source: no clamping
      for (int r = 0; r < h; r++) {
        const auto* s = &sp[static_cast<size_t>(y0 + r) * ss + x0];
        const uint16_t* d = dst + r * dst_stride;
        for (int c = 0; c < w; c++) res[r * w + c] = static_cast<int16_t>(s[c] - d[c]);
      }
    } else {
      for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++) {
          const int sy = std::min(y0 + r, sh - 1), sx = std::min(x0 + c, sw - 1);
          res[r * w + c] = static_cast<int16_t>(sp[sy * ss + sx] - dst[r * dst_stride + c]);
        }
    }
  };
  if (src_->bit_depth > 8) fill(plane == 0 ? src_->hy : plane == 1 ? src_->hu : src_->hv);
  else fill(plane == 0 ? src_->y : plane == 1 ? src_->u : src_->v);
}

// Luma only: adding chroma changed anime A 1080p (120 frames, qindex 130 / 190) by +0.5% / -0.7% in
// bytes at -0.01 / +0.04 dB, at 0.87x the encode speed.
double TileCoder::pred_rd(double bound) {
  AV2_COMMON* const cm = cm_;
  MACROBLOCKD* const xd = xd_;
  MB_MODE_INFO* const mbmi = xd->mi[0];
  predict_luma();
  struct macroblockd_plane* const pd = &xd->plane[0];
  const BLOCK_SIZE bsize = mbmi->sb_type[PLANE_TYPE_Y];
  const TX_SIZE tx = max_txsize_rect_lookup[bsize];
  const bool tcq = tcq_enable(cm->features.tcq_mode, xd->lossless[mbmi->segment_id], 0, TX_CLASS_2D);
  int16_t res[64 * 64];
  tran_low_t coeff[1024];
  double total = 0;
  for (int r = 0; r < max_block_high(xd, bsize, 0); r += tx_size_high_unit[tx])
    for (int c = 0; c < max_block_wide(xd, bsize, 0); c += tx_size_wide_unit[tx]) {
      residual_of(0, tx, &pd->dst.buf[(r * pd->dst.stride + c) << MI_SIZE_LOG2], pd->dst.stride, res);
      const double scale = enc_.coef_scale(0, tx, xd);
      fwd_dct(res, tx_size_wide[tx], tx, scale, coeff);
      total += quantize_cost(coeff, tx, q_, DCT_DCT, nullptr, tcq) +
               dropped_cost(res, tx_size_wide[tx], tx, scale, coeff, q_);
      if (total > bound) return total;
    }
  return total;
}

void TileCoder::residual_code(int plane, TX_SIZE tx_size, uint16_t* dst, int dst_stride,
                            int blk_row, int blk_col) {
  AV2_COMMON* const cm = cm_;
  MACROBLOCKD* const xd = xd_;
  const int w = tx_size_wide[tx_size], h = tx_size_high[tx_size];
  int16_t res[64 * 64];
  residual_of(plane, tx_size, dst, dst_stride, res);
  // AVM layout keeps at most 32x32 coefficients, but the inverse reads tx_size_2d[tx] values.
  // Luma up to 16x16 (inter and intra) tries the types of its transform set and keeps the one of
  // least quantize() cost, a type other than DCT charged kTypeBits more (chroma is DCT only in the
  // sequence header). DCT/IDTX only, IDTX at any size and uncharged: anime clips -1.4% / -2.4%
  // BD-rate but a 1080p anime +0.3% at 0.84x the speed; up to 16x16 and 8 bits: -0.6% / -1.7%, -0.1%.
  // Every type (ADST, FLIPADST, H/V too), inter and intra: another -1.3% / -1.1%, -0.2%, 0.95x.
  constexpr double kTypeBits = 8;
  tran_low_t coeff[1024], qcoeff[1024], dqcoeff[MAX_TX_SQUARE];
  tran_low_t try_q[1024], try_dq[MAX_TX_SQUARE];
  std::memset(dqcoeff, 0, sizeof(tran_low_t) * tx_size_2d[tx_size]);
  const bool inter_luma = plane == 0 && is_inter_block(xd->mi[0], xd->tree_type);
  const int reduced_set = cm->features.reduced_tx_set_used;
  const TxSetType set = av2_get_ext_tx_set_type(tx_size, inter_luma, reduced_set);
  TX_TYPE types[TX_TYPES] = {DCT_DCT};
  int ntypes = 1;
  const bool long_side = set == EXT_TX_SET_LONG_SIDE_64 || set == EXT_TX_SET_LONG_SIDE_32;
  if (plane == 0 && !long_side && get_ext_tx_types(tx_size, inter_luma, reduced_set) > 1 && w <= 16 && h <= 16) {
    if (inter_luma) {
      for (int i = 0; i < av2_num_ext_tx_set[set]; i++)
        if (av2_ext_tx_inv[set][i] != DCT_DCT) types[ntypes++] = static_cast<TX_TYPE>(av2_ext_tx_inv[set][i]);
    } else if (reduced_set != 2) {  // intra: the set's order depends on the mode (av2_tx_idx_to_type)
      const PREDICTION_MODE mode = get_intra_mode(xd->mi[0], PLANE_TYPE_Y);
      for (int i = 0; i < av2_num_ext_tx_set_intra[set]; i++) {
        const TX_TYPE t = av2_tx_idx_to_type(i, set, mode, av2_size_class[tx_size]);
        if (t != DCT_DCT) types[ntypes++] = t;
      }
    }
  }
  const int nq = std::min(w, 32) * std::min(h, 32);
  // DCT first; where it leaves a residual, of the other types only the most compact one (least
  // coefficient magnitude) is quantized against it: quantizing all 16 halved the encode speed.
  int eob = -1;
  TX_TYPE chosen = DCT_DCT;
  double best = 1e300;
  for (int k = 0; k < ntypes; k++) {
    if (k == 1) {
      if (eob == 0) break;
      types[1] = most_compact(res, w, tx_size, types + 1, ntypes - 1, coeff), ntypes = 2;
    }
    const TX_TYPE type = types[k];
    // Inter IDTX with forward skip coding has no TCQ (av2_read_coeffs_txb_skip()).
    const bool tcq = tcq_enable(cm->features.tcq_mode, xd->lossless[xd->mi[0]->segment_id], plane,
                                tx_type_to_class[get_primary_tx_type(type)]) &&
                     !use_inter_fsc(cm, get_plane_type(plane), type, is_inter_block(xd->mi[0], xd->tree_type));
    if (type == DCT_DCT) fwd_dct(res, w, tx_size, enc_.coef_scale(plane, tx_size, xd_), coeff);
    double cost = 0;
    if (k == 1) {  // the other type is quantized only if it wins
      int e = 0;
      cost = quantize_cost(coeff, tx_size, q_, type, &e, tcq) + kTypeBits * 0.2;
      // An intra block with only a DC level codes no type: the decoder takes DCT.
      if ((!inter_luma && e == 1) || cost >= best) continue;
    }
    std::memset(try_dq, 0, sizeof(tran_low_t) * tx_size_2d[tx_size]);
    const int e = quantize(coeff, try_q, try_dq, tx_size, q_, plane, type,
                           k == 0 && ntypes > 1 ? &cost : nullptr, tcq, cm->seq_params.bit_depth);
    best = cost, eob = e, chosen = e ? type : DCT_DCT;
    std::memcpy(qcoeff, try_q, sizeof(tran_low_t) * nq);
    std::memcpy(dqcoeff, try_dq, sizeof(tran_low_t) * tx_size_2d[tx_size]);
  }
  bw_->coeffs(plane, blk_row, blk_col, tx_size, qcoeff, eob, chosen);
  if (eob) {
    const PLANE_TYPE plane_type = get_plane_type(plane);
    const uint8_t reduced = is_reduced_tx_set_used(cm, plane_type);
    const TX_TYPE tx_type = av2_get_tx_type(xd, plane_type, blk_row, blk_col, tx_size, reduced);
    if (tx_type != chosen) throw std::runtime_error("unexpected transform type");
    av2_inverse_transform_block(
        xd, dqcoeff, plane, tx_type, tx_size, dst, dst_stride, eob,
        replace_adst_by_ddt(cm->seq_params.enable_inter_ddt, cm->features.allow_screen_content_tools,
                            xd),
        reduced);
  }
}

Encoder::Encoder(const SeqConfig& c) : cfg_(c), st_(std::make_unique<CodecState>(c)) {}

double Encoder::sse_per_bit(int coded_qindex) {
  const double step = av2_ac_quant_QTX(coded_qindex, 0, 0, kQDepth) / static_cast<double>(1 << QUANT_TABLE_BITS) /
                      luma_scale(TX_8X8);
  return 0.025 * step * step;
}

double Encoder::luma_scale(TX_SIZE tx) {
  // Before any tile is set up: a copy of tile 0's xd on a blank block (the DCT reads its segment
  // and mode only for lossless and secondary transforms).
  MACROBLOCKD xd = *st_->xd(0);
  MB_MODE_INFO blank{};
  MB_MODE_INFO* mi = &blank;
  xd.mi = &mi;
  return coef_scale(0, tx, &xd);
}

int TileCoder::parse_tree(size_t& pos, int& leaf, int row, int col, int high, int wide) {
  if (pos >= sb_->tree.size()) throw std::invalid_argument("DecisionMap tree is truncated");
  const int id = static_cast<int>(nodes_.size());
  nodes_.push_back(Node{sb_->tree[pos++]});
  nodes_[id].row = row, nodes_[id].col = col, nodes_[id].high = high, nodes_[id].wide = wide;
  const int hh = high / 2, hw = wide / 2;
  if (nodes_[id].part == Part::Split && wide * MI_SIZE < 128) {
    // AV2 has square SPLIT only at 128x128 and up; below that the same quadrants are coded as
    // HORZ and then VERT on each half, in the same order.
    nodes_[id].part = Part::Horz;
    for (int half = 0; half < 2; half++) {
      const int v = static_cast<int>(nodes_.size());
      nodes_.push_back(Node{Part::Vert});
      Node& n = nodes_[v];
      n.row = row + half * hh, n.col = col, n.high = hh, n.wide = wide;
      nodes_[id].child[half] = v;
      for (int i = 0; i < 2; i++) {
        const int ch = parse_tree(pos, leaf, row + half * hh, col + i * hw, hh, hw);
        nodes_[v].child[i] = ch;
      }
    }
    return id;
  }
  const Part part = nodes_[id].part;
  const int n = num_children(part);
  if (n == 0) {
    if (leaf >= static_cast<int>(sb_->blocks.size()))
      throw std::invalid_argument("DecisionMap has fewer blocks than leaves");
    nodes_[id].leaf = leaf++;
  }
  for (int i = 0; i < n; i++) {
    const int r = part == Part::Horz ? row + i * hh : part == Part::Split ? row + (i >> 1) * hh : row;
    const int c = part == Part::Vert ? col + i * hw : part == Part::Split ? col + (i & 1) * hw : col;
    const int ch = parse_tree(pos, leaf, r, c, part == Part::Vert ? high : hh, part == Part::Horz ? wide : hw);
    nodes_[id].child[i] = ch;
  }
  return id;
}

int TileCoder::node_holding(int row, int col, int high, int wide) const {
  int node = 0;
  for (;;) {
    int next = -1;
    for (const int ch : nodes_[node].child) {
      if (ch < 0) break;
      const Node& n = nodes_[ch];
      if (row >= n.row && col >= n.col && row + high <= n.row + n.high && col + wide <= n.col + n.wide) next = ch;
    }
    if (next < 0) return node;
    node = next;
  }
}

// A 128x128 superblock's decision: SPLIT into the four 64x64 ones under it (raster order; one past
// the frame edge is a stand-in leaf the coder never reaches). Merging four equal inter blocks into
// one 128x128 coded the same content 30% larger in a test and changed nothing on 1080p sources.
static SbDecision compose_sb(const DecisionMap& dm, int sb_col, int sb_row) {
  const SbDecision* q[4];
  for (int i = 0; i < 4; i++) {
    const int c = 2 * sb_col + (i & 1), r = 2 * sb_row + (i >> 1);
    q[i] = c < dm.sb_cols && r < dm.sb_rows ? &dm.sbs[static_cast<size_t>(r) * dm.sb_cols + c] : nullptr;
  }
  SbDecision out;
  out.tree = {Part::Split};
  for (const SbDecision* s : q) {
    if (s) {
      out.tree.insert(out.tree.end(), s->tree.begin(), s->tree.end());
      out.blocks.insert(out.blocks.end(), s->blocks.begin(), s->blocks.end());
    } else {
      out.tree.push_back(Part::None);
      out.blocks.push_back(q[0]->blocks[0]);
    }
  }
  return out;
}

void TileCoder::code_row(BlockWriter& bw, int mi_row, int r, Wavefront& wf) {
  bw_ = &bw;
  const SeqConfig& cfg = enc_.cfg_;
  auto intra_lambda = [&](int q) { return lambda_q8(cfg.qindex_8bit(q)) << (2 * (cfg.bit_depth - 8)); };
  bw.set_intra_search([this](int joint) { return intra_satd(joint); }, intra_lambda(cm_->quant_params.base_qindex));
  if (!frame_is_intra_only(cm_))
    bw.set_inter_search([this](double bound) { return pred_rd(bound); });
  // decode_tile(), one row
  const TileInfo tile_info = xd_->tile;
  {
    av2_zero_left_context(xd_);
    av2_zero(xd_->ref_mv_bank);
    av2_zero(xd_->warp_param_bank);
    // The qindex the delta of the row's first superblock steps from is the end of the row above
    // (SymbolLog::delta_q() takes it at replay); here it only steers the CDF adaptation.
    if (mi_row > tile_info.mi_row_start)
      xd_->current_base_qindex = enc_.st_->sb_qindex(mi_row - cm_->mib_size, tile_info.mi_col_end - 1);
    int c = 0;
    for (int mi_col = tile_info.mi_col_start; mi_col < tile_info.mi_col_end;
         mi_col += cm_->mib_size, c++) {
      wf.wait(r, c);
      if (c == 0 && r > 0) *xd_->tile_ctx = wf.start(r);  // written once row r - 1 got this far
      av2_reset_is_mi_coded_map(xd_, cm_->mib_size);
      av2_set_sb_info(cm_, xd_, mi_row, mi_col, BRU_ACTIVE_SB);
      av2_reset_refmv_bank(cm_, xd_, &tile_info, mi_row, mi_col);
      const int q = enc_.st_->sb_qindex(mi_row, mi_col);
      if (q != q_) bw.set_intra_lambda(intra_lambda(q));
      q_ = q;
      const int sb_px = block_size_wide[cm_->sb_size];
      if (dm_.sb_size == sb_px) {
        if (gate_) gate_->wait_row(mi_row >> cm_->mib_size_log2);
        sb_ = &dm_.sbs[static_cast<size_t>(mi_row >> cm_->mib_size_log2) * dm_.sb_cols +
                       (mi_col >> cm_->mib_size_log2)];
      } else {  // 128x128 over 64x64 decisions: both decision rows under it
        const int row = mi_row >> cm_->mib_size_log2;
        if (gate_)  // rows may be published in any order (RowGate::publish_one())
          for (int r = 2 * row; r <= std::min(2 * row + 1, dm_.sb_rows - 1); r++) gate_->wait_row(r);
        composed_ = compose_sb(dm_, mi_col >> cm_->mib_size_log2, row);
        sb_ = &composed_;
      }
      nodes_.clear();
      size_t pos = 0;
      int leaf = 0;
      parse_tree(pos, leaf, mi_row, mi_col, cm_->mib_size, cm_->mib_size);
      // decode_partition_sb()
      xd_->tree_type = SHARED_PART;
      av2_reset_ptree_in_sbi(xd_->sbi, xd_->tree_type);
      code_partition(mi_row, mi_col, cm_->sb_size, xd_->sbi->ptree_root[av2_get_sdp_idx(xd_->tree_type)]);
      wf.finish(r, *xd_->tile_ctx);
    }
  }
  bw_ = nullptr;
}

int Encoder::tile_rows() const { return st_->cm()->tiles.rows; }

int Encoder::tile_threads() const {
  const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
  const int n = st_->num_tiles();
  return std::min({n, static_cast<int>(hw), cfg_.threads > 0 ? cfg_.threads : n});
}

std::vector<uint8_t> Encoder::encode(const Frame420& src, const DecisionMap& dm, bool key,
                                     const RowGate* gate) {
  begin(FrameSpec{key});
  std::vector<uint8_t> tu = make_obu(OBU_TEMPORAL_DELIMITER, {});
  const std::vector<uint8_t> obus = code(src, dm, gate);
  tu.insert(tu.end(), obus.begin(), obus.end());
  return tu;
}

std::vector<int> Encoder::begin(const FrameSpec& f) {
  key_ = f.key;
  st_->begin_frame(f);
  return st_->ref_display_hints();
}

std::vector<uint8_t> Encoder::show_existing(int display_order_hint) {
  const int slot = st_->slot_of(display_order_hint);
  if (slot < 0) throw std::runtime_error("show_existing: no slot holds that frame");
  // read_tilegroup_obu() + read_show_existing_frame(): no first-tile-group flag for a SEF, the
  // frame header's MFH / sequence header ids, the slot, derive_sef_order_hint = 1 (the hidden
  // frame's own order hint), then trailing bits.
  std::vector<uint8_t> buf(16, 0);
  avm_write_bit_buffer wb = {buf.data(), 0};
  avm_wb_write_uvlc(&wb, 0);  // cur_mfh_id
  avm_wb_write_uvlc(&wb, st_->cm()->seq_params.seq_header_id);
  avm_wb_write_literal(&wb, slot, st_->cm()->seq_params.ref_frames_log2);
  avm_wb_write_bit(&wb, 1);
  avm_wb_write_bit(&wb, 1);  // trailing one bit, then zeros to the byte boundary
  buf.resize(avm_wb_bytes_written(&wb));
  return make_obu(OBU_REGULAR_SEF, buf);
}

// read_global_motion() as our header codes it: an IDENTITY base, then each of the two nearest
// references' model (estimate_rotzoom(), IDENTITY where it finds none) before any block uses them.
static void set_global_motion(AV2_COMMON* cm, const Frame420& src, int* dist) {
  cm->base_global_motion_model = default_warp_params;
  cm->base_global_motion_distance = 1;
  for (int r = 0; r < cm->ref_frames_info.num_total_refs; r++) {
    const RefCntBuffer* const b = get_ref_frame_buf(cm, r);
    dist[r] = b->is_restricted ? 0
                               : get_relative_dist(&cm->seq_params.order_hint_info, cm->cur_frame->display_order_hint,
                                                   b->display_order_hint);
    // The two nearest references: estimating one halves the gain on a zoom (-4.0% against -6.6%)
    // for 3% of the speed.
    if (dist[r] == 0 || r >= 2) continue;
    WarpedMotionParams wm;
    if (estimate_rotzoom(src.y.data(), src.stride_y(), b->buf.y_buffer, b->buf.y_stride, cm->seq_params.bit_depth - 8,
                         cm->width, cm->height,
                         get_ref_scale_factors_const(cm, r), &wm))
      cm->global_motion[r] = wm;
  }
  std::memcpy(cm->cur_frame->global_motion, cm->global_motion, INTER_REFS_PER_FRAME * sizeof(WarpedMotionParams));
}

std::vector<uint8_t> Encoder::code(const Frame420& src, const DecisionMap& dm, const RowGate* gate) {
  const bool key = key_;
  if (src.width != cfg_.width || src.height != cfg_.height)
    throw std::invalid_argument("source frame size does not match the sequence");
  AV2_COMMON* const cm = st_->cm();
  // Sizes only: GpuEngine::launch() sets them before any row is published.
  // 64x64 decisions also serve 128x128 superblocks (compose_sb()).
  if ((dm.sb_size != block_size_wide[cm->seq_params.sb_size] &&
       !(dm.sb_size == 64 && cm->seq_params.sb_size == BLOCK_128X128)) ||
      dm.sb_cols != (cfg_.width + dm.sb_size - 1) / dm.sb_size ||
      dm.sb_rows != (cfg_.height + dm.sb_size - 1) / dm.sb_size ||
      dm.sbs.size() != static_cast<size_t>(dm.sb_cols) * dm.sb_rows)
    throw std::invalid_argument("DecisionMap does not match the frame");
  if (!key && cm->seq_params.enable_global_motion) set_global_motion(cm, src, st_->gm_ref_dist);

  // Tiles are independent (own contexts, CDFs and above-context range); within a tile the
  // superblock rows run in a wavefront (Wavefront), each logging its symbols, and the rows' logs
  // are then range coded in raster order with the tile's CDFs, from where begin_tile() left them.
  const int n = st_->num_tiles();
  struct TileOut {
    std::vector<uint8_t> buf;
    size_t size = 0;
    std::vector<SymbolLog> logs;  // per superblock row
    std::unique_ptr<Wavefront> wf;
  };
  std::vector<TileOut> out(n);
  const bool update = !cm->features.disable_cdf_update;
  struct Row {
    int t, r, mi_row;
  };
  std::vector<Row> rows;  // row-major over the tiles: a row only waits on rows listed before it
  for (int t = 0; t < n; t++) {
    st_->begin_tile(t);
    const TileInfo& ti = st_->xd(t)->tile;
    const int tile_rows = (ti.mi_row_end - ti.mi_row_start + cm->mib_size - 1) >> cm->mib_size_log2;
    const int tile_cols = (ti.mi_col_end - ti.mi_col_start + cm->mib_size - 1) >> cm->mib_size_log2;
    out[t].logs.resize(tile_rows);
    out[t].wf = std::make_unique<Wavefront>(*st_, t, tile_rows, tile_cols);
  }
  for (int r = 0;; r++) {
    const size_t before = rows.size();
    for (int t = 0; t < n; t++)
      if (r < static_cast<int>(out[t].logs.size()))
        rows.push_back({t, r, st_->xd(t)->tile.mi_row_start + (r << cm->mib_size_log2)});
    if (rows.size() == before) break;
  }
  auto code_row = [&](int k) {
    const Row& row = rows[k];
    Wavefront& wf = *out[row.t].wf;
    try {
      CodecState::RowWorker* const w = st_->acquire_row_worker();
      struct Release {
        CodecState& st;
        CodecState::RowWorker* w;
        ~Release() { st.release_row_worker(w); }
      } release{*st_, w};
      st_->begin_row(*w, row.t);
      if (row.r == 0) *w->ctx = *st_->xd(row.t)->tile_ctx;  // later rows: TileCoder::code_row()
      SymbolLog& log = out[row.t].logs[row.r] = SymbolLog(w->ctx, update);
      BlockWriter bw(*st_, w->xd, &log);
      TileCoder(*this, *st_, w->xd, w->mc_buf, src, dm, gate).code_row(bw, row.mi_row, row.r, wf);
    } catch (...) {
      wf.fail();  // rows waiting on this one stop
      throw;
    }
  };
  if (cfg_.threads == 1) {
    for (size_t k = 0; k < rows.size(); k++) code_row(static_cast<int>(k));
  } else {
    celv2::parallel(static_cast<int>(rows.size()), code_row);
  }
  celv2::parallel(n, [&](int t) {
    const TileInfo& ti = st_->xd(t)->tile;
    const size_t pels = static_cast<size_t>(ti.mi_row_end - ti.mi_row_start) *
                        (ti.mi_col_end - ti.mi_col_start) * MI_SIZE * MI_SIZE;
    out[t].buf.resize(pels * 4 + 4096);
    avm_writer w{};
    avm_start_encode(&w, out[t].buf.data());
    w.allow_update_cdf = update;
    int base_qindex = cm->quant_params.base_qindex;
    try {
      // tile_ctx: the CDFs as begin_tile() set them; the replay leaves the tile's final ones there.
      for (const SymbolLog& log : out[t].logs) log.replay(&w, st_->xd(t)->tile_ctx, &base_qindex);
    } catch (...) {
      avm_stop_encode(&w);
      throw;
    }
    avm_stop_encode(&w);
    out[t].size = w.pos;
  });
  st_->end_frame(src, sse_per_bit(cm->quant_params.base_qindex));

  // After the in-loop filters: the header carries the CDEF strengths chosen on the frame.
  std::vector<uint8_t> payload = write_frame_header(cfg_, *st_, key);
  // get_tile_buffers(): every tile but the last is preceded by its size minus 1, little-endian.
  for (int t = 0; t < n; t++) {
    if (t + 1 < n) {
      if (out[t].size < 1) throw std::runtime_error("empty tile");
      const uint64_t v = out[t].size - 1;
      for (int b = 0; b < kTileSizeBytes; b++) payload.push_back(static_cast<uint8_t>(v >> (8 * b)));
    }
    payload.insert(payload.end(), out[t].buf.begin(), out[t].buf.begin() + out[t].size);
  }

  std::vector<uint8_t> tu;
  if (key) {
    tu = make_obu(OBU_SEQUENCE_HEADER, write_sequence_header(cfg_));
    auto add = [&](const std::vector<uint8_t>& o) { tu.insert(tu.end(), o.begin(), o.end()); };
    if (cfg_.colour) add(content_interpretation_obu(cfg_.colour, cfg_.full_range));
    if (!cfg_.mdcv.empty()) add(metadata_obu(OBU_METADATA_TYPE_HDR_MDCV, cfg_.mdcv));
    if (!cfg_.cll.empty()) add(metadata_obu(OBU_METADATA_TYPE_HDR_CLL, cfg_.cll));
  }
  const auto frame = make_obu(key ? OBU_CLOSED_LOOP_KEY : OBU_REGULAR_TILE_GROUP, payload);
  tu.insert(tu.end(), frame.begin(), frame.end());
  return tu;
}

// decodeframe.c decode_partition(), shared luma/chroma tree (SDP is off).
void TileCoder::code_partition(int mi_row, int mi_col, BLOCK_SIZE bsize, PARTITION_TREE* ptree) {
  AV2_COMMON* const cm = cm_;
  MACROBLOCKD* const xd = xd_;
  const int ss_x = xd->plane[1].subsampling_x;
  const int ss_y = xd->plane[1].subsampling_y;
  const int hbs_w = mi_size_wide[bsize] / 2;
  const int hbs_h = mi_size_high[bsize] / 2;
  if (mi_row >= cm->mi_params.mi_rows || mi_col >= cm->mi_params.mi_cols) {
    av2_mark_block_as_pseudo_coded(xd, mi_row, mi_col, bsize, cm->sb_size);
    return;
  }
  if (ptree->parent) {
    if (ptree->parent->bsize == cm->sb_size) {
      if (ptree->parent->partition == PARTITION_VERT)
        ptree->sb_root_partition_info = SB_VERT_PARTITION;
      else if (ptree->parent->partition == PARTITION_HORZ ||
               ptree->parent->partition == PARTITION_SPLIT)
        ptree->sb_root_partition_info = SB_HORZ_OR_QUAD_PARTITION;
      else
        ptree->sb_root_partition_info = INVALID_INTRABC_SB_PARTITION;
    } else {
      ptree->sb_root_partition_info = ptree->parent->sb_root_partition_info;
    }
  }
  const int is_sb_root = bsize == cm->sb_size;
  if (is_sb_root) {
    xd->is_cfl_allowed_in_sdp = is_cfl_allowed_for_sdp(cm, xd, nullptr, PARTITION_NONE, bsize);
    ptree->is_cfl_allowed_for_this_chroma_partition = CFL_DISALLOWED_FOR_CHROMA;
    if (!frame_is_intra_only(cm)) {
      ptree->region_type = MIXED_INTER_INTRA_REGION;
      ptree->extended_sdp_allowed_flag = cm->seq_params.enable_extended_sdp;
    } else {
      ptree->region_type = INTRA_REGION;
      ptree->extended_sdp_allowed_flag = 0;
    }
    xd->sbi->sb_mv_precision = cm->features.fr_mv_precision;  // set_sb_mv_precision()
  }
  ptree->bsize = bsize;
  ptree->mi_row = mi_row;
  ptree->mi_col = mi_col;
  ptree->is_settled = 1;
  PARTITION_TREE* parent = ptree->parent;
  set_chroma_ref_info(xd->tree_type, mi_row, mi_col, ptree->index, bsize,
                      &ptree->chroma_ref_info, parent ? &parent->chroma_ref_info : nullptr,
                      parent ? parent->bsize : BLOCK_INVALID,
                      parent ? parent->partition : PARTITION_NONE, ss_x, ss_y);

  // The decision for this block: the node of the same block, else (a partition forced at a frame
  // edge, or a decision leaf this block lies inside) the split along the decision boundary this
  // block straddles, or none inside a leaf, coded with the leaf at the block's top left.
  const int high = mi_size_high[bsize], wide = mi_size_wide[bsize];
  const int node = node_holding(mi_row, mi_col, high, wide);
  const Node& nd = nodes_[node];
  PARTITION_TYPE wanted = PARTITION_NONE;
  if (nd.row == mi_row && nd.col == mi_col && nd.high == high && nd.wide == wide)
    wanted = to_avm(nd.part);
  else if (nd.leaf < 0)
    wanted = mi_col < nd.col + nd.wide / 2 && mi_col + wide > nd.col + nd.wide / 2 ? PARTITION_VERT : PARTITION_HORZ;
  const PARTITION_TYPE partition =
      !is_partition_point(bsize) ? PARTITION_NONE
                                 : bw_->partition(mi_row, mi_col, bsize, ptree, wanted);

  ptree->is_cfl_allowed_for_this_chroma_partition |=
      is_cfl_allowed_for_sdp(cm, xd, nullptr, partition, bsize);
  const CFL_ALLOWED_FOR_SDP_TYPE is_cfl_allowed_in_sdp =
      ptree->is_cfl_allowed_for_this_chroma_partition;
  ptree->partition = partition;
  if (!is_sb_root && parent) {
    ptree->extended_sdp_allowed_flag =
        parent->extended_sdp_allowed_flag
            ? is_extended_sdp_allowed(cm->seq_params.enable_extended_sdp, parent->bsize,
                                      parent->partition)
            : 0;
    if (!frame_is_intra_only(cm) && ptree->partition && parent->region_type != INTRA_REGION &&
        ptree->extended_sdp_allowed_flag && is_bsize_allowed_for_extended_sdp(bsize, partition))
      throw std::runtime_error("extended SDP region type is not supported");
    ptree->region_type = frame_is_intra_only(cm) ? INTRA_REGION : parent->region_type;
  }
  const int n_sub = partition == PARTITION_SPLIT ? 4 : partition == PARTITION_NONE ? 0 : 2;
  if (partition != PARTITION_NONE && partition != PARTITION_SPLIT &&
      partition != PARTITION_HORZ && partition != PARTITION_VERT)
    throw std::runtime_error("unexpected partition type");
  for (int i = 0; i < n_sub; i++) {
    ptree->sub_tree[i] = av2_alloc_ptree_node(ptree, i);
    ptree->sub_tree[i]->is_cfl_allowed_for_this_chroma_partition = is_cfl_allowed_in_sdp;
  }
  if (partition == PARTITION_NONE) xd->is_cfl_allowed_in_sdp = is_cfl_allowed_in_sdp;

  const BLOCK_SIZE subsize = get_partition_subsize(bsize, partition);
  if (subsize == BLOCK_INVALID) throw std::runtime_error("invalid partition subsize");
  switch (partition) {
    case PARTITION_NONE: {
      const int at = node_holding(mi_row, mi_col, 1, 1);  // the leaf at the block's top left
      code_block(mi_row, mi_col, partition, subsize, ptree, sb_->blocks[nodes_[at].leaf]);
      break;
    }
    case PARTITION_HORZ:
      code_partition(mi_row, mi_col, subsize, ptree->sub_tree[0]);
      code_partition(mi_row + hbs_h, mi_col, subsize, ptree->sub_tree[1]);
      break;
    case PARTITION_VERT:
      code_partition(mi_row, mi_col, subsize, ptree->sub_tree[0]);
      code_partition(mi_row, mi_col + hbs_w, subsize, ptree->sub_tree[1]);
      break;
    default:  // PARTITION_SPLIT
      code_partition(mi_row, mi_col, subsize, ptree->sub_tree[0]);
      code_partition(mi_row, mi_col + hbs_w, subsize, ptree->sub_tree[1]);
      code_partition(mi_row + hbs_h, mi_col, subsize, ptree->sub_tree[2]);
      code_partition(mi_row + hbs_h, mi_col + hbs_w, subsize, ptree->sub_tree[3]);
      break;
  }
  update_ext_partition_context(xd, mi_row, mi_col, subsize, bsize, partition);
}

// decodeframe.c parse_decode_block() + decode_token_recon_block(), intra path.
// The block's MVs for later frames' temporal MVs: decodemv.c intra_copy_frame_mvs() in intra
// frames, decodeframe.c copy_frame_mvs_inter_block() in inter frames.
static void copy_frame_mvs(AV2_COMMON* cm, MACROBLOCKD* xd, int x_mis, int y_mis) {
  if (!cm->seq_params.order_hint_info.enable_ref_frame_mvs) return;
  MB_MODE_INFO* const mi = xd->mi[0];
  if (!frame_is_intra_only(cm)) {
    if (enable_refined_mvs_in_tmvp(cm, xd, mi))
      av2_copy_frame_refined_mvs(cm, xd, mi, xd->mi_row, xd->mi_col, x_mis, y_mis);
    else
      av2_copy_frame_mvs(cm, xd, mi, xd->mi_row, xd->mi_col, x_mis, y_mis);
    return;
  }
  const int cols = ROUND_POWER_OF_TWO(cm->mi_params.mi_cols, TMVP_SHIFT_BITS);
  MV_REF* row = cm->cur_frame->mvs + (xd->mi_row >> TMVP_SHIFT_BITS) * cols + (xd->mi_col >> TMVP_SHIFT_BITS);
  for (int h = 0; h < ROUND_POWER_OF_TWO(y_mis, TMVP_SHIFT_BITS); h++, row += cols)
    for (int w = 0; w < ROUND_POWER_OF_TWO(x_mis, TMVP_SHIFT_BITS); w++)
      row[w].ref_frame[0] = row[w].ref_frame[1] = NONE_FRAME;
}

void TileCoder::code_block(int mi_row, int mi_col, PARTITION_TYPE partition, BLOCK_SIZE bsize,
                         PARTITION_TREE* parent, const BlockDecision& d) {
  AV2_COMMON* const cm = cm_;
  MACROBLOCKD* const xd = xd_;
  const int num_planes = av2_num_planes(cm);
  const int bw = mi_size_wide[bsize];
  const int bh = mi_size_high[bsize];
  const int x_mis = AVMMIN(bw, cm->mi_params.mi_cols - mi_col);
  const int y_mis = AVMMIN(bh, cm->mi_params.mi_rows - mi_row);
  // decode_mbmi_block()
  set_offsets(cm, xd, bsize, mi_row, mi_col, bw, bh, x_mis, y_mis, parent, 0);
  MB_MODE_INFO* const mbmi = xd->mi[0];
  mbmi->partition = partition;
  mbmi->region_type = parent->region_type;
  mbmi->tree_type = xd->tree_type;
  mbmi->sb_root_partition_info = parent->sb_root_partition_info;
  mbmi->local_rest_type = 1;
  mbmi->local_ccso_blk_flag = 1;
  mbmi->local_gdf_mode = 1;
  // av2_read_mode_info()
  mbmi->use_intrabc[xd->tree_type == CHROMA_PART] = 0;
  mbmi->warpmv_with_mvd_flag = 0;
  if (xd->tree_type == SHARED_PART) mbmi->sb_type[PLANE_TYPE_UV] = mbmi->sb_type[PLANE_TYPE_Y];
  if (frame_is_intra_only(cm))
    bw_->intra_frame_mode_info(d);
  else
    bw_->inter_frame_mode_info(d);

  // parse_decode_block(): transform size. With TX_MODE_SELECT an inter block codes each unit's
  // partition once predicted (code_inter_block); an intra block codes none (read_tx_size()).
  xd->reduced_tx_part_set = cm->seq_params.reduced_tx_part_set;
  std::memset(mbmi->tx_partition_type, TX_PARTITION_NONE, sizeof(mbmi->tx_partition_type));
  if (block_signals_txsize(bsize) && cm->features.tx_mode == TX_MODE_SELECT && !is_inter_block(mbmi, xd->tree_type))
    bw_->tx_partition(max_txsize_rect_lookup[bsize], 0, 0, TX_PARTITION_NONE);
  else
    mbmi->tx_size = block_signals_txsize(bsize) ? tx_size_from_tx_mode(bsize, cm->features.tx_mode)
                                                : max_txsize_rect_lookup[bsize];

  mbmi->partition = partition;
  if (is_inter_block(mbmi, xd->tree_type)) {
    code_inter_block(bsize);
    copy_frame_mvs(cm, xd, x_mis, y_mis);
    av2_mark_block_as_coded(xd, bsize, cm->sb_size);
    return;
  }
  // decode_token_recon_block(), intra
  const int plane_start = get_partition_plane_start(xd->tree_type);
  const int plane_end = get_partition_plane_end(xd->tree_type, num_planes);
  av2_init_txk_skip_array(cm, xd->mi_row, xd->mi_col, bsize, 0, xd->tree_type,
                          &mbmi->chroma_ref_info, plane_start, plane_end);
  xd->cfl.use_dc_pred_cache = 0;
  xd->cfl.dc_pred_is_cached[0] = 0;
  xd->cfl.dc_pred_is_cached[1] = 0;
  const int max_blocks_wide = max_block_wide(xd, bsize, 0);
  const int max_blocks_high = max_block_high(xd, bsize, 0);
  const int mu_blocks_wide = AVMMIN(max_blocks_wide, mi_size_wide[BLOCK_64X64]);
  const int mu_blocks_high = AVMMIN(max_blocks_high, mi_size_high[BLOCK_64X64]);
  const int mu128_wide = mi_size_wide[BLOCK_128X128];
  const int mu128_high = mi_size_high[BLOCK_128X128];
  for (int row128 = 0; row128 < max_blocks_high; row128 += mu128_high) {
    for (int col128 = 0; col128 < max_blocks_wide; col128 += mu128_wide) {
      for (int row = row128; row < AVMMIN(row128 + mu128_high, max_blocks_high);
           row += mu_blocks_high) {
        for (int col = col128; col < AVMMIN(col128 + mu128_wide, max_blocks_wide);
             col += mu_blocks_wide) {
          for (int plane = plane_start; plane < plane_end; ++plane) {
            const struct macroblockd_plane* const pd = &xd->plane[plane];
            const int ss_x = pd->subsampling_x;
            const int ss_y = pd->subsampling_y;
            const BLOCK_SIZE plane_bsize = get_mb_plane_block_size(xd, mbmi, plane, ss_x, ss_y);
            const int plane_unit_height = get_plane_tx_unit_height(xd, plane_bsize, plane, row, ss_y);
            const int plane_unit_width = get_plane_tx_unit_width(xd, plane_bsize, plane, col, ss_x);
            if (plane == AVM_PLANE_Y) {
              const TX_SIZE max_tx_size = max_txsize_rect_lookup[plane_bsize];
              get_tx_partition_sizes(mbmi->tx_partition_type[0], max_tx_size, &mbmi->txb_pos,
                                     mbmi->sub_txs, xd->error_info);
              for (int txb_idx = 0; txb_idx < mbmi->txb_pos.n_partitions; ++txb_idx) {
                const TX_SIZE tx_size = mbmi->sub_txs[txb_idx];
                mbmi->txb_idx = txb_idx;
                const int blk_row = row + mbmi->txb_pos.row_offset[txb_idx];
                const int blk_col = col + mbmi->txb_pos.col_offset[txb_idx];
                if (blk_row >= plane_unit_height || blk_col >= plane_unit_width) continue;
                code_intra_txb(plane, blk_row, blk_col, tx_size);
              }
            } else {
              if (!xd->is_chroma_ref) break;
              const TX_SIZE tx_size = av2_get_tx_size(plane, xd);
              const int stepr = tx_size_high_unit[tx_size];
              const int stepc = tx_size_wide_unit[tx_size];
              const bool need_parsing = !skip_parsing_recon(row, col, ss_x, ss_y, false);
              for (int blk_row = row >> ss_y; blk_row < plane_unit_height; blk_row += stepr) {
                for (int blk_col = col >> ss_x; blk_col < plane_unit_width; blk_col += stepc) {
                  if (!need_parsing) continue;
                  // The decoder predicts/reconstructs chroma at (blk & 0xf0); with 64x64
                  // superblocks the chroma transform covers the block, so both positions agree.
                  if ((blk_row & 0xf0) != blk_row || (blk_col & 0xf0) != blk_col)
                    throw std::runtime_error("chroma transform block inside a 64x64 unit");
                  code_intra_txb(plane, blk_row, blk_col, tx_size);
                }
              }
            }
          }
        }
      }
    }
  }
  copy_frame_mvs(cm, xd, x_mis, y_mis);
  av2_mark_block_as_coded(xd, bsize, cm->sb_size);
}
