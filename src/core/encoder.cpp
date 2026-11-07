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
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "bitstream/frame_header.h"
#include "bitstream/obu.h"
#include "core/film_grain.h"
#include "core/profile.h"
#include "core/psy.h"
#include "core/quant.h"
#include "core/global_motion.h"
#include "gpu/sb_blocks.h"
#include "core/threads.h"
#include "core/ist.h"
#include "core/transform.h"
#include "core/wide.h"
#include "gpu/intra_common.h"
#ifdef CELV2_ORACLE
#include "oracle.h"
extern "C" {
#include "av2/common/hr_coding.h"
#include "av2/common/scan.h"
#include "av2/common/txb_common.h"
}
#include <atomic>
#include <limits>
#endif

extern "C" {
#include "av2/common/idct.h"
#include "av2/common/pred_common.h"
#include "av2/common/quant_common.h"
#include "avm_dsp/bitwriter_buffer.h"
#include "av2/common/reconinter.h"
#include "av2/common/reconintra.h"
#include "av2/common/secondary_tx.h"
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
  void trial_64(int mi_row, int mi_col, PARTITION_TREE* ptree);
  double block_sse(int p, int mi_row, int mi_col) const;
  int sb_row_ = 0, sb_col_ = 0, pick_[4] = {};
  bool trying_ = false;
  void code_block(int mi_row, int mi_col, PARTITION_TYPE partition, BLOCK_SIZE bsize,
                  PARTITION_TREE* ptree, const BlockDecision& d);
  void code_intra_txb(int plane, int blk_row, int blk_col, TX_SIZE tx_size);
  // BlockWriter::IntraCost: the current block's luma prediction with joint mode `joint` (its
  // largest transform block, predicted into dst as code_intra_txb does), SATD against the source.
  int64_t intra_satd(int joint);
  void prepare_uv(UV_PREDICTION_MODE uv);  // mbmi's chroma mode set to uv; CfL's luma made ready
  double intra_uv_rd(UV_PREDICTION_MODE uv);
  // BlockWriter::IntraRd: intra_satd()'s prediction, its residual quantized (luma_tx_cost()).
  double intra_rd(int joint);
  // BlockWriter::PaletteSource: the block's source luma as its palette where it has 2-8 values (exact).
  int palette_source(uint16_t* colors);
  // BlockWriter::IntraBcSearch: valid vectors to exact repeats of the block's source (8x8 and up,
  // IntrabcIndex::repeats()), then near copies along its period (near_copies()); BlockWriter::IntraBcRd: the
  // copy's luma_tx_cost().
  int intrabc_candidates(MV* dvs, int max);
  MV near_dv_[3];  // intrabc_candidates()'s copies that are not exact
  int near_n_ = 0;
  double intrabc_rd(MV dv);
  std::vector<uint8_t> color_map_ = std::vector<uint8_t>(MAX_SB_SQUARE);  // xd->plane[0].color_index_map
  // Both chroma planes' SATD of the current intra block predicted with chroma mode uv (BlockWriter::UvCost).
  int64_t intra_uv_satd(UV_PREDICTION_MODE uv);
  // dec_build_inter_predictor()'s inter-intra blend of plane (nothing unless mbmi is inter-intra).
  void blend_interintra(int plane);
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
  // keep: the chosen partition's reconstruction stays in the frame (recon_luma_for_cfl()).
  TX_PARTITION_TYPE choose_intra_tx_partition(BLOCK_SIZE bsize, bool keep = false);
  // Before an MHCCP prediction of a block past the frame's edge: what av2_predict_intra_block_facade()
  // reads of the block's luma but does not fetch.
  void mhccp_fill_outside(int blk_col, int blk_row, TX_SIZE tx);
  // --cfl-rd: the block's luma reconstructed (DCT, as choose_intra_tx_partition() quantizes) for the
  // CfL / MHCCP estimates, once a block; its transform partition kept for coding (cfl_txp_).
  void recon_luma_for_cfl();
  int cfl_txp_ = -1;  // the partition recon_luma_for_cfl() chose (-1: none yet)
  bool cfl_recon_ = false;  // recon_luma_for_cfl() ran for this block
  // About quantize()'s cost of the luma DCT of residual tx (quantize_cost).
  // dst (with its stride): the block's prediction, for --psy-rd's term (null: no term).
  double luma_tx_cost(const int16_t* res, int stride, TX_SIZE tx, const uint16_t* dst = nullptr, int dst_stride = 0);
  // --psy-rd: the strength times the weight of the luma tx block whose prediction is at dst (0: off).
  double psy_scale_at(const uint16_t* dst, int dst_stride, TX_SIZE tx) const;
  // --psy-rd's term (core/psy.h) of a luma tx block: res its residual (stride res_stride), dst its
  // prediction (the source is their sum), coeff / dq its coefficients and dequantized levels.
  double psy_cost(const int16_t* res, int res_stride, const uint16_t* dst, int dst_stride, TX_SIZE tx,
                  const tran_low_t* coeff, const tran_low_t* dq);
  // Residual of the prediction already in dst: transform, quantize, write, reconstruct into dst.
  // Where CCTX is allowed: U and V of one transform block (both predicted into dst), coded together.
  void residual_code_uv(TX_SIZE tx_size, int blk_row, int blk_col);
  void residual_code(int plane, TX_SIZE tx_size, uint16_t* dst, int dst_stride, int blk_row,
                     int blk_col);
  // A luma block's secondary transform: prim holds the primary coefficients of `type`, quantized
  // into qcoeff / dqcoeff with end of block eob. Where an IST codes cheaper, these and type take it.
  void ist_search(TX_SIZE tx, const tran_low_t* prim, int& eob, TX_TYPE& type, tran_low_t* qcoeff,
                  tran_low_t* dqcoeff);

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
  double trellis_ = 0;  // --psy-trellis: the strength times the current block's weight (luma only)
  std::vector<Node> nodes_;
#ifdef CELV2_ORACLE  // tools/oracle/tile_import.inc
  const celv2::oracle::Frame* imp_frame_ = nullptr;  // from L3 on
  bool imp_again_ = false;  // reconstructing what is already written
  std::unique_ptr<FRAME_CONTEXT> imp_fc_;  // Qc: the superblock's CDFs
  double imp_rate_[2] = {};  // real_quant()'s last bits, by plane > 0 + V
  TXB_CTX imp_ctx_[2] = {};  // and the contexts
  void real_check(int plane, int blk_row, int blk_col, TX_SIZE tx, TX_TYPE type, const tran_low_t* coeff, const tran_low_t* q, int eob);
  bool real_on(int plane, TX_TYPE type, bool tcq) const;
  int real_quant(int plane, int blk_row, int blk_col, TX_SIZE tx, TX_TYPE type, const tran_low_t* coeff, tran_low_t* q, tran_low_t* dq, bool tcq, bool eob_u);
  const OracleTx* imp_tx(int plane, int blk_row, int blk_col, TX_SIZE tx) const;
  TX_PARTITION_TYPE import_partition(BLOCK_SIZE bsize, TX_SIZE max_tx, int blk_row, int blk_col);
  void import_forward(const int16_t* res, int w, TX_SIZE tx, TX_TYPE primary, int plane, tran_low_t* out);
  void import_residual(int plane, TX_SIZE tx, uint16_t* dst, int dst_stride, int blk_row, int blk_col);
  void import_residual_uv(TX_SIZE tx, int blk_row, int blk_col);
  void import_skip_block(BLOCK_SIZE bsize, int plane_start, int plane_end);
  int import_levels(const OracleTx* t, TX_SIZE tx, tran_low_t* q, tran_low_t* dq) const;
#endif
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
  // predict_inter_block(): one reference, or two averaged; IntraBC reads the frame itself.
  for (int ref = 0; ref < 1 + has_second_ref(mbmi) && !is_intrabc_block(mbmi, xd->tree_type); ++ref) {
    const RefCntBuffer* ref_buf = get_ref_frame_buf(cm, mbmi->ref_frame[ref]);
    const struct scale_factors* sf = get_ref_scale_factors_const(cm, mbmi->ref_frame[ref]);
    xd->block_ref_scale_factors[ref] = sf;
    av2_setup_pre_planes(xd, ref, &ref_buf->buf, xd->mi_row, xd->mi_col, sf, num_planes,
                         &mbmi->chroma_ref_info);
  }
  if (mbmi->refinemv_flag || mbmi->morph_pred) throw std::runtime_error("unsupported inter prediction tool");
  for (int plane = 0; plane < num_planes; ++plane) {
    if (plane && !xd->is_chroma_ref) break;
    av2_build_inter_predictors(cm, xd, plane, mbmi, nullptr, 0, 1, xd->plane[plane].width,
                               xd->plane[plane].height, xd->mi_col * MI_SIZE,
                               xd->mi_row * MI_SIZE, mc_buf_, calc_subpel_params_and_extend);
    blend_interintra(plane);
  }
#ifdef CELV2_ORACLE
  if (mbmi->skip_txfm[0]) return import_skip_block(bsize, plane_start, plane_end);  // L5 on
#endif
  // parse_decode_block(): each max-size luma transform unit's partition, by the cost of its parts.
  if (cm->features.tx_mode == TX_MODE_SELECT && block_signals_txsize(bsize)) {
    const TX_SIZE max_tx = max_txsize_rect_lookup[bsize];
    for (int idy = 0; idy < mi_size_high[bsize]; idy += tx_size_high_unit[max_tx])
      for (int idx = 0; idx < mi_size_wide[bsize]; idx += tx_size_wide_unit[max_tx])
#ifdef CELV2_ORACLE
        if (imp_frame_) bw_->tx_partition(max_tx, idy, idx, import_partition(bsize, max_tx, idy, idx)); else
#endif
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
            const bool cctx = plane && is_cctx_allowed(cm, xd);
            if (cctx && plane == AVM_PLANE_U) continue;  // with V
            for (int blk_row = row >> ss_y; blk_row < plane_unit_height; blk_row += bh_var_tx)
              for (int blk_col = col >> ss_x; blk_col < plane_unit_width; blk_col += bw_var_tx) {
                // decode_reconstruct_tx()
                const int txp_index = av2_get_txb_size_index(plane_bsize, blk_row, blk_col);
                if (blk_row >= max_block_high(xd, plane_bsize, plane) ||
                    blk_col >= max_block_wide(xd, plane_bsize, plane))
                  continue;
                if (cctx) {
                  residual_code_uv(max_tx_size, blk_row, blk_col);
                  continue;
                }
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

double TileCoder::psy_scale_at(const uint16_t* dst, int dst_stride, TX_SIZE tx) const {
  const celv2::AqMap* const m = enc_.map_.get();
  if (!m || enc_.cfg_.psy_rd <= 0) return 0;
  const ptrdiff_t off = dst - cm_->cur_frame->buf.y_buffer;
  const int y0 = static_cast<int>(off / dst_stride), x0 = static_cast<int>(off % dst_stride);
  return enc_.cfg_.psy_rd * celv2::psy_weight(*m, x0, y0, tx_size_wide[tx], tx_size_high[tx],
                                              enc_.cfg_.psy_lines ? celv2::PsyRule::Line : celv2::PsyRule::Texture);
}

double TileCoder::psy_cost(const int16_t* res, int res_stride, const uint16_t* dst, int dst_stride, TX_SIZE tx,
                           const tran_low_t* coeff, const tran_low_t* dq) {
  const double k = psy_scale_at(dst, dst_stride, tx);
  if (k <= 0) return 0;
  const int w = tx_size_wide[tx], h = tx_size_high[tx];
  int16_t s[64 * 64];
  for (int r = 0; r < h; r++)
    for (int c = 0; c < w; c++) s[r * w + c] = static_cast<int16_t>(res[r * res_stride + c] + dst[r * dst_stride + c]);
  tran_low_t S[1024];
  fwd_dct(s, w, tx, enc_.coef_scale(0, tx, xd_), S);
  const double step = av2_ac_quant_QTX(q_, 0, 0, kQDepth) / static_cast<double>(1 << (QUANT_TABLE_BITS + av2_get_tx_scale(tx)));
  return k * celv2::kPsyScale * celv2::psy_term(S, coeff, dq, std::min(w, 32) * std::min(h, 32), step);
}

double TileCoder::luma_tx_cost(const int16_t* res, int stride, TX_SIZE tx, const uint16_t* dst, int dst_stride) {
  tran_low_t coeff[1024];
  const double scale = enc_.coef_scale(0, tx, xd_);
  fwd_dct(res, stride, tx, scale, coeff);
  if (dst && psy_scale_at(dst, dst_stride, tx) > 0) {  // the levels kept, for the term
    tran_low_t q[1024], dq[MAX_TX_SQUARE];
    std::memset(dq, 0, sizeof(tran_low_t) * tx_size_2d[tx]);
    double cost = 0;
    quantize(coeff, q, dq, tx, q_, 0, DCT_DCT, &cost, tcq_enable(cm_->features.tcq_mode, 0, AVM_PLANE_Y, TX_CLASS_2D),
             cm_->seq_params.bit_depth, trellis_);
    return cost + dropped_cost(res, stride, tx, scale, coeff, q_) + psy_cost(res, stride, dst, dst_stride, tx, coeff, dq);
  }
  return quantize_cost(coeff, tx, q_, DCT_DCT, nullptr,
                       tcq_enable(cm_->features.tcq_mode, 0, AVM_PLANE_Y, TX_CLASS_2D), trellis_) +
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
        cost += luma_tx_cost(&res[(pos.row_offset[i] * uw + pos.col_offset[i]) * MI_SIZE], uw, subs[i],
                             &dst[(pos.row_offset[i] * pd->dst.stride + pos.col_offset[i]) * MI_SIZE], pd->dst.stride);
    }
    if (cost < least) least = cost, best = k.p;
  }
  return best;
}
// The intra block's luma transform partition (its mode chosen): each partition's transform blocks
// predicted in turn as code_intra_txb() predicts them, from the parts before them quantized (DCT)
// and reconstructed, and costed by that quantization.
TX_PARTITION_TYPE TileCoder::choose_intra_tx_partition(BLOCK_SIZE bsize, bool keep) {
  MACROBLOCKD* const xd = xd_;
  MB_MODE_INFO* const mbmi = xd->mi[0];
  const TX_SIZE max_tx = max_txsize_rect_lookup[bsize];
  const bool horz = allow_tx_horz_split(bsize, max_tx), vert = allow_tx_vert_split(bsize, max_tx);
  if (!horz && !vert && !keep) return TX_PARTITION_NONE;
  struct macroblockd_plane* const pd = &xd->plane[AVM_PLANE_Y];
  uint16_t* const dst = pd->dst.buf;
  const int stride = pd->dst.stride;
  const ptrdiff_t off = dst - cm_->cur_frame->buf.y_buffer;
  const int y0 = static_cast<int>(off / stride), x0 = static_cast<int>(off % stride);
  const int bw = std::min<int>(block_size_wide[bsize], src_->pw - x0), bh = std::min<int>(block_size_high[bsize], src_->ph - y0);
  auto src_at = [&](int r, int c) -> uint16_t {
    const size_t at = static_cast<size_t>(std::min(y0 + r, src_->ph - 1)) * src_->stride_y() + std::min(x0 + c, src_->pw - 1);
    return src_->bit_depth > 8 ? src_->hy[at] : src_->y[at];
  };
  auto put_source = [&](int r0, int c0, int h, int w) {
    for (int r = r0; r < std::min(r0 + h, bh); r++)
      for (int c = c0; c < std::min(c0 + w, bw); c++) dst[r * stride + c] = src_at(r, c);
  };
  std::vector<uint16_t> saved(static_cast<size_t>(bw) * bh);
  for (int r = 0; r < bh; r++) std::copy(dst + r * stride, dst + r * stride + bw, saved.begin() + static_cast<size_t>(r) * bw);
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
  int16_t res[64 * 64];
  // One candidate's cost, its parts reconstructed in turn (stopping once past `bound`).
  auto run = [&](const auto& k, double bound) {
    TXB_POS_INFO pos;
    TX_SIZE subs[MAX_TX_PARTITIONS];
    const int n = get_tx_partition_sizes(k.p, max_tx, &pos, subs, xd->error_info);
    mbmi->tx_partition_type[0] = k.p;
    put_source(0, 0, bh, bw);  // nothing of the last candidate's reconstruction
    double cost = 0.2 * k.bits;  // quantize()'s weight per bit
    for (int i = 0; i < n && cost < bound; i++) {
      const int br = pos.row_offset[i], bc = pos.col_offset[i];
      if (br >= max_block_high(xd, bsize, 0) || bc >= max_block_wide(xd, bsize, 0)) continue;
      const int tw = tx_size_wide[subs[i]], th = tx_size_high[subs[i]], r0 = br * MI_SIZE, c0 = bc * MI_SIZE;
      mbmi->txb_idx = i;
      mbmi->tx_size = subs[i];
      bw_->pre_txb(AVM_PLANE_Y, subs[i]);
      av2_predict_intra_block_facade(cm_, xd, AVM_PLANE_Y, bc, br, subs[i]);
      for (int r = 0; r < th; r++)
        for (int c = 0; c < tw; c++)
          res[r * tw + c] = static_cast<int16_t>(r0 + r < bh && c0 + c < bw ? src_at(r0 + r, c0 + c) - dst[(r0 + r) * stride + c0 + c] : 0);
      // DCT, quantized as residual_code() quantizes, and reconstructed for the parts after it.
      tran_low_t coeff[1024], qcoeff[1024], dqcoeff[MAX_TX_SQUARE];
      std::memset(dqcoeff, 0, sizeof(tran_low_t) * tx_size_2d[subs[i]]);
      const double scale = enc_.coef_scale(0, subs[i], xd);
      fwd_dct(res, tw, subs[i], scale, coeff);
      double part = 0;
      const int eob = quantize(coeff, qcoeff, dqcoeff, subs[i], q_, 0, DCT_DCT, &part,
                               tcq_enable(cm_->features.tcq_mode, 0, AVM_PLANE_Y, TX_CLASS_2D), cm_->seq_params.bit_depth, trellis_);
      cost += part + dropped_cost(res, tw, subs[i], scale, coeff, q_) +
              psy_cost(res, tw, &dst[r0 * stride + c0], stride, subs[i], coeff, dqcoeff);
      if (eob)
        av2_inverse_transform_block(xd, dqcoeff, AVM_PLANE_Y, DCT_DCT, subs[i], &dst[r0 * stride + c0], stride, eob, 0,
                                    is_reduced_tx_set_used(cm_, PLANE_TYPE_Y));
    }
    return cost;
  };
  for (const auto& k : cands) {
    if (!k.ok) continue;
    const double cost = run(k, least);
    if (cost < least) least = cost, best = k.p;
  }
  if (keep) {  // the chosen partition's reconstruction left in place
    for (const auto& k : cands)
      if (k.ok && k.p == best) run(k, 1e300);
  } else {
    for (int r = 0; r < bh; r++) std::copy(saved.begin() + static_cast<size_t>(r) * bw, saved.begin() + static_cast<size_t>(r + 1) * bw, dst + r * stride);
  }
  mbmi->tx_partition_type[0] = TX_PARTITION_NONE;
  mbmi->txb_idx = 0;
  mbmi->tx_size = max_tx;
  return best;
}

void TileCoder::recon_luma_for_cfl() {
  if (cfl_recon_) return;
  cfl_recon_ = true;
  const BLOCK_SIZE bsize = xd_->mi[0]->sb_type[PLANE_TYPE_Y];
  const TX_PARTITION_TYPE p = choose_intra_tx_partition(bsize, true);  // NONE's reconstruction where it cannot split
  if (block_signals_txsize(bsize) && cm_->features.tx_mode == TX_MODE_SELECT && enc_.cfg_.intra_tx_part) cfl_txp_ = p;
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
  if (plane != AVM_PLANE_Y) mhccp_fill_outside(blk_col, blk_row, tx_size);
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

double TileCoder::intra_rd(int joint) {
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
  int16_t res[64 * 64];
  residual_of(AVM_PLANE_Y, tx, pd->dst.buf, pd->dst.stride, res);
  return luma_tx_cost(res, tx_size_wide[tx], tx, pd->dst.buf, pd->dst.stride);
}

int TileCoder::intrabc_candidates(MV* dvs, int max) {
  near_n_ = 0;
  MACROBLOCKD* const xd = xd_;
  const celv2::IntrabcIndex* const index = enc_.ibc_.get();
  const BLOCK_SIZE bsize = xd->mi[0]->sb_type[PLANE_TYPE_Y];
  const int bw = block_size_wide[bsize], bh = block_size_high[bsize];
  if (!index || bw < 8 || bh < 8) return 0;
  const int x = xd->mi_col * MI_SIZE, y = xd->mi_row * MI_SIZE, sb = index->band_rows(), sb_x = x / sb * sb;
  const bool global = cm_->features.allow_global_intrabc;
  int n = 0;
  index->repeats(*src_, x, y, bw, bh, global ? 0 : std::max(0, sb_x - 2 * sb), global ? cm_->width : sb_x + sb, [&](int px, int py) {
    const MV dv{static_cast<MV_COMP_DATA_TYPE>((py - y) * 8), static_cast<MV_COMP_DATA_TYPE>((px - x) * 8)};
    // Rows above are coded at once, Wavefront's way: row r - k has finished superblocks up to c + k.
    const int k = y / sb - (py + bh - 1) / sb;
    if (k > 0 && (px + bw - 1) / sb > x / sb + k) return true;
    if (av2_is_dv_valid(dv, cm_, xd, xd->mi_row, xd->mi_col, bsize, cm_->mib_size_log2)) dvs[n++] = dv;
    return n < max;
  }, global);
  // Stage B1: near copies along a halftone's period, after the exact ones.
  if (n < max && bw <= 64 && bh <= 64) {
    std::pair<int, int> nc[3];
    const int x_lo = global ? 0 : std::max(0, sb_x - 2 * sb), x_hi = global ? cm_->width : sb_x + sb;
    const int got = index->near_copies(*src_, x, y, bw, bh, [&](int px, int py) {
      if (px < x_lo || px >= x_hi) return false;
      const int k = y / sb - (py + bh - 1) / sb;
      if (k > 0 && (px + bw - 1) / sb > x / sb + k) return false;
      const MV dv{static_cast<MV_COMP_DATA_TYPE>((py - y) * 8), static_cast<MV_COMP_DATA_TYPE>((px - x) * 8)};
      return static_cast<bool>(av2_is_dv_valid(dv, cm_, xd, xd->mi_row, xd->mi_col, bsize, cm_->mib_size_log2));
    }, nc, std::min(3, max - n));
    for (int i = 0; i < got; i++) {
      near_dv_[i] = MV{static_cast<MV_COMP_DATA_TYPE>((nc[i].second - y) * 8), static_cast<MV_COMP_DATA_TYPE>((nc[i].first - x) * 8)};
      dvs[n++] = near_dv_[i];
    }
    near_n_ = got;
  }
  return n;
}

double TileCoder::intrabc_rd(MV dv) {
  MACROBLOCKD* const xd = xd_;
  const BLOCK_SIZE bsize = xd->mi[0]->sb_type[PLANE_TYPE_Y];
  const TX_SIZE tx = max_txsize_rect_lookup[bsize];
  struct macroblockd_plane* const pd = &xd->plane[AVM_PLANE_Y];
  const int s = pd->dst.stride, w = block_size_wide[bsize], h = block_size_high[bsize];
  // The copy, as the decoder's integer-vector prediction from the frame's unfiltered reconstruction.
  const uint16_t* const ref = pd->dst.buf + (dv.row >> 3) * s + (dv.col >> 3);
  for (int r = 0; r < h; r++) std::memcpy(pd->dst.buf + r * s, ref + r * s, sizeof(uint16_t) * w);
  int16_t res[64 * 64];
  residual_of(AVM_PLANE_Y, tx, pd->dst.buf, s, res);
  // kNearCopyCharge: a near copy's charge over its luma cost, for the chroma it leaves out: 128 bits' worth.
  // 0 / 16 / 48 / 128 / 256 / 512 bits: Flash -3.81 / -4.28 / -4.53 / -4.79 / -4.69 / -4.69%, AIR +0.30 / +0.11 /
  // +0.10 / -0.00 / +0.05 / +0.05%, Angel Beats -7.78 / -7.59 / -8.54 / -8.28 / -8.62 / -6.70%.
  constexpr double kNearCopyCharge = 6.4;
  double extra = 0;
  for (int i = 0; i < near_n_; i++)
    if (near_dv_[i].row == dv.row && near_dv_[i].col == dv.col) extra = kNearCopyCharge;
  return luma_tx_cost(res, tx_size_wide[tx], tx, pd->dst.buf, s) + extra;
}

int TileCoder::palette_source(uint16_t* colors) {
  MACROBLOCKD* const xd = xd_;
  int width, height, rows, cols;
  av2_get_block_dimensions(xd->mi[0]->sb_type[PLANE_TYPE_Y], 0, xd, &width, &height, &rows, &cols);
  const struct macroblockd_plane* const pd = &xd->plane[AVM_PLANE_Y];
  const ptrdiff_t off = pd->dst.buf - cm_->cur_frame->buf.y_buffer;
  const int y0 = static_cast<int>(off / pd->dst.stride), x0 = static_cast<int>(off % pd->dst.stride);
  auto sample = [&](int r, int c) -> uint16_t {
    const size_t at = static_cast<size_t>(y0 + r) * src_->stride_y() + x0 + c;
    return src_->bit_depth > 8 ? src_->hy[at] : src_->y[at];
  };
  uint16_t v[PALETTE_MAX_SIZE];
  int n = 0;
  for (int r = 0; r < rows; r++)
    for (int c = 0; c < cols; c++) {
      const uint16_t s = sample(r, c);
      if (std::find(v, v + n, s) != v + n) continue;
      if (n == PALETTE_MAX_SIZE) return 0;
      v[n++] = s;
    }
  if (n < PALETTE_MIN_SIZE) return 0;
  std::sort(v, v + n);
  uint8_t* const map = xd->plane[0].color_index_map;
  for (int r = 0; r < rows; r++)
    for (int c = 0; c < cols; c++) map[r * width + c] = static_cast<uint8_t>(std::find(v, v + n, sample(r, c)) - v);
  // decode_color_map_tokens()'s edges: the last column, then the last row, repeated.
  for (int r = 0; r < rows; r++) std::fill(map + r * width + cols, map + (r + 1) * width, map[r * width + cols - 1]);
  for (int r = rows; r < height; r++) std::copy(map + (rows - 1) * width, map + rows * width, map + r * width);
  std::copy(v, v + n, colors);
  return n;
}

// mhccp_implicit_fetch_neighbor_luma() stops at the frame's edge, and the prediction reads the transform
// block's whole size: past the edge it found what earlier blocks left in cfl.mhccp_ref_buf_q3. Those rows are not
// shown, but their residual is coded, so the bytes depended on the blocks before (which worker coded which
// chunk). The edge repeated: what a block past the edge is coded from, as the source's edge is repeated there.
void TileCoder::mhccp_fill_outside(int blk_col, int blk_row, TX_SIZE tx) {
  MACROBLOCKD* const xd = xd_;
  const MB_MODE_INFO* const mbmi = xd->mi[0];
  if (mbmi->uv_mode != UV_CFL_PRED || mbmi->cfl_idx != CFL_MULTI_PARAM) return;
  CFL_CTX* const cfl = &xd->cfl;
  // CELV2_TEST_MHCCP_STALE: what earlier blocks left there, for test_cli.
  static const char* const stale = std::getenv("CELV2_TEST_MHCCP_STALE");
  if (stale) std::fill_n(cfl->mhccp_ref_buf_q3[0], CFL_BUF_SQUARE * 4, static_cast<uint16_t>(std::atoi(stale)));
  if (xd->mb_to_bottom_edge >= 0 && xd->mb_to_right_edge >= 0) return;
  const int sub_x = cfl->subsampling_x, sub_y = cfl->subsampling_y;
  const int row_start = (mbmi->chroma_ref_info.mi_row_chroma_base + (blk_row << sub_y)) << MI_SIZE_LOG2;
  int above = 0, left = 0, ref_w = 0, ref_h = 0;
  mhccp_implicit_fetch_neighbor_luma(cm_, xd, blk_row << sub_y, blk_col << sub_x, tx, &above, &left,
                                     !(row_start % block_size_high[cm_->sb_size]), &ref_w, &ref_h);
  above >>= sub_y, left >>= sub_x, ref_w >>= sub_x, ref_h >>= sub_y;
  uint16_t* const q3 = cfl->mhccp_ref_buf_q3[0];
  constexpr int kStride = CFL_BUF_LINE * 2;
  const int w = left + tx_size_wide[tx], h = above + tx_size_high[tx];
  for (int r = above; r < std::min(h, ref_h); r++)
    for (int c = ref_w; c < w; c++) q3[r * kStride + c] = q3[r * kStride + ref_w - 1];
  for (int r = ref_h; r < h; r++) std::copy(q3 + (ref_h - 1) * kStride, q3 + (ref_h - 1) * kStride + w, q3 + r * kStride);
}

void TileCoder::prepare_uv(UV_PREDICTION_MODE uv) {
  MACROBLOCKD* const xd = xd_;
  MB_MODE_INFO* const mbmi = xd->mi[0];
  mbmi->uv_mode = uv;
  mbmi->angle_delta[PLANE_TYPE_UV] = uv == static_cast<UV_PREDICTION_MODE>(mbmi->mode) ? mbmi->angle_delta[PLANE_TYPE_Y] : 0;
  if (uv == UV_CFL_PRED && enc_.cfg_.cfl_rd) {
    recon_luma_for_cfl();  // CfL predicts from this block's reconstructed luma: made now (and again when coded)
  } else if (uv == UV_CFL_PRED) {
    // CfL predicts from this block's reconstructed luma, which the coder makes only later: the source
    // luma stands in for it (the block's luma area is predicted and reconstructed afresh afterwards).
    const struct macroblockd_plane* const pd = &xd->plane[AVM_PLANE_Y];
    const ptrdiff_t off = pd->dst.buf - cm_->cur_frame->buf.y_buffer;
    const int y0 = static_cast<int>(off / pd->dst.stride), x0 = static_cast<int>(off % pd->dst.stride);
    const BLOCK_SIZE bsize = mbmi->sb_type[PLANE_TYPE_Y];
    const int w = std::min<int>(block_size_wide[bsize], src_->pw - x0), h = std::min<int>(block_size_high[bsize], src_->ph - y0);
    for (int r = 0; r < h; r++)
      for (int c = 0; c < w; c++) {
        const size_t at = static_cast<size_t>(y0 + r) * src_->stride_y() + x0 + c;
        pd->dst.buf[r * pd->dst.stride + c] = src_->bit_depth > 8 ? src_->hy[at] : src_->y[at];
      }
  }
}

int64_t TileCoder::intra_uv_satd(UV_PREDICTION_MODE uv) {
  MACROBLOCKD* const xd = xd_;
  prepare_uv(uv);
  const TX_SIZE tx = av2_get_tx_size(AVM_PLANE_U, xd);
  int64_t total = 0;
  for (int plane = AVM_PLANE_U; plane <= AVM_PLANE_V; plane++) {
    bw_->pre_txb(plane, tx);
    mhccp_fill_outside(0, 0, tx);
    av2_predict_intra_block_facade(cm_, xd, plane, 0, 0, tx);
    const struct macroblockd_plane* const pd = &xd->plane[plane];
    const ptrdiff_t off = pd->dst.buf - (plane == AVM_PLANE_U ? cm_->cur_frame->buf.u_buffer : cm_->cur_frame->buf.v_buffer);
    const int y0 = static_cast<int>(off / pd->dst.stride), x0 = static_cast<int>(off % pd->dst.stride);
    const int w = std::min<int>(tx_size_wide[tx], src_->pw / 2 - x0), h = std::min<int>(tx_size_high[tx], src_->ph / 2 - y0);
    const size_t at = static_cast<size_t>(y0) * src_->stride_uv() + x0;
    if (src_->bit_depth > 8)
      total += satd_src(&(plane == AVM_PLANE_U ? src_->hu : src_->hv)[at], src_->stride_uv(), pd->dst.buf, pd->dst.stride, w & ~3, h & ~3);
    else
      total += satd_src(&(plane == AVM_PLANE_U ? src_->u : src_->v)[at], src_->stride_uv(), pd->dst.buf, pd->dst.stride, w & ~3, h & ~3);
  }
  return total;
}

// --cfl-rd: both chroma blocks predicted with uv as intra_uv_satd() predicts them, costed as
// luma_tx_cost() costs luma (DCT quantized, squared error in steps plus quantize()'s weight per bit).
double TileCoder::intra_uv_rd(UV_PREDICTION_MODE uv) {
  MACROBLOCKD* const xd = xd_;
  prepare_uv(uv);
  const TX_SIZE tx = av2_get_tx_size(AVM_PLANE_U, xd);
  const int w = tx_size_wide[tx];
  double total = 0;
  for (int plane = AVM_PLANE_U; plane <= AVM_PLANE_V; plane++) {
    bw_->pre_txb(plane, tx);
    mhccp_fill_outside(0, 0, tx);
    av2_predict_intra_block_facade(cm_, xd, plane, 0, 0, tx);
    const struct macroblockd_plane* const pd = &xd->plane[plane];
    int16_t res[64 * 64];
    residual_of(plane, tx, pd->dst.buf, pd->dst.stride, res);
    tran_low_t coeff[1024];
    const double scale = enc_.coef_scale(plane, tx, xd);
    fwd_dct(res, w, tx, scale, coeff);
    total += quantize_cost(coeff, tx, q_, DCT_DCT, nullptr, tcq_enable(cm_->features.tcq_mode, 0, plane, TX_CLASS_2D),
                           plane ? 0 : trellis_) +
             dropped_cost(res, w, tx, scale, coeff, q_);
  }
  return total;
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
  blend_interintra(0);
}

// dec_build_inter_predictor()'s inter-intra step for one plane: the intra prediction from the
// reconstructed neighbours, blended into the inter prediction in dst.
void TileCoder::blend_interintra(int plane) {
  MACROBLOCKD* const xd = xd_;
  const MB_MODE_INFO* const mbmi = xd->mi[0];
  if (!is_interintra_pred(mbmi)) return;
  if (plane && mbmi->tree_type == SHARED_PART && mbmi->region_type == MIXED_INTER_INTRA_REGION &&
      mbmi->chroma_ref_info.offset_started)
    return;
  const BUFFER_SET ctx = {{xd->plane[0].dst.buf, xd->plane[1].dst.buf, xd->plane[2].dst.buf},
                          {xd->plane[0].dst.stride, xd->plane[1].dst.stride, xd->plane[2].dst.stride}};
  av2_build_interintra_predictor(cm_, xd, xd->plane[plane].dst.buf, xd->plane[plane].dst.stride, &ctx, plane,
                                 mbmi->sb_type[PLANE_TYPE_Y]);
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
      const uint16_t* const at = &pd->dst.buf[(r * pd->dst.stride + c) << MI_SIZE_LOG2];
      residual_of(0, tx, at, pd->dst.stride, res);
      if (psy_scale_at(at, pd->dst.stride, tx) > 0) {
        total += luma_tx_cost(res, tx_size_wide[tx], tx, at, pd->dst.stride);
      } else {
        const double scale = enc_.coef_scale(0, tx, xd);
        fwd_dct(res, tx_size_wide[tx], tx, scale, coeff);
        total += quantize_cost(coeff, tx, q_, DCT_DCT, nullptr, tcq, trellis_) +
                 dropped_cost(res, tx_size_wide[tx], tx, scale, coeff, q_);
      }
      if (total > bound) return total;
    }
  return total;
}

void TileCoder::residual_code(int plane, TX_SIZE tx_size, uint16_t* dst, int dst_stride,
                            int blk_row, int blk_col) {
#ifdef CELV2_ORACLE
  if (imp_frame_) return import_residual(plane, tx_size, dst, dst_stride, blk_row, blk_col);
#endif
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
  tran_low_t try_q[1024], try_dq[MAX_TX_SQUARE], prim[1024];
  std::memset(dqcoeff, 0, sizeof(tran_low_t) * tx_size_2d[tx_size]);
  const bool inter_luma = plane == 0 && is_inter_block(xd->mi[0], xd->tree_type);
  const int reduced_set = cm->features.reduced_tx_set_used;
  const TxSetType set = av2_get_ext_tx_set_type(tx_size, inter_luma, reduced_set);
  TX_TYPE types[TX_TYPES] = {DCT_DCT};
  int ntypes = 1;
  const bool long_side = set == EXT_TX_SET_LONG_SIDE_64 || set == EXT_TX_SET_LONG_SIDE_32;
  static const bool all_types = celv2::wide('5');
  if (plane == 0 && !long_side && get_ext_tx_types(tx_size, inter_luma, reduced_set) > 1 && ((w <= 16 && h <= 16) || all_types)) {
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
    if (k >= 1) {
      if (eob == 0) break;
      if (all_types) fwd_transform(res, w, tx_size, types[k], coeff);
      else types[1] = most_compact(res, w, tx_size, types + 1, ntypes - 1, coeff), ntypes = 2;
    }
    const TX_TYPE type = types[k];
    // Inter IDTX with forward skip coding has no TCQ (av2_read_coeffs_txb_skip()).
    const bool tcq = tcq_enable(cm->features.tcq_mode, xd->lossless[xd->mi[0]->segment_id], plane,
                                tx_type_to_class[get_primary_tx_type(type)]) &&
                     !use_inter_fsc(cm, get_plane_type(plane), type, is_inter_block(xd->mi[0], xd->tree_type));
    if (type == DCT_DCT) fwd_dct(res, w, tx_size, enc_.coef_scale(plane, tx_size, xd_), coeff);
    double cost = 0;
    // Rates weighted x0.5 against quantize()'s in the choice (33 frames 1080p q100-190, BD-rate AM demo /
    // live action / AIR OP: x0.5 -0.10 / -0.05 / -0.25%, x0.25 -0.16 / +0.28 / +0.11%, the same speed).
    constexpr double kf = 0.5;
    std::memset(try_dq, 0, sizeof(tran_low_t) * tx_size_2d[tx_size]);
    const int e = quantize(coeff, try_q, try_dq, tx_size, q_, plane, type, ntypes > 1 ? &cost : nullptr, tcq,
                           cm->seq_params.bit_depth, plane ? 0 : trellis_);
    if (ntypes > 1) {
      const double step = av2_ac_quant_QTX(q_, 0, 0, kQDepth) /
                          static_cast<double>(1 << (QUANT_TABLE_BITS + av2_get_tx_scale(tx_size)));
      double d = 0;
      for (int i = 0; i < nq; i++) d += static_cast<double>(coeff[i] - try_dq[i]) * (coeff[i] - try_dq[i]);
      d /= step * step;
      cost = d + kf * (cost - d + (k >= 1 ? kTypeBits * 0.2 : 0));
    }
    // An intra block with only a DC level codes no type: the decoder takes DCT.
    if (k >= 1 && ((!inter_luma && e == 1) || cost >= best)) continue;
    best = cost, eob = e, chosen = e ? type : DCT_DCT;
    std::memcpy(qcoeff, try_q, sizeof(tran_low_t) * nq);
    std::memcpy(dqcoeff, try_dq, sizeof(tran_low_t) * tx_size_2d[tx_size]);
    std::memcpy(prim, coeff, sizeof(tran_low_t) * nq);
  }
  if (plane == 0 && eob > 0) ist_search(tx_size, prim, eob, chosen, qcoeff, dqcoeff);
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

// -log2 of a symbol's probability in an AVM inverse CDF (cdf[i] = 32768 - P(x <= i) * 32768).
static double symbol_bits(const avm_cdf_prob* cdf, int sym) {
  const int hi = sym ? cdf[sym - 1] : CDF_PROB_TOP;
  return -std::log2(std::max(1, hi - cdf[sym]) / static_cast<double>(CDF_PROB_TOP));
}

// decode_reconstruct_tx() / the intra chroma loop with CCTX allowed: both chroma residuals transformed,
// then coded unrotated or rotated by the type nearest their principal axis (the transpose of
// av2_inv_cross_chroma_tx_block()), whichever quantize() costs less; written U then V, rotated back
// and reconstructed as decodeframe.c does.
void TileCoder::residual_code_uv(TX_SIZE tx_size, int blk_row, int blk_col) {
#ifdef CELV2_ORACLE
  if (imp_frame_) return import_residual_uv(tx_size, blk_row, blk_col);
#endif
  AV2_COMMON* const cm = cm_;
  MACROBLOCKD* const xd = xd_;
  const int w = tx_size_wide[tx_size], h = tx_size_high[tx_size];
  const int nq = std::min(w, 32) * std::min(h, 32);
  const int stride = xd->plane[AVM_PLANE_U].dst.stride;
  const bool is_inter = is_inter_block(xd->mi[0], xd->tree_type);
  uint16_t* dst[2];
  int16_t res[64 * 64];
  tran_low_t coeff[2][1024], rot[2][1024], q[2][1024], dq[2][MAX_TX_SQUARE], try_q[2][1024], try_dq[2][MAX_TX_SQUARE];
  double s11 = 0, s22 = 0, s12 = 0;
  for (int p = 0; p < 2; p++) {
    struct macroblockd_plane* const pd = &xd->plane[AVM_PLANE_U + p];
    dst[p] = &pd->dst.buf[(blk_row * pd->dst.stride + blk_col) << MI_SIZE_LOG2];
    residual_of(AVM_PLANE_U + p, tx_size, dst[p], stride, res);
    fwd_dct(res, w, tx_size, enc_.coef_scale(AVM_PLANE_U + p, tx_size, xd), coeff[p]);
  }
  for (int i = 0; i < nq; i++) {
    s11 += static_cast<double>(coeff[0][i]) * coeff[0][i];
    s22 += static_cast<double>(coeff[1][i]) * coeff[1][i];
    s12 += static_cast<double>(coeff[0][i]) * coeff[1][i];
  }
  // The angle of (U, V) holding most of their energy, mod 180 degrees; the nearest type's.
  const double axis = 0.5 * std::atan2(2 * s12, s11 - s22);
  CctxType axis_type = CCTX_START;
  double near_d = 1e9;
  for (int t = CCTX_START; t < CCTX_TYPES; t++) {
    const double a = std::atan2(cctx_mtx[t - CCTX_START][1], cctx_mtx[t - CCTX_START][0]);
    double d = std::fabs(std::remainder(axis - a, 3.14159265358979323846));
    if (d < near_d) near_d = d, axis_type = static_cast<CctxType>(t);
  }
  const bool tcq = tcq_enable(cm->features.tcq_mode, xd->lossless[xd->mi[0]->segment_id], AVM_PLANE_U, TX_CLASS_2D);
  int eob[2] = {0, 0};
  CctxType chosen = CCTX_NONE;
  double best = 1e300;
  for (const CctxType t : {CctxType(CCTX_NONE), axis_type}) {
    if (t == CCTX_NONE) {
      std::memcpy(rot, coeff, sizeof(rot));
    } else {
      const int c = cctx_mtx[t - CCTX_START][0], s = cctx_mtx[t - CCTX_START][1];
      for (int i = 0; i < nq; i++) {
        rot[0][i] = static_cast<tran_low_t>(std::lround((c * static_cast<double>(coeff[0][i]) + s * static_cast<double>(coeff[1][i])) / (1 << CCTX_PREC_BITS)));
        rot[1][i] = static_cast<tran_low_t>(std::lround((c * static_cast<double>(coeff[1][i]) - s * static_cast<double>(coeff[0][i])) / (1 << CCTX_PREC_BITS)));
      }
    }
    double cost = 0;
    int e[2];
    for (int p = 0; p < 2; p++) {
      double c = 0;
      std::memset(try_dq[p], 0, sizeof(tran_low_t) * tx_size_2d[tx_size]);
      e[p] = quantize(rot[p], try_q[p], try_dq[p], tx_size, q_, AVM_PLANE_U + p, DCT_DCT, &c, tcq, cm->seq_params.bit_depth);
      cost += c;
    }
    // BlockWriter::coeffs() codes the type only where U keeps a level (not an intra DC alone).
    const bool coded = e[0] > 0 && (is_inter || e[0] != 1);
    if (t != CCTX_NONE && !coded) continue;
    if (coded) cost += kBitWeight * symbol_bits(xd->tile_ctx->cctx_type_cdf, t);
    if (cost >= best) continue;
    best = cost, chosen = t, eob[0] = e[0], eob[1] = e[1];
    std::memcpy(q, try_q, sizeof(q));
    for (int p = 0; p < 2; p++) std::memcpy(dq[p], try_dq[p], sizeof(tran_low_t) * tx_size_2d[tx_size]);
  }
  bw_->coeffs(AVM_PLANE_U, blk_row, blk_col, tx_size, q[0], eob[0], DCT_DCT, chosen);
  bw_->coeffs(AVM_PLANE_V, blk_row, blk_col, tx_size, q[1], eob[1]);
  av2_inv_cross_chroma_tx_block(dq[0], dq[1], tx_size, chosen, xd->bd);
  const int max_eob = std::max(eob[0], eob[1]);
  const uint8_t reduced = is_reduced_tx_set_used(cm, PLANE_TYPE_UV);
  for (int p = 0; p < 2; p++) {
    // V is reconstructed from a rotated U alone too.
    if (!eob[p] && !(p == 1 && chosen != CCTX_NONE && eob[0])) continue;
    av2_inverse_transform_block(
        xd, dq[p], AVM_PLANE_U + p, DCT_DCT, tx_size, dst[p], stride, max_eob,
        replace_adst_by_ddt(cm->seq_params.enable_inter_ddt, cm->features.allow_screen_content_tools, xd),
        reduced);
  }
}

// Every kernel set and type the block can code, quantized, against the primary coding: quantize()'s
// cost with its distortion replaced by that of the reconstruction through AVM's inverse (the 7-bit
// kernels are only about orthonormal: the forward's own estimate came out 20% low and IST lost 0.07 dB),
// plus kBitWeight per bit of the stx symbols (from the tile's CDFs), the rates weighted by kf.
void TileCoder::ist_search(TX_SIZE tx, const tran_low_t* prim, int& eob, TX_TYPE& type, tran_low_t* qcoeff,
                           tran_low_t* dqcoeff) {
  AV2_COMMON* const cm = cm_;
  MACROBLOCKD* const xd = xd_;
  MB_MODE_INFO* const mbmi = xd->mi[0];
  const bool inter = is_inter_block(mbmi, xd->tree_type);
  if (!(inter ? cm->seq_params.enable_inter_ist : cm->seq_params.enable_ist) || xd->lossless[mbmi->segment_id]) return;
  const int w = tx_size_wide[tx], h = tx_size_high[tx];
  const PREDICTION_MODE mode = inter ? DC_PRED : static_cast<PREDICTION_MODE>(get_intra_mode(mbmi, PLANE_TYPE_Y));
  // The decoder checks the primary type and end of block (block_signals_sec_tx_type()).
  if (!(type == DCT_DCT || type == ADST_ADST) || (inter ? type != DCT_DCT || w < 16 || h < 16 : mode >= PAETH_PRED))
    return;
  const bool tcq = tcq_enable(cm->features.tcq_mode, 0, 0, tx_type_to_class[type]);
  const int bd = cm->seq_params.bit_depth;
  const FRAME_CONTEXT* const fc = xd->tile_ctx;
  const avm_cdf_prob* const stx_cdf = fc->stx_cdf[inter][txsize_sqr_map[tx]];
  const int nq = std::min(w, 32) * std::min(h, 32);
  const double step = av2_ac_quant_QTX(q_, 0, 0, kQDepth) / static_cast<double>(1 << (QUANT_TABLE_BITS + av2_get_tx_scale(tx)));
  const double inv_step2 = 1 / (step * step);  // equal_ac_dc_q: one step for DC and AC
  // The rate's weight in this choice, against quantize()'s (33 frames 1080p q100-190, BD-rate am / yo / air):
  // x1 +0.36 / +0.89 / -0.16% (fewer bytes, more lost PSNR), x0.5 -0.32 / -0.15 / -1.48%,
  // x0.25 -0.27 / -0.27 / -1.70%, x0.125 +0.38 / +0.21 / -1.00%.
  constexpr double kf = 0.25;
  double base_d = 0;
  for (int i = 0; i < nq; i++) base_d += static_cast<double>(prim[i] - dqcoeff[i]) * (prim[i] - dqcoeff[i]);
  base_d *= inv_step2;
  double best = base_d + kf * (quantize_cost(prim, tx, q_, type, nullptr, tcq, trellis_) - base_d);
  if (block_signals_sec_tx_type(xd, tx, type, eob)) best += kf * kBitWeight * symbol_bits(stx_cdf, 0);
  const bool adst8 = type == ADST_ADST && w >= 8 && h >= 8;
  const int sets = inter ? 1 : adst8 ? IST_REDUCED_SET_SIZE : IST_SET_SIZE;
  TX_TYPE best_type = type;
  int best_eob = 0;
  tran_low_t c[1024], q[1024], dq[MAX_TX_SQUARE], r[MAX_TX_SQUARE], best_q[1024], best_dq[MAX_TX_SQUARE];
  for (int set = 0; set < sets; set++) {
    const int full_set = set + (type == ADST_ADST ? IST_SET_SIZE : 0);
    double set_bits = 0;
    if (!inter) {
      const int sym = adst8 ? most_probable_stx_mapping_ADST_ADST[mode][set] : most_probable_stx_mapping[mode][set];
      set_bits = symbol_bits(adst8 ? fc->most_probable_stx_set_cdf_ADST_ADST : fc->most_probable_stx_set_cdf, sym);
    }
    for (int stx = 1; stx < STX_TYPES; stx++) {
      std::memcpy(c, prim, sizeof(tran_low_t) * nq);
      fwd_ist(c, tx, type, mode, full_set, stx, bd);
      TX_TYPE full = type;
      set_secondary_tx_type(&full, stx);
      set_secondary_tx_set(&full, full_set);
      std::memset(dq, 0, sizeof(tran_low_t) * tx_size_2d[tx]);
      double cost = 0;
      const int e = quantize(c, q, dq, tx, q_, 0, full, &cost, tcq, bd, trellis_);
      if (!block_signals_sec_tx_type(xd, tx, type, e)) continue;
      std::memcpy(r, dq, sizeof(tran_low_t) * nq);
      TxfmParam p{};
      p.tx_type = type, p.sec_tx_set = full_set, p.sec_tx_type = stx, p.intra_mode = mode, p.is_inter = inter;
      p.tx_size = tx, p.bd = bd;
      av2_inv_stxfm(r, &p);
      double own = 0, actual = 0;
      for (int i = 0; i < nq; i++) {
        own += static_cast<double>(c[i] - dq[i]) * (c[i] - dq[i]);
        actual += static_cast<double>(prim[i] - r[i]) * (prim[i] - r[i]);
      }
      cost = actual * inv_step2 + kf * (cost - own * inv_step2 + kBitWeight * (symbol_bits(stx_cdf, stx) + set_bits));
      if (cost >= best) continue;
      best = cost, best_type = full, best_eob = e;
      std::memcpy(best_q, q, sizeof(tran_low_t) * nq);
      std::memcpy(best_dq, dq, sizeof(tran_low_t) * tx_size_2d[tx]);
    }
  }
  if (best_type == type) return;
  eob = best_eob, type = best_type;
  std::memcpy(qcoeff, best_q, sizeof(tran_low_t) * nq);
  std::memcpy(dqcoeff, best_dq, sizeof(tran_low_t) * tx_size_2d[tx]);
}

#ifdef CELV2_ORACLE
#include "tile_import.inc"
#endif

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
// pick (null: none): each quarter's choice, 0 its tree and blocks, k its alts[k - 1] (CELV2_WIDE W3).
static SbDecision compose_sb(const DecisionMap& dm, int sb_col, int sb_row, const int* pick = nullptr) {
  const SbDecision* q[4];
  for (int i = 0; i < 4; i++) {
    const int c = 2 * sb_col + (i & 1), r = 2 * sb_row + (i >> 1);
    q[i] = c < dm.sb_cols && r < dm.sb_rows ? &dm.sbs[static_cast<size_t>(r) * dm.sb_cols + c] : nullptr;
  }
  SbDecision out;
  out.tree = {Part::Split};
  for (int i = 0; i < 4; i++) {
    const SbDecision* const s = q[i];
    if (s) {
      const int k = pick ? pick[i] : 0;
      const auto& tree = k ? s->alts[k - 1].tree : s->tree;
      const auto& blocks = k ? s->alts[k - 1].blocks : s->blocks;
      out.tree.insert(out.tree.end(), tree.begin(), tree.end());
      out.blocks.insert(out.blocks.end(), blocks.begin(), blocks.end());
    } else {
      out.tree.push_back(Part::None);
      out.blocks.push_back(q[0]->blocks[0]);
    }
  }
  return out;
}

void TileCoder::code_row(BlockWriter& bw, int mi_row, int r, Wavefront& wf) {
  bw_ = &bw;
  g_rd_zero = enc_.cfg_.rd_zero;
  const SeqConfig& cfg = enc_.cfg_;
  auto intra_lambda = [&](int q) { return lambda_q8(cfg.qindex_8bit(q)) << (2 * (cfg.bit_depth - 8)); };
#ifdef CELV2_ORACLE
  const bool own = celv2::oracle::level() < 2;  // L2: no re-scoring, avmenc's modes as they are
#else
  constexpr bool own = true;
#endif
  if (own) bw.set_intra_search([this](int joint) { return intra_satd(joint); }, intra_lambda(cm_->quant_params.base_qindex));
  if (own) bw.set_intra_rd([this](int joint) { return intra_rd(joint); });
  xd_->plane[0].color_index_map = color_map_.data();
  xd_->color_index_map_offset[0] = 0;
  if (cfg.screen) {
    bw.set_palette_source([this](uint16_t* colors) { return palette_source(colors); });
    bw.set_intrabc_search([this](MV* dvs, int max) { return intrabc_candidates(dvs, max); },
                          [this](MV dv) { return intrabc_rd(dv); });
  }
  // Chroma modes searched too (33 frames 1080p q100-190, BD-rate Y / YUV 6:1:1: AM demo -0.20 / -0.32%,
  // live action -0.09 / +0.13%, AIR OP -0.40 / -0.49%; before, chroma took the luma mode).
  if (own) bw.set_uv_search([this](UV_PREDICTION_MODE uv) { return intra_uv_satd(uv); });
  bw.set_cfl_factor(cfg.cfl_rd ? 1.0 : 1.2);
  if (cfg.cfl_rd && own) bw.set_uv_rd([this](UV_PREDICTION_MODE uv) { return intra_uv_rd(uv); });
  if (!frame_is_intra_only(cm_) && own)
    bw.set_inter_search([this](double bound) { return pred_rd(bound); });
  if (cfg.warp)
    bw.set_motion_field([this](int r, int x, int y, MV* mv) {
      const int s = dm_.sb_size, sc = x / s, sr = y / s;
      if (s != 64 || sc >= dm_.sb_cols || sr >= dm_.sb_rows || r >= BlockCandidates::kRefs) return false;
      const SbDecision& sb = dm_.sbs[static_cast<size_t>(sr) * dm_.sb_cols + sc];
      if (sb.cands.empty()) return false;
      const BlockCandidates& c = sb.cands[celv2::sb_square_id(3, x % s / 8, y % s / 8) * 5];
      if (c.ref_cost[r] == UINT32_MAX) return false;
      mv->row = c.ref_mv_row[r], mv->col = c.ref_mv_col[r];
      return true;
    });
#ifdef CELV2_ORACLE
  if (celv2::oracle::level() >= 3) imp_frame_ = celv2::oracle::frame(static_cast<int>(cm_->current_frame.display_order_hint));
#endif
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
#ifdef CELV2_ORACLE
      if (celv2::oracle::real_cost() & celv2::oracle::kRealRefresh) {
        if (!imp_fc_) imp_fc_ = std::make_unique<FRAME_CONTEXT>();
        *imp_fc_ = *xd_->tile_ctx;
      }
#endif
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
      sb_row_ = mi_row, sb_col_ = mi_col;
      std::fill(pick_, pick_ + 4, 0);
      // decode_partition_sb()
      xd_->tree_type = SHARED_PART;
      av2_reset_ptree_in_sbi(xd_->sbi, xd_->tree_type);
      bw.lr_units(mi_row, mi_col, cm_->sb_size);
      code_partition(mi_row, mi_col, cm_->sb_size, xd_->sbi->ptree_root[av2_get_sdp_idx(xd_->tree_type)]);
      wf.finish(r, *xd_->tile_ctx);
    }
  }
  bw_ = nullptr;
}

int Encoder::sb_px() const { return block_size_wide[st_->cm()->seq_params.sb_size]; }

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
  map_ = f.map;
  ibc_ = f.ibc;
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
  if (st_->cm()->seq_params.film_grain_params_present)
    write_film_grain(&wb, st_->cm()->ref_frame_map[slot]->film_grain_params);
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
  celv2::StageScope prof(celv2::Stage::Coder);
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
  // --grain: the decoder adds the source's grain when it shows this frame (now or by a SEF, from the
  // frame buffer as AVM keeps it); the frame header and SEF write it.
  cm->cur_frame->film_grain_params = avm_film_grain_t{};
  if (cm->seq_params.film_grain_params_present && src.grain) {
    cm->cur_frame->film_grain_params = src.grain->p;
    cm->cur_frame->film_grain_params.random_seed = grain_seed(static_cast<int>(cm->cur_frame->display_order_hint));
  }
#ifdef CELV2_ORACLE
  if (celv2::oracle::level() < 2)  // L2: identity, avmenc's sequence has no global motion
#endif
  if (!key && cm->seq_params.enable_global_motion) {
    celv2::StageScope gm(celv2::Stage::GlobalMotion);
    set_global_motion(cm, src, st_->gm_ref_dist);
  }

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
  trial_lambda_ = sse_per_bit(cm->quant_params.base_qindex);
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
  // The in-loop filters before the range coding: the restoration flags the tiles defer come from them.
  st_->end_frame(src, sse_per_bit(cm->quant_params.base_qindex));
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
      celv2::StageScope ec(celv2::Stage::Entropy);
      for (const SymbolLog& log : out[t].logs) log.replay(&w, st_->xd(t)->tile_ctx, &base_qindex, st_->lr_flags().data());
    } catch (...) {
      avm_stop_encode(&w);
      throw;
    }
    avm_stop_encode(&w);
    out[t].size = w.pos;
  });
  st_->save_contexts();

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

// CELV2_WIDE W3: each alternative of the 64x64 at (mi_row, mi_col) coded for real, its CDF bits and
// squared error (luma and chroma) compared, everything put back, and the best left in pick_ / nodes_.
void TileCoder::trial_64(int mi_row, int mi_col, PARTITION_TREE* ptree) {
  const int dc = mi_col >> 4, dr = mi_row >> 4, quarter = (dr & 1) * 2 + (dc & 1);
  const SbDecision& d = dm_.sbs[static_cast<size_t>(dr) * dm_.sb_cols + dc];
  if (d.alts.empty()) return;
  AV2_COMMON* const cm = cm_;
  MACROBLOCKD* const xd = xd_;
  const CommonModeInfoParams& mp = cm->mi_params;
  const int rows = std::min(16, mp.mi_rows - mi_row), cols = std::min(16, mp.mi_cols - mi_col);
  const int planes = av2_num_planes(cm);
  SymbolLog* const log = bw_->log();
  // What a coded block reads before it writes it.
  auto saved_xd = std::make_unique<MACROBLOCKD>(*xd);
  auto saved_fc = std::make_unique<FRAME_CONTEXT>(*xd->tile_ctx);
  const PARTITION_TREE saved_tree = *ptree;
  const size_t log_size = log->size();
  std::vector<ENTROPY_CONTEXT> ent[MAX_MB_PLANE];
  std::vector<PARTITION_CONTEXT> par[MAX_MB_PLANE];
  for (int p = 0; p < planes; p++) {
    const int ss = xd->plane[p].subsampling_x;
    ent[p].assign(xd->above_entropy_context[p] + (mi_col >> ss), xd->above_entropy_context[p] + ((mi_col + 16) >> ss));
    par[p].assign(xd->above_partition_context[p] + mi_col, xd->above_partition_context[p] + mi_col + 16);
  }
  std::vector<MB_MODE_INFO*> grid(static_cast<size_t>(rows) * cols);
  std::vector<SUBMB_INFO*> subgrid(grid.size());
  std::vector<MB_MODE_INFO> mis(grid.size());
  std::vector<SUBMB_INFO> subs(grid.size());
  for (int r = 0; r < rows; r++)
    for (int c = 0; c < cols; c++) {
      const int g = get_mi_grid_idx(&mp, mi_row + r, mi_col + c), a = get_alloc_mi_idx(&mp, mi_row + r, mi_col + c);
      const size_t k = static_cast<size_t>(r) * cols + c;
      grid[k] = mp.mi_grid_base[g], subgrid[k] = mp.submi_grid_base[g];
      mis[k] = mp.mi_alloc[a], subs[k] = mp.mi_alloc_sub[a];
    }
  auto restore = [&] {
    for (int i = 0; i < 4; i++) {
      av2_free_ptree_recursive(ptree->sub_tree[i]);
      ptree->sub_tree[i] = nullptr;
    }
    *ptree = saved_tree;
    *xd = *saved_xd;
    *xd->tile_ctx = *saved_fc;
    log->truncate(log_size);
    for (int p = 0; p < planes; p++) {
      std::copy(ent[p].begin(), ent[p].end(), xd->above_entropy_context[p] + (mi_col >> xd->plane[p].subsampling_x));
      std::copy(par[p].begin(), par[p].end(), xd->above_partition_context[p] + mi_col);
    }
    for (int r = 0; r < rows; r++)
      for (int c = 0; c < cols; c++) {
        const int g = get_mi_grid_idx(&mp, mi_row + r, mi_col + c), a = get_alloc_mi_idx(&mp, mi_row + r, mi_col + c);
        const size_t k = static_cast<size_t>(r) * cols + c;
        mp.mi_grid_base[g] = grid[k], mp.submi_grid_base[g] = subgrid[k];
        mp.mi_alloc[a] = mis[k], mp.mi_alloc_sub[a] = subs[k];
      }
  };
  auto use = [&](int k) {
    pick_[quarter] = k;
    composed_ = compose_sb(dm_, sb_col_ >> cm->mib_size_log2, sb_row_ >> cm->mib_size_log2, pick_);
    nodes_.clear();
    size_t pos = 0;
    int leaf = 0;
    parse_tree(pos, leaf, sb_row_, sb_col_, cm->mib_size, cm->mib_size);
  };
  static const double kScale = std::getenv("X_TL") ? std::atof(std::getenv("X_TL")) : 1;
  static const double kChroma = std::getenv("X_TC") ? std::atof(std::getenv("X_TC")) : 1;
  const double base = av2_ac_quant_QTX(cm->quant_params.base_qindex, 0, 0, AVM_BITS_8), ac = av2_ac_quant_QTX(q_, 0, 0, AVM_BITS_8);
  const double lambda = kScale * enc_.trial_lambda_ * (ac / base) * (ac / base);
  double least = 1e300;
  int best = 0;
  trying_ = true;
  for (int k = 0; k <= static_cast<int>(d.alts.size()); k++) {
    if (k) {
      const SbDecision::Alt& a = d.alts[k - 1];
      if (a.tree == d.tree && a.blocks.size() == d.blocks.size() &&
          std::equal(a.blocks.begin(), a.blocks.end(), d.blocks.begin(), [](const BlockDecision& x, const BlockDecision& y) {
            return x.inter == y.inter && x.intra == y.intra && x.ref == y.ref && x.ref2 == y.ref2 && x.mv_row == y.mv_row &&
                   x.mv_col == y.mv_col && x.mv2_row == y.mv2_row && x.mv2_col == y.mv2_col;
          }))
        continue;
    }
    use(k);
    log->count(true);
    code_partition(mi_row, mi_col, BLOCK_64X64, ptree);
    const double bits = log->bits();
    log->count(false);
    double sse = 0;
    for (int p = 0; p < planes; p++) sse += (p ? kChroma : 1) * block_sse(p, mi_row, mi_col);
    const double j = sse + lambda * bits;
    if (j < least) least = j, best = k;
    restore();
  }
  trying_ = false;
  use(best);
}

// Squared error of the 64x64's reconstruction in plane p against the source, inside the padded frame.
double TileCoder::block_sse(int p, int mi_row, int mi_col) const {
  const YV12_BUFFER_CONFIG& fb = cm_->cur_frame->buf;
  const uint16_t* const base = p == 0 ? fb.y_buffer : p == 1 ? fb.u_buffer : fb.v_buffer;
  const int stride = p ? fb.uv_stride : fb.y_stride, ss = p ? src_->stride_uv() : src_->stride_y(), sh = p ? 1 : 0;
  const int x0 = mi_col * MI_SIZE >> sh, y0 = mi_row * MI_SIZE >> sh;
  const int x1 = std::min(x0 + (64 >> sh), cm_->width >> sh), y1 = std::min(y0 + (64 >> sh), cm_->height >> sh);
  double total = 0;
  auto add = [&](const auto& sp) {
    for (int y = y0; y < y1; y++)
      for (int x = x0; x < x1; x++) {
        const int e = sp[static_cast<size_t>(y) * ss + x] - base[static_cast<size_t>(y) * stride + x];
        total += static_cast<double>(e) * e;
      }
  };
  if (src_->bit_depth > 8) add(p == 0 ? src_->hy : p == 1 ? src_->hu : src_->hv);
  else add(p == 0 ? src_->y : p == 1 ? src_->u : src_->v);
  return total;
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
  if (bsize == BLOCK_64X64 && sb_ == &composed_ && !trying_) trial_64(mi_row, mi_col, ptree);
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
  cfl_txp_ = -1, cfl_recon_ = false;
  const celv2::AqMap* const map = enc_.map_.get();
  trellis_ = map && enc_.cfg_.psy_trellis > 0
                 ? enc_.cfg_.psy_trellis * celv2::psy_weight(*map, mi_col * MI_SIZE, mi_row * MI_SIZE, block_size_wide[bsize], block_size_high[bsize],
                                                             enc_.cfg_.psy_lines ? celv2::PsyRule::Line : celv2::PsyRule::Texture)
                 : 0;
  // No palette unless the mode info chooses one (the mi grid keeps the last frame's, which the next
  // blocks' palette caches would read).
  mbmi->palette_mode_info.palette_size[0] = mbmi->palette_mode_info.palette_size[1] = 0;
  if (frame_is_intra_only(cm))
    bw_->intra_frame_mode_info(d);
  else
    bw_->inter_frame_mode_info(d);
  // av2_visit_palette(): a palette block's colour index map follows its mode info.
  if (xd->tree_type != CHROMA_PART && !is_inter_block(mbmi, xd->tree_type) && mbmi->palette_mode_info.palette_size[0] > 0)
    bw_->palette_tokens();

  // parse_decode_block(): transform size. With TX_MODE_SELECT an inter block codes each unit's
  // partition once predicted (code_inter_block); an intra block codes none (read_tx_size()).
  xd->reduced_tx_part_set = cm->seq_params.reduced_tx_part_set;
  std::memset(mbmi->tx_partition_type, TX_PARTITION_NONE, sizeof(mbmi->tx_partition_type));
#ifdef CELV2_ORACLE
  if (imp_frame_ && block_signals_txsize(bsize) && cm->features.tx_mode == TX_MODE_SELECT && !is_inter_block(mbmi, xd->tree_type))
    bw_->tx_partition(max_txsize_rect_lookup[bsize], 0, 0, import_partition(bsize, max_txsize_rect_lookup[bsize], 0, 0));
  else
#endif
  if (block_signals_txsize(bsize) && cm->features.tx_mode == TX_MODE_SELECT && !is_inter_block(mbmi, xd->tree_type))
    bw_->tx_partition(max_txsize_rect_lookup[bsize], 0, 0,
                      cfl_txp_ >= 0             ? static_cast<TX_PARTITION_TYPE>(cfl_txp_)
                      : enc_.cfg_.intra_tx_part ? choose_intra_tx_partition(bsize)
                                                : TX_PARTITION_NONE);
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
#ifdef CELV2_ORACLE
  // CfL over more than one 64x64 unit: chroma's levels are written after the first unit, but the decoder predicts it after the
  // last unit's luma. So all the luma first, unwritten; the units below then give it again.
  if (imp_frame_ && plane_end > 1 && xd->is_chroma_ref && mbmi->uv_mode == UV_CFL_PRED &&
      (mu_blocks_wide < max_blocks_wide || mu_blocks_high < max_blocks_high)) {
    const BLOCK_SIZE plane_bsize = get_mb_plane_block_size(xd, mbmi, AVM_PLANE_Y, 0, 0);
    TXB_POS_INFO* const pos = &mbmi->txb_pos;
    get_tx_partition_sizes(mbmi->tx_partition_type[0], max_txsize_rect_lookup[plane_bsize], pos, mbmi->sub_txs, xd->error_info);
    // X_ORACLE_Q prices by the entropy contexts, so they move here as they will below, and go back after.
    struct macroblockd_plane* const luma = &xd->plane[AVM_PLANE_Y];
    ENTROPY_CONTEXT above[32], left[32];
    std::memcpy(above, luma->above_entropy_context, mi_size_wide[bsize]);
    std::memcpy(left, luma->left_entropy_context, mi_size_high[bsize]);
    imp_again_ = true;
    for (int row = 0; row < max_blocks_high; row += mu_blocks_high)
      for (int col = 0; col < max_blocks_wide; col += mu_blocks_wide)
        for (int i = 0; i < pos->n_partitions; ++i) {
          mbmi->txb_idx = i;
          const int blk_row = row + pos->row_offset[i], blk_col = col + pos->col_offset[i];
          if (blk_row < get_plane_tx_unit_height(xd, plane_bsize, AVM_PLANE_Y, row, 0) &&
              blk_col < get_plane_tx_unit_width(xd, plane_bsize, AVM_PLANE_Y, col, 0))
            code_intra_txb(AVM_PLANE_Y, blk_row, blk_col, mbmi->sub_txs[i]);
        }
    imp_again_ = false;
    std::memcpy(luma->above_entropy_context, above, mi_size_wide[bsize]);
    std::memcpy(luma->left_entropy_context, left, mi_size_high[bsize]);
  }
#endif
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
              const bool cctx = is_cctx_allowed(cm, xd);
              if (cctx && plane == AVM_PLANE_U) continue;  // with V
              for (int blk_row = row >> ss_y; blk_row < plane_unit_height; blk_row += stepr) {
                for (int blk_col = col >> ss_x; blk_col < plane_unit_width; blk_col += stepc) {
                  if (!need_parsing) continue;
                  // The decoder predicts/reconstructs chroma at (blk & 0xf0); with 64x64
                  // superblocks the chroma transform covers the block, so both positions agree.
                  if ((blk_row & 0xf0) != blk_row || (blk_col & 0xf0) != blk_col)
                    throw std::runtime_error("chroma transform block inside a 64x64 unit");
                  if (!cctx) {
                    code_intra_txb(plane, blk_row, blk_col, tx_size);
                    continue;
                  }
                  for (int p = AVM_PLANE_U; p <= AVM_PLANE_V; p++) {
                    bw_->pre_txb(p, tx_size);
                    if (p != AVM_PLANE_Y) mhccp_fill_outside(blk_col, blk_row, tx_size);
                    av2_predict_intra_block_facade(cm, xd, p, blk_col, blk_row, tx_size);
                  }
                  residual_code_uv(tx_size, blk_row, blk_col);
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
