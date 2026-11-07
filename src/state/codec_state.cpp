// Portions of this file are derived from the AVM v1.0.0 decoder (av2/decoder),
// Copyright (c) 2021, Alliance for Open Media, under the BSD 3-Clause Clear License
// and the Alliance for Open Media Patent License 1.0: see third_party/avm/.

#include "state/codec_state.h"

#include "core/ccso_apply.h"
#include "core/ccso_search.h"
#include "core/profile.h"
#include "core/threads.h"
#ifdef CELV2_ORACLE
#include "oracle.h"
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <mutex>
#include <stdexcept>
#include <thread>

extern "C" {
#include "config/av2_rtcd.h"
#include "av2/common/alloccommon.h"
#include "av2/common/av2_loopfilter.h"
#include "av2/common/bru.h"
#include "av2/common/ccso.h"
#include "av2/common/cdef.h"
#include "av2/common/cdef_block.h"
#include "av2/common/pred_common.h"
#include "av2/common/reconinter.h"
#include "av2/common/entropymode.h"
#include "av2/common/mvref_common.h"
#include "av2/common/quant_common.h"
#include "av2/common/reconintra.h"
#include "av2/common/tile_common.h"
#include "config/avm_dsp_rtcd.h"
#include "avm_mem/avm_mem.h"
#include "avm_dsp/bitwriter_buffer.h"
#include "config/avm_scale_rtcd.h"
}

namespace {

// decodeframe.c MC_TEMP_BUF_PELS
constexpr int kMcTempBufPels =
    (MAX_SB_SIZE * 2 + AVM_INTERP_EXTEND * 2) * (MAX_SB_SIZE * 2 + AVM_INTERP_EXTEND * 2);

// Same as the decoder's dec_set_mb_mi / dec_setup_mi / dec_free_mi (decoder.c): the mi grid
// covers the frame rounded up to 8 luma pixels, one MB_MODE_INFO per 4x4.
void set_mb_mi(CommonModeInfoParams* mi, int width, int height) {
  const int aligned_width = ALIGN_POWER_OF_TWO(width, 3);
  const int aligned_height = ALIGN_POWER_OF_TWO(height, 3);
  mi->mi_cols = aligned_width >> MI_SIZE_LOG2;
  mi->mi_rows = aligned_height >> MI_SIZE_LOG2;
  mi->mi_stride = calc_mi_size(mi->mi_cols);
  mi->mb_cols = (mi->mi_cols + 2) >> 2;
  mi->mb_rows = (mi->mi_rows + 2) >> 2;
  mi->MBs = mi->mb_rows * mi->mb_cols;
  mi->mi_alloc_bsize = BLOCK_4X4;
  mi->mi_alloc_stride = mi->mi_stride;
}

void setup_mi(CommonModeInfoParams* mi) {
  const int mi_grid_size = mi->mi_stride * calc_mi_size(mi->mi_rows);
  std::memset(mi->mi_grid_base, 0, mi_grid_size * sizeof(*mi->mi_grid_base));
  std::memset(mi->submi_grid_base, 0, mi_grid_size * sizeof(*mi->submi_grid_base));
  std::memset(mi->cctx_type_map, 0, mi_grid_size * sizeof(*mi->cctx_type_map));
  av2_reset_txk_skip_array_using_mi_params(mi);
}

void free_mi(CommonModeInfoParams* mi) {
  avm_free(mi->mi_alloc);
  mi->mi_alloc = nullptr;
  avm_free(mi->mi_grid_base);
  mi->mi_grid_base = nullptr;
  avm_free(mi->mi_alloc_sub);
  mi->mi_alloc_sub = nullptr;
  avm_free(mi->submi_grid_base);
  mi->submi_grid_base = nullptr;
  mi->mi_alloc_size = 0;
  avm_free(mi->tx_type_map);
  mi->tx_type_map = nullptr;
  avm_free(mi->cctx_type_map);
  mi->cctx_type_map = nullptr;
  av2_dealloc_class_id_array(mi);
  av2_dealloc_txk_skip_array(mi);
}

void init_tables_once() {
  static std::once_flag once;
  std::call_once(once, [] {
    av2_rtcd();
    avm_dsp_rtcd();
    avm_scale_rtcd();
    av2_init_intra_predictors();
    av2_init_stxfm_kernels();
    av2_init_wedge_masks();  // global tables: once, before any state codes (states may run at once)
  });
}

template <typename T>
T* alloc_zeroed() {
  void* p = avm_memalign(32, sizeof(T));
  if (!p) throw std::bad_alloc();
  std::memset(p, 0, sizeof(T));
  return static_cast<T*>(p);
}

// decodeframe.c setup_segmentation_dequant() with segmentation and quant matrices off.
void setup_dequant(AV2_COMMON* cm) {
  CommonQuantParams* q = &cm->quant_params;
  const SequenceHeader* s = &cm->seq_params;
  const avm_bit_depth_t bd = s->bit_depth;
  const int qindex = q->base_qindex;
  q->y_dequant_QTX[0][0] = av2_dc_quant_QTX(qindex, q->y_dc_delta_q, s->base_y_dc_delta_q, bd);
  q->y_dequant_QTX[0][1] = av2_ac_quant_QTX(qindex, 0, 0, bd);
  q->u_dequant_QTX[0][0] = av2_dc_quant_QTX(qindex, q->u_dc_delta_q, s->base_uv_dc_delta_q, bd);
  q->u_dequant_QTX[0][1] = av2_ac_quant_QTX(qindex, q->u_ac_delta_q, s->base_uv_ac_delta_q, bd);
  q->v_dequant_QTX[0][0] = av2_dc_quant_QTX(qindex, q->v_dc_delta_q, s->base_uv_dc_delta_q, bd);
  q->v_dequant_QTX[0][1] = av2_ac_quant_QTX(qindex, q->v_ac_delta_q, s->base_uv_ac_delta_q, bd);
  for (int j = 0; j < TX_SIZES_ALL; ++j) {
    q->y_iqmatrix[0][j] = av2_iqmatrix(q, NUM_QM_LEVELS - 1, AVM_PLANE_Y, j);
    q->u_iqmatrix[0][j] = av2_iqmatrix(q, NUM_QM_LEVELS - 1, AVM_PLANE_U, j);
    q->v_iqmatrix[0][j] = av2_iqmatrix(q, NUM_QM_LEVELS - 1, AVM_PLANE_V, j);
  }
}

// The source's rows of a bs x bs block (bs 8 or 4) as 16-bit samples: rows r and r + 1 (bs 8), or
// all four (bs 4).
inline __m256i src_rows(const uint8_t* o, int os, int bs, int r) {
  if (bs == 8)
    return _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(o + r * os)),
                                                   _mm_loadl_epi64(reinterpret_cast<const __m128i*>(o + (r + 1) * os))));
  int32_t w[4];
  for (int i = 0; i < 4; i++) std::memcpy(&w[i], o + i * os, 4);
  return _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w)));
}
inline __m256i src_rows(const uint16_t* o, int os, int bs, int r) {
  if (bs == 8)
    return _mm256_set_m128i(_mm_loadu_si128(reinterpret_cast<const __m128i*>(o + (r + 1) * os)),
                            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o + r * os)));
  return _mm256_set_epi64x(*reinterpret_cast<const int64_t*>(o + 3 * os), *reinterpret_cast<const int64_t*>(o + 2 * os),
                           *reinterpret_cast<const int64_t*>(o + os), *reinterpret_cast<const int64_t*>(o));
}

// The squared error of a packed bs x bs block f (bs 8 or 4) to the source at o, stride os
// (differences within 10 bits: each pair of squares fits madd's 32 bits).
template <typename S>
int64_t block_sse(const uint16_t* f, int bs, const S* o, int os) {
  __m256i acc = _mm256_setzero_si256();
  for (int r = 0; r < bs; r += 16 / bs) {
    const __m256i d = _mm256_sub_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(f + r * bs)), src_rows(o, os, bs, r));
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d, d));
  }
  const __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
  return static_cast<int64_t>(_mm_cvtsi128_si32(s)) + _mm_extract_epi32(s, 1) + _mm_extract_epi32(s, 2) + _mm_extract_epi32(s, 3);
}

}  // namespace

CodecState::CodecState(const SeqConfig& c) : cfg_(c) {
  c.validate();
  init_tables_once();
  cm_ = alloc_zeroed<AV2_COMMON>();
  pool_ = alloc_zeroed<BufferPool>();
  cm_->fc = alloc_zeroed<FRAME_CONTEXT>();
  cm_->default_frame_context = alloc_zeroed<FRAME_CONTEXT>();
  cm_->buffer_pool = pool_;
  cm_->mi_params.free_mi = free_mi;
  cm_->mi_params.setup_mi = setup_mi;
  cm_->mi_params.set_mb_mi = set_mb_mi;
  fill_sequence_params(c, &cm_->seq_params);
  init_ibp_info(cm_->ibp_directional_weights);  // av2_dx_iface.c: weights intra bi-prediction reads
  for (int i = 0; i < MAX_NUM_MLAYERS; i++) {  // as reset by the decoder
    cm_->olk_refresh_frame_flags[i] = -1;
    cm_->prev_olk_co_vcl_refresh_frame_flags[i] = -1;
  }
  av2_loop_filter_init(cm_);

  // decodeframe.c resize_context_buffers(): mi grid, superblock info, above contexts.
  if (av2_alloc_context_buffers(cm_, c.width, c.height))
    throw std::runtime_error("av2_alloc_context_buffers failed");
  av2_init_mi_buffers(&cm_->mi_params);
  cm_->width = c.width;
  cm_->height = c.height;
  cm_->render_width = c.width;
  cm_->render_height = c.height;
  av2_set_frame_sb_size(cm_, cm_->seq_params.sb_size);
  if (av2_alloc_superblock_info_buffers(cm_))
    throw std::runtime_error("av2_alloc_superblock_info_buffers failed");
  setup_tiles();
  // decodeframe.c alloc_above_context_buffers(): one set of above contexts per tile row.
  if (av2_alloc_above_context_buffers(&cm_->above_contexts, cm_->tiles.rows,
                                      cm_->mi_params.mi_cols, av2_num_planes(cm_)))
    throw std::runtime_error("av2_alloc_above_context_buffers failed");
  recon_ = make_frame(c.width, c.height, c.bit_depth);
  tiles_.resize(static_cast<size_t>(cm_->tiles.cols) * cm_->tiles.rows);
  for (Tile& t : tiles_) {
    alloc_worker(t);
    for (auto& h : t.handoff) h = alloc_zeroed<FRAME_CONTEXT>();
  }
}

void CodecState::alloc_worker(RowWorker& w) {
  w.xd = alloc_zeroed<MACROBLOCKD>();
  w.ctx = alloc_zeroed<FRAME_CONTEXT>();
  // decodeframe.c allocate_mc_tmp_buf() / allocate_opfl_tmp_bufs()
  for (auto& b : w.mc_buf) {
    b = static_cast<uint16_t*>(avm_memalign(16, kMcTempBufPels << 1));
    if (!b) throw std::bad_alloc();
    std::memset(b, 0, kMcTempBufPels << 1);
  }
  MACROBLOCKD* const xd = w.xd;
  xd->tmp_conv_dst = static_cast<CONV_BUF_TYPE*>(
      avm_memalign(32, MAX_SB_SIZE * MAX_SB_SIZE * sizeof(*xd->tmp_conv_dst)));
  xd->opfl_vxy_bufs =
      static_cast<int*>(avm_memalign(32, N_OF_OFFSETS * 4 * sizeof(*xd->opfl_vxy_bufs)));
  xd->opfl_gxy_bufs = static_cast<int16_t*>(
      avm_memalign(32, MAX_SB_SQUARE * 4 * sizeof(*xd->opfl_gxy_bufs)));
  xd->opfl_dst_bufs = static_cast<uint16_t*>(
      avm_memalign(32, MAX_SB_SQUARE * 2 * sizeof(*xd->opfl_dst_bufs)));
  if (!xd->tmp_conv_dst || !xd->opfl_vxy_bufs || !xd->opfl_gxy_bufs || !xd->opfl_dst_bufs)
    throw std::bad_alloc();
}

void CodecState::free_worker(RowWorker& w) {
  for (auto* b : w.mc_buf) avm_free(b);
  if (w.xd) {
    avm_free(w.xd->tmp_conv_dst);
    avm_free(w.xd->opfl_vxy_bufs);
    avm_free(w.xd->opfl_gxy_bufs);
    avm_free(w.xd->opfl_dst_bufs);
  }
  avm_free(w.ctx);
  avm_free(w.xd);
}

CodecState::RowWorker* CodecState::acquire_row_worker() {
  std::lock_guard<std::mutex> lock(rows_mu_);
  if (free_rows_.empty()) {
    auto w = std::make_unique<RowWorker>();
    try {
      alloc_worker(*w);
    } catch (...) {
      free_worker(*w);
      throw;
    }
    rows_.push_back(std::move(w));
    return rows_.back().get();
  }
  RowWorker* const w = free_rows_.back();
  free_rows_.pop_back();
  return w;
}

void CodecState::release_row_worker(RowWorker* w) {
  std::lock_guard<std::mutex> lock(rows_mu_);
  free_rows_.push_back(w);
}

void CodecState::begin_row(RowWorker& w, int t) {
  MACROBLOCKD* const xd = w.xd;
  CONV_BUF_TYPE* const conv = xd->tmp_conv_dst;
  int* const vxy = xd->opfl_vxy_bufs;
  int16_t* const gxy = xd->opfl_gxy_bufs;
  uint16_t* const dst = xd->opfl_dst_bufs;
  *xd = *tiles_[t].xd;  // the scratch pointers above are the only ones into the tile's own memory
  xd->tmp_conv_dst = conv, xd->opfl_vxy_bufs = vxy, xd->opfl_gxy_bufs = gxy, xd->opfl_dst_bufs = dst;
  xd->tile_ctx = w.ctx;
}

CodecState::~CodecState() {
  for (auto& fb : pool_->frame_bufs) {
    avm_free(fb.mvs);
    avm_free(fb.seg_map);
    for (auto* p : fb.ccso_info.sb_filter_control) avm_free(p);
    avm_free_frame_buffer(&fb.buf);
  }
  for (Tile& t : tiles_) {
    free_worker(t);
    for (auto* h : t.handoff) avm_free(h);
  }
  for (auto& w : rows_) free_worker(*w);
  for (AVxWorker& w : lf_workers_) avm_get_worker_interface()->end(&w);
  av2_loop_filter_dealloc(&lf_sync_);
  av2_loop_restoration_dealloc(&lr_sync_, static_cast<int>(lf_workers_.size()));
  if (cdef_workers_) av2_free_cdef_buffers(cm_, &cdef_worker_, &cdef_sync_, cdef_workers_);
  av2_free_cdef_sync(&cdef_sync_);
  avm_free(cm_->bru.active_mode_map);
  avm_free(cm_->bru.active_region);
  av2_free_above_context_buffers(&cm_->above_contexts);
  av2_free_context_buffers(cm_);
  avm_free(cm_->fc);
  avm_free(cm_->default_frame_context);
  avm_free(pool_);
  avm_free(cm_);
}

RefCntBuffer* CodecState::free_buffer() {
  for (auto& fb : pool_->frame_bufs) {
    bool used = false;
    for (auto* r : cm_->ref_frame_map) used |= (r == &fb);
    if (!used) return &fb;
  }
  throw std::runtime_error("no free frame buffer");
}

// A copy of s (another state's frame) in d, keeping d's own allocations: the fields as they are,
// then the pixels (the planes with their borders, laid out alike at one frame size).
void snap_sb_dq(std::vector<int>& sb_dq, int width, int height, int coded_sb_px) {
  if (sb_dq.empty()) return;
  const int cols = (width + 63) / 64, rows = (height + 63) / 64, n = std::max(1, coded_sb_px / 64);
  if (sb_dq.size() != static_cast<size_t>(cols) * rows) throw std::invalid_argument("snap_sb_dq: one offset per 64x64 superblock");
  for (int r = 0; r < rows; r += n)
    for (int c = 0; c < cols; c += n) {
      int sum = 0, cnt = 0;
      for (int y = r; y < std::min(rows, r + n); y++)
        for (int x = c; x < std::min(cols, c + n); x++) sum += sb_dq[y * cols + x], cnt++;
      const int d = static_cast<int>(std::lround(static_cast<double>(sum) / cnt / kDeltaQRes)) * kDeltaQRes;
      for (int y = r; y < std::min(rows, r + n); y++)
        for (int x = c; x < std::min(cols, c + n); x++) sb_dq[y * cols + x] = d;
    }
}

void CodecState::copy_frame_buffer(RefCntBuffer* d, const RefCntBuffer* s) {
  const YV12_BUFFER_CONFIG keep_buf = d->buf;
  const auto keep_raw = d->raw_frame_buffer;
  MV_REF* const keep_mvs = d->mvs;
  uint8_t* const keep_seg = d->seg_map;
  const auto keep_ccso = d->ccso_info;
  decltype(d->rst_info) keep_rst;
  std::memcpy(keep_rst, d->rst_info, sizeof(keep_rst));
  const int keep_ref_count = d->ref_count;
  *d = *s;
  d->buf = keep_buf;
  d->raw_frame_buffer = keep_raw;
  d->mvs = keep_mvs;
  d->seg_map = keep_seg;
  d->ccso_info = keep_ccso;
  std::memcpy(d->rst_info, keep_rst, sizeof(keep_rst));
  for (int p = 0; p < MAX_MB_PLANE; p++) av2_copy_rst_frame_filters(&d->rst_info[p], &s->rst_info[p]);  // later frames' dictionaries
  d->ref_count = keep_ref_count;
  ensure_mv_buffer(d, cm_);  // allocates when d had none
  if (s->mvs)  // later frames' temporal MVs
    std::memcpy(d->mvs, s->mvs,
                sizeof(MV_REF) * ROUND_POWER_OF_TWO(cm_->mi_params.mi_rows, TMVP_SHIFT_BITS) *
                    ROUND_POWER_OF_TWO(cm_->mi_params.mi_cols, TMVP_SHIFT_BITS));
  const SequenceHeader* const seq = &cm_->seq_params;
  if (avm_realloc_frame_buffer(&d->buf, cm_->width, cm_->height, seq->subsampling_x, seq->subsampling_y,
                               AVM_DEC_BORDER_IN_PIXELS, cm_->features.byte_alignment, &d->raw_frame_buffer, nullptr,
                               nullptr, false))
    throw std::runtime_error("failed to allocate frame buffer");
  if (d->buf.frame_size != s->buf.frame_size) throw std::runtime_error("sync: frame buffers differ in layout");
  // In parts on several threads (16K: 0.4 GB a frame), 8 MB a thread at least.
  const size_t size = s->buf.frame_size;
  const int n = static_cast<int>(std::max<size_t>(1, std::min<size_t>(16, size >> 23)));
  auto part = [&](int k) {
    std::memcpy(d->buf.buffer_alloc + size * k / n, s->buf.buffer_alloc + size * k / n, size * (k + 1) / n - size * k / n);
  };
  celv2::parallel(n, part);
  d->buf.bit_depth = s->buf.bit_depth;
  d->buf.render_width = s->buf.render_width;
  d->buf.render_height = s->buf.render_height;
  d->buf.max_width = s->buf.max_width;
  d->buf.max_height = s->buf.max_height;
}

void CodecState::sync_refs_from(const CodecState& o) {
  const AV2_COMMON* const ocm = o.cm_;
  RefCntBuffer* map[REF_FRAMES] = {};
  bool taken[FRAME_BUFFERS] = {};
  auto gen_of = [&](const RefCntBuffer* b) { return o.gen_[b - o.pool_->frame_bufs]; };
  // Buffers already holding a slot's frame keep it; the other frames are copied into free ones.
  for (int i = 0; i < REF_FRAMES; i++) {
    const RefCntBuffer* s = ocm->ref_frame_map[i];
    if (!s) continue;
    for (int k = 0; k < FRAME_BUFFERS; k++)
      if (mirror_of_[k] == s && mirror_gen_[k] == gen_of(s)) map[i] = &pool_->frame_bufs[k], taken[k] = true;
  }
  for (int i = 0; i < REF_FRAMES; i++) {
    const RefCntBuffer* s = ocm->ref_frame_map[i];
    if (!s || map[i]) continue;
    for (int j = 0; j < i && !map[i]; j++)
      if (ocm->ref_frame_map[j] == s) map[i] = map[j];  // a frame in several slots
    if (map[i]) continue;
    int k = 0;
    while (k < FRAME_BUFFERS && taken[k]) k++;
    if (k == FRAME_BUFFERS) throw std::runtime_error("sync: no free frame buffer");
    copy_frame_buffer(&pool_->frame_bufs[k], s);
    gen_[k]++;  // a new frame here: states mirroring this one must copy it again
    mirror_of_[k] = s, mirror_gen_[k] = gen_of(s);
    map[i] = &pool_->frame_bufs[k], taken[k] = true;
  }
  for (int i = 0; i < REF_FRAMES; i++) cm_->ref_frame_map[i] = map[i];
  *cm_->default_frame_context = *ocm->default_frame_context;
  frame_number_ = o.frame_number_;
  since_key_ = o.since_key_;
  display_counter_ = o.display_counter_;
  realloc_bru_info(cm_);  // begin_frame's reference ranking writes there (o's first keyframe made o's)
}

// decodeframe.c get_disp_order_hint() for a regular (non-key) frame.
int CodecState::derive_display_order_hint(int order_hint) const {
  const SequenceHeader* const seq = &cm_->seq_params;
  int max_doh = 0;
  for (int i = 0; i < seq->ref_frames; i++) {
    const RefCntBuffer* const b = cm_->ref_frame_map[i];
    if (b == nullptr || (!b->implicit_output_picture && !b->immediate_output_picture) ||
        b->is_restricted)
      continue;
    if (static_cast<int>(b->display_order_hint) > max_doh) max_doh = b->display_order_hint;
  }
  int cur = order_hint;
  const int factor = 1 << (seq->order_hint_info.order_hint_bits_minus_1 + 1);
  while (std::abs(max_doh - cur) >= (factor >> 1)) {
    if (cur > max_doh) return cur;
    cur += factor;
  }
  return cur;
}

std::vector<int> CodecState::ref_display_hints() const {
  std::vector<int> out;
  if (cm_->current_frame.frame_type == KEY_FRAME) return out;
  for (int i = 0; i < cm_->ref_frames_info.num_total_refs; i++)
    out.push_back(static_cast<int>(get_ref_frame_buf(cm_, i)->display_order_hint));
  return out;
}

int CodecState::slot_of(int display_order_hint) const {
  for (int i = 0; i < cm_->seq_params.ref_frames; i++)
    if (cm_->ref_frame_map[i] && static_cast<int>(cm_->ref_frame_map[i]->display_order_hint) == display_order_hint)
      return i;
  return -1;
}

void CodecState::begin_frame(const FrameSpec& spec) {
  const bool key = spec.key;
  qindex_delta_ = spec.qindex_delta;
  AV2_COMMON* const cm = cm_;
  const SequenceHeader* const seq = &cm->seq_params;
  CurrentFrame* const cf = &cm->current_frame;
  FeatureFlags* const f = &cm->features;
  const int num_planes = av2_num_planes(cm);
  if (!key && cm->ref_frame_map[0] == nullptr)
    throw std::runtime_error("an inter frame needs a preceding keyframe");

  cm->cur_frame = free_buffer();
  RefCntBuffer* const buf = cm->cur_frame;
  gen_[buf - pool_->frame_bufs]++;
  mirror_of_[buf - pool_->frame_bufs] = nullptr;  // no longer a copy of another state's frame (sync_refs_from)

  // read_uncompressed_header(): frame type, output flags, order hint, primary reference.
  init_bru_params(cm);
  cf->frame_type = key ? KEY_FRAME : INTER_FRAME;
  const int oh_bits = seq->order_hint_info.order_hint_bits_minus_1 + 1;
  if (spec.display >= 0)
    display_counter_ = spec.display;
  else
    display_counter_ = key ? frame_number_ % (1 << oh_bits) : display_counter_ + 1;
  cf->order_hint = display_counter_ % (1 << oh_bits);
  cf->display_order_hint = key ? cf->order_hint : derive_display_order_hint(cf->order_hint);
  cf->frame_number = cf->order_hint;
  cf->display_order_hint_restricted = cf->display_order_hint;
  cf->long_term_id = -1;
  // Slots 0..refs-1 take turns; the slots above hold the newest shown frame (with every frame shown,
  // the newest reference frame: a duplicate AV2 ranks once). One reference: every slot, as a
  // keyframe does. The display order hint of a frame is derived from the shown frames in the slots
  // (hidden ones do not count); with only hidden ones left it would restart at the 7-bit order
  // hint every 128 frames.
  const int all = (1 << seq->ref_frames) - 1, turns = (1 << cfg_.refs) - 1;
  if (key) {
    since_key_ = 0;
    cf->refresh_frame_flags = all;
  } else if (spec.reference) {
    since_key_++;
    cf->refresh_frame_flags = (1 << (since_key_ % cfg_.refs)) | (spec.shown ? all & ~turns : 0);
  } else {
    cf->refresh_frame_flags = spec.shown && spec.keep_shown ? all & ~turns : 0;
  }
  if (!key && spec.refresh >= 0) cf->refresh_frame_flags = spec.refresh;
  if (key && !spec.shown) throw std::runtime_error("keyframes are shown");
  cm->show_existing_frame = 0;
  cm->immediate_output_picture = spec.shown;
  cm->implicit_output_picture = 0;  // monotonic output order: a hidden frame waits for its SEF
  buf->immediate_output_picture = spec.shown;
  buf->implicit_output_picture = 0;
  buf->frame_output_done = 0;
  av2_set_frame_sb_size(cm, seq->sb_size);
  f->allow_intrabc = 0;
  f->allow_global_intrabc = 0;
  f->allow_local_intrabc = 0;
  f->primary_ref_frame = PRIMARY_REF_NONE;
  f->derived_primary_ref_frame = PRIMARY_REF_NONE;
  f->cross_frame_context = CROSS_FRAME_CONTEXT_FORWARD;  // signal_primary_ref_frame = 0
  f->allow_lf_sub_pu = 0;
  f->tip_frame_mode = TIP_FRAME_DISABLED;
  f->allow_ref_frame_mvs = 0;
  cf->pyramid_level = 1;
  if (key) {
    cm->prev_frame = nullptr;
    buf->num_ref_frames = 0;
  } else {
    init_ref_map_pair(cm, cm->ref_frame_map_pairs, 0, 0, 0);
    av2_get_ref_frames(cm, cf->display_order_hint, 0, 0, cm->ref_frame_map_pairs);
  }

  // setup_frame_size(): sequence size, frame buffer.
  ensure_mv_buffer(buf, cm);
  buf->width = cm->width;
  buf->height = cm->height;
  if (avm_realloc_frame_buffer(&buf->buf, cm->width, cm->height, seq->subsampling_x,
                               seq->subsampling_y, AVM_DEC_BORDER_IN_PIXELS, f->byte_alignment,
                               &buf->raw_frame_buffer, nullptr, nullptr, false))
    throw std::runtime_error("failed to allocate frame buffer");
  buf->buf.bit_depth = seq->bit_depth;
  buf->buf.render_width = cm->render_width;
  buf->buf.render_height = cm->render_height;
  buf->buf.max_width = seq->max_frame_width;
  buf->buf.max_height = seq->max_frame_height;
  realloc_bru_info(cm);

  // Screen content tools (--screen) as the frame's spec says, with local IntraBC and on key frames
  // global IntraBC; no forced integer MVs.
  if (spec.screen && !cfg_.screen) throw std::runtime_error("screen content tools need --screen");
  f->allow_screen_content_tools = spec.screen;
  f->cur_frame_force_integer_mv = 0;
  f->allow_intrabc = f->allow_local_intrabc = spec.screen;
  f->allow_global_intrabc = spec.screen && key;
  f->max_bvp_drl_bits = seq->def_max_bvp_drl_bits;
  if (!key) {
    av2_get_ref_frames(cm, cf->display_order_hint, 1, 0, cm->ref_frame_map_pairs);
    // cfg.refs reference frames, and with B frames the newest shown one (AV2 ranks at most 7)
    if (cm->ref_frames_info.num_total_refs < 1 ||
        cm->ref_frames_info.num_total_refs > std::min(INTER_REFS_PER_FRAME, cfg_.refs + 1))
      throw std::runtime_error("unexpected number of reference frames");
    // setup_bru_active_info(): BRU disabled in the sequence header.
    std::memset(cm->bru.active_mode_map, 2, sizeof(uint8_t) * cm->bru.total_units);
    cm->ref_frames_info.num_same_ref_compound =
        AVMMIN(seq->num_same_ref_compound, cm->ref_frames_info.num_total_refs);
    buf->num_ref_frames = cm->ref_frames_info.num_total_refs;
    cm->tmvp_sample_step = 1;
    cm->tmvp_sample_stepl2 = 0;
#ifdef CELV2_ORACLE
    if (celv2::oracle::level() >= 5 && celv2::oracle::frame(spec.display)->head.tmvp_sample_step == 2) cm->tmvp_sample_step = 2, cm->tmvp_sample_stepl2 = 1;
#endif
    cm->tip_global_motion.as_int = 0;
    cm->tip_interp_filter = MULTITAP_SHARP;
    cm->tip_global_wtd_index = 0;
    cm->has_both_sides_refs =
        cm->ref_frames_info.num_future_refs > 0 && cm->ref_frames_info.num_past_refs > 0;
    f->opfl_refine_type = seq->enable_opfl_refine;  // read_frame_opfl_refine_type(), not AUTO
    f->max_drl_bits = seq->def_max_drl_bits;        // read_frame_max_drl_bits()
    f->fr_mv_precision = MV_PRECISION_QTR_PEL;      // header bit 1
    f->most_probable_fr_mv_precision = f->fr_mv_precision;
    f->use_pb_mv_precision = seq->enable_flex_mvres;
    f->interp_filter = EIGHTTAP_REGULAR;  // not switchable, literal 0
#ifdef CELV2_ORACLE
    if (celv2::oracle::level() >= 2) {
      const OracleFrame& o = celv2::oracle::frame(spec.display)->head;
      f->fr_mv_precision = f->most_probable_fr_mv_precision = static_cast<MvSubpelPrecision>(o.fr_mv_precision);
      f->interp_filter = static_cast<InterpFilter>(o.interp_filter);
    }
#endif
    f->enabled_motion_modes = seq->seq_enabled_motion_modes;
    for (int i = 0; i < cm->ref_frames_info.num_total_refs; ++i) {
      const RefCntBuffer* const ref = get_ref_frame_buf(cm, i);
      av2_setup_scale_factors_for_frame(get_ref_scale_factors(cm, i), ref->buf.y_crop_width,
                                        ref->buf.y_crop_height, cm->width, cm->height);
    }
  }

  av2_setup_frame_buf_refs(cm);
  av2_setup_frame_sign_bias(cm);
  buf->frame_type = cf->frame_type;
  buf->order_hint = cf->order_hint;
  buf->display_order_hint = cf->display_order_hint;

  f->disable_cdf_update = 0;

  setup_tiles();
  if (static_cast<size_t>(cm->tiles.cols) * cm->tiles.rows != tiles_.size())
    throw std::runtime_error("tile layout changed between frames");

  // setup_quantization(): base qindex only.
  CommonQuantParams* const q = &cm->quant_params;
  q->base_qindex = cfg_.coded_qindex(std::max(1, std::min(255, cfg_.qindex + qindex_delta_)));
  q->y_dc_delta_q = q->u_dc_delta_q = q->u_ac_delta_q = q->v_dc_delta_q = q->v_ac_delta_q = 0;
  buf->base_qindex = q->base_qindex;
  buf->u_ac_delta_q = buf->v_ac_delta_q = 0;

  // set_primary_ref_frame_and_ctx()
  if (!key) {
    int tmp[2] = {0, 0};
    choose_primary_secondary_ref_frame(cm, tmp, 0);
    f->derived_primary_ref_frame = tmp[0];
    f->derived_secondary_ref_frame = tmp[1];
    f->primary_ref_frame = f->derived_primary_ref_frame;
  }
  if (f->derived_primary_ref_frame == PRIMARY_REF_NONE || f->primary_ref_frame == PRIMARY_REF_NONE) {
    f->primary_ref_frame = PRIMARY_REF_NONE;
    f->derived_primary_ref_frame = PRIMARY_REF_NONE;
    f->cross_frame_context = CROSS_FRAME_CONTEXT_DISABLED;
  }
  if (f->primary_ref_frame == PRIMARY_REF_NONE ||
      f->cross_frame_context == CROSS_FRAME_CONTEXT_DISABLED)
    av2_setup_past_independence(cm);
  else
    *cm->fc = get_primary_ref_frame_buf(cm, f->primary_ref_frame)->frame_context;
  if (!key) cm->prev_frame = get_primary_ref_frame_buf(cm, f->derived_primary_ref_frame);

  // setup_segmentation(): disabled.
  if (buf->seg_map) std::memset(buf->seg_map, 0, buf->mi_rows * buf->mi_cols);
  std::memset(&cm->seg, 0, sizeof(cm->seg));
  cm->seg.enable_ext_seg = seq->enable_ext_seg;
  segfeatures_copy(&buf->seg, &cm->seg);

  // Delta qindex per superblock (FrameSpec::sb_dq), in steps of kDeltaQRes from the frame's.
  cm->delta_q_info.delta_q_res = 1;
  cm->delta_q_info.delta_q_present_flag = 0;
  sb_q_.clear();
  if (!spec.sb_dq.empty()) {
    const int sb64_cols = (cfg_.width + 63) / 64, sb64_rows = (cfg_.height + 63) / 64;
    if (spec.sb_dq.size() != static_cast<size_t>(sb64_cols) * sb64_rows)
      throw std::invalid_argument("FrameSpec::sb_dq: one offset per 64x64 superblock");
    const int n = block_size_wide[cm->sb_size] / 64;  // 64x64 superblocks per coded one, each way
    sb_q_cols_ = (sb64_cols + n - 1) / n;
    const int rows = (sb64_rows + n - 1) / n, base = q->base_qindex;
    const int maxq = seq->bit_depth == AVM_BITS_8 ? MAXQ_8_BITS : MAXQ_10_BITS;
    bool any = false;
    for (int r = 0; r < rows; r++)
      for (int c = 0; c < sb_q_cols_; c++) {
        int sum = 0, cnt = 0;
        for (int y = r * n; y < std::min(sb64_rows, (r + 1) * n); y++)
          for (int x = c * n; x < std::min(sb64_cols, (c + 1) * n); x++) sum += spec.sb_dq[y * sb64_cols + x], cnt++;
        // the mean, rounded to the nearest step, kept within 1..maxq
        int d = static_cast<int>(std::lround(static_cast<double>(sum) / cnt / kDeltaQRes)) * kDeltaQRes;
        while (base + d < 1) d += kDeltaQRes;
        while (base + d > maxq) d -= kDeltaQRes;
        any = any || d != 0;
        sb_q_.push_back(base + d);
      }
    if (any) {
      cm->delta_q_info.delta_q_present_flag = 1;
      cm->delta_q_info.delta_q_res = kDeltaQRes;
    } else {
      sb_q_.clear();
    }
  }
#ifdef CELV2_ORACLE
  if (celv2::oracle::level() >= 5 && celv2::oracle::frame(spec.display)->head.delta_q_present) {  // the stream's, from its blocks
    const celv2::oracle::Frame& o = *celv2::oracle::frame(spec.display);
    const int sb = cm->mib_size;
    sb_q_cols_ = (cm->mi_params.mi_cols + sb - 1) / sb;
    sb_q_.assign(static_cast<size_t>(sb_q_cols_) * ((cm->mi_params.mi_rows + sb - 1) / sb), q->base_qindex);
    for (const OracleBlock& b : o.leaves)
      if (!(b.mi_row % sb) && !(b.mi_col % sb)) sb_q_[static_cast<size_t>(b.mi_row / sb) * sb_q_cols_ + b.mi_col / sb] = b.qindex;
    cm->delta_q_info.delta_q_present_flag = 1;
    cm->delta_q_info.delta_q_res = o.head.delta_q_res;
  }
#endif
  f->has_lossless_segment = 0;
  for (int i = 0; i < MAX_SEGMENTS; i++) f->lossless_segment[i] = 0;
  f->coded_lossless = 0;
  f->all_lossless = 0;
  f->tcq_mode = seq->enable_tcq;
  f->allow_parity_hiding = false;
  setup_dequant(cm);

  // setup_loopfilter(): deblocking on every plane (or none) with no strength deltas, so the
  // decoder derives the levels from qindex. CDEF, LR, CCSO, GDF: off.
  struct loopfilter* const lf = &cm->lf;
  lf->apply_deblocking_filter[0] = lf->apply_deblocking_filter[1] = cfg_.deblock;
  lf->apply_deblocking_filter_u = lf->apply_deblocking_filter_v = cfg_.deblock;
  const int dd = cfg_.deblock ? cfg_.deblock_delta : 0;  // pick_deblock() may change it after the tiles
  lf->delta_q_luma[0] = lf->delta_q_luma[1] = lf->delta_side_luma[0] = lf->delta_side_luma[1] = dd;
  lf->delta_q_u = lf->delta_q_v = lf->delta_side_u = lf->delta_side_v = dd;
  cm->gdf_info.gdf_mode = 0;
  cm->cdef_info.cdef_frame_enable = 0;
  for (int p = 0; p < MAX_MB_PLANE; p++) {
    cm->rst_info[p].frame_restoration_type = RESTORE_NONE;
    cm->rst_info[p].frame_filters_on = 0;
    buf->rst_info[p].frame_filters_on = 0;
  }
  // Loop restoration (SeqConfig::lr): luma PC_WIENER in 128x128 units (one per superblock), each
  // switched by restore() after the tiles; chroma, with lr_uv, WIENER_NONSEP in 64x64 units with one
  // filter per plane in the frame header (frame_filters_on, not predicted from a reference), else
  // none. The sizes decode_restoration_mode() sets.
  if (seq->enable_restoration) {
    for (int p = 0; p < num_planes; p++) av2_set_lr_tools(seq->lr_tools_disable_mask[p > 0], p, f);
    cm->rst_info[0].frame_restoration_type = RESTORE_PC_WIENER;
    cm->rst_info[0].restoration_unit_size = RESTORATION_UNITSIZE_MAX >> 2;
    const int ss = AVMMAX(seq->subsampling_x, seq->subsampling_y);
    const bool uv = num_planes > 1 && !((seq->lr_tools_disable_mask[1] >> RESTORE_WIENER_NONSEP) & 1);
    cm->rst_info[1].restoration_unit_size = cm->rst_info[2].restoration_unit_size =
        RESTORATION_UNITSIZE_MAX >> (uv ? 2 + ss : 3 + ss);
    for (int p = 1; uv && p < num_planes; p++) {
      RestorationInfo* const rsi = &cm->rst_info[p];
      rsi->frame_restoration_type = RESTORE_WIENER_NONSEP;
      rsi->frame_filters_on = 1;
      rsi->frame_filters_initialized = 0;
      rsi->temporal_pred_flag = 0;
      rsi->rst_ref_pic_idx = 0;
      rsi->num_filter_classes = 1;
      rsi->frame_filters.num_classes = 1;
    }
#ifdef CELV2_ORACLE
    if (celv2::oracle::filters()) import_lr_types(*celv2::oracle::frame(spec.display));
#endif
    av2_alloc_restoration_buffers(cm);
#ifdef CELV2_ORACLE
    if (celv2::oracle::filters()) import_lr_units(*celv2::oracle::frame(spec.display));
#endif
  }
  // CCSO on every plane and every unit (BlockWriter::ccso() codes the unit flags as tiles are
  // coded); the offsets come after CDEF (ccso()), in time for the frame header.
  CcsoInfo* const ci = &cm->ccso_info;
  ci->ccso_frame_flag = seq->enable_ccso;
  ci->ccso_blk_size = buf->ccso_info.ccso_blk_size =
      get_ccso_unit_size_log2_adaptive_tile(cm, cm->mib_size_log2 + MI_SIZE_LOG2, CCSO_BLK_SIZE);
  for (int p = 0; p < CCSO_NUM_COMPONENTS; p++) {
    ci->ccso_enable[p] = buf->ccso_info.ccso_enable[p] = seq->enable_ccso;
    ci->reuse_ccso[p] = ci->sb_reuse_ccso[p] = false;
  }

#ifdef CELV2_ORACLE
  if (celv2::oracle::filters()) import_filters(*celv2::oracle::frame(spec.display));
#endif
  f->tx_mode = TX_MODE_SELECT;  // inter blocks code their transform partitions
  cf->reference_mode =
      cfg_.compound && !key && cm->ref_frames_info.num_total_refs >= 2 ? REFERENCE_MODE_SELECT : SINGLE_REFERENCE;
#ifdef CELV2_ORACLE
  if (!key && celv2::oracle::level() >= 2) cf->reference_mode = static_cast<REFERENCE_MODE>(celv2::oracle::frame(spec.display)->head.reference_mode);
#endif
  av2_setup_skip_mode_allowed(cm);
  cf->skip_mode_info.skip_mode_flag = cf->skip_mode_info.skip_mode_allowed;
#ifdef CELV2_ORACLE
  if (!key && celv2::oracle::level() >= 2 && !celv2::oracle::frame(spec.display)->head.skip_mode_flag) cf->skip_mode_info.skip_mode_flag = 0;
#endif
  f->enable_bawp = !key && seq->enable_bawp;
  f->enable_intra_bawp = seq->enable_bawp;
  f->enable_cwp = seq->enable_cwp;
  f->allow_warpmv_mode = !key && (f->enabled_motion_modes & (1 << WARP_DELTA));  // WARP_NEWMV needs it
  f->enable_imp_msk_bld = seq->enable_imp_msk_bld;
  f->reduced_tx_set_used = 0;  // the full transform sets (intra luma codes its type too)
  f->enable_ext_seg = seq->enable_ext_seg;

  // av2_read_tilegroup_header(), first tile group.
  for (int i = 0; i < INTER_REFS_PER_FRAME; ++i) {
    cm->global_motion[i] = default_warp_params;
    buf->global_motion[i] = default_warp_params;
  }
  cm->mi_params.setup_mi(&cm->mi_params);
  // IntraBC predicts from the frame itself, unscaled (decodeframe.c, after the header).
  av2_setup_scale_factors_for_frame(&cm->sf_identity, buf->buf.y_crop_width, buf->buf.y_crop_height,
                                    buf->buf.y_crop_width, buf->buf.y_crop_height);
  // The decoder sets these on its one MACROBLOCKD; celv2 has one per tile.
  for (Tile& t : tiles_) {
    MACROBLOCKD* const xd = t.xd;
    xd->bd = static_cast<int>(seq->bit_depth);
    xd->cur_frame_force_integer_mv = f->cur_frame_force_integer_mv;
    for (int i = 0; i < MAX_SEGMENTS; i++) {
      xd->lossless[i] = 0;
      xd->qindex[i] = av2_get_qindex(&cm->seg, i, q->base_qindex, seq->bit_depth);
    }
    xd->global_motion = cm->global_motion;
    xd->cur_buf = &buf->buf;
    av2_setup_block_planes(xd, seq->subsampling_x, seq->subsampling_y, num_planes);
  }
  // read_uncompressed_header(): temporal MVs wherever the frame may use them.
  f->allow_ref_frame_mvs = frame_might_allow_ref_frame_mvs(cm);
  if (f->allow_ref_frame_mvs) {
    av2_setup_motion_field(cm);
    av2_fill_tpl_mvs_sample_gap(cm);  // process_tip_mode() without TIP
  } else {
    av2_setup_ref_frame_sides(cm);
  }
  if (f->primary_ref_frame == PRIMARY_REF_NONE ||
      f->cross_frame_context == CROSS_FRAME_CONTEXT_DISABLED)
    *cm->fc = *cm->default_frame_context;
  else
    *cm->fc = get_primary_ref_frame_buf(cm, f->primary_ref_frame)->frame_context;
}

void CodecState::setup_tiles() {
  AV2_COMMON* const cm = cm_;
  const SequenceHeader* const seq = &cm->seq_params;
  CommonTileParams* const tiles = &cm->tiles;
  av2_get_tile_limits(tiles, cm->mi_params.mi_rows, cm->mi_params.mi_cols, cm->mib_size_log2,
                      seq->mib_size_log2, seq->seq_max_level_idx, seq->seq_tier);
  tiles->uniform_spacing = 1;
  tiles->log2_cols =
      AVMMAX(tiles->min_log2_cols, AVMMIN(cfg_.tile_cols_log2, tiles->max_log2_cols));
  av2_calculate_tile_cols(tiles);
  tiles->log2_rows =
      AVMMAX(tiles->min_log2_rows, AVMMIN(cfg_.tile_rows_log2, tiles->max_log2_rows));
  av2_calculate_tile_rows(tiles);
  if (!tiles_warned_ &&
      (tiles->log2_cols != cfg_.tile_cols_log2 || tiles->log2_rows != cfg_.tile_rows_log2)) {
    std::fprintf(stderr,
                 "celv2: warning: tile log2 columns/rows %d/%d adjusted to %d/%d (allowed %d..%d/%d..%d)\n",
                 cfg_.tile_cols_log2, cfg_.tile_rows_log2, tiles->log2_cols, tiles->log2_rows,
                 tiles->min_log2_cols, tiles->max_log2_cols, tiles->min_log2_rows,
                 tiles->max_log2_rows);
    tiles_warned_ = true;
  }
}

int CodecState::sb_qindex(int mi_row, int mi_col) const {
  if (sb_q_.empty()) return cm_->quant_params.base_qindex;
  return sb_q_[static_cast<size_t>(mi_row >> cm_->mib_size_log2) * sb_q_cols_ + (mi_col >> cm_->mib_size_log2)];
}

void CodecState::begin_tile(int t) {
  AV2_COMMON* const cm = cm_;
  MACROBLOCKD* const xd = tiles_[t].xd;
  const int num_planes = av2_num_planes(cm);
  const int row = t / cm->tiles.cols, col = t % cm->tiles.cols;
  // decode_tiles()
  av2_tile_init(&xd->tile, cm, row, col);
  xd->current_base_qindex = cm->quant_params.base_qindex;
  av2_init_macroblockd(cm, xd);
  av2_init_above_context(&cm->above_contexts, num_planes, row, xd);
  xd->tile.tile_active_mode = 1;
  *tiles_[t].ctx = *cm->fc;
  xd->tile_ctx = tiles_[t].ctx;
  // decode_tile()
  av2_zero_above_context(cm, xd, xd->tile.mi_col_start, xd->tile.mi_col_end, row);
  int num_filter_classes[MAX_MB_PLANE];
  for (int p = 0; p < num_planes; ++p) num_filter_classes[p] = cm->rst_info[p].num_filter_classes;
  av2_reset_loop_restoration(xd, 0, num_planes, num_filter_classes);
}

void CodecState::pick_deblock(const Frame420& src) {
  AV2_COMMON* const cm = cm_;
  YV12_BUFFER_CONFIG& b = cm->cur_frame->buf;
  deblock_keep_.assign(b.buffer_alloc, b.buffer_alloc + b.frame_size);
  const bool high = src.bit_depth > 8;
  const int cw = (src.width + 1) / 2, ch = (src.height + 1) / 2;
  // Squared error of the deblocked frame against src, by rows of luma (and the chroma rows under them).
  auto sse = [&] {
    const int bands = std::max(1, std::min(static_cast<int>(std::thread::hardware_concurrency()), ch / 16));
    std::vector<double> part(bands, 0);
    celv2::parallel(bands, [&](int k) {
      double e = 0;
      auto add = [&](const uint16_t* r, auto s, int n) {
        for (int c = 0; c < n; c++) e += static_cast<double>(r[c] - s[c]) * (r[c] - s[c]);
      };
      for (int r = ch * k / bands; r < ch * (k + 1) / bands; r++) {
        for (int rr = 2 * r; rr < std::min(2 * r + 2, src.height); rr++) {
          const uint16_t* const y = b.y_buffer + static_cast<size_t>(rr) * b.y_stride;
          const size_t at = static_cast<size_t>(rr) * src.stride_y();
          high ? add(y, src.hy.data() + at, src.width) : add(y, src.y.data() + at, src.width);
        }
        const size_t at = static_cast<size_t>(r) * src.stride_uv();
        const uint16_t* const u = b.u_buffer + static_cast<size_t>(r) * b.uv_stride;
        const uint16_t* const v = b.v_buffer + static_cast<size_t>(r) * b.uv_stride;
        high ? add(u, src.hu.data() + at, cw) : add(u, src.u.data() + at, cw);
        high ? add(v, src.hv.data() + at, cw) : add(v, src.v.data() + at, cw);
      }
      part[k] = e;
    });
    double t = 0;
    for (double e : part) t += e;
    return t;
  };
  loopfilter* const lf = &cm->lf;
  auto set = [&](int dd) {
    lf->delta_q_luma[0] = lf->delta_q_luma[1] = lf->delta_side_luma[0] = lf->delta_side_luma[1] = dd;
    lf->delta_q_u = lf->delta_q_v = lf->delta_side_u = lf->delta_side_v = dd;
  };
  int best = 0;
  double best_e = 0;
  for (int dd : {-4, -2, 0, 2, 4}) {
    set(dd);
    deblock();
    const double e = sse();
    if (dd == -4 || e < best_e) best = dd, best_e = e;
    std::memcpy(b.buffer_alloc, deblock_keep_.data(), deblock_keep_.size());
  }
  set(best);
}

namespace {

// CELV2_PROFILE: the CPU cycles of AVM's worker threads so far. avm_thread.c keeps AVxWorkerImpl
// private; on Windows it is a CRITICAL_SECTION (40 bytes), a CONDITION_VARIABLE and the thread's HANDLE.
uint64_t workers_cycles(const std::vector<AVxWorker>& ws) {
  if (!celv2::profiling()) return 0;
  struct Impl {
    unsigned char mutex[40];
    void* cond;
    void* thread;
  };
  uint64_t c = 0;
  for (const AVxWorker& w : ws)
    if (w.impl_) c += celv2::thread_cycles(reinterpret_cast<const Impl*>(w.impl_)->thread);
  return c;
}

// Adds what the workers spent from construction to destruction to a stage.
struct WorkersScope {
  WorkersScope(celv2::Stage s, const std::vector<AVxWorker>& ws) : stage(s), workers(ws), before(workers_cycles(ws)) {}
  ~WorkersScope() {
    if (celv2::profiling()) celv2::add_cycles(stage, workers_cycles(workers) - before);
  }
  celv2::Stage stage;
  const std::vector<AVxWorker>& workers;
  uint64_t before;
};

}  // namespace

void CodecState::deblock() {
  celv2::StageScope prof(celv2::Stage::Deblock);
  AV2_COMMON* const cm = cm_;
  if (lf_workers_.empty()) {
    // One thread per superblock row at most; --threads caps it like the tile threads.
    const int sb_rows = (cm->mi_params.mi_rows + cm->mib_size - 1) >> cm->mib_size_log2;
    const int cores = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    const int n = std::max(1, std::min(sb_rows, cfg_.threads > 0 ? cfg_.threads : cores));
    const AVxWorkerInterface* const wi = avm_get_worker_interface();
    lf_workers_.resize(n);
    for (int i = 0; i < n; i++) {
      wi->init(&lf_workers_[i]);
      lf_workers_[i].thread_name = "celv2 deblock";
      if (i + 1 < n && !wi->reset(&lf_workers_[i])) {
        for (int k = 0; k < i; k++) wi->end(&lf_workers_[k]);
        lf_workers_.clear();
        throw std::runtime_error("deblocking thread creation failed");
      }
    }
  }
  WorkersScope avm_workers(celv2::Stage::Deblock, lf_workers_);
  if (lf_workers_.size() == 1)
    av2_loop_filter_frame(&cm->cur_frame->buf, cm, tiles_[0].xd, 0, av2_num_planes(cm), 0);
  else
    av2_loop_filter_frame_mt(&cm->cur_frame->buf, cm, tiles_[0].xd, 0, av2_num_planes(cm), 0,
                             lf_workers_.data(), static_cast<int>(lf_workers_.size()), &lf_sync_);
}

void CodecState::cdef(const Frame420& src) {
  celv2::StageScope prof(celv2::Stage::Cdef);
  WorkersScope avm_workers(celv2::Stage::Cdef, lf_workers_);
  AV2_COMMON* const cm = cm_;
  CdefInfo* const ci = &cm->cdef_info;
  const CommonModeInfoParams* const mi = &cm->mi_params;
  // read_cdef() with one strength and no filtering of skipped blocks: a 64x64 unit's first block
  // gets strength 0 if any of its blocks codes a residual, else -1 (not filtered).
  const int nvfb = (mi->mi_rows + MI_SIZE_64X64 - 1) / MI_SIZE_64X64;
  const int nhfb = (mi->mi_cols + MI_SIZE_64X64 - 1) / MI_SIZE_64X64;
  std::vector<int> units;  // filtered units, fbr * nhfb + fbc
  for (int fbr = 0; fbr < nvfb; fbr++)
    for (int fbc = 0; fbc < nhfb; fbc++) {
      bool coded = false;
      for (int r = fbr * MI_SIZE_64X64; r < std::min(mi->mi_rows, (fbr + 1) * MI_SIZE_64X64) && !coded; r++)
        for (int c = fbc * MI_SIZE_64X64; c < std::min(mi->mi_cols, (fbc + 1) * MI_SIZE_64X64); c++)
          if (!mi->mi_grid_base[r * mi->mi_stride + c]->skip_txfm[PLANE_TYPE_Y]) {
            coded = true;
            break;
          }
      mi->mi_grid_base[fbr * MI_SIZE_64X64 * mi->mi_stride + fbc * MI_SIZE_64X64]->cdef_strength = coded ? 0 : -1;
      if (coded) units.push_back(fbr * nhfb + fbc);
    }
  ci->cdef_frame_enable = 0;
  ci->cdef_on_skip_txfm_frame_enable = 0;
  ci->nb_cdef_strengths = 1;
  ci->cdef_damping = std::min(6, 3 + (cfg_.qindex_8bit(cm->quant_params.base_qindex) >> 6));
  if (units.empty()) return;

  // The one strength pair (luma, chroma) of least squared error against the source over at most
  // kSample of those units, each filtered from the deblocked frame as av2_cdef_frame() would (1080p:
  // 128 instead of all 510 took the encode from 19.3 to 24.9 fps for -0.01 dB at the same size).
  constexpr int kSample = 128;
  constexpr int kStrengths = CDEF_PRI_STRENGTHS * CDEF_SEC_STRENGTHS;
  const size_t step = (units.size() + kSample - 1) / kSample;
  const YV12_BUFFER_CONFIG& b = cm->cur_frame->buf;
  const int cores = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  // One unit a thread at least (the sums are integers: the same at any split).
  const int nt = std::max(1, std::min({cores, cfg_.threads > 0 ? cfg_.threads : cores,
                                       static_cast<int>((units.size() + step - 1) / step)}));
  std::vector<std::array<int64_t, 2 * kStrengths>> sse(nt);
  int damping = ci->cdef_damping;
  int only[2] = {-1, -1};  // >= 0: the one strength tried per plane type (else all)
  auto search = [&](int t) {
    sse[t].fill(0);
    std::vector<uint16_t> in(CDEF_INBUF_SIZE), out(64 * 64);
    cdef_list dlist[MI_SIZE_64X64 * MI_SIZE_64X64];
    int dir[CDEF_NBLOCKS][CDEF_NBLOCKS], var[CDEF_NBLOCKS][CDEF_NBLOCKS];
    for (size_t k = t * step; k < units.size(); k += nt * step) {
      const int fbr = units[k] / nhfb, fbc = units[k] % nhfb;
      const int count = av2_cdef_compute_sb_list(cm, mi, fbr * MI_SIZE_64X64, fbc * MI_SIZE_64X64,
                                                 dlist, BLOCK_64X64, MAX_MB_PLANE);
      if (!count) continue;
      int dirinit = 0;
      for (int p = 0; p < MAX_MB_PLANE; p++) {
        const int ss = p ? 1 : 0, bs = 8 >> ss, size = 64 >> ss;
        const uint16_t* const rec = p == 0 ? b.y_buffer : p == 1 ? b.u_buffer : b.v_buffer;
        const int rs = p ? b.uv_stride : b.y_stride;
        // the source at the coded depth (16-bit planes above 8 bits)
        const uint8_t* const org8 = p == 0 ? src.y.data() : p == 1 ? src.u.data() : src.v.data();
        const uint16_t* const org16 =
            src.bit_depth == 8 ? nullptr : p == 0 ? src.hy.data() : p == 1 ? src.hu.data() : src.hv.data();
        const int os = p ? src.stride_uv() : src.stride_y();
        const int fw = (mi->mi_cols * MI_SIZE) >> ss, fh = (mi->mi_rows * MI_SIZE) >> ss;
        const int vw = (src.width + ss) >> ss, vh = (src.height + ss) >> ss;
        const int y0 = fbr * size, x0 = fbc * size;
        for (int r = -CDEF_VBORDER; r < size + CDEF_VBORDER; r++)
          for (int c = -CDEF_HBORDER; c < size + CDEF_HBORDER; c++) {
            const int y = y0 + r, x = x0 + c;
            in[(r + CDEF_VBORDER) * CDEF_BSTRIDE + c + CDEF_HBORDER] =
                y >= 0 && y < fh && x >= 0 && x < fw ? rec[static_cast<size_t>(y) * rs + x] : CDEF_VERY_LARGE;
          }
        uint16_t* const blk = &in[CDEF_VBORDER * CDEF_BSTRIDE + CDEF_HBORDER];
        // Squared error of the packed 8x8 (4x4 chroma) blocks out[] (or the input) to the source.
        auto error = [&](const uint16_t* f, bool packed) {
          int64_t e = 0;
          for (int i = 0; i < count; i++) {
            const int by = y0 + dlist[i].by * bs, bx = x0 + dlist[i].bx * bs;
            if (packed && by + bs <= vh && bx + bs <= vw) {  // inside the picture: AVX2
              const size_t o = static_cast<size_t>(by) * os + bx;
              e += org16 ? block_sse(f + i * bs * bs, bs, org16 + o, os) : block_sse(f + i * bs * bs, bs, org8 + o, os);
              continue;
            }
            for (int r = 0; r < bs; r++) {
              const int y = y0 + dlist[i].by * bs + r;
              if (y >= vh) break;
              for (int c = 0; c < bs; c++) {
                const int x = x0 + dlist[i].bx * bs + c;
                if (x >= vw) break;
                const int v = packed ? f[i * bs * bs + r * bs + c] : f[(dlist[i].by * bs + r) * CDEF_BSTRIDE + dlist[i].bx * bs + c];
                const size_t o = static_cast<size_t>(y) * os + x;
                const int d = v - (org16 ? org16[o] : org8[o]);
                e += d * d;
              }
            }
          }
          return e;
        };
        int64_t* const acc = &sse[t][p ? kStrengths : 0];
        const int one = only[p > 0];
        if (one <= 0) acc[0] += error(blk, false);
        // Luma first: its directions (found on the first filtered strength) steer chroma too.
        for (int s = one > 0 ? one : 1; s < (one > 0 ? one + 1 : one == 0 ? 1 : kStrengths); s++) {
          const int pri = s / CDEF_SEC_STRENGTHS, sec = s % CDEF_SEC_STRENGTHS + (s % CDEF_SEC_STRENGTHS == 3);
          av2_cdef_filter_fb(nullptr, out.data(), 0, blk, ss, ss, dir, &dirinit, var, p, dlist, count, pri, sec,
                             damping, cm->seq_params.bit_depth - 8);  // coeff_shift, as av2_cdef_frame()
          acc[s] += error(out.data(), true);
        }
      }
    }
  };
  auto run = [&] { celv2::parallel(nt, search); };
  auto total = [&](int p, int s) {
    int64_t e = 0;
    for (int t = 0; t < nt; t++) e += sse[t][p * kStrengths + s];
    return e;
  };
  run();
  int best[2] = {0, 0};
  for (int p = 0; p < 2; p++) {
    int64_t least = INT64_MAX;
    for (int s = 0; s < kStrengths; s++)
      if (total(p, s) < least) least = total(p, s), best[p] = s;
  }
  // Then the damping (3..6) for that pair: on anime clips -0.9% / -0.1% BD-rate at qindex 45-180 /
  // 30-75 (searching every strength at every damping gained the same at 1.5x the encode time).
  int64_t least = total(0, best[0]) + total(1, best[1]);
  const int start = ci->cdef_damping;
  only[0] = best[0], only[1] = best[1];
  // (luma filtered: chroma takes its directions)
  for (damping = 3; damping <= 6 && best[0]; damping++) {
    if (damping == start) continue;
    run();
    const int64_t e = total(0, best[0]) + total(1, best[1]);
    if (e < least) least = e, ci->cdef_damping = damping;
  }
  if (!best[0] && !best[1]) return;
  ci->cdef_frame_enable = 1;
  ci->cdef_strengths[0] = best[0];
  ci->cdef_uv_strengths[0] = best[1];
  const int workers = static_cast<int>(lf_workers_.size());
  if (workers > 1) {
    if (cdef_workers_ != workers) {
      av2_alloc_cdef_buffers(cm, &cdef_worker_, &cdef_sync_, workers);
      av2_alloc_cdef_sync(cm, &cdef_sync_, workers);
      cdef_workers_ = workers;
    }
    av2_cdef_frame_mt(cm, tiles_[0].xd, cdef_worker_, lf_workers_.data(), &cdef_sync_, workers, av2_cdef_init_fb_row_mt);
  } else {
    if (cdef_workers_ != 1) av2_alloc_cdef_buffers(cm, &cdef_worker_, &cdef_sync_, 1), cdef_workers_ = 1;
    av2_cdef_frame(&cm->cur_frame->buf, cm, tiles_[0].xd, av2_cdef_init_fb_row);
  }
}

void CodecState::ccso(const Frame420& src, std::vector<uint16_t>& luma, double sse_per_bit) {
  celv2::StageScope prof(celv2::Stage::Ccso);
  AV2_COMMON* const cm = cm_;
  CcsoInfo* const ci = &cm->ccso_info;
  const YV12_BUFFER_CONFIG& b = cm->cur_frame->buf;
  const int stride = b.y_width + 2 * CCSO_PADDING_SIZE;
  const int threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  // The planes' searches at once (each one's counting is on its own threads; 1080p chroma has one).
  CcsoChoice choice[3];
  auto search = [&](int p) {
    const bool uv = p > 0;
    const uint16_t* const rec = p == 0 ? b.y_buffer : p == 1 ? b.u_buffer : b.v_buffer;
    const uint16_t* const lum = luma.data() + CCSO_PADDING_SIZE * stride + CCSO_PADDING_SIZE;
    const int rs = uv ? b.uv_stride : b.y_stride, ss = uv ? src.stride_uv() : src.stride_y();
    const int w = uv ? b.uv_crop_width : b.y_crop_width, h = uv ? b.uv_crop_height : b.y_crop_height;
    const int sx = uv ? cm->seq_params.subsampling_x : 0, sy = uv ? cm->seq_params.subsampling_y : 0;
    if (src.bit_depth > 8)  // the full-depth source
      choice[p] = ccso_search(lum, stride, rec, rs, p == 0 ? src.hy.data() : p == 1 ? src.hu.data() : src.hv.data(), ss,
                              w, h, sx, sy, sse_per_bit, threads, src.bit_depth);
    else
      choice[p] = ccso_search(lum, stride, rec, rs, p == 0 ? src.y.data() : p == 1 ? src.u.data() : src.v.data(), ss,
                              w, h, sx, sy, sse_per_bit, threads);
  };
  celv2::parallel(3, [&](int p) {
    if (ci->ccso_enable[p]) search(p);
  });
  for (int p = 0; p < 3; p++) {
    if (!ci->ccso_enable[p]) continue;
    const CcsoChoice& c = choice[p];
    ci->ccso_bo_only[p] = c.bo_only;
    ci->scale_idx[p] = static_cast<uint8_t>(c.scale_idx);
    ci->quant_idx[p] = static_cast<uint8_t>(c.quant_idx);
    ci->ext_filter_support[p] = static_cast<uint8_t>(c.ext_filter_support);
    ci->edge_clf[p] = static_cast<uint8_t>(c.edge_clf);
    ci->max_band_log2[p] = c.max_band_log2;
    std::memcpy(ci->filter_offset[p], c.offset, sizeof(c.offset));
    av2_copy_ccso_filters(&cm->cur_frame->ccso_info, ci, p, 1, 0, 0);
  }
  apply_ccso(luma);
}

// ccso.c ccso_frame(), with celv2's own filter: which 32x32 blocks av2_apply_ccso_filter_for_row()
// filters (the CCSO unit's flag and the superblock's local flag), then the rows of blocks of every
// plane on threads of their own (the AVM function took 0.22 s a 16K frame on one thread).
void CodecState::apply_ccso(const std::vector<uint16_t>& luma) {
  AV2_COMMON* const cm = cm_;
  const CcsoInfo* const ci = &cm->ccso_info;
  if (cm->seq_params.disable_loopfilters_across_tiles || cm->features.has_lossless_segment || cm->bru.enabled ||
      cm->bridge_frame_info.is_bridge_frame)
    throw std::logic_error("apply_ccso: tiles, lossless segments, BRU and bridge frames are not supported");
  YV12_BUFFER_CONFIG& b = cm->cur_frame->buf;
  const CommonModeInfoParams* const mi = &cm->mi_params;
  const int luma_stride = b.y_width + 2 * CCSO_PADDING_SIZE;
  const int blk_log2 = get_ccso_unit_size_log2_adaptive_tile(cm, cm->mib_size_log2 + MI_SIZE_LOG2, CCSO_BLK_SIZE);
  const int blk_size = 1 << blk_log2;
  celv2::CcsoPlane planes[3];
  std::vector<uint8_t> on[3];
  int rows[3] = {};
  for (int p = 0; p < 3; p++) {
    if (!ci->ccso_enable[p]) continue;
    celv2::CcsoPlane& c = planes[p];
    const bool uv = p > 0;
    c.luma = luma.data() + CCSO_PADDING_SIZE * luma_stride + CCSO_PADDING_SIZE;
    c.luma_stride = luma_stride;
    c.dst = p == 0 ? b.y_buffer : p == 1 ? b.u_buffer : b.v_buffer;
    c.dst_stride = uv ? b.uv_stride : b.y_stride;
    c.w = uv ? b.uv_width : b.y_width;
    c.h = uv ? b.uv_height : b.y_height;
    c.sx = uv ? cm->seq_params.subsampling_x : 0;
    c.sy = uv ? cm->seq_params.subsampling_y : 0;
    c.bit_depth = cm->seq_params.bit_depth;
    c.bo_only = ci->ccso_bo_only[p];
    c.thr = quant_sz[ci->scale_idx[p]][ci->quant_idx[p]];
    c.edge_clf = ci->edge_clf[p];
    celv2::ccso_neighbours(ci->ext_filter_support[p], luma_stride, c.loc);
    c.max_band_log2 = ci->max_band_log2[p];
    c.offset = ci->filter_offset[p];
    // av2_apply_ccso_filter_for_row()'s skips, per block.
    const int blk_log2_x = blk_log2 - c.sx, blk_log2_y = blk_log2 - c.sy;
    const int unit_log2 = cm->mib_size_log2 - std::max(c.sx, c.sy) + MI_SIZE_LOG2;
    const int unit_log2_x = std::min(unit_log2, blk_log2_x), unit_log2_y = std::min(unit_log2, blk_log2_y);
    const int cols = (c.w + (1 << celv2::kCcsoBlockLog2) - 1) >> celv2::kCcsoBlockLog2;
    rows[p] = (c.h + (1 << celv2::kCcsoBlockLog2) - 1) >> celv2::kCcsoBlockLog2;
    on[p].assign(static_cast<size_t>(cols) * rows[p], 0);
    for (int r = 0; r < rows[p]; r++)
      for (int k = 0; k < cols; k++) {
        const int y = r << celv2::kCcsoBlockLog2, x = k << celv2::kCcsoBlockLog2;
        const MB_MODE_INFO* const unit = mi->mi_grid_base[(blk_size >> MI_SIZE_LOG2) * (y >> blk_log2_y) * mi->mi_stride +
                                                         (blk_size >> MI_SIZE_LOG2) * (x >> blk_log2_x)];
        if (!(p == 0 ? unit->ccso_blk_y : p == 1 ? unit->ccso_blk_u : unit->ccso_blk_v)) continue;
        const int ym = (y >> unit_log2_y) << unit_log2_y, xm = (x >> unit_log2_x) << unit_log2_x;
        on[p][static_cast<size_t>(r) * cols + k] =
            mi->mi_grid_base[get_mi_grid_idx(mi, ym >> (MI_SIZE_LOG2 - c.sy), xm >> (MI_SIZE_LOG2 - c.sx))]->local_ccso_blk_flag;
      }
    c.on = on[p].data();
    c.on_stride = cols;
  }
  const int tasks = rows[0] + rows[1] + rows[2];
  // A thread per million samples or so: starting threads costs more than 1080p's filtering saves.
  int64_t samples = 0;
  for (int p = 0; p < 3; p++) samples += static_cast<int64_t>(planes[p].w) * planes[p].h * (rows[p] > 0);
  const int threads = static_cast<int>(std::clamp<int64_t>(samples >> 20, 1, std::min<int64_t>(std::thread::hardware_concurrency(), tasks)));
  std::atomic<int> next{0};
  celv2::parallel(threads, [&](int) {
    for (int t; (t = next.fetch_add(1)) < tasks;) {
      const int p = t < rows[0] ? 0 : t < rows[0] + rows[1] ? 1 : 2;
      celv2::ccso_apply_row(planes[p], t - (p > 0 ? rows[0] : 0) - (p > 1 ? rows[1] : 0));
    }
  });
}

void CodecState::save_contexts() {
  AV2_COMMON* const cm = cm_;
  // av2_decode_tg_tiles_and_wrapup(): no CDF averaging (enable_avg_cdf = 0), so the CDFs of
  // tile context_update_tile_id (always 0) become the frame's.
  if (!cm->features.disable_cdf_update) {
    *cm->fc = *tiles_[0].ctx;
    av2_reset_cdf_symbol_counters(cm->fc);
  }
  cm->cur_frame->frame_context = *cm->fc;
}

// Least squares for plane p's chroma WIENER_NONSEP taps (wienerns_filter_uv: 6 symmetric chroma pairs
// and 12 luma taps, all against the centre sample) against src, over the samples 2 or more from the
// edges, the luma downsampled as the decoder's filter has it (cfl_ds_filter_index) without its stripe
// boundaries: the normal equations' matrix a (18x18) and vector v. Taps t then change the plane's
// squared error by about (t.a.t / 128 - 2 t.v) / 128 (wienerns_gain()).
void CodecState::fit_wienerns(const Frame420& src, int p, double* a, double* v) const {
  const AV2_COMMON* const cm = cm_;
  const YV12_BUFFER_CONFIG& b = cm->cur_frame->buf;
  const NonsepFilterConfig& nc = get_wienerns_parameters(cm->quant_params.base_qindex, 1)->nsfilter_config;
  static constexpr int N = kWienerTaps;
  const int w = b.uv_crop_width, h = b.uv_crop_height, st = b.uv_stride, ys = b.y_stride;
  const int ds = cm->seq_params.cfl_ds_filter_index;
  std::vector<int16_t> luma(static_cast<size_t>(w) * h);
  celv2::parallel(h, [&](int r) {
    for (int c = 0; c < w; c++) {
      const uint16_t* const q = b.y_buffer + static_cast<size_t>(2 * r) * ys + 2 * c;
      luma[static_cast<size_t>(r) * w + c] =
          static_cast<int16_t>(ds == 1 ? (q[0] + q[ys]) >> 1 : ds == 2 ? q[0] : (q[0] + q[1] + q[ys] + q[ys + 1]) >> 2);
    }
  });
  const uint16_t* const x = b.buffers[p];
  const uint16_t* const s16 = src.bit_depth > 8 ? (p == 1 ? src.hu : src.hv).data() : nullptr;
  const uint8_t* const s8 = (p == 1 ? src.u : src.v).data();
  // The sums in doubles (v as column N of a): every product is an integer below 2^24 and every sum one
  // below 2^53 (a 16K 10-bit plane), so they are exact in any order, the int64 sums' values. A row's
  // samples first, then each 4x4 tile of the upper triangle summed over them in registers (AM 1080p 120
  // frames: 2.53 -> 0.95 s, the encode 5.26 -> 5.43 fps).
  static constexpr int NP = 20;  // N, t, padded to 4 lanes
  struct alignas(32) Sums {
    double a[NP][NP] = {};
  };
  const int bands = std::max(1, std::min(static_cast<int>(std::thread::hardware_concurrency()), (h - 4) / 16));
  std::vector<Sums> part(bands);
  celv2::parallel(bands, [&](int k) {
    Sums& m = part[k];
    const int n = w - 4;
    std::vector<double> rowf(static_cast<size_t>(std::max(n, 0)) * NP);
    for (int r = 2 + (h - 4) * k / bands; r < 2 + (h - 4) * (k + 1) / bands; r++) {
      for (int c = 2; c < w - 2; c++) {
        const uint16_t* const o = x + static_cast<size_t>(r) * st + c;
        const int16_t* const l = &luma[static_cast<size_t>(r) * w + c];
        double* const f = &rowf[static_cast<size_t>(c - 2) * NP];
        for (int i = 0; i < nc.num_pixels; i += 2) {
          const int d = nc.config[i][NONSEP_ROW_ID] * st + nc.config[i][NONSEP_COL_ID];
          f[nc.config[i][NONSEP_BUF_POS]] = o[d] + o[-d] - 2 * o[0];
        }
        for (int i = 0; i < nc.num_pixels2; i++)
          f[nc.config2[i][NONSEP_BUF_POS]] = l[nc.config2[i][NONSEP_ROW_ID] * w + nc.config2[i][NONSEP_COL_ID]] - l[0];
        const size_t at = static_cast<size_t>(r) * src.stride_uv() + c;
        f[N] = (s16 ? s16[at] : s8[at]) - o[0];
      }
      for (int i0 = 0; i0 < N; i0 += 4)
        for (int j0 = i0; j0 < NP; j0 += 4) {
          __m256d s0 = _mm256_setzero_pd(), s1 = s0, s2 = s0, s3 = s0;
          for (int c = 0; c < n; c++) {
            const double* const f = &rowf[static_cast<size_t>(c) * NP];
            const __m256d fj = _mm256_loadu_pd(f + j0);
            s0 = _mm256_add_pd(s0, _mm256_mul_pd(_mm256_broadcast_sd(f + i0), fj));
            s1 = _mm256_add_pd(s1, _mm256_mul_pd(_mm256_broadcast_sd(f + i0 + 1), fj));
            s2 = _mm256_add_pd(s2, _mm256_mul_pd(_mm256_broadcast_sd(f + i0 + 2), fj));
            s3 = _mm256_add_pd(s3, _mm256_mul_pd(_mm256_broadcast_sd(f + i0 + 3), fj));
          }
          const __m256d s[4] = {s0, s1, s2, s3};
          for (int q = 0; q < 4; q++)
            _mm256_store_pd(&m.a[i0 + q][j0], _mm256_add_pd(_mm256_load_pd(&m.a[i0 + q][j0]), s[q]));
        }
    }
  });
  for (int i = 0; i < N; i++)
    for (int j = i; j <= N; j++) {
      double sum = 0;
      for (const Sums& m : part) sum += m.a[i][j];
      if (j < N) a[i * N + j] = a[j * N + i] = sum;
      else v[i] = sum;
    }
}

namespace {

constexpr int kWienerN = CodecState::kWienerTaps;

// The squared error taps t (in 1/128) take off the plane, by the normal equations of fit_wienerns().
double wienerns_gain(const double* a, const double* v, const int16_t* t) {
  double g = 0;
  for (int i = 0; i < kWienerN; i++) {
    double r = 0;
    for (int j = 0; j < kWienerN; j++) r += a[i * kWienerN + j] * t[j];
    g += 2 * v[i] * t[i] * 128 - r * t[i];
  }
  return g / (128.0 * 128.0);
}

// The taps of subset s (the others 0) with the least squared error, a little ridge so flat planes stay
// solvable, by Gaussian elimination; rounded and clipped to the coded ranges.
void wienerns_solve(const double* a0, const double* v0, const WienernsFilterParameters* prm, int s, int16_t* t) {
  int idx[kWienerN], n = 0;
  for (int i = 0; i < kWienerN; i++)
    if (prm->subset_config[s][i]) idx[n++] = i;
  double m[kWienerN][kWienerN + 1];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) m[i][j] = a0[idx[i] * kWienerN + idx[j]];
    m[i][i] += 1e-3 * m[i][i] + 1;
    m[i][n] = v0[idx[i]];
  }
  for (int i = 0; i < n; i++) {
    int piv = i;
    for (int r = i + 1; r < n; r++)
      if (std::abs(m[r][i]) > std::abs(m[piv][i])) piv = r;
    std::swap(m[i], m[piv]);
    for (int r = 0; r < n; r++) {
      if (r == i || m[i][i] == 0) continue;
      const double q = m[r][i] / m[i][i];
      for (int j = i; j <= n; j++) m[r][j] -= q * m[i][j];
    }
  }
  std::fill(t, t + kWienerN, 0);
  for (int i = 0; i < n; i++) {
    const int k = idx[i], lo = prm->coeffs[k][WIENERNS_MIN_ID], hi = lo + (1 << prm->coeffs[k][WIENERNS_BIT_ID]) - 1;
    const double x = m[i][i] != 0 ? m[i][n] / m[i][i] * (1 << prm->nsfilter_config.prec_bits) : 0;
    t[k] = static_cast<int16_t>(std::clamp(static_cast<int>(std::lround(x)), lo, hi));
  }
}

}  // namespace

// read_wienerns_framefilters_hdr() for plane p's one class, in the group the decoder predicts:
// `match` (from wienerns_matches()) the dictionary filter it predicts from, into *ref; with s < 0 the
// filter is that one (exact_match), else the taps t of subset s coded against it (no symmetric
// pairs). Returns the bits; *out, when given, gets them one a byte. Before the reference map moves
// on: the dictionary holds the references' filters.
int CodecState::wienerns_syntax(int p, int match, int s, const int16_t* t, int16_t* ref, std::vector<uint8_t>* out) {
  AV2_COMMON* const cm = cm_;
  RestorationInfo* const rsi = &cm->rst_info[p];
  const int nopcw = disable_pcwiener_filters_in_framefilters(&cm->seq_params);
  const int base_qindex = cm->quant_params.base_qindex;
  *cm->num_ref_filters = set_frame_filter_dictionary(p, cm, rsi->num_filter_classes, cm->frame_filter_dictionary,
                                                     cm->frame_filter_dictionary_stride);
  WienerNonsepInfo* const ff = &rsi->frame_filters;
  ff->num_classes = rsi->num_filter_classes;
  ff->num_ref_filters = *cm->num_ref_filters;
  int group_counts[NUM_MATCH_GROUPS];
  set_group_counts(p, ff->num_classes, ff->num_ref_filters, group_counts, nopcw);
  int only;
  const int group = predict_group(0, ff->match_indices, group_counts, &only);
  const int base = get_group_base(group, group_counts), n = group == 0 ? 1 : group_counts[group];
  const int pred = predict_within_group(group, 0, ff->match_indices, group_counts);
  ff->match_indices[0] = match;
  uint8_t buf[256] = {};
  avm_write_bit_buffer wb = {buf, 0};
  if (!only) avm_wb_write_bit(&wb, 0);  // the predicted group
  if (n > 1) avm_wb_write_primitive_refsubexpfin(&wb, n, 4, pred - base, match - base);
  WienerNonsepInfoBank bank = {};
  bank.filter[0].num_classes = ff->num_classes;
  fill_first_slot_of_bank_with_filter_match(p, &bank, ff, ff->match_indices, base_qindex, 0, cm->frame_filter_dictionary,
                                            cm->frame_filter_dictionary_stride, nopcw);
  const WienernsFilterParameters* const prm = get_wienerns_parameters(base_qindex, p != AVM_PLANE_Y);
  std::copy_n(const_nsfilter_taps(av2_constref_from_wienerns_bank(&bank, 0, 0), 0), prm->ncoeffs, ref);
  const bool exact = s < 0 || std::equal(t, t + prm->ncoeffs, ref);
  avm_wb_write_bit(&wb, exact);
  if (!exact) {
    for (int i = 0; i < prm->nsubsets - 1; i++) {
      avm_wb_write_bit(&wb, i < s);
      if (i >= s) break;
    }
    if (!skip_sym_bit(prm, s)) avm_wb_write_bit(&wb, 0);
    for (int i = 0; i < prm->ncoeffs; i++) {
      if (!prm->subset_config[s][i]) {
        if (t[i]) throw std::logic_error("a Wiener tap outside its subset");
        continue;
      }
      const int lo = prm->coeffs[i][WIENERNS_MIN_ID], nb = prm->coeffs[i][WIENERNS_BIT_ID];
      avm_wb_write_primitive_refsubexpfin(&wb, 1 << nb, nb - 3, ref[i] - lo, t[i] - lo);
    }
  }
  if (out) {
    out->clear();
    for (uint32_t i = 0; i < wb.bit_offset; i++) out->push_back((buf[i >> 3] >> (7 - (i & 7))) & 1);
  }
  return static_cast<int>(wb.bit_offset);
}

// The match indices wienerns_syntax() can take: the predicted group's.
std::pair<int, int> CodecState::wienerns_matches(int p) {
  AV2_COMMON* const cm = cm_;
  const RestorationInfo& rsi = cm->rst_info[p];
  const int nopcw = disable_pcwiener_filters_in_framefilters(&cm->seq_params);
  const int nref = set_frame_filter_dictionary(p, cm, rsi.num_filter_classes, cm->frame_filter_dictionary,
                                               cm->frame_filter_dictionary_stride);
  int group_counts[NUM_MATCH_GROUPS];
  set_group_counts(p, rsi.num_filter_classes, nref, group_counts, nopcw);
  int only;
  const int group = predict_group(0, rsi.frame_filters.match_indices, group_counts, &only);
  const int base = get_group_base(group, group_counts);
  return {base, base + (group == 0 ? 1 : group_counts[group])};
}

void CodecState::restore(const Frame420& src, bool optimized, double luma_sse_per_bit) {
  celv2::StageScope prof(celv2::Stage::Restoration);
  WorkersScope avm_workers(celv2::Stage::Restoration, lf_workers_);
  AV2_COMMON* const cm = cm_;
  // A chroma bit is weighed as 4 bits against chroma error: weights 1 / 2 / 4 / 8 gave BD-rate Y on
  // AM demo, live action, AIR OP and kaguya of up to +0.66 / +0.65 / -0.01 / +0.09% (4: all below 0).
  constexpr double uv_k = 4;
  YV12_BUFFER_CONFIG& b = cm->cur_frame->buf;
  const int planes = av2_num_planes(cm);
  // A WIENER_NONSEP plane's filter: of a dictionary filter as it is and the least-squares taps of each
  // subset coded against one, the one whose estimated error less its bits' worth is best.
  struct Choice {
    int match = 0, s = -1, bits = 0;
    int16_t t[kWienerTaps] = {};
  };
  Choice chosen[MAX_MB_PLANE];
  // The filter also writes the rows and columns past the visible frame up to the aligned size; an
  // unit put back takes those too, its error counts only the visible pixels.
  for (int p = 0; p < planes; p++) {
    RestorationInfo* const rsi = &cm->rst_info[p];
    if (rsi->frame_restoration_type == RESTORE_NONE) continue;
    lr_keep_[p].resize(static_cast<size_t>(b.strides[p > 0]) * b.heights[p > 0]);
    std::memcpy(lr_keep_[p].data(), b.buffers[p], sizeof(uint16_t) * lr_keep_[p].size());
    const int n = rsi->horz_units_per_frame * rsi->vert_units_per_frame;
    if (rsi->frame_restoration_type == RESTORE_PC_WIENER) {
      for (int i = 0; i < n; i++) rsi->unit_info[i].restoration_type = RESTORE_PC_WIENER;
      continue;
    }
    const double sse_per_bit = luma_sse_per_bit * uv_k;
    const WienernsFilterParameters* const prm = get_wienerns_parameters(cm->quant_params.base_qindex, 1);
    if (prm->ncoeffs != kWienerTaps) throw std::logic_error("chroma Wiener taps other than 18");
    double a[kWienerTaps * kWienerTaps], v[kWienerTaps];
    fit_wienerns(src, p, a, v);
    int16_t fit[3][kWienerTaps], ref[kWienerTaps];
    for (int s = 0; s < prm->nsubsets; s++) wienerns_solve(a, v, prm, s, fit[s]);
    double best = -1e300;
    const auto [m0, m1] = wienerns_matches(p);
    for (int m = m0; m < m1; m++) {
      const int bits = wienerns_syntax(p, m, -1, nullptr, ref, nullptr);
      const double e = wienerns_gain(a, v, ref) - sse_per_bit * bits;
      if (e > best) {
        best = e, chosen[p] = {m, -1, bits, {}};
        std::copy_n(ref, kWienerTaps, chosen[p].t);
      }
      for (int s = 0; s < prm->nsubsets; s++) {
        const int fb = wienerns_syntax(p, m, s, fit[s], ref, nullptr);
        const double f = wienerns_gain(a, v, fit[s]) - sse_per_bit * fb;
        if (f > best) {
          best = f, chosen[p] = {m, s, fb, {}};
          std::copy_n(fit[s], kWienerTaps, chosen[p].t);
        }
      }
    }
    std::copy_n(chosen[p].t, kWienerTaps, nsfilter_taps(&rsi->frame_filters, 0));
    for (int i = 0; i < n; i++) {
      RestorationUnitInfo* const rui = &rsi->unit_info[i];
      rui->restoration_type = RESTORE_WIENER_NONSEP;
      rui->wienerns_info.num_classes = rsi->num_filter_classes;
      copy_nsfilter_taps(&rui->wienerns_info, &rsi->frame_filters);
    }
  }
  // On the deblocking workers when there are several (1 thread: 0.91x the speed at 1080p).
  if (lf_workers_.size() > 1)
    av2_loop_restoration_filter_frame_mt(&b, cm, optimized, lf_workers_.data(), static_cast<int>(lf_workers_.size()),
                                         &lr_sync_, &lr_ctxt_);
  else
    av2_loop_restoration_filter_frame(&b, cm, optimized, &lr_ctxt_);
  // Per plane, each unit kept where it brings the plane nearer src by more than a bit's worth; the
  // planes' flags one after the other.
  lr_flags_.clear();
  for (int p = 0; p < planes; p++) {
    RestorationInfo* const rsi = &cm->rst_info[p];
    if (rsi->frame_restoration_type == RESTORE_NONE) continue;
    const int uv = p > 0;
    const double sse_per_bit = uv ? luma_sse_per_bit * uv_k : luma_sse_per_bit;
    const int w = b.crop_widths[uv], h = b.crop_heights[uv], aw = b.widths[uv], ah = b.heights[uv], stride = b.strides[uv];
    const int size = rsi->restoration_unit_size;
    const int cols = rsi->horz_units_per_frame, rows = rsi->vert_units_per_frame;
    const size_t base = lr_flags_.size();
    lr_flags_.resize(base + static_cast<size_t>(cols) * rows, 0);
    const uint16_t* const s16 = src.bit_depth > 8 ? (p == 0 ? src.hy : p == 1 ? src.hu : src.hv).data() : nullptr;
    const uint8_t* const s8 = (p == 0 ? src.y : p == 1 ? src.u : src.v).data();
    const int sstride = uv ? src.stride_uv() : src.stride_y();
    uint16_t* const buf = b.buffers[p];
    const std::vector<uint16_t>& kept = lr_keep_[p];
    // The unit's rectangle (av2_get_rutile_rect(): units start again at each tile), to the aligned
    // size at the frame's right and bottom.
    auto unit = [&](int r, int c, int* x0, int* x1, int* y0, int* y1) {
      const AV2PixelRect u = av2_get_rutile_rect(cm, p, r, r + 1, c, c + 1, size, size);
      *x0 = u.left, *x1 = u.right >= w ? aw : u.right, *y0 = u.top, *y1 = u.bottom >= h ? ah : u.bottom;
    };
    auto put_back = [&](int r, int c) {
      int x0, x1, y0, y1;
      unit(r, c, &x0, &x1, &y0, &y1);
      rsi->unit_info[r * cols + c].restoration_type = RESTORE_NONE;
      lr_flags_[base + static_cast<size_t>(r) * cols + c] = 0;
      for (int y = y0; y < y1; y++)
        std::memcpy(buf + static_cast<size_t>(y) * stride + x0, kept.data() + static_cast<size_t>(y) * stride + x0,
                    sizeof(uint16_t) * (x1 - x0));
    };
    std::vector<double> row_gain(rows, 0.0);
    celv2::parallel(rows, [&](int r) {
      for (int c = 0; c < cols; c++) {
        int x0, x1, y0, y1;
        unit(r, c, &x0, &x1, &y0, &y1);
        int64_t on = 0, no = 0;
        for (int y = y0; y < AVMMIN(y1, h); y++)
          for (int x = x0; x < AVMMIN(x1, w); x++) {
            const size_t at = static_cast<size_t>(y) * sstride + x;
            const int s = s16 ? s16[at] : s8[at];
            const int a = buf[static_cast<size_t>(y) * stride + x] - s;
            const int k = kept[static_cast<size_t>(y) * stride + x] - s;
            on += a * a, no += k * k;
          }
        const bool keep = static_cast<double>(on) + sse_per_bit < static_cast<double>(no);
        lr_flags_[base + static_cast<size_t>(r) * cols + c] = keep;
        if (keep)
          row_gain[r] += static_cast<double>(no - on);
        else
          put_back(r, c);
      }
    });
    if (rsi->frame_restoration_type != RESTORE_WIENER_NONSEP) continue;
    // Units keeping the filter for less than its bits and the flags' (their entropy): the plane goes
    // without restoration (frame type NONE, the tiles' flags for it not coded, SymbolLog::replay()).
    double gain = 0;
    int kept_units = 0;
    for (const double g : row_gain) gain += g;
    for (size_t i = base; i < lr_flags_.size(); i++) kept_units += lr_flags_[i];
    const double q = static_cast<double>(kept_units) / (cols * rows);
    const double flag_bits = q > 0 && q < 1 ? -(cols * rows) * (q * std::log2(q) + (1 - q) * std::log2(1 - q)) : 0;
    const Choice* const c = &chosen[p];
    if (gain <= sse_per_bit * (c->bits + flag_bits)) {
      for (int r = 0; r < rows; r++)
        for (int k = 0; k < cols; k++)
          if (lr_flags_[base + static_cast<size_t>(r) * cols + k]) put_back(r, k);
      std::fill(lr_flags_.begin() + base, lr_flags_.end(), kLrSkip);
      rsi->frame_restoration_type = RESTORE_NONE;
      rsi->frame_filters_on = 0;
      cm->cur_frame->rst_info[p].frame_filters_on = 0;
      continue;
    }
    int16_t ref[kWienerTaps];
    wienerns_syntax(p, c->match, c->s, c->t, ref, &lr_filter_bits_[p]);
    rsi->frame_filters_initialized = 1;
    av2_copy_rst_frame_filters(&cm->cur_frame->rst_info[p], rsi);
  }
}

#ifdef CELV2_ORACLE
#include "filter_import.inc"
#endif

void CodecState::end_frame(const Frame420& src, double sse_per_bit) {
  AV2_COMMON* const cm = cm_;
  // decode_tiles() then av2_decode_tg_tiles_and_wrapup(): the in-loop filter runs on the whole
  // frame before it becomes a reference.
#ifdef CELV2_ORACLE
  const bool own_filters = !celv2::oracle::filters();
  if (!own_filters) import_end_frame();
#else
  constexpr bool own_filters = true;
#endif
  if (own_filters) {
  if (cm->lf.apply_deblocking_filter[0] || cm->lf.apply_deblocking_filter[1]) {
    if (cfg_.deblock_search) pick_deblock(src);
    deblock();
  }
  // av2_decode_tg_tiles_and_wrapup(): CCSO reads the deblocked luma, before CDEF.
  std::vector<uint16_t>& ccso_luma = ccso_luma_;
  const CcsoInfo* const ci = &cm->ccso_info;
  const bool use_ccso = ci->ccso_enable[0] || ci->ccso_enable[1] || ci->ccso_enable[2];
  if (use_ccso) {
    // The luma padded as extend_ccso_border() pads it, in bands of rows on several threads (one
    // thread, with a fresh buffer each frame, took about 0.1 s a 16K frame).
    const YV12_BUFFER_CONFIG& b = cm->cur_frame->buf;
    const int d = CCSO_PADDING_SIZE, w = b.y_width, h = b.y_height, stride = w + 2 * d;
    ccso_luma.resize(static_cast<size_t>(stride) * (h + 2 * d));
    auto row = [&](int r) { return &ccso_luma[static_cast<size_t>(r + d) * stride]; };
    const int bands = std::max(1, std::min(static_cast<int>(std::thread::hardware_concurrency()), h / 32));
    celv2::parallel(bands, [&](int k) {
      const int r0 = h * k / bands, r1 = h * (k + 1) / bands;
      for (int r = r0; r < r1; r++) {
        uint16_t* const p = row(r);
        std::memcpy(p + d, b.y_buffer + static_cast<size_t>(r) * b.y_stride, sizeof(uint16_t) * w);
        std::fill(p, p + d, p[d]);
        std::fill(p + d + w, p + stride, p[d + w - 1]);
      }
      for (int y = 1; k == 0 && y <= d; y++) std::memcpy(row(-y), row(0), sizeof(uint16_t) * stride);
      for (int y = 1; k == bands - 1 && y <= d; y++) std::memcpy(row(h - 1 + y), row(h - 1), sizeof(uint16_t) * stride);
    });
  }
  // av2_decode_tg_tiles_and_wrapup(): with loop restoration, the deblocked frame's stripe
  // boundaries are saved before CDEF and the CCSO output's after (unless no filter runs between).
  const bool lr = cm->rst_info[0].frame_restoration_type != RESTORE_NONE;
  if (lr) av2_loop_restoration_save_boundary_lines(&cm->cur_frame->buf, cm, 0);
  if (cm->seq_params.enable_cdef) cdef(src);
  if (use_ccso) ccso(src, ccso_luma, sse_per_bit);
  if (lr) {
    const bool optimized = !cm->seq_params.disable_loopfilters_across_tiles && !use_ccso &&
                           !(cm->seq_params.enable_cdef && cm->cdef_info.cdef_frame_enable);
    if (!optimized) av2_loop_restoration_save_boundary_lines(&cm->cur_frame->buf, cm, 1);
    restore(src, optimized, sse_per_bit);
  }
  }
  for (int i = 0; i < REF_FRAMES; i++)
    if (cm->current_frame.refresh_frame_flags & (1 << i)) cm->ref_frame_map[i] = cm->cur_frame;

  // The 8-bit copy (and above 8 bits the full-depth one), in bands of rows on several threads (one
  // thread took 0.16 s a 16K frame).
  const YV12_BUFFER_CONFIG& b = cm->cur_frame->buf;
  const int cw = (recon_.width + 1) / 2, ch = (recon_.height + 1) / 2;
  const int bands = std::max(1, std::min(static_cast<int>(std::thread::hardware_concurrency()), ch / 32));
  const int shift = recon_.bit_depth - 8;
  auto copy_band = [&](int k) {
    auto row = [&](std::vector<uint8_t>& p8, std::vector<uint16_t>& p16, size_t at, const uint16_t* s, int n) {
      if (!shift) {
        for (int c = 0; c < n; c++) p8[at + c] = static_cast<uint8_t>(s[c]);
        return;
      }
      for (int c = 0; c < n; c++)
        p16[at + c] = s[c], p8[at + c] = static_cast<uint8_t>(std::min(255, (s[c] + (1 << (shift - 1))) >> shift));
    };
    for (int r = ch * k / bands; r < ch * (k + 1) / bands; r++) {
      for (int rr = 2 * r; rr < std::min(2 * r + 2, recon_.height); rr++)
        row(recon_.y, recon_.hy, static_cast<size_t>(rr) * recon_.stride_y(), b.y_buffer + static_cast<size_t>(rr) * b.y_stride,
            recon_.width);
      row(recon_.u, recon_.hu, static_cast<size_t>(r) * recon_.stride_uv(), b.u_buffer + static_cast<size_t>(r) * b.uv_stride, cw);
      row(recon_.v, recon_.hv, static_cast<size_t>(r) * recon_.stride_uv(), b.v_buffer + static_cast<size_t>(r) * b.uv_stride, cw);
    }
  };
  celv2::parallel(bands, copy_band);
  pad_frame(recon_);
  frame_number_++;
}
