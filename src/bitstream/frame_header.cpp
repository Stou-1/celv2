// Portions of this file are derived from the AVM v1.0.0 decoder (av2/decoder),
// Copyright (c) 2021, Alliance for Open Media, under the BSD 3-Clause Clear License
// and the Alliance for Open Media Patent License 1.0: see third_party/avm/.

#include "bitstream/frame_header.h"

#include <stdexcept>

extern "C" {
#include "av2/common/ccso.h"
#include "av2/common/cdef.h"
#include "av2/common/filter.h"
#include "av2/common/warped_motion.h"
#include "avm_dsp/bitwriter_buffer.h"
}

namespace {

// decodeframe.c read_global_motion_params(), against ref.
void write_gm_params(avm_write_bit_buffer* wb, const WarpedMotionParams& wm, const WarpedMotionParams& ref) {
  avm_wb_write_bit(wb, wm.wmtype != IDENTITY);
  if (wm.wmtype == IDENTITY) return;
  if (wm.wmtype != ROTZOOM && wm.wmtype != AFFINE) throw std::runtime_error("global motion: ROTZOOM or AFFINE only");
  avm_wb_write_bit(wb, wm.wmtype == ROTZOOM);
  const int one = 1 << GM_ALPHA_PREC_BITS;
  auto alpha = [&](int i, int off) {
    avm_wb_write_signed_primitive_refsubexpfin(wb, GM_ALPHA_MAX + 1, SUBEXPFIN_K, (ref.wmmat[i] >> GM_ALPHA_PREC_DIFF) - off,
                                               (wm.wmmat[i] >> GM_ALPHA_PREC_DIFF) - off);
  };
  alpha(2, one);
  alpha(3, 0);
  if (wm.wmtype == AFFINE) alpha(4, 0), alpha(5, one);
  for (int i = 0; i < 2; i++)
    avm_wb_write_signed_primitive_refsubexpfin(wb, GM_TRANS_MAX + 1, SUBEXPFIN_K, ref.wmmat[i] >> GM_TRANS_PREC_DIFF,
                                               wm.wmmat[i] >> GM_TRANS_PREC_DIFF);
}

// decodeframe.c read_global_motion(): the models are coded against the IDENTITY base
// (our_ref = num_total_refs). dist: each reference's distance as the frame was coded (the
// reference map has moved on by the time the header is written), 0 where the decoder reads no model.
void write_global_motion(avm_write_bit_buffer* wb, const AV2_COMMON* cm, const int* dist) {
  const int n = cm->ref_frames_info.num_total_refs;
  bool use = false;
  for (int r = 0; r < n; r++) use |= cm->global_motion[r].wmtype != IDENTITY;
  avm_wb_write_bit(wb, use);
  if (!use) return;
  if (!frame_is_sframe(cm)) avm_wb_write_primitive_quniform(wb, n + 1, n);
  for (int r = 0; r < n; r++) {
    if (dist[r] == 0) continue;
    WarpedMotionParams ref;
    av2_scale_warp_model(&cm->base_global_motion_model, cm->base_global_motion_distance, &ref, dist[r]);
    write_gm_params(wb, cm->global_motion[r], ref);
  }
}

}  // namespace

std::vector<uint8_t> write_frame_header(const SeqConfig& c, CodecState& st, bool key) {
  (void)c;
  AV2_COMMON* const cm = st.cm();
  const FeatureFlags* const f = &cm->features;
  const SequenceHeader* const seq = &cm->seq_params;
  const CommonTileParams* const tiles = &cm->tiles;
  std::vector<uint8_t> buf(256, 0);
  avm_write_bit_buffer wb = {buf.data(), 0};
  avm_wb_write_bit(&wb, 1);   // first_tile_group_in_frame
  avm_wb_write_uvlc(&wb, 0);  // cur_mfh_id; 0 -> seq_header_id follows
  avm_wb_write_uvlc(&wb, seq->seq_header_id);
  // OBU_CLOSED_LOOP_KEY: frame type implied, long_term_id has 0 bits. Regular OBU: INTER bit.
  if (!key) avm_wb_write_bit(&wb, 1);
  avm_wb_write_bit(&wb, cm->immediate_output_picture);
  avm_wb_write_bit(&wb, 0);  // frame_size_override_flag
  avm_wb_write_literal(&wb, cm->current_frame.order_hint,
                       seq->order_hint_info.order_hint_bits_minus_1 + 1);
  if (key) {
    // Closed-loop key with one mlayer: refresh flags implied.
  } else {
    avm_wb_write_bit(&wb, 0);  // signal_primary_ref_frame (derived)
    avm_wb_write_bit(&wb, 0);  // cross_frame_context: forward
    avm_wb_write_literal(&wb, cm->current_frame.refresh_frame_flags, seq->ref_frames);
    // Implicit reference map, no BRU: no bits. Temporal MVs, with superblocks over 64 high and 2+
    // references their sampling step (tmvp_sample_step - 1), then no TIP and opfl refine fixed.
    if (frame_might_allow_ref_frame_mvs(cm)) avm_wb_write_bit(&wb, f->allow_ref_frame_mvs);
    if (f->allow_ref_frame_mvs && cm->ref_frames_info.num_total_refs > 1 && block_size_high[seq->sb_size] > 64)
      avm_wb_write_bit(&wb, cm->tmvp_sample_step - 1);
  }
  // Screen content tools fixed off.
  avm_wb_write_bit(&wb, f->allow_intrabc);
  if (!key) {
    // read_frame_max_drl_bits(): nothing. MV precision, interpolation filter.
    if (f->fr_mv_precision != MV_PRECISION_QTR_PEL || f->interp_filter > 3)
      throw std::runtime_error("unsupported inter frame parameters");
    avm_wb_write_bit(&wb, 1);  // MV_PRECISION_QTR_PEL
    avm_wb_write_bit(&wb, 0);  // not switchable
    avm_wb_write_literal(&wb, f->interp_filter, LOG_SWITCHABLE_FILTERS);
  }
  avm_wb_write_bit(&wb, f->disable_cdf_update);
  // read_tile_info_max_tile(): uniform spacing, one increment bit per step above the minimum.
  avm_wb_write_bit(&wb, 1);
  for (int i = tiles->min_log2_cols; i < tiles->log2_cols; i++) avm_wb_write_bit(&wb, 1);
  if (tiles->log2_cols < tiles->max_log2_cols) avm_wb_write_bit(&wb, 0);
  for (int i = tiles->min_log2_rows; i < tiles->log2_rows; i++) avm_wb_write_bit(&wb, 1);
  if (tiles->log2_rows < tiles->max_log2_rows) avm_wb_write_bit(&wb, 0);
  const bool multi_tile = tiles->rows * tiles->cols > 1;
  if (multi_tile) {  // read_tile_info(): enable_avg_cdf is 0
    avm_wb_write_literal(&wb, 0, tiles->log2_rows + tiles->log2_cols);  // context_update_tile_id
    avm_wb_write_literal(&wb, kTileSizeBytes - 1, 2);
  }
  avm_wb_write_literal(&wb, cm->quant_params.base_qindex,
                       cm->seq_params.bit_depth == AVM_BITS_8 ? QINDEX_BITS_UNEXT : QINDEX_BITS);
  avm_wb_write_bit(&wb, 0);  // segmentation enabled
  avm_wb_write_bit(&wb, 0);  // using_qmatrix
  avm_wb_write_bit(&wb, cm->delta_q_info.delta_q_present_flag);  // (base_qindex > 0)
  if (cm->delta_q_info.delta_q_present_flag)
    avm_wb_write_literal(&wb, get_msb(cm->delta_q_info.delta_q_res), 2);  // log2 of delta_q_res
  // setup_loopfilter(): mfh 0 has no deblocking update, so every flag is coded here, then one
  // "no delta" bit per filtered plane.
  const struct loopfilter* const lf = &cm->lf;
  avm_wb_write_bit(&wb, lf->apply_deblocking_filter[0]);
  avm_wb_write_bit(&wb, lf->apply_deblocking_filter[1]);
  if (lf->apply_deblocking_filter[0] || lf->apply_deblocking_filter[1]) {
    avm_wb_write_bit(&wb, lf->apply_deblocking_filter_u);
    avm_wb_write_bit(&wb, lf->apply_deblocking_filter_v);
  }
  // One delta per filtered plane (luma vertical, horizontal, U, V): a presence bit, then the
  // value offset by half its range (a horizontal delta left out repeats the vertical one).
  const int par_bits = cm->seq_params.df_par_bits_minus2 + 2;
  const int deltas[4] = {lf->delta_q_luma[0], lf->delta_q_luma[1], lf->delta_q_u, lf->delta_q_v};
  const int on[4] = {lf->apply_deblocking_filter[0], lf->apply_deblocking_filter[1],
                     lf->apply_deblocking_filter_u, lf->apply_deblocking_filter_v};
  for (int p = 0; p < 4; p++) {
    if (!on[p]) continue;
    const int inherited = p == 1 && on[0] ? deltas[0] : 0;
    avm_wb_write_bit(&wb, deltas[p] != inherited);
    if (deltas[p] != inherited) avm_wb_write_literal(&wb, deltas[p] + (1 << (par_bits - 1)), par_bits);
  }
  if (seq->enable_cdef) {  // setup_cdef(): a strength under 4 takes 2 bits
    const CdefInfo* const ci = &cm->cdef_info;
    avm_wb_write_bit(&wb, ci->cdef_frame_enable);
    if (ci->cdef_frame_enable) {
      avm_wb_write_literal(&wb, ci->cdef_damping - 3, 2);
      avm_wb_write_literal(&wb, ci->nb_cdef_strengths - 1, 3);
      for (int i = 0; i < ci->nb_cdef_strengths; i++)
        for (const int s : {ci->cdef_strengths[i], ci->cdef_uv_strengths[i]}) {
          avm_wb_write_bit(&wb, s < 4);
          avm_wb_write_literal(&wb, s, s < 4 ? 2 : CDEF_STRENGTH_BITS);
        }
    }
  }
  if (seq->enable_ccso) {  // setup_ccso(): no reuse of a reference's filters
    const CcsoInfo* const ci = &cm->ccso_info;
    avm_wb_write_bit(&wb, ci->ccso_frame_flag);
    for (int p = 0; ci->ccso_frame_flag && p < 3; p++) {
      avm_wb_write_bit(&wb, ci->ccso_enable[p]);
      if (!ci->ccso_enable[p]) continue;
      if (!frame_is_intra_only(cm)) avm_wb_write_bit(&wb, 0), avm_wb_write_bit(&wb, 0);  // reuse_ccso, sb_reuse_ccso
      avm_wb_write_bit(&wb, ci->ccso_bo_only[p]);
      avm_wb_write_literal(&wb, ci->scale_idx[p], 2);
      if (ci->ccso_bo_only[p]) {
        avm_wb_write_literal(&wb, ci->max_band_log2[p], 3);
      } else {
        avm_wb_write_literal(&wb, ci->quant_idx[p], 2);
        avm_wb_write_literal(&wb, ci->ext_filter_support[p], 3);
        if (quant_sz[ci->scale_idx[p]][ci->quant_idx[p]]) avm_wb_write_bit(&wb, ci->edge_clf[p]);
        avm_wb_write_literal(&wb, ci->max_band_log2[p], 2);
      }
      static const int kOffset[8] = {0, 1, -1, 3, -3, 7, -7, -10};  // read_ccso_offset_idx(): truncated unary
      const int scale = ci->scale_idx[p] + 1;
      const int n = ci->ccso_bo_only[p] ? 1 : edge_clf_to_edge_interval[ci->edge_clf[p]];
      for (int d0 = 0; d0 < n; d0++)
        for (int d1 = 0; d1 < n; d1++)
          for (int band = 0; band < 1 << ci->max_band_log2[p]; band++) {
            const int v = ci->filter_offset[p][(band << 4) + (d0 << 2) + d1];
            int idx = 0;
            while (idx < 8 && kOffset[idx] * scale != v) idx++;
            if (idx == 8) throw std::logic_error("CCSO offset not in the table");
            for (int k = 0; k < idx; k++) avm_wb_write_bit(&wb, 1);
            if (idx < 7) avm_wb_write_bit(&wb, 0);
          }
    }
  }
  avm_wb_write_bit(&wb, f->tx_mode == TX_MODE_SELECT);
  if (!key) {
    avm_wb_write_bit(&wb, cm->current_frame.reference_mode == REFERENCE_MODE_SELECT);
    if (cm->current_frame.skip_mode_info.skip_mode_allowed)
      avm_wb_write_bit(&wb, cm->current_frame.skip_mode_info.skip_mode_flag);
    if (seq->enable_bawp) avm_wb_write_bit(&wb, f->enable_bawp);
  }
  avm_wb_write_literal(&wb, f->reduced_tx_set_used, 2);
  if (!key && seq->enable_global_motion) write_global_motion(&wb, cm, st.gm_ref_dist);
  // read_tile_indices_in_tilegroup(): all tiles in this one tile group.
  if (multi_tile) avm_wb_write_bit(&wb, 0);  // tile_start_and_end_present_flag
  buf.resize(avm_wb_bytes_written(&wb));  // byte alignment with zero bits
  return buf;
}
