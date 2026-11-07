// Portions of this file are derived from the AVM v1.0.0 decoder (av2/decoder),
// Copyright (c) 2021, Alliance for Open Media, under the BSD 3-Clause Clear License
// and the Alliance for Open Media Patent License 1.0: see third_party/avm/.

#ifdef CELV2_ORACLE
#include "oracle.h"
#endif
#include "bitstream/frame_header.h"

#include <stdexcept>

extern "C" {
#include "av2/common/ccso.h"
#include "av2/common/cdef.h"
#include "av2/common/filter.h"
#include "av2/common/restoration.h"
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

void write_film_grain(avm_write_bit_buffer* wb, const avm_film_grain_t& g) {
  avm_wb_write_bit(wb, g.apply_grain);
  if (!g.apply_grain) return;
  avm_wb_write_literal(wb, 0, FGM_ID_BITS);
  avm_wb_write_literal(wb, g.random_seed, 16);
}

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
  // read_screen_content_params(): the frame's choice where the sequence leaves it to frames, then (on)
  // no forced integer MVs; read_intrabc_params(): key frames global (then local) IntraBC, inter
  // frames local, the sequence's BVP DRL bits.
  if (seq->force_screen_content_tools == 2) avm_wb_write_bit(&wb, f->allow_screen_content_tools);
  else if (f->allow_screen_content_tools != seq->force_screen_content_tools)
    throw std::runtime_error("screen content tools against the sequence header");
  if (f->allow_screen_content_tools) avm_wb_write_bit(&wb, f->cur_frame_force_integer_mv);
  if (f->allow_intrabc && (!f->allow_local_intrabc || seq->allow_frame_max_bvp_drl_bits))
    throw std::runtime_error("IntraBC without local, or with per-frame BVP DRL bits");
  avm_wb_write_bit(&wb, f->allow_intrabc);
  if (f->allow_intrabc && frame_is_intra_only(cm)) {
    avm_wb_write_bit(&wb, f->allow_global_intrabc);
    if (f->allow_global_intrabc) avm_wb_write_bit(&wb, f->allow_local_intrabc);
  }
  if (!key) {
    // read_frame_max_drl_bits(): nothing. MV precision, interpolation filter.
#ifdef CELV2_ORACLE  // L2: avmenc's
    avm_wb_write_bit(&wb, f->fr_mv_precision == MV_PRECISION_QTR_PEL);
    if (f->fr_mv_precision != MV_PRECISION_QTR_PEL) avm_wb_write_bit(&wb, f->fr_mv_precision == MV_PRECISION_ONE_EIGHTH_PEL);
    avm_wb_write_bit(&wb, f->interp_filter == SWITCHABLE);
    if (f->interp_filter != SWITCHABLE) avm_wb_write_literal(&wb, f->interp_filter, LOG_SWITCHABLE_FILTERS);
#else
    if (f->fr_mv_precision != MV_PRECISION_QTR_PEL || f->interp_filter > 3)
      throw std::runtime_error("unsupported inter frame parameters");
    avm_wb_write_bit(&wb, 1);  // MV_PRECISION_QTR_PEL
    avm_wb_write_bit(&wb, 0);  // not switchable
    avm_wb_write_literal(&wb, f->interp_filter, LOG_SWITCHABLE_FILTERS);
#endif
    if (seq->seq_frame_motion_modes_present_flag)
      for (int mm = INTERINTRA; mm < MOTION_MODES; mm++)
        if (seq->seq_enabled_motion_modes & (1 << mm)) avm_wb_write_bit(&wb, (f->enabled_motion_modes >> mm) & 1);
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
#ifdef CELV2_ORACLE
  if (celv2::oracle::filters() && seq->enable_restoration) {  // decode_restoration_mode(), all of it
    bool luma = false, chroma = false;
    for (int p = 0; p < av2_num_planes(cm); p++) {
      const FeatureFlags& f = cm->features;
      const RestorationInfo& rsi = cm->rst_info[p];
      int ndx = 0;
      for (int r = RESTORE_NONE; r < rsi.frame_restoration_type; r++) ndx += !((f.lr_tools_disable_mask[p] >> r) & 1);
      const int n = f.lr_frame_tools_count[p], l = get_unsigned_bits(n), m = (1 << l) - n;
      if (ndx < m) {
        avm_wb_write_literal(&wb, ndx, l - 1);
      } else {
        avm_wb_write_literal(&wb, m + ((ndx - m) >> 1), l - 1);
        avm_wb_write_bit(&wb, (ndx - m) & 1);
      }
      if (rsi.frame_restoration_type != RESTORE_NONE) (p ? chroma : luma) = true;
      if ((rsi.frame_restoration_type == RESTORE_WIENER_NONSEP || rsi.frame_restoration_type == RESTORE_SWITCHABLE) && is_frame_filters_enabled(p)) {
        avm_wb_write_bit(&wb, rsi.frame_filters_on);
        if (rsi.frame_filters_on) {
          const int refs = frame_is_intra_only(cm) || frame_is_sframe(cm) ? 0 : cm->ref_frames_info.num_total_refs;
          if (refs > 0) avm_wb_write_bit(&wb, rsi.temporal_pred_flag);
          if (rsi.temporal_pred_flag && refs > 1) avm_wb_write_literal(&wb, rsi.rst_ref_pic_idx, avm_ceil_log2(refs));
          if (!rsi.temporal_pred_flag && max_num_classes(p) > 1)
            avm_wb_write_literal(&wb, encode_num_filter_classes(rsi.num_filter_classes), NUM_FILTER_CLASSES_BITS);
        }
      }
    }
    if (cm->mib_size != 32) throw std::logic_error("restoration units under superblocks other than 128x128");
    for (int k = 0; k < 2; k++) {
      if (!(k ? chroma : luma)) continue;
      const int size = RESTORATION_UNITSIZE_MAX >> k, u = cm->rst_info[k].restoration_unit_size;
      avm_wb_write_bit(&wb, u == size >> 1);
      if (u != size >> 1) avm_wb_write_bit(&wb, u == size);
    }
    for (int p = 0; p < av2_num_planes(cm); p++)
      if (is_frame_filters_enabled(p) && to_readwrite_framefilters(&cm->rst_info[p], 0, 0))
        for (const uint8_t bit : st.lr_filter_bits(p)) avm_wb_write_bit(&wb, bit);
  } else
#endif
  if (seq->enable_restoration) {  // decode_restoration_mode(): CodecState::begin_frame() set the tools
    for (int p = 0; p < av2_num_planes(cm); p++) {
      const FeatureFlags& f = cm->features;
      int ndx = 0;  // the inverse of index_to_frame_restoration_type()
      for (int r = RESTORE_NONE; r < cm->rst_info[p].frame_restoration_type; r++) ndx += !((f.lr_tools_disable_mask[p] >> r) & 1);
      // rb_read_uniform()
      const int n = f.lr_frame_tools_count[p], l = get_unsigned_bits(n), m = (1 << l) - n;
      if (ndx < m) {
        avm_wb_write_literal(&wb, ndx, l - 1);
      } else {
        avm_wb_write_literal(&wb, m + ((ndx - m) >> 1), l - 1);
        avm_wb_write_bit(&wb, (ndx - m) & 1);
      }
      const RestorationInfo& rsi = cm->rst_info[p];
      if (rsi.frame_restoration_type == RESTORE_WIENER_NONSEP) {  // one filter for the frame, not a reference's
        if (!rsi.frame_filters_on || rsi.temporal_pred_flag || rsi.num_filter_classes != 1 || p == 0)
          throw std::logic_error("Wiener-nonsep other than one chroma frame filter");
        avm_wb_write_bit(&wb, 1);  // frame_filters_on
        if (!frame_is_intra_only(cm) && !frame_is_sframe(cm) && cm->ref_frames_info.num_total_refs > 0)
          avm_wb_write_bit(&wb, 0);  // temporal_pred_flag
      }
    }
    if (cm->mib_size != 32) throw std::logic_error("restoration units under superblocks other than 128x128");
    if (cm->rst_info[0].frame_restoration_type != RESTORE_NONE) {  // 128x128 luma units
      if (cm->rst_info[0].restoration_unit_size != RESTORATION_UNITSIZE_MAX >> 2)
        throw std::logic_error("luma restoration units other than 128x128");
      avm_wb_write_bit(&wb, 0);  // not 256
      avm_wb_write_bit(&wb, 0);  // not 512
    }
    if (cm->rst_info[1].frame_restoration_type != RESTORE_NONE || cm->rst_info[2].frame_restoration_type != RESTORE_NONE) {
      if (cm->rst_info[1].restoration_unit_size != RESTORATION_UNITSIZE_MAX >> 3)  // 64x64 at 4:2:0
        throw std::logic_error("chroma restoration units other than 64x64");
      avm_wb_write_bit(&wb, 0);  // not 128
      avm_wb_write_bit(&wb, 0);  // not 256
    }
    for (int p = 1; p < av2_num_planes(cm); p++)  // read_wienerns_framefilters_hdr(), from CodecState::restore()
      if (cm->rst_info[p].frame_restoration_type == RESTORE_WIENER_NONSEP)
        for (const uint8_t bit : st.lr_filter_bits(p)) avm_wb_write_bit(&wb, bit);
  }
  if (seq->enable_ccso) {  // setup_ccso(): no reuse of a reference's filters
    const CcsoInfo* const ci = &cm->ccso_info;
    avm_wb_write_bit(&wb, ci->ccso_frame_flag);
    for (int p = 0; ci->ccso_frame_flag && p < 3; p++) {
      avm_wb_write_bit(&wb, ci->ccso_enable[p]);
      if (!ci->ccso_enable[p]) continue;
#ifdef CELV2_ORACLE
      if (celv2::oracle::filters()) {
        if (!frame_is_intra_only(cm)) avm_wb_write_bit(&wb, ci->reuse_ccso[p]), avm_wb_write_bit(&wb, ci->sb_reuse_ccso[p]);
        const int refs = frame_is_intra_only(cm) || frame_is_sframe(cm) ? 0 : cm->ref_frames_info.num_total_refs;
        if ((ci->reuse_ccso[p] || ci->sb_reuse_ccso[p]) && refs > 1) avm_wb_write_literal(&wb, ci->ccso_ref_idx[p], avm_ceil_log2(refs));
        if (ci->reuse_ccso[p]) continue;
      } else
#endif
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
    if (f->enabled_motion_modes & (1 << WARP_DELTA)) avm_wb_write_bit(&wb, f->allow_warpmv_mode);
  }
  avm_wb_write_literal(&wb, f->reduced_tx_set_used, 2);
  if (!key && seq->enable_global_motion) write_global_motion(&wb, cm, st.gm_ref_dist);
  if (seq->film_grain_params_present && (cm->immediate_output_picture || cm->implicit_output_picture))
    write_film_grain(&wb, cm->cur_frame->film_grain_params);
  // read_tile_indices_in_tilegroup(): all tiles in this one tile group.
  if (multi_tile) avm_wb_write_bit(&wb, 0);  // tile_start_and_end_present_flag
  buf.resize(avm_wb_bytes_written(&wb));  // byte alignment with zero bits
  return buf;
}
