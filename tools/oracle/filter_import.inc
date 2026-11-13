// CodecState from L4 on (tools/oracle/oracle.h): avmenc's in-loop filter settings, on CelV2's reconstruction
// (codec_state.cpp, CELV2_ORACLE only).

namespace {
// X_ORACLE_FILT r: plane p's loop restoration is CelV2's own (restore()); with f luma keeps F0's types, refit; with u
// chroma keeps F0's restoration as it is (types, unit sizes, unit types and taps).
bool own_lr(int p) { return celv2::oracle::filt_cut('r') && !celv2::oracle::filt_cut(p > 0 ? 'u' : 'f'); }
}  // namespace

// begin_frame(): what the block syntax needs before the tiles.
void CodecState::import_filters(const celv2::oracle::Frame& o) {
  AV2_COMMON* const cm = cm_;
  const OracleFilters& f = o.filters;
  struct loopfilter* const lf = &cm->lf;
  lf->apply_deblocking_filter[0] = f.lf_apply[0], lf->apply_deblocking_filter[1] = f.lf_apply[1];
  lf->apply_deblocking_filter_u = f.lf_apply[2], lf->apply_deblocking_filter_v = f.lf_apply[3];
  lf->delta_q_luma[0] = lf->delta_side_luma[0] = f.lf_delta[0], lf->delta_q_luma[1] = lf->delta_side_luma[1] = f.lf_delta[1];
  lf->delta_q_u = lf->delta_side_u = f.lf_delta[2], lf->delta_q_v = lf->delta_side_v = f.lf_delta[3];
  if (celv2::oracle::filt_cut('d')) {
    lf->delta_q_luma[0] = lf->delta_side_luma[0] = lf->delta_q_luma[1] = lf->delta_side_luma[1] = 0;
    lf->delta_q_u = lf->delta_side_u = lf->delta_q_v = lf->delta_side_v = 0;
  }
  CdefInfo* const ci = &cm->cdef_info;
  ci->cdef_frame_enable = f.cdef_enable;
  ci->cdef_damping = f.cdef_damping;
  ci->nb_cdef_strengths = f.cdef_n;
  ci->cdef_on_skip_txfm_frame_enable = celv2::oracle::filt_cut('s') ? 0 : f.cdef_on_skip;
  for (int i = 0; i < f.cdef_n; i++) ci->cdef_strengths[i] = f.cdef_y[i], ci->cdef_uv_strengths[i] = f.cdef_uv[i];
  if (celv2::oracle::filt_cut('c') && f.cdef_n > 1) {  // one strength: the one F0 gave the most 64x64 units
    int uses[CDEF_MAX_STRENGTHS] = {};
    for (const int8_t v : o.cdef_idx)
      if (v >= 0 && v < f.cdef_n) uses[v]++;
    const int best = static_cast<int>(std::max_element(uses, uses + f.cdef_n) - uses);
    ci->nb_cdef_strengths = 1, ci->cdef_strengths[0] = f.cdef_y[best], ci->cdef_uv_strengths[0] = f.cdef_uv[best];
  }
  CcsoInfo* const c = &cm->ccso_info;
  c->ccso_frame_flag = f.ccso_frame != 0;
  for (int p = 0; p < 3; p++) {
    const OracleCcso& s = f.ccso[p];
    c->ccso_enable[p] = cm->cur_frame->ccso_info.ccso_enable[p] = s.enable != 0;
    c->reuse_ccso[p] = s.reuse != 0, c->sb_reuse_ccso[p] = s.sb_reuse != 0 && !celv2::oracle::filt_cut('o');
    c->ccso_bo_only[p] = static_cast<uint8_t>(s.bo_only), c->scale_idx[p] = static_cast<uint8_t>(s.scale), c->quant_idx[p] = static_cast<uint8_t>(s.quant);
    c->ext_filter_support[p] = static_cast<uint8_t>(s.ext), c->edge_clf[p] = static_cast<uint8_t>(s.edge_clf);
    c->max_band_log2[p] = s.band, c->ccso_ref_idx[p] = static_cast<uint8_t>(s.ref_idx);
    std::memcpy(c->filter_offset[p], f.ccso_offset[p], sizeof(c->filter_offset[p]));
    if (s.enable) av2_copy_ccso_filters(&cm->cur_frame->ccso_info, c, p, 1, 0, 0);
  }
}

// begin_frame(), the restoration types and sizes before av2_alloc_restoration_buffers().
void CodecState::import_lr_types(const celv2::oracle::Frame& o) {
  AV2_COMMON* const cm = cm_;
  const OracleFilters& f = o.filters;
  for (int p = 0; p < av2_num_planes(cm); p++) {
    if (own_lr(p)) continue;  // begin_frame()'s, for restore()
    RestorationInfo* const rsi = &cm->rst_info[p];
    rsi->frame_restoration_type = static_cast<RestorationType>(f.lr_type[p]);
    rsi->restoration_unit_size = f.lr_size[p];
    rsi->frame_filters_on = f.lr_frame_filters[p];
    rsi->temporal_pred_flag = static_cast<uint8_t>(f.lr_temporal[p]);
    rsi->rst_ref_pic_idx = static_cast<uint8_t>(f.lr_ref[p]);
    const bool pc = p == 0 && celv2::oracle::filt_cut('w');  // C2 w: luma's Wiener units as PC Wiener, no frame filters
    if (pc && rsi->frame_restoration_type == RESTORE_WIENER_NONSEP) rsi->frame_restoration_type = RESTORE_PC_WIENER;
    const int classes = celv2::oracle::filt_cut('k') ? 1 : f.lr_classes[p];
    rsi->num_filter_classes = classes;
    rsi->frame_filters_on = pc ? 0 : f.lr_frame_filters[p];
    rsi->frame_filters_initialized = rsi->frame_filters_on;
    if (rsi->frame_filters_on) std::memcpy(&rsi->frame_filters, o.lr_frame_filter[p].data(), sizeof(WienerNonsepInfo));
    rsi->frame_filters.num_classes = classes;
    if (p == 0 && celv2::oracle::filt_cut('r') && (rsi->frame_restoration_type == RESTORE_WIENER_NONSEP || rsi->frame_restoration_type == RESTORE_SWITCHABLE)) {
      rsi->frame_filters_on = rsi->frame_filters_initialized = 1;  // rf: frame filters, refit_luma_lr()'s taps
      rsi->temporal_pred_flag = 0, rsi->rst_ref_pic_idx = 0;
      std::memset(&rsi->frame_filters, 0, sizeof(WienerNonsepInfo));
      rsi->frame_filters.num_classes = classes;
    }
  }
}

// After the allocation: each unit's type and taps, and what its syntax codes against (the bank, unit by unit in the
// order the superblocks read them: rows of superblocks are coded at once, so it cannot be followed there).
void CodecState::import_lr_units(const celv2::oracle::Frame& o) {
  AV2_COMMON* const cm = cm_;
  WienerNonsepInfoBank bank[MAX_MB_PLANE];
  for (int p = 0; p < av2_num_planes(cm); p++) {
    RestorationInfo* const rsi = &cm->rst_info[p];
    lr_plan_[p].clear();
    if (rsi->frame_restoration_type == RESTORE_NONE || own_lr(p)) continue;
    const int n = rsi->horz_units_per_frame * rsi->vert_units_per_frame;
    if (n != static_cast<int>(o.lr_unit_type[p].size())) throw std::runtime_error("X_ORACLE: restoration units differ from the stream's");
    lr_plan_[p].resize(n);
    for (int i = 0; i < n; i++) {
      RestorationUnitInfo* const rui = &rsi->unit_info[i];
      rui->restoration_type = static_cast<RestorationType>(o.lr_unit_type[p][i]);
      if (p == 0 && rui->restoration_type == RESTORE_WIENER_NONSEP && celv2::oracle::filt_cut('w')) rui->restoration_type = RESTORE_PC_WIENER;
      rui->wienerns_info.num_classes = rsi->num_filter_classes;
      if (rui->restoration_type == RESTORE_WIENER_NONSEP) {
        if (rsi->frame_filters_on) copy_nsfilter_taps(&rui->wienerns_info, &rsi->frame_filters);
        else std::memcpy(&rui->wienerns_info, o.lr_unit_filter[p].data() + static_cast<size_t>(o.lr_unit_at[p][i]) * sizeof(WienerNonsepInfo), sizeof(WienerNonsepInfo));
        rui->wienerns_info.num_classes = rsi->num_filter_classes;  // C2 k: class 0's taps alone
      }
    }
    if (rsi->frame_filters_on) av2_copy_rst_frame_filters(&cm->cur_frame->rst_info[p], rsi);
    av2_reset_wienerns_bank(&bank[p], cm->quant_params.base_qindex, rsi->num_filter_classes, p != AVM_PLANE_Y);
  }
  for (int r = 0; r < cm->mi_params.mi_rows; r += cm->mib_size)
    for (int c = 0; c < cm->mi_params.mi_cols; c += cm->mib_size)
      for (int p = 0; p < av2_num_planes(cm); p++) {
        RestorationInfo* const rsi = &cm->rst_info[p];
        int r0, r1, c0, c1;
        if (rsi->frame_restoration_type == RESTORE_NONE || own_lr(p) || !av2_loop_restoration_corners_in_sb(cm, p, r, c, cm->sb_size, &c0, &c1, &r0, &r1)) continue;
        for (int rr = r0; rr < r1; rr++)
          for (int rc = c0; rc < c1; rc++) {
            const int i = rc + rr * rsi->horz_units_per_frame;
            RestorationUnitInfo* const rui = &rsi->unit_info[i];
            if (rui->restoration_type != RESTORE_WIENER_NONSEP || rsi->frame_filters_on) continue;
            LrPlan& plan = lr_plan_[p][i];
            const WienernsFilterParameters* const prm = get_wienerns_parameters(cm->quant_params.base_qindex, p != AVM_PLANE_Y);
            for (int k = 0; k < rsi->num_filter_classes; k++) {
              plan.size[k] = bank[p].bank_size_for_class[k];
              plan.ref[k] = rui->wienerns_info.bank_ref_for_class[k];
              const int16_t* const ref = const_nsfilter_taps(av2_constref_from_wienerns_bank(&bank[p], plan.ref[k], k), k);
              std::copy_n(ref, prm->ncoeffs, plan.taps[k]);
              plan.exact[k] = std::equal(ref, ref + prm->ncoeffs, const_nsfilter_taps(&rui->wienerns_info, k));
            }
            for (int k = 0; k < rsi->num_filter_classes; k++)
              if (!plan.exact[k] || bank[p].bank_size_for_class[k] == 0) av2_add_to_wienerns_bank(&bank[p], &rui->wienerns_info, k);
          }
      }
}

// end_frame() from L4 on, in av2_decode_tg_tiles_and_wrapup()'s order.
void CodecState::import_end_frame(const Frame420& src, double sse_per_bit) {
  AV2_COMMON* const cm = cm_;
  YV12_BUFFER_CONFIG& b = cm->cur_frame->buf;
  const bool own = celv2::oracle::filt_cut('r');  // restore() and refit_luma_lr() write their own filter bits
  // Before the reference slots change: the dictionary is the references'.
  for (int p = 0; p < av2_num_planes(cm); p++)
    if ((!own || (p > 0 && !own_lr(p))) && is_frame_filters_enabled(p) && to_readwrite_framefilters(&cm->rst_info[p], 0, 0)) frame_filter_bits(p);
  if (cm->lf.apply_deblocking_filter[0] || cm->lf.apply_deblocking_filter[1]) deblock();
  const CcsoInfo* const ci = &cm->ccso_info;
  const bool use_ccso = ci->ccso_enable[0] || ci->ccso_enable[1] || ci->ccso_enable[2];
  if (use_ccso) {
    const int d = CCSO_PADDING_SIZE, w = b.y_width, h = b.y_height, stride = w + 2 * d;
    ccso_luma_.resize(static_cast<size_t>(stride) * (h + 2 * d));
    auto row = [&](int r) { return &ccso_luma_[static_cast<size_t>(r + d) * stride]; };
    for (int r = 0; r < h; r++) {
      uint16_t* const p = row(r);
      std::memcpy(p + d, b.y_buffer + static_cast<size_t>(r) * b.y_stride, sizeof(uint16_t) * w);
      std::fill(p, p + d, p[d]);
      std::fill(p + d + w, p + stride, p[d + w - 1]);
    }
    for (int y = 1; y <= d; y++) std::memcpy(row(-y), row(0), sizeof(uint16_t) * stride), std::memcpy(row(h - 1 + y), row(h - 1), sizeof(uint16_t) * stride);
  }
  bool lr = false;
  for (int p = 0; p < 3; p++) lr = lr || cm->rst_info[p].frame_restoration_type != RESTORE_NONE;
  const bool do_cdef = cm->cdef_info.cdef_frame_enable;
  const bool optimized = !cm->seq_params.disable_loopfilters_across_tiles && !use_ccso && !do_cdef;
  if (!optimized) {
    if (lr) av2_loop_restoration_save_boundary_lines(&b, cm, 0);
    if (do_cdef) {
      if (cdef_workers_ != 1) av2_alloc_cdef_buffers(cm, &cdef_worker_, &cdef_sync_, 1), cdef_workers_ = 1;
      av2_cdef_frame(&b, cm, tiles_[0].xd, av2_cdef_init_fb_row);
    }
    if (use_ccso) apply_ccso(ccso_luma_);
    if (lr) av2_loop_restoration_save_boundary_lines(&b, cm, 1);
  }
  if (!lr) return;
  if (!own) {
    av2_loop_restoration_filter_frame(&b, cm, optimized, &lr_ctxt_);
  } else if (celv2::oracle::filt_cut('u')) {  // chroma F0's on the luma before restoration (luma NONE), then restore()'s luma
    RestorationInfo* const y = &cm->rst_info[0];
    const RestorationType type = y->frame_restoration_type, chroma[2] = {cm->rst_info[1].frame_restoration_type, cm->rst_info[2].frame_restoration_type};
    y->frame_restoration_type = RESTORE_NONE;
    if (chroma[0] != RESTORE_NONE || chroma[1] != RESTORE_NONE) av2_loop_restoration_filter_frame(&b, cm, optimized, &lr_ctxt_);
    y->frame_restoration_type = type;
    cm->rst_info[1].frame_restoration_type = cm->rst_info[2].frame_restoration_type = RESTORE_NONE;
    restore(src, optimized, sse_per_bit);
    cm->rst_info[1].frame_restoration_type = chroma[0], cm->rst_info[2].frame_restoration_type = chroma[1];
  } else if (!celv2::oracle::filt_cut('f')) {
    restore(src, optimized, sse_per_bit);
  } else {  // chroma by restore() on the luma before restoration, as the decoder's one pass has it; then luma
    RestorationInfo* const y = &cm->rst_info[0];
    const RestorationType type = y->frame_restoration_type;
    y->frame_restoration_type = RESTORE_NONE;
    restore(src, optimized, sse_per_bit);
    y->frame_restoration_type = type;
    if (type != RESTORE_NONE) refit_luma_lr(src, sse_per_bit, optimized);
  }
}

// X_ORACLE_FILT rf (C2 part 3): luma's WIENER_NONSEP taps by least squares on CelV2's reconstruction, per class over the
// units F0 gave WIENER_NONSEP; each class's subset, then its dictionary match, by squared error less the bits' worth.
void CodecState::refit_luma_lr(const Frame420& src, double sse_per_bit, bool optimized) {
  AV2_COMMON* const cm = cm_;
  YV12_BUFFER_CONFIG& b = cm->cur_frame->buf;
  RestorationInfo* const rsi = &cm->rst_info[0];
  const RestorationType chroma[2] = {cm->rst_info[1].frame_restoration_type, cm->rst_info[2].frame_restoration_type};
  cm->rst_info[1].frame_restoration_type = cm->rst_info[2].frame_restoration_type = RESTORE_NONE;  // restore() filtered them
  const int classes = rsi->num_filter_classes, n = rsi->horz_units_per_frame * rsi->vert_units_per_frame;
  WienerNonsepInfo* const ff = &rsi->frame_filters;
  bool any = false;
  for (int i = 0; i < n; i++) any = any || rsi->unit_info[i].restoration_type == RESTORE_WIENER_NONSEP;
  if (rsi->frame_filters_on && any) {
    if (classes > 1) {  // the class map, from a pass with any taps (it is of the restoration's input)
      const std::vector<uint16_t> keep(b.y_buffer, b.y_buffer + static_cast<size_t>(b.y_stride) * b.heights[0]);
      for (int i = 0; i < n; i++) copy_nsfilter_taps(&rsi->unit_info[i].wienerns_info, ff);
      av2_loop_restoration_filter_frame(&b, cm, optimized, &lr_ctxt_);
      std::copy(keep.begin(), keep.end(), b.y_buffer);
    }
    const WienernsFilterParameters* const prm = get_wienerns_parameters(cm->quant_params.base_qindex, 0);
    const NonsepFilterConfig& nc = prm->nsfilter_config;
    static constexpr int N = kWienerTaps;
    int nf = 0, dx[N], px[N];
    for (int i = 0; i < nc.num_pixels; i += 2, nf++)
      dx[nf] = nc.config[i][NONSEP_ROW_ID] * b.y_stride + nc.config[i][NONSEP_COL_ID], px[nf] = nc.config[i][NONSEP_BUF_POS];
    // The 4x4 blocks in a WIENER_NONSEP unit.
    const int w = b.y_crop_width, h = b.y_crop_height, bw = (w + 3) / 4, bh = (h + 3) / 4, size = rsi->restoration_unit_size;
    std::vector<uint8_t> on(static_cast<size_t>(bw) * bh, 0);
    for (int r = 0; r < rsi->vert_units_per_frame; r++)
      for (int c = 0; c < rsi->horz_units_per_frame; c++) {
        if (rsi->unit_info[r * rsi->horz_units_per_frame + c].restoration_type != RESTORE_WIENER_NONSEP) continue;
        const AV2PixelRect u = av2_get_rutile_rect(cm, 0, r, r + 1, c, c + 1, size, size);
        for (int y = u.top / 4; y < std::min(bh, (u.bottom + 3) / 4); y++)
          for (int x = u.left / 4; x < std::min(bw, (u.right + 3) / 4); x++) on[static_cast<size_t>(y) * bw + x] = 1;
      }
    const uint8_t* const map = cm->mi_params.wiener_class_id[0];
    const int mstride = cm->mi_params.wiener_class_id_stride[0];
    const uint8_t* const sub = get_pc_wiener_sub_classifier(classes, get_filter_set_index(cm->quant_params.base_qindex, 0));
    // The normal equations per class (fit_wienerns()'s features: a pair's two samples less twice the centre), in bands.
    const int bands = std::max(1, std::min(64, (h - 8) / 16)), per = classes * N * (N + 1);
    std::vector<double> sums(static_cast<size_t>(bands) * per, 0.0);
    celv2::parallel(bands, [&](int k) {
      double* const s0 = &sums[static_cast<size_t>(k) * per];
      for (int y = 4 + (h - 8) * k / bands; y < 4 + (h - 8) * (k + 1) / bands; y++)
        for (int x = 4; x < w - 4; x++) {
          if (!on[static_cast<size_t>(y / 4) * bw + x / 4]) continue;
          const int cls = classes > 1 ? sub[map[static_cast<size_t>(y / 4) * mstride + x / 4]] : 0;
          const uint16_t* const o = b.y_buffer + static_cast<size_t>(y) * b.y_stride + x;
          const size_t at = static_cast<size_t>(y) * src.stride_y() + x;
          double f[N + 1] = {};
          for (int i = 0; i < nf; i++) f[px[i]] = o[dx[i]] + o[-dx[i]] - 2.0 * o[0];
          f[N] = (src.bit_depth > 8 ? src.hy[at] : src.y[at]) - static_cast<double>(o[0]);
          double* const s = s0 + static_cast<size_t>(cls) * N * (N + 1);
          for (int i = 0; i < N; i++)
            for (int j = i; j <= N; j++) s[i * (N + 1) + j] += f[i] * f[j];
        }
    });
    const int nopcw = disable_pcwiener_filters_in_framefilters(&cm->seq_params);
    const int nref = set_frame_filter_dictionary(0, cm, classes, cm->frame_filter_dictionary, cm->frame_filter_dictionary_stride);
    int counts[NUM_MATCH_GROUPS];
    set_group_counts(0, classes, nref, counts, nopcw);
    const int total = counts[0] + counts[1] + counts[2];
    std::fill(ff->match_indices, ff->match_indices + WIENERNS_MAX_CLASSES, 0);
    auto bits = [&] { return frame_filter_bits(0), static_cast<double>(lr_filter_bits_[0].size()); };
    for (int c = 0; c < classes; c++) {
      double a[N * N], v[N];
      for (int i = 0; i < N; i++)
        for (int j = i; j <= N; j++) {
          double s = 0;
          for (int k = 0; k < bands; k++) s += sums[static_cast<size_t>(k) * per + static_cast<size_t>(c) * N * (N + 1) + i * (N + 1) + j];
          if (j < N) a[i * N + j] = a[j * N + i] = s;
          else v[i] = s;
        }
      // The subset at match 0 (the zero filter), then the match with the fewest bits at that subset.
      int16_t fit[N] = {};
      double best = -1e300;
      for (int s = 0; s < prm->nsubsets; s++) {
        int16_t t[N];
        wienerns_solve(a, v, prm, s, t);
        std::copy_n(t, prm->ncoeffs, nsfilter_taps(ff, c));
        const double j = wienerns_gain(a, v, t) - sse_per_bit * bits();
        if (j > best) best = j, std::copy_n(t, N, fit);
      }
      std::copy_n(fit, prm->ncoeffs, nsfilter_taps(ff, c));
      int best_m = 0;
      double fewest = 1e300;
      for (int m = 0; m < total; m++) {
        if (m < counts[0] && m > c) continue;  // group 0: the zero filter and the classes before this one
        ff->match_indices[c] = m;
        const double nb = bits();
        if (nb < fewest) fewest = nb, best_m = m;
      }
      ff->match_indices[c] = best_m;
    }
    for (int i = 0; i < n; i++)
      if (rsi->unit_info[i].restoration_type == RESTORE_WIENER_NONSEP) {
        copy_nsfilter_taps(&rsi->unit_info[i].wienerns_info, ff);
        rsi->unit_info[i].wienerns_info.num_classes = classes;
      }
  }
  if (rsi->frame_filters_on) frame_filter_bits(0);  // with no WIENER_NONSEP unit, the zero filters as coded
  av2_loop_restoration_filter_frame(&b, cm, optimized, &lr_ctxt_);
  cm->rst_info[1].frame_restoration_type = chroma[0], cm->rst_info[2].frame_restoration_type = chroma[1];
}
