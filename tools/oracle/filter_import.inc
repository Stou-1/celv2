// CodecState from L4 on (tools/oracle/oracle.h): avmenc's in-loop filter settings, on CelV2's reconstruction
// (codec_state.cpp, CELV2_ORACLE only).

// begin_frame(): what the block syntax needs before the tiles.
void CodecState::import_filters(const celv2::oracle::Frame& o) {
  AV2_COMMON* const cm = cm_;
  const OracleFilters& f = o.filters;
  struct loopfilter* const lf = &cm->lf;
  lf->apply_deblocking_filter[0] = f.lf_apply[0], lf->apply_deblocking_filter[1] = f.lf_apply[1];
  lf->apply_deblocking_filter_u = f.lf_apply[2], lf->apply_deblocking_filter_v = f.lf_apply[3];
  lf->delta_q_luma[0] = lf->delta_side_luma[0] = f.lf_delta[0], lf->delta_q_luma[1] = lf->delta_side_luma[1] = f.lf_delta[1];
  lf->delta_q_u = lf->delta_side_u = f.lf_delta[2], lf->delta_q_v = lf->delta_side_v = f.lf_delta[3];
  CdefInfo* const ci = &cm->cdef_info;
  ci->cdef_frame_enable = f.cdef_enable;
  ci->cdef_damping = f.cdef_damping;
  ci->nb_cdef_strengths = f.cdef_n;
  ci->cdef_on_skip_txfm_frame_enable = f.cdef_on_skip;
  for (int i = 0; i < f.cdef_n; i++) ci->cdef_strengths[i] = f.cdef_y[i], ci->cdef_uv_strengths[i] = f.cdef_uv[i];
  CcsoInfo* const c = &cm->ccso_info;
  c->ccso_frame_flag = f.ccso_frame != 0;
  for (int p = 0; p < 3; p++) {
    const OracleCcso& s = f.ccso[p];
    c->ccso_enable[p] = cm->cur_frame->ccso_info.ccso_enable[p] = s.enable != 0;
    c->reuse_ccso[p] = s.reuse != 0, c->sb_reuse_ccso[p] = s.sb_reuse != 0;
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
    RestorationInfo* const rsi = &cm->rst_info[p];
    rsi->frame_restoration_type = static_cast<RestorationType>(f.lr_type[p]);
    rsi->restoration_unit_size = f.lr_size[p];
    rsi->frame_filters_on = f.lr_frame_filters[p];
    rsi->temporal_pred_flag = static_cast<uint8_t>(f.lr_temporal[p]);
    rsi->rst_ref_pic_idx = static_cast<uint8_t>(f.lr_ref[p]);
    rsi->num_filter_classes = f.lr_classes[p];
    rsi->frame_filters_initialized = f.lr_frame_filters[p];
    if (f.lr_frame_filters[p]) std::memcpy(&rsi->frame_filters, o.lr_frame_filter[p].data(), sizeof(WienerNonsepInfo));
    rsi->frame_filters.num_classes = f.lr_classes[p];
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
    if (rsi->frame_restoration_type == RESTORE_NONE) continue;
    const int n = rsi->horz_units_per_frame * rsi->vert_units_per_frame;
    if (n != static_cast<int>(o.lr_unit_type[p].size())) throw std::runtime_error("X_ORACLE: restoration units differ from the stream's");
    lr_plan_[p].resize(n);
    for (int i = 0; i < n; i++) {
      RestorationUnitInfo* const rui = &rsi->unit_info[i];
      rui->restoration_type = static_cast<RestorationType>(o.lr_unit_type[p][i]);
      rui->wienerns_info.num_classes = rsi->num_filter_classes;
      if (rui->restoration_type == RESTORE_WIENER_NONSEP) {
        if (rsi->frame_filters_on) copy_nsfilter_taps(&rui->wienerns_info, &rsi->frame_filters);
        else std::memcpy(&rui->wienerns_info, o.lr_unit_filter[p].data() + static_cast<size_t>(o.lr_unit_at[p][i]) * sizeof(WienerNonsepInfo), sizeof(WienerNonsepInfo));
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
        if (rsi->frame_restoration_type == RESTORE_NONE || !av2_loop_restoration_corners_in_sb(cm, p, r, c, cm->sb_size, &c0, &c1, &r0, &r1)) continue;
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

// The subset of taps s and whether they go in pairs (sym), for t against prm.
void wienerns_shape(const WienernsFilterParameters* prm, const int16_t* t, int* s, int* sym) {
  int ncoeffs1, ncoeffs2;
  config2ncoeffs(&prm->nsfilter_config, &ncoeffs1, &ncoeffs2);
  for (*s = 0; *s < prm->nsubsets - 1; ++*s) {
    bool fits = true;
    for (int i = 0; i < prm->ncoeffs; i++) fits = fits && (prm->subset_config[*s][i] || !t[i]);
    if (fits) break;
  }
  *sym = 1;
  for (int i = 0; i < prm->ncoeffs; i++) {
    const bool asym = i < prm->nsfilter_config.asymmetric || (i >= ncoeffs1 && i - ncoeffs1 < prm->nsfilter_config.asymmetric2);
    if (!prm->subset_config[*s][i] || !asym) continue;
    if (t[i] != t[i + 1]) *sym = 0;
    i++;
  }
}

// read_match_indices_hdr() and read_wienerns_framefilters_hdr(), written, for the frame header.
void CodecState::import_frame_filter_bits(int p) {
  AV2_COMMON* const cm = cm_;
  RestorationInfo* const rsi = &cm->rst_info[p];
  WienerNonsepInfo* const ff = &rsi->frame_filters;
  const int nopcw = disable_pcwiener_filters_in_framefilters(&cm->seq_params), base_qindex = cm->quant_params.base_qindex;
  const int classes = rsi->num_filter_classes;
  if (cm->frame_filter_dictionary == nullptr) throw std::logic_error("no frame filter dictionary");
  *cm->num_ref_filters = set_frame_filter_dictionary(p, cm, classes, cm->frame_filter_dictionary, cm->frame_filter_dictionary_stride);
  ff->num_classes = classes, ff->num_ref_filters = *cm->num_ref_filters;
  std::vector<uint8_t> buf(1 << 14);
  avm_write_bit_buffer wb = {buf.data(), 0};
  int group_counts[NUM_MATCH_GROUPS];
  set_group_counts(p, classes, ff->num_ref_filters, group_counts, nopcw);
  int matches[WIENERNS_MAX_CLASSES];
  std::copy_n(ff->match_indices, WIENERNS_MAX_CLASSES, matches);
  for (int c = 0; c < classes; c++) {
    int only;
    ff->match_indices[c] = matches[c];  // predict_*() read the earlier ones
    const int pred = predict_group(c, ff->match_indices, group_counts, &only);
    const int group = index_to_group(matches[c], group_counts);
    if (!only) {
      avm_wb_write_bit(&wb, group != pred);
      if (group != pred) {
        int zero = -1;
        for (int i = 0; i < NUM_MATCH_GROUPS && zero < 0; i++)
          if (i != pred && group_counts[i] == 0) zero = i;
        if (zero < 0) {
          const int larger[] = {2, 2, 1};
          avm_wb_write_bit(&wb, group == larger[pred]);
        }
      }
    }
    const int ref = predict_within_group(group, c, ff->match_indices, group_counts), base = get_group_base(group, group_counts);
    const int n = group == 0 ? c + 1 : group_counts[group];
    if (n > 1) avm_wb_write_primitive_refsubexpfin(&wb, n, 4, ref - base, matches[c] - base);
  }
  const WienernsFilterParameters* const prm = get_wienerns_parameters(base_qindex, p != AVM_PLANE_Y);
  WienerNonsepInfoBank bank = {};
  bank.filter[0].num_classes = classes;
  std::vector<std::vector<int16_t>> refs(classes);
  bool exact[WIENERNS_MAX_CLASSES];
  for (int c = 0; c < classes; c++) {
    fill_first_slot_of_bank_with_filter_match(p, &bank, ff, ff->match_indices, base_qindex, c, cm->frame_filter_dictionary,
                                              cm->frame_filter_dictionary_stride, nopcw);
    const int16_t* const ref = const_nsfilter_taps(av2_constref_from_wienerns_bank(&bank, 0, c), c);
    refs[c].assign(ref, ref + prm->ncoeffs);
    exact[c] = std::equal(ref, ref + prm->ncoeffs, const_nsfilter_taps(ff, c));
    avm_wb_write_bit(&wb, exact[c]);
  }
  int ncoeffs1, ncoeffs2;
  config2ncoeffs(&prm->nsfilter_config, &ncoeffs1, &ncoeffs2);
  for (int c = 0; c < classes; c++) {
    if (exact[c]) continue;
    const int16_t* const t = const_nsfilter_taps(ff, c);
    int s, sym;
    wienerns_shape(prm, t, &s, &sym);
    for (int i = 0; i < prm->nsubsets - 1; i++) {
      avm_wb_write_bit(&wb, i < s);
      if (i >= s) break;
    }
    if (!skip_sym_bit(prm, s)) avm_wb_write_bit(&wb, sym);
    for (int i = 0; i < prm->ncoeffs; i++) {
      if (!prm->subset_config[s][i]) continue;
      const int lo = prm->coeffs[i][WIENERNS_MIN_ID], nb = prm->coeffs[i][WIENERNS_BIT_ID];
      avm_wb_write_primitive_refsubexpfin(&wb, 1 << nb, nb - 3, refs[c][i] - lo, t[i] - lo);
      if (sym && (i < prm->nsfilter_config.asymmetric || (i >= ncoeffs1 && i - ncoeffs1 < prm->nsfilter_config.asymmetric2))) i++;
    }
  }
  lr_filter_bits_[p].clear();
  for (uint32_t i = 0; i < wb.bit_offset; i++) lr_filter_bits_[p].push_back((buf[i >> 3] >> (7 - (i & 7))) & 1);
  rsi->frame_filters_initialized = 1;
  av2_copy_rst_frame_filters(&cm->cur_frame->rst_info[p], rsi);
}

// end_frame() from L4 on, in av2_decode_tg_tiles_and_wrapup()'s order.
void CodecState::import_end_frame() {
  AV2_COMMON* const cm = cm_;
  YV12_BUFFER_CONFIG& b = cm->cur_frame->buf;
  // Before the reference slots change: the dictionary is the references'.
  for (int p = 0; p < av2_num_planes(cm); p++)
    if (is_frame_filters_enabled(p) && to_readwrite_framefilters(&cm->rst_info[p], 0, 0)) import_frame_filter_bits(p);
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
  if (lr) av2_loop_restoration_filter_frame(&b, cm, optimized, &lr_ctxt_);
}
