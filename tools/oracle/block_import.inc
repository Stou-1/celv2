// BlockWriter's L2 (tools/oracle/oracle.h): avmenc's block coded as it chose it, where the syntax still
// allows it (block_writer.cpp, CELV2_ORACLE only).

#include "avm_dsp/recenter.h"

namespace oracle = celv2::oracle;
using celv2::oracle::count;

namespace {
bool same_mv(MV a, MV b) { return a.row == b.row && a.col == b.col; }

// What the decoder adds the difference to (assign_mv()).
MV mvd_base(MV ref, MvSubpelPrecision p, bool amvd) {
  if (!amvd && p < MV_PRECISION_HALF_PEL) lower_mv_precision(&ref, p);
  return ref;
}

bool reachable(MV want, MV ref, MvSubpelPrecision p) {
  const MV base = mvd_base(ref, p, false);
  const int step = 1 << (MV_PRECISION_ONE_EIGHTH_PEL - p);
  return (want.row - base.row) % step == 0 && (want.col - base.col) % step == 0;
}

bool in_amvd_set(MV want, MV ref) {
  for (const int d : {want.row - ref.row, want.col - ref.col}) {
    const int i = get_amvd_index_from_mvd(d);
    if (std::abs(i) > MAX_AMVD_INDEX || get_mvd_from_amvd_index(i) != d) return false;
  }
  return true;
}
}  // namespace

long long BlockWriter::block_area() const {
  const BLOCK_SIZE bsize = xd_->mi[0]->sb_type[PLANE_TYPE_Y];
  return static_cast<long long>(block_size_wide[bsize]) * block_size_high[bsize];
}

// A single-reference block: which of NEARMV / GLOBALMV / NEWMV, its DRL index, precision and adaptive
// difference. near_idx and new_idx come in as the MV search left them (near_idx: a candidate that is the
// MV, or -1). A 64x64 after the first of a 128 block, or the first reference of a compound the sequence
// cannot code, keeps only the MV: NEARMV where a candidate holds it.
void BlockWriter::import_single(const BlockDecision& d, MV_REFERENCE_FRAME ref_frame, MV gmv, MV wanted, int& near_idx,
                                int& new_idx, bool& global, bool& amvd, MvSubpelPrecision& new_prec) {
  AV2_COMMON* const cm = st_.cm();
  MACROBLOCKD* const xd = xd_;
  const OracleBlock& b = *d.imp;
  const int max_idx = cm->features.max_drl_bits;
  auto candidate = [&](int i) { return xd->ref_mv_stack[ref_frame][i].this_mv.as_mv; };
  if (b.ref[1] >= 0 && b.ref[0] == b.ref[1]) count(oracle::kSameRefCompound, block_area());
  if (b.y_mode == GLOBALMV && same_mv(wanted, gmv)) {
    global = true, near_idx = -1;
    return;
  }
  if (b.y_mode == NEARMV && !d.piece) {
    if (near_idx < 0) count(oracle::kNearLost, block_area());
    else if (b.ref_mv_idx[0] <= max_idx && same_mv(candidate(b.ref_mv_idx[0]), wanted)) near_idx = b.ref_mv_idx[0];
  }
  const bool newmv = b.y_mode == NEWMV && !d.piece && b.ref[1] < 0;
  if (newmv) {
    near_idx = -1;
    if (b.ref_mv_idx[0] <= max_idx) new_idx = b.ref_mv_idx[0];
    if (b.amvd && cm->seq_params.enable_adaptive_mvd) {  // its candidate, else another that gives a difference in the set
      int i = new_idx;
      if (!in_amvd_set(wanted, candidate(i)))
        for (i = 0; i <= max_idx && !in_amvd_set(wanted, candidate(i)); i++) {
        }
      amvd = i <= max_idx;
      if (amvd) new_idx = i;
      else count(oracle::kAmvdLost, block_area());
    }
    if (!amvd && cm->seq_params.enable_flex_mvres && cm->features.use_pb_mv_precision) {
      const MvSubpelPrecision p = static_cast<MvSubpelPrecision>(b.precision);
      if (reachable(wanted, candidate(new_idx), p)) new_prec = p;
      else count(oracle::kPrecisionLost, block_area());
    }
  }
  // A difference at the block's precision moves the MV unless its predictor is on the same grid: another
  // candidate's may be (an MV of avmenc's can be odd at quarter pel, by a candidate of its own).
  if (near_idx < 0 && !global && !amvd && !reachable(wanted, candidate(new_idx), new_prec)) {
    int i = 0;
    while (i <= max_idx && !reachable(wanted, candidate(i), new_prec)) i++;
    if (i <= max_idx) new_idx = i;
    else count(oracle::kMvMoved, block_area());
  }
}

// read_bawp flags of a single-reference block, as read_inter_block_mode_info() reads them.
void BlockWriter::import_bawp(const OracleBlock& b) {
  AV2_COMMON* const cm = st_.cm();
  MACROBLOCKD* const xd = xd_;
  MB_MODE_INFO* const mbmi = xd->mi[0];
  FRAME_CONTEXT* const ec_ctx = xd->tile_ctx;
  const int flag = b.bawp[0];
  mbmi->bawp_flag[0] = mbmi->bawp_flag[1] = 0;
  if (cm->features.enable_bawp && av2_allow_bawp(cm, mbmi, xd->mi_row, xd->mi_col)) {
    w_->symbol(flag > 0, ec_ctx->bawp_cdf[0], 2);
    if (flag && av2_allow_explicit_bawp(mbmi))
      w_->symbol(flag > 1, ec_ctx->explicit_bawp_cdf[mbmi->mode == NEARMV ? 0 : mbmi->mode == NEWMV && mbmi->use_amvd ? 1 : 2], 2);
    if (flag > 1) w_->symbol(flag - 2, ec_ctx->explicit_bawp_scale_cdf, EXPLICIT_BAWP_SCALE_CNT);
    mbmi->bawp_flag[0] = static_cast<int8_t>(flag);
  } else if (flag) {
    count(oracle::kBawpLost, block_area());
  }
  if (!cm->seq_params.monochrome && xd->is_chroma_ref && mbmi->bawp_flag[0]) {
    w_->symbol(b.bawp[1] != 0, ec_ctx->bawp_cdf[1], 2);
    mbmi->bawp_flag[1] = b.bawp[1];
  }
}

// One difference's magnitude, as read_mv() reads it.
void BlockWriter::import_magnitude(MV diff) {
  MB_MODE_INFO* const mbmi = xd_->mi[0];
  nmv_context* const nmvc = &xd_->tile_ctx->nmvc;
  if (!mbmi->use_amvd) return write_mv_magnitude(w_, diff, nmvc, mbmi->pb_mv_precision);
  const int ir = get_amvd_index_from_mvd(diff.row), ic = get_amvd_index_from_mvd(diff.col);
  w_->symbol((ir ? MV_JOINT_HZVNZ : 0) | (ic ? MV_JOINT_HNZVZ : 0), nmvc->amvd_joints_cdf, MV_JOINTS);
  if (ir) w_->symbol(std::abs(ir) - 1, nmvc->comps[0].amvd_indices_cdf, MAX_AMVD_INDEX);
  if (ic) w_->symbol(std::abs(ic) - 1, nmvc->comps[1].amvd_indices_cdf, MAX_AMVD_INDEX);
}

// The compound branch of inter_block_mode_info(): r0 < r1, the references already coded.
void BlockWriter::import_compound(const BlockDecision& d, int r0, MV_REFERENCE_FRAME ref_frame, int16_t* inter_mode_ctx,
                                  int_mv (*ref_mvs)[MAX_MV_REF_CANDIDATES], int* pts0, int* pts0_inref) {
  AV2_COMMON* const cm = st_.cm();
  MACROBLOCKD* const xd = xd_;
  MB_MODE_INFO* const mbmi = xd->mi[0];
  FRAME_CONTEXT* const ec_ctx = xd->tile_ctx;
  const BLOCK_SIZE bsize = mbmi->sb_type[PLANE_TYPE_Y];
  const OracleBlock& b = *d.imp;
  const int max_idx = cm->features.max_drl_bits;
  const MV m{d.mv_row, d.mv_col}, m2{d.mv2_row, d.mv2_col};
  const MV want[2] = {d.ref == r0 ? m : m2, d.ref == r0 ? m2 : m};
  int idx[2] = {b.ref_mv_idx[0], b.ref_mv_idx[1]};
  int_mv pred[2] = {};
  auto cand = [&](int list, int i) {
    if (has_second_drl(mbmi)) return xd->ref_mv_stack[mbmi->ref_frame[list]][i].this_mv;
    return list ? xd->ref_mv_stack[ref_frame][i].comp_mv : xd->ref_mv_stack[ref_frame][i].this_mv;
  };
  auto find = [&] {
    av2_find_mv_refs(cm, xd, mbmi, ref_frame, ref_mv_count_, xd->ref_mv_stack, xd->weight, ref_mvs, nullptr,
                     xd->warp_param_stack, 0, xd->valid_num_warp_candidates);
  };
  auto predictors = [&] {
    for (int r = 0; r < 2; r++) pred[r] = cand(r, idx[has_second_drl(mbmi) ? r : 0]);
  };
  // Whether the mode can code the pair; sets idx and the predictors.
  auto plan = [&](PREDICTION_MODE mode) {
    mbmi->mode = mode;
    mbmi->ref_mv_idx[0] = mbmi->ref_mv_idx[1] = 0;
    mbmi->use_amvd = 0;
    find();
    if (mode == GLOBAL_GLOBALMV) {
      for (int r = 0; r < 2; r++) {
        const MV g = get_warp_motion_vector(xd, &cm->global_motion[mbmi->ref_frame[r]], cm->features.fr_mv_precision, bsize,
                                            xd->mi_col, xd->mi_row).as_mv;
        if (!same_mv(g, want[r])) return false;
      }
      return true;
    }
    const bool second = has_second_drl(mbmi);
    const bool at_near[2] = {mode == NEAR_NEARMV || mode == NEAR_NEWMV, mode == NEAR_NEARMV || mode == NEW_NEARMV};
    for (int r = 0; r < 1 + second; r++) idx[r] = std::min(idx[r], max_idx);
    for (int r = 0; r < 2; r++) {
      const int k = second ? r : 0;
      if (!at_near[r]) continue;
      if (same_mv(cand(r, idx[k]).as_mv, want[r])) continue;
      int i = 0;
      while (i <= max_idx && !same_mv(cand(r, i).as_mv, want[r])) i++;
      if (i > max_idx) return false;
      idx[k] = i;
    }
    // NEW lists: a predictor on the MV's grid (see import_single()), where the list has its own index.
    for (int r = 0; r < 2 && mode != NEW_NEWMV; r++) {
      const int k = second ? r : 0;
      if (at_near[r] || (!second && at_near[1 - r]) || reachable(want[r], cand(r, idx[k]).as_mv, mbmi->max_mv_precision)) continue;
      for (int i = 0; i <= max_idx; i++)
        if (reachable(want[r], cand(r, i).as_mv, mbmi->max_mv_precision)) {
          idx[k] = i;
          break;
        }
    }
    if (mode == NEW_NEWMV && !(reachable(want[0], cand(0, idx[0]).as_mv, mbmi->max_mv_precision) && reachable(want[1], cand(1, idx[0]).as_mv, mbmi->max_mv_precision)))
      for (int i = 0; i <= max_idx; i++)
        if (reachable(want[0], cand(0, i).as_mv, mbmi->max_mv_precision) && reachable(want[1], cand(1, i).as_mv, mbmi->max_mv_precision)) {
          idx[0] = i;
          break;
        }
    predictors();
    return true;
  };
  const PREDICTION_MODE first = b.skip_mode || d.piece ? NEAR_NEARMV : static_cast<PREDICTION_MODE>(b.y_mode);
  PREDICTION_MODE mode = first;
  if (first > NEW_NEWMV || b.ref[0] != r0 || !plan(mode)) {
    count(first == NEAR_NEARMV ? oracle::kNearLost : oracle::kCompoundMode, block_area());
    mode = NEW_NEWMV;
    idx[0] = std::min<int>(b.ref_mv_idx[0], max_idx);
    plan(mode);
  }
  const bool has_new[2] = {mode == NEW_NEWMV || mode == NEW_NEARMV, mode == NEW_NEWMV || mode == NEAR_NEWMV};
  bool amvd = false;
  if (b.amvd && mode == first && cm->seq_params.enable_adaptive_mvd && allow_amvd_mode(mode)) {
    amvd = true;
    for (int r = 0; r < 2; r++) amvd = amvd && (!has_new[r] || in_amvd_set(want[r], pred[r].as_mv));
    if (!amvd) count(oracle::kAmvdLost, block_area());
  }

  const int16_t mode_ctx = av2_mode_context_analyzer(inter_mode_ctx, mbmi->ref_frame);
  w_->symbol(0, ec_ctx->inter_compound_mode_is_joint_cdf[get_inter_compound_mode_is_joint_context(cm, mbmi)], NUM_OPTIONS_IS_JOINT);
  w_->symbol(INTER_COMPOUND_OFFSET(mode), ec_ctx->inter_compound_mode_non_joint_type_cdf[mode_ctx], NUM_OPTIONS_NON_JOINT_TYPE);
  if (cm->features.opfl_refine_type == REFINE_SWITCHABLE) throw std::runtime_error("optical flow unsupported");
  mbmi->use_amvd = amvd;
  if (cm->seq_params.enable_adaptive_mvd && allow_amvd_mode(mode))
    w_->symbol(amvd, ec_ctx->amvd_mode_cdf[amvd_mode_to_index(mode)][get_amvd_context(xd)], 2);
  mbmi->ref_mv_idx[0] = mbmi->ref_mv_idx[1] = 0;
  find();
  if (mode != GLOBAL_GLOBALMV) predictors();
  mbmi->bawp_flag[1] = 0;
  for (int r = 0; r < 2; r++) xd->block_ref_scale_factors[r] = get_ref_scale_factors_const(cm, mbmi->ref_frame[r]);
  if (is_motion_variation_allowed_bsize(bsize, xd->mi_row, xd->mi_col) && is_compound_warp_causal_allowed(cm, xd, mbmi)) {
    mbmi->num_proj_ref[0] = av2_findSamples(cm, xd, pts0, pts0_inref, 0);
    mbmi->num_proj_ref[1] = av2_findSamples(cm, xd, pts0, pts0_inref, 1);
  }
  mbmi->max_num_warp_candidates = 0;
  if (motion_mode_allowed(cm, xd, xd->ref_mv_stack[mbmi->ref_frame[0]], mbmi) != (1 << SIMPLE_TRANSLATION))
    throw std::runtime_error("only translation is supported");
  mbmi->use_wedge_interintra = 0;
  mbmi->motion_mode = SIMPLE_TRANSLATION;
  mbmi->warpmv_with_mvd_flag = 0;
  mbmi->jmvd_scale_mode = 0;
  // read_drl_idx()
  if (have_drl_index(mode)) {
    const int16_t pristine = av2_mode_context_pristine(inter_mode_ctx, mbmi->ref_frame);
    for (int r = 0; r < 1 + has_second_drl(mbmi); r++) {
      for (int i = 0; i < max_idx; ++i) {
        const int bit = idx[r] > i;
        w_->symbol(bit, av2_get_drl_cdf(mbmi, ec_ctx, pristine, i), 2);
        if (!bit) break;
      }
      mbmi->ref_mv_idx[r] = idx[r];
    }
  }
  set_mv_precision(mbmi, mbmi->max_mv_precision);
  MvSubpelPrecision prec = mbmi->max_mv_precision;
  if (is_pb_mv_precision_active(cm, mbmi, bsize)) {
    const MvSubpelPrecision p = static_cast<MvSubpelPrecision>(b.precision);
    bool ok = mode == first;
    for (int r = 0; r < 2; r++) ok = ok && (!has_new[r] || reachable(want[r], pred[r].as_mv, p));
    if (ok) prec = p;
    else if (mode == first) count(oracle::kPrecisionLost, block_area());
    mv_precision(prec);
  }
  if (enable_adaptive_mvd_resolution(cm, mbmi)) set_amvd_mv_precision(mbmi, mbmi->max_mv_precision);
  if (is_mvd_sign_derive_allowed(cm, xd, mbmi)) throw std::runtime_error("MVD sign derivation unsupported");
  // assign_mv(): the differences' magnitudes, then their signs.
  if (mode == GLOBAL_GLOBALMV) {
    mbmi->mv[0].as_mv = want[0], mbmi->mv[1].as_mv = want[1];
  } else {
    MV diff[2] = {};
    for (int r = 0; r < 2; r++) {
      if (!has_new[r]) continue;
      const MV base = mvd_base(pred[r].as_mv, mbmi->pb_mv_precision, amvd);
      const int step = amvd ? 1 : 1 << (MV_PRECISION_ONE_EIGHTH_PEL - mbmi->pb_mv_precision);
      diff[r].row = static_cast<int16_t>((want[r].row - base.row) / step * step);
      diff[r].col = static_cast<int16_t>((want[r].col - base.col) / step * step);
      import_magnitude(diff[r]);
    }
    for (int r = 0; r < 2; r++) {
      if (has_new[r]) {
        if (diff[r].row) w_->literal(diff[r].row < 0, 1);
        if (diff[r].col) w_->literal(diff[r].col < 0, 1);
      }
      const MV base = has_new[r] ? mvd_base(pred[r].as_mv, mbmi->pb_mv_precision, amvd) : pred[r].as_mv;
      mbmi->mv[r].as_mv.row = static_cast<int16_t>(base.row + diff[r].row);
      mbmi->mv[r].as_mv.col = static_cast<int16_t>(base.col + diff[r].col);
      if (!same_mv(mbmi->mv[r].as_mv, want[r])) count(oracle::kMvMoved, block_area());
      mv_clamp_to_integer(&mbmi->mv[r].as_mv);
    }
  }
  mbmi->interinter_comp.type = COMPOUND_AVERAGE;
}

// read_cdef(): the 64x64's strength index at its first block that reads one.
void BlockWriter::import_cdef() {
  AV2_COMMON* const cm = st_.cm();
  MACROBLOCKD* const xd = xd_;
  if (cm->features.coded_lossless || !cm->cdef_info.cdef_frame_enable) return;
  const int skip = xd->mi[0]->skip_txfm[0], sb_mask = cm->mib_size - 1;
  if (!(xd->mi_row & sb_mask) && !(xd->mi_col & sb_mask)) av2_zero(xd->cdef_transmitted);
  const int index = av2_get_cdef_transmitted_index(xd->mi_row, xd->mi_col);
  CommonModeInfoParams* const mi = &cm->mi_params;
  MB_MODE_INFO* const unit = mi->mi_grid_base[fetch_cdef_mi_grid_index(cm, xd)];
  if (xd->cdef_transmitted[index]) return;
  if (!cm->cdef_info.cdef_on_skip_txfm_frame_enable && skip) {
    unit->cdef_strength = -1;
    return;
  }
  const oracle::Frame* const f = oracle::frame(static_cast<int>(cm->current_frame.display_order_hint));
  const int nhfb = (mi->mi_cols + 15) / 16;
  int v = f->cdef_idx[(xd->mi_row >> 4) * nhfb + (xd->mi_col >> 4)];
  if (v < 0) v = 0;  // not filtered in avmenc, where this block was skipped
  const int n = cm->cdef_info.nb_cdef_strengths;
  if (n > 1) {
    w_->symbol(v == 0, xd->tile_ctx->cdef_strength_index0_cdf[av2_get_cdef_context(cm, xd)], 2);
    if (v && n > 2) w_->symbol(v - 1, xd->tile_ctx->cdef_cdf[n - 3], n - 1);
  }
  unit->cdef_strength = static_cast<int8_t>(v);
  xd->cdef_transmitted[index] = true;
}

// loop_restoration_read_sb_coeffs() and read_wienerns_filter(), the unit's plan from CodecState::import_lr_units().
void BlockWriter::import_lr_unit(int plane, int unit) {
  AV2_COMMON* const cm = st_.cm();
  MACROBLOCKD* const xd = xd_;
  FRAME_CONTEXT* const fc = xd->tile_ctx;
  const RestorationInfo& rsi = cm->rst_info[plane];
  const RestorationUnitInfo& rui = rsi.unit_info[unit];
  const int type = rui.restoration_type;
  const bool is_uv = plane != AVM_PLANE_Y;
  if (rsi.frame_restoration_type == RESTORE_SWITCHABLE) {
    for (int re = 0; re <= RESTORE_SWITCHABLE - 2; re++) {
      if (cm->features.lr_tools_disable_mask[plane] & (1 << re)) continue;
      w_->symbol(type == re, fc->switchable_flex_restore_cdf[re][plane], 2);
      if (type == re) break;
    }
  } else if (rsi.frame_restoration_type == RESTORE_WIENER_NONSEP) {
    w_->symbol(type == RESTORE_WIENER_NONSEP, fc->wienerns_restore_cdf, 2);
  } else {
    w_->symbol(type == RESTORE_PC_WIENER, fc->pc_wiener_restore_cdf, 2);
  }
  if (type != RESTORE_WIENER_NONSEP || rsi.frame_filters_on) return;
  const CodecState::LrPlan& plan = st_.lr_plan(plane)[unit];
  const int classes = rsi.num_filter_classes;
  for (int c = 0; c < classes; c++) {
    w_->literal(plan.exact[c], 1);
    for (int ref = 0; ref < plan.size[c] - 1; ref++) {
      w_->literal(ref == plan.ref[c], 1);
      if (ref == plan.ref[c]) break;
    }
  }
  const WienernsFilterParameters* const prm = get_wienerns_parameters(cm->quant_params.base_qindex, is_uv);
  int ncoeffs1, ncoeffs2;
  config2ncoeffs(&prm->nsfilter_config, &ncoeffs1, &ncoeffs2);
  for (int c = 0; c < classes; c++) {
    if (plan.exact[c]) continue;
    const int16_t* const t = const_nsfilter_taps(&rui.wienerns_info, c);
    int s, sym;
    wienerns_shape(prm, t, &s, &sym);
    for (int i = 0; i < prm->nsubsets - 1; i++) {
      w_->symbol(i < s, fc->wienerns_length_cdf[is_uv], 2);
      if (i >= s) break;
    }
    if (!skip_sym_bit(prm, s)) w_->symbol(sym, fc->wienerns_uv_sym_cdf, 2);
    for (int i = 0; i < prm->ncoeffs; i++) {
      if (!prm->subset_config[s][i]) continue;
      const int lo = prm->coeffs[i][WIENERNS_MIN_ID], nb = prm->coeffs[i][WIENERNS_BIT_ID];
      // avm_write_4part_wref()
      const int v = recenter_finite_nonneg(1 << nb, plan.taps[c][i] - lo, t[i] - lo);
      const int part_bits[4] = {nb - 3, nb - 3, nb - 2, nb - 1}, part_offs[4] = {0, 1 << (nb - 3), 1 << (nb - 2), 1 << (nb - 1)};
      int part = 3;
      while (v < part_offs[part]) part--;
      w_->symbol(part, fc->wienerns_4part_cdf[prm->coeffs[i][WIENERNS_PAR_ID]], 4);
      w_->literal(v - part_offs[part], part_bits[part]);
      if (sym && (i < prm->nsfilter_config.asymmetric || (i >= ncoeffs1 && i - ncoeffs1 < prm->nsfilter_config.asymmetric2))) i++;
    }
  }
}

double BlockWriter::oracle_bits(int plane, int blk_row, int blk_col, TX_SIZE tx, const tran_low_t* q, int eob, FRAME_CONTEXT& fc, const TXB_CTX& ctx) {
  MACROBLOCKD* const xd = xd_;
  MB_MODE_INFO* const mbmi = xd->mi[0];
  SymbolLog scratch(&fc, false);  // no update: fc stays as it is
  scratch.count(true);
  SymbolLog* const w = w_;
  FRAME_CONTEXT* const tile_ctx = xd->tile_ctx;
  w_ = &scratch, xd->tile_ctx = &fc;
  const int all_zero = eob == 0;
  if (plane == AVM_PLANE_V) {
    w_->symbol(all_zero, fc.v_txb_skip_cdf[ctx.txb_skip_ctx + (xd->eob_u_flag ? V_TXB_SKIP_CONTEXT_OFFSET : 0)], 2);
  } else {
    const int pred_mode_ctx = (is_inter_block(mbmi, xd->tree_type) || mbmi->fsc_mode[xd->tree_type == CHROMA_PART]) ? 1 : 0;
    w_->symbol(all_zero, fc.txb_skip_cdf[pred_mode_ctx][get_txsize_entropy_ctx(tx)][ctx.txb_skip_ctx], 2);
  }
  if (!all_zero) {
    eob_symbols(plane, tx, eob);
    levels_and_signs(plane, tx, ctx, q, eob, blk_row, blk_col);
  }
  w_ = w, xd->tile_ctx = tile_ctx;
  return scratch.bits();
}
