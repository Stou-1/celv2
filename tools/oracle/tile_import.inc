// TileCoder from L3 on (tools/oracle/oracle.h): avmenc's transform splits and types, then its zero blocks and levels
// (encoder.cpp, CELV2_ORACLE only).

using celv2::oracle::count;
namespace oracle = celv2::oracle;

#include "real_trellis.inc"

// X_ORACLE_Q takes this block: TCQ's, and with d the plain path's but for inter forward skip coding (its own writer).
bool TileCoder::real_on(int plane, TX_TYPE type, bool tcq) const {
  const int m = oracle::real_cost();
  if (!m || use_inter_fsc(cm_, get_plane_type(plane), type, is_inter_block(xd_->mi[0], xd_->tree_type))) return false;
  return tcq || (m & oracle::kRealPlain);
}

// real::levels() for a transform block about to be coded: the contexts as BlockWriter::coeffs() takes them. eob_u: V's flag context.
int TileCoder::real_quant(int plane, int blk_row, int blk_col, TX_SIZE tx, TX_TYPE type, const tran_low_t* coeff, tran_low_t* q, tran_low_t* dq, bool tcq, bool eob_u) {
  AV2_COMMON* const cm = cm_;
  MACROBLOCKD* const xd = xd_;
  MB_MODE_INFO* const mbmi = xd->mi[0];
  struct macroblockd_plane* const pd = &xd->plane[plane];
  const BLOCK_SIZE plane_bsize = get_mb_plane_block_size(xd, mbmi, plane, pd->subsampling_x, pd->subsampling_y);
  const bool inter = is_inter_block(mbmi, xd->tree_type), fsc = mbmi->fsc_mode[xd->tree_type == CHROMA_PART] != 0;
  TXB_CTX& ctx = imp_ctx_[plane == AVM_PLANE_V];
  get_txb_ctx(plane_bsize, tx, plane, pd->above_entropy_context + blk_col, pd->left_entropy_context + blk_row, &ctx, fsc && cm->seq_params.enable_fsc);
  const int mode = oracle::real_cost();
  const FRAME_CONTEXT* const fc = (mode & oracle::kRealRefresh) ? imp_fc_.get() : cm->fc;
  const avm_cdf_prob* const flag = plane == AVM_PLANE_V ? fc->v_txb_skip_cdf[ctx.txb_skip_ctx + (eob_u ? V_TXB_SKIP_CONTEXT_OFFSET : 0)]
                                                       : fc->txb_skip_cdf[inter || fsc][get_txsize_entropy_ctx(tx)][ctx.txb_skip_ctx];
  // avmenc's weight: RDMULT_FROM_Q2_NUM / _DEN, plane_rd_mult, rshift 2 and RDCOST's scale; of the DC step.
  static const int plane_mult[2][2] = {{17, 13}, {16, 10}};
  static const double depth_factor[6] = {1, 1, 1.125, 1.25, 1.25, 1.40625};  // rd_layer_depth_factor
  double weight = 84.0 / 32 * plane_mult[inter][plane > 0] / 4 / 128;
  if (mode & oracle::kRealDepth) weight *= depth_factor[oracle::depth(static_cast<int>(cm->current_frame.display_order_hint))];
  const real::Block b{fc, tx, type, plane, get_eob_plane_ctx(plane, inter), ctx.dc_sign_ctx, q_, cm->seq_params.bit_depth, tcq,
                      {real::bits(flag, 0), real::bits(flag, 1)}, weight};
  return real::levels(b, coeff, q, dq, &imp_rate_[plane == AVM_PLANE_V]);
}

// X_ORACLE_QCHECK: on fixed CDFs, the path's rate is the bits of the symbols BlockWriter writes for its levels, here and on the
// same coefficients taken far past the range, into the Golomb steps.
void TileCoder::real_check(int plane, int blk_row, int blk_col, TX_SIZE tx, TX_TYPE type, const tran_low_t* coeff, const tran_low_t* q, int eob) {
  static std::atomic<long long> blocks[TX_SIZES_ALL], golomb;
  static struct Print {
    ~Print() {
      std::fprintf(stderr, "oracle: rate check:");
      for (int t = 0; t < TX_SIZES_ALL; t++) std::fprintf(stderr, " %dx%d %lld", tx_size_wide[t], tx_size_high[t], blocks[t].load());
      std::fprintf(stderr, ", past the range %lld\n", golomb.load());
    }
  } print;
  const int mode = oracle::real_cost();
  FRAME_CONTEXT* const fc = (mode & oracle::kRealRefresh) ? imp_fc_.get() : cm_->fc;
  const int v = plane == AVM_PLANE_V;
  auto same = [&](const tran_low_t* levels, int end, double rate) {
    const double got = bw_->oracle_bits(plane, blk_row, blk_col, tx, levels, end, *fc, imp_ctx_[v]);
    if (std::abs(got - rate) > 1e-6) {
      std::fprintf(stderr, "oracle: rate %.6f, written %.6f: plane %d tx %d type %d end %d\n", rate, got, plane, static_cast<int>(tx), static_cast<int>(type), end);
      throw std::runtime_error("X_ORACLE_QCHECK: the path's rate is not the symbols'");
    }
  };
  same(q, eob, imp_rate_[v]);
  blocks[tx]++;
  if (!eob) return;  // the writer's type is gone: DCT_DCT in the map
  const bool tcq = tcq_enable(cm_->features.tcq_mode, 0, plane, tx_type_to_class[get_primary_tx_type(type)]);
  const int n = get_txb_wide(tx) * get_txb_high(tx);
  tran_low_t big[1024], bq[1024], bdq[MAX_TX_SQUARE];
  for (int i = 0; i < n; i++) big[i] = coeff[i] * 64;
  const double saved = imp_rate_[v];
  const int end = real_quant(plane, blk_row, blk_col, tx, type, big, bq, bdq, tcq, xd_->eob_u_flag != 0);
  same(bq, end, imp_rate_[v]);
  imp_rate_[v] = saved;
  for (int i = 0; i < n; i++)
    if (std::abs(static_cast<int>(bq[i])) > MAX_BASE_BR_RANGE) {
      golomb++;
      break;
    }
}

const OracleTx* TileCoder::imp_tx(int plane, int blk_row, int blk_col, TX_SIZE tx) const {
  const CHROMA_REF_INFO& c = xd_->mi[0]->chroma_ref_info;
  const int r = (plane ? c.mi_row_chroma_base >> 1 : xd_->mi_row) + blk_row;
  const int col = (plane ? c.mi_col_chroma_base >> 1 : xd_->mi_col) + blk_col;
  const OracleTx* const t = imp_frame_->tx_at(plane, r, col);
  return t && t->tx_size == tx ? t : nullptr;
}

// The partition of the max-size unit at (blk_row, blk_col), where this block can code it.
TX_PARTITION_TYPE TileCoder::import_partition(BLOCK_SIZE bsize, TX_SIZE max_tx, int blk_row, int blk_col) {
  MACROBLOCKD* const xd = xd_;
  const int r = xd->mi_row + blk_row, c = xd->mi_col + blk_col;
  if (r >= imp_frame_->head.mi_rows || c >= imp_frame_->head.mi_cols) return TX_PARTITION_NONE;
  const TX_PARTITION_TYPE p = static_cast<TX_PARTITION_TYPE>(imp_frame_->tx_part[static_cast<size_t>(r) * imp_frame_->head.mi_cols + c]);
  if (p == TX_PARTITION_NONE) return p;
  const bool horz = allow_tx_horz_split(bsize, max_tx), vert = allow_tx_vert_split(bsize, max_tx);
  const int h_or_v = get_vert_or_horz_group(bsize);
  const bool four = p == TX_PARTITION_HORZ4 || p == TX_PARTITION_VERT4;
  const bool ok = horz && vert ? true
                  : !horz && !vert ? false
                  : p == (four ? (horz ? TX_PARTITION_HORZ4 : TX_PARTITION_VERT4) : (horz ? TX_PARTITION_HORZ : TX_PARTITION_VERT)) &&
                        (!four || (h_or_v && !xd->reduced_tx_part_set));
  if (ok) return p;
  count(oracle::kTxSplitLost, static_cast<long long>(tx_size_wide[max_tx]) * tx_size_high[max_tx]);
  return TX_PARTITION_NONE;
}

// The coefficients of tx_type's primary transform of res, AVM layout. Above 32 samples only the DCT has a kernel here
// but for the 64-sample sets, fwd_long().
void TileCoder::import_forward(const int16_t* res, int w, TX_SIZE tx, TX_TYPE primary, int plane, tran_low_t* out) {
  if (primary == DCT_DCT) fwd_dct(res, w, tx, enc_.coef_scale(plane, tx, xd_), out);
  else if (std::max(tx_size_wide[tx], tx_size_high[tx]) > 32) fwd_long(res, w, tx, primary, out);
  else fwd_transform(res, w, tx, primary, out);
}

// decode_token_recon_block() for a block with skip_txfm: nothing coded, the states the decoder leaves.
void TileCoder::import_skip_block(BLOCK_SIZE bsize, int plane_start, int plane_end) {
  AV2_COMMON* const cm = cm_;
  MACROBLOCKD* const xd = xd_;
  MB_MODE_INFO* const mbmi = xd->mi[0];
  av2_reset_entropy_context(xd, bsize, av2_num_planes(cm));
  av2_init_txk_skip_array(cm, xd->mi_row, xd->mi_col, bsize, 1, xd->tree_type, &mbmi->chroma_ref_info, plane_start, plane_end);
  if (is_cctx_enabled(cm, xd) && xd->is_chroma_ref) {
    const struct macroblockd_plane* const pd = &xd->plane[AVM_PLANE_U];
    const BLOCK_SIZE uv = get_mb_plane_block_size(xd, mbmi, AVM_PLANE_U, pd->subsampling_x, pd->subsampling_y);
    int row_offset, col_offset;
    get_chroma_mi_offsets(xd, &row_offset, &col_offset);
    update_cctx_array(xd, 0, 0, row_offset, col_offset, get_vartx_max_txsize(xd, uv, AVM_PLANE_U), CCTX_NONE);
  }
}

// t's levels and dequantized values into q and dq (zeroed), its end of block returned.
int TileCoder::import_levels(const OracleTx* t, TX_SIZE tx, tran_low_t* q, tran_low_t* dq) const {
  std::memset(q, 0, sizeof(tran_low_t) * 1024);
  std::memset(dq, 0, sizeof(tran_low_t) * tx_size_2d[tx]);
  if (!t || !t->eob) return 0;
  const OracleCoef* c = &imp_frame_->coefs[imp_frame_->first_coef[t - imp_frame_->txs.data()]];
  for (int i = 0; i < t->n_coef; i++, c++) q[c->pos] = c->level, dq[c->pos] = c->dq;
  return t->eob;
}

void TileCoder::import_residual(int plane, TX_SIZE tx, uint16_t* dst, int dst_stride, int blk_row, int blk_col) {
  AV2_COMMON* const cm = cm_;
  MACROBLOCKD* const xd = xd_;
  MB_MODE_INFO* const mbmi = xd->mi[0];
  const int w = tx_size_wide[tx], h = tx_size_high[tx], nq = std::min(w, 32) * std::min(h, 32), bd = cm->seq_params.bit_depth;
  const bool inter = is_inter_block(mbmi, xd->tree_type);
  const uint8_t reduced = is_reduced_tx_set_used(cm, get_plane_type(plane));
  const OracleTx* const t = imp_tx(plane, blk_row, blk_col, tx);
  int16_t res[64 * 64];
  residual_of(plane, tx, dst, dst_stride, res);
  tran_low_t coeff[1024], q[1024], dq[MAX_TX_SQUARE];
  TX_TYPE full = DCT_DCT;
  if (plane) full = av2_get_tx_type(xd, PLANE_TYPE_UV, blk_row, blk_col, tx, reduced);
  else if (t && t->eob) full = static_cast<TX_TYPE>(t->tx_type);
  if (!plane && !inter && get_primary_tx_type(full) == IDTX && oracle::level() < 5) full = DCT_DCT, count(oracle::kFsc, static_cast<long long>(w) * h);
  auto code = [&](TX_TYPE type) {
    const TX_TYPE primary = get_primary_tx_type(type);
    const int stx = plane ? 0 : get_secondary_tx_type(type);
    import_forward(res, w, tx, primary, plane, coeff);
    const PREDICTION_MODE mode = inter ? DC_PRED : static_cast<PREDICTION_MODE>(get_intra_mode(mbmi, PLANE_TYPE_Y));
    if (stx) fwd_ist(coeff, tx, primary, mode, get_secondary_tx_set(type), stx, bd);
    const bool tcq = tcq_enable(cm->features.tcq_mode, xd->lossless[mbmi->segment_id], plane, tx_type_to_class[primary]) &&
                     !use_inter_fsc(cm, get_plane_type(plane), type, inter);
    std::memset(dq, 0, sizeof(tran_low_t) * tx_size_2d[tx]);
    if (real_on(plane, type, tcq)) return real_quant(plane, blk_row, blk_col, tx, type, coeff, q, dq, tcq, plane == AVM_PLANE_V && xd->eob_u_flag);
    return quantize(coeff, q, dq, tx, q_, plane, type, nullptr, tcq, bd, plane ? 0 : trellis_);
  };
  const int lv = oracle::level();
  int eob = 0;
  if (lv >= 6) {
    if (!plane && t) full = mbmi->fsc_mode[0] ? IDTX : static_cast<TX_TYPE>(t->tx_type);
    eob = import_levels(t, tx, q, dq);
  } else {
    eob = code(full);
    if (!plane && get_secondary_tx_type(full) &&
        ((inter ? eob <= 3 : eob == 1) || !block_signals_sec_tx_type(xd, tx, get_primary_tx_type(full), eob))) {
      full = get_primary_tx_type(full);
      eob = code(full);
    }
    if (!plane && !inter && eob == 1 && full != DCT_DCT) full = DCT_DCT, eob = code(full);  // a lone DC codes no type
    if (lv >= 5 && eob && (!t || !t->eob)) {
      eob = 0;
      full = plane ? full : DCT_DCT;
      std::memset(q, 0, sizeof(tran_low_t) * nq);
    }
  }
  if (!imp_again_) bw_->coeffs(plane, blk_row, blk_col, tx, q, eob, full);
  else {  // the contexts coeffs() leaves (levels_and_signs()'s cul_level)
    const int16_t* const scan = get_scan(tx, full)->scan;
    int cul = 0;
    for (int c = 0; c < eob; c++) cul += std::abs(static_cast<int>(q[scan[c]]));
    cul = AVMMIN(COEFF_CONTEXT_MASK, cul);
    if (eob) set_dc_sign(&cul, q[scan[0]]);
    struct macroblockd_plane* const pd = &xd->plane[plane];
    av2_set_entropy_contexts(xd, pd, plane, get_mb_plane_block_size(xd, mbmi, plane, pd->subsampling_x, pd->subsampling_y), tx, cul, blk_col, blk_row);
  }
  if (!imp_again_ && lv < 5 && oracle::real_check() && real_on(plane, full, tcq_enable(cm->features.tcq_mode, xd->lossless[mbmi->segment_id], plane, tx_type_to_class[get_primary_tx_type(full)])))
    real_check(plane, blk_row, blk_col, tx, full, coeff, q, eob);
  if (eob) {
    const PLANE_TYPE plane_type = get_plane_type(plane);
    const TX_TYPE tx_type = imp_again_ ? full : av2_get_tx_type(xd, plane_type, blk_row, blk_col, tx, reduced);
    if (tx_type != full) throw std::runtime_error("unexpected transform type");
    av2_inverse_transform_block(xd, dq, plane, tx_type, tx, dst, dst_stride, eob,
                                replace_adst_by_ddt(cm->seq_params.enable_inter_ddt, cm->features.allow_screen_content_tools, xd), reduced);
  }
}

// residual_code_uv() with avmenc's rotation and the chroma type the sequence derives.
void TileCoder::import_residual_uv(TX_SIZE tx, int blk_row, int blk_col) {
  AV2_COMMON* const cm = cm_;
  MACROBLOCKD* const xd = xd_;
  const int w = tx_size_wide[tx], h = tx_size_high[tx], nq = std::min(w, 32) * std::min(h, 32);
  const int stride = xd->plane[AVM_PLANE_U].dst.stride, bd = cm->seq_params.bit_depth;
  const bool inter = is_inter_block(xd->mi[0], xd->tree_type);
  const uint8_t reduced = is_reduced_tx_set_used(cm, PLANE_TYPE_UV);
  const TX_TYPE type = av2_get_tx_type(xd, PLANE_TYPE_UV, blk_row, blk_col, tx, reduced);
  const OracleTx* const t = imp_tx(AVM_PLANE_U, blk_row, blk_col, tx);
  CctxType cctx = t ? static_cast<CctxType>(t->cctx) : CCTX_NONE;
  uint16_t* dst[2];
  int16_t res[64 * 64];
  tran_low_t coeff[2][1024], rot[2][1024], q[2][1024], dq[2][MAX_TX_SQUARE];
  for (int p = 0; p < 2; p++) {
    struct macroblockd_plane* const pd = &xd->plane[AVM_PLANE_U + p];
    dst[p] = &pd->dst.buf[(blk_row * pd->dst.stride + blk_col) << MI_SIZE_LOG2];
    residual_of(AVM_PLANE_U + p, tx, dst[p], stride, res);
    import_forward(res, w, tx, get_primary_tx_type(type), AVM_PLANE_U + p, coeff[p]);
  }
  const bool tcq = tcq_enable(cm->features.tcq_mode, xd->lossless[xd->mi[0]->segment_id], AVM_PLANE_U, tx_type_to_class[get_primary_tx_type(type)]);
  int eob[2] = {0, 0};
  auto code = [&](CctxType c) {
    std::memcpy(rot, coeff, sizeof(rot));
    if (c != CCTX_NONE) {
      const int a = cctx_mtx[c - CCTX_START][0], s = cctx_mtx[c - CCTX_START][1];
      for (int i = 0; i < nq; i++) {
        rot[0][i] = static_cast<tran_low_t>(std::lround((a * static_cast<double>(coeff[0][i]) + s * static_cast<double>(coeff[1][i])) / (1 << CCTX_PREC_BITS)));
        rot[1][i] = static_cast<tran_low_t>(std::lround((a * static_cast<double>(coeff[1][i]) - s * static_cast<double>(coeff[0][i])) / (1 << CCTX_PREC_BITS)));
      }
    }
    for (int p = 0; p < 2; p++) {
      std::memset(dq[p], 0, sizeof(tran_low_t) * tx_size_2d[tx]);
      eob[p] = real_on(AVM_PLANE_U + p, type, tcq) ? real_quant(AVM_PLANE_U + p, blk_row, blk_col, tx, type, rot[p], q[p], dq[p], tcq, p && eob[0])
                                                   : quantize(rot[p], q[p], dq[p], tx, q_, AVM_PLANE_U + p, type, nullptr, tcq, bd);
    }
  };
  const OracleTx* const tv = imp_tx(AVM_PLANE_V, blk_row, blk_col, tx);
  if (oracle::level() >= 6) {
    eob[0] = import_levels(t, tx, q[0], dq[0]), eob[1] = import_levels(tv, tx, q[1], dq[1]);
  } else {
    code(cctx);
    if (cctx != CCTX_NONE && !(eob[0] > 0 && (inter || eob[0] != 1))) cctx = CCTX_NONE, code(cctx);  // the type is not coded then
    for (int p = 0; p < 2 && oracle::level() >= 5; p++) {
      const OracleTx* const o = p ? tv : t;
      if (o && o->eob) continue;
      eob[p] = 0;
      std::memset(q[p], 0, sizeof(tran_low_t) * nq);
      std::memset(dq[p], 0, sizeof(tran_low_t) * tx_size_2d[tx]);
    }
    if (cctx != CCTX_NONE && !(eob[0] > 0 && (inter || eob[0] != 1))) cctx = CCTX_NONE;  // reconstruct as coded
  }
  bw_->coeffs(AVM_PLANE_U, blk_row, blk_col, tx, q[0], eob[0], type, cctx);
  bw_->coeffs(AVM_PLANE_V, blk_row, blk_col, tx, q[1], eob[1], type);
  if (oracle::level() < 5 && oracle::real_check() && real_on(AVM_PLANE_U, type, false))
    for (int p = 0; p < 2; p++) real_check(AVM_PLANE_U + p, blk_row, blk_col, tx, type, rot[p], q[p], eob[p]);
  av2_inv_cross_chroma_tx_block(dq[0], dq[1], tx, cctx, xd->bd);
  const int max_eob = std::max(eob[0], eob[1]);
  for (int p = 0; p < 2; p++) {
    if (!eob[p] && !(p == 1 && cctx != CCTX_NONE && eob[0])) continue;
    av2_inverse_transform_block(xd, dq[p], AVM_PLANE_U + p, type, tx, dst[p], stride, max_eob,
                                replace_adst_by_ddt(cm->seq_params.enable_inter_ddt, cm->features.allow_screen_content_tools, xd), reduced);
  }
}
