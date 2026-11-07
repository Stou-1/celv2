// oracle_dump <ivf> <out>: the decisions of an AV2 stream, read through the inspection decoder
// (docs/superpowers/specs/2026-11-04-av1-gap-breakdown-design.md, stage 3). Decoder headers only.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avm/avm_decoder.h"
#include "avm/avmdx.h"
#include "av2/common/av2_common_int.h"
#include "av2/decoder/accounting.h"
#include "av2/decoder/decoder.h"
#include "oracle_format.h"

static FILE* out;
static unsigned char* tree;
static OracleBlock* blocks;
static int n_tree, n_blocks, cap_blocks, cap_tree;
static unsigned char* tx_part;
static OracleTx* txs;
static OracleCoef* coefs;
static int n_tx, n_coef, cap_tx, cap_coef;
static uint32_t offset[3];  // this superblock's coefficient buffers, as far as the walk has gone

static void put_tree(int v) {
  if (n_tree == cap_tree) tree = realloc(tree, cap_tree = cap_tree ? 2 * cap_tree : 1 << 16);
  tree[n_tree++] = (unsigned char)v;
}

static void put_block(const AV2_COMMON* cm, int mi_row, int mi_col) {
  const MB_MODE_INFO* m = cm->mi_params.mi_grid_base[mi_row * cm->mi_params.mi_stride + mi_col];
  if (n_blocks == cap_blocks) blocks = realloc(blocks, sizeof(OracleBlock) * (cap_blocks = cap_blocks ? 2 * cap_blocks : 1 << 14));
  OracleBlock* b = &blocks[n_blocks++];
  memset(b, 0, sizeof(*b));
  const BLOCK_SIZE bs = m->sb_type[0];
  b->mi_row = (int16_t)mi_row, b->mi_col = (int16_t)mi_col;
  b->w4 = mi_size_wide[bs], b->h4 = mi_size_high[bs];
  b->inter = is_inter_ref_frame(m->ref_frame[0]);
  b->skip_mode = m->skip_mode, b->skip_txfm = m->skip_txfm[0];
  b->qindex = (int16_t)m->current_qindex;
  b->y_mode = m->mode, b->joint = m->joint_y_mode_delta_angle, b->uv_mode = m->uv_mode, b->mrl = m->mrl_index, b->multi_mrl = m->multi_line_mrl;
  b->cfl_idx = m->cfl_idx, b->cfl_alpha_idx = m->cfl_alpha_idx, b->mh_dir = m->mh_dir;
  b->angle_y = m->angle_delta[0], b->angle_uv = m->angle_delta[1], b->cfl_alpha_signs = m->cfl_alpha_signs;
  b->precision = m->pb_mv_precision, b->amvd = (uint8_t)m->use_amvd;
  b->interp = (uint8_t)m->interp_fltr, b->motion_mode = m->motion_mode;
  b->comp_type = m->interinter_comp.type, b->comp_group = m->comp_group_idx, b->cwp = (uint8_t)m->cwp_idx;
  b->fsc = m->fsc_mode[0];
  for (int i = 0; i < 2; i++) {
    b->ref_mv_idx[i] = (uint8_t)m->ref_mv_idx[i];
    b->bawp[i] = m->bawp_flag[i];
    b->ref[i] = m->ref_frame[i];
    b->mv[i][0] = m->mv[i].as_mv.row, b->mv[i][1] = m->mv[i].as_mv.col;
  }
}

static void walk(const AV2_COMMON* cm, const PARTITION_TREE* t, int mi_row, int mi_col, BLOCK_SIZE bsize) {
  if (mi_row >= cm->mi_params.mi_rows || mi_col >= cm->mi_params.mi_cols) {
    put_tree(255);
    return;
  }
  const int hw = mi_size_wide[bsize] / 2, hh = mi_size_high[bsize] / 2;
  const BLOCK_SIZE sub = get_partition_subsize(bsize, t->partition);
  switch (t->partition) {
    case PARTITION_NONE:
      put_tree(0);
      put_block(cm, mi_row, mi_col);
      break;
    case PARTITION_HORZ:
      put_tree(1);
      walk(cm, t->sub_tree[0], mi_row, mi_col, sub);
      walk(cm, t->sub_tree[1], mi_row + hh, mi_col, sub);
      break;
    case PARTITION_VERT:
      put_tree(2);
      walk(cm, t->sub_tree[0], mi_row, mi_col, sub);
      walk(cm, t->sub_tree[1], mi_row, mi_col + hw, sub);
      break;
    case PARTITION_SPLIT:
      put_tree(3);
      for (int i = 0; i < 4; i++) walk(cm, t->sub_tree[i], mi_row + (i >> 1) * hh, mi_col + (i & 1) * hw, sub);
      break;
    default:
      fprintf(stderr, "oracle_dump: partition type %d: extended partitions are not expected\n", t->partition);
      exit(5);
  }
}

// One transform block, in the plane's coding order: the next in that plane's coefficient buffers.
static void put_tx(AV2Decoder* pbi, const MACROBLOCKD* xd, int plane, TX_SIZE tx, int blk_row, int blk_col) {
  const AV2_COMMON* cm = &pbi->common;
  const DecoderCodingBlock* d = &pbi->td.dcb;
  const MB_MODE_INFO* m = xd->mi[0];
  const CHROMA_REF_INFO* c = &m->chroma_ref_info;
  const int stride = cm->mi_params.mi_stride;
  const int base_r = plane ? c->mi_row_chroma_base >> 1 : xd->mi_row, base_c = plane ? c->mi_col_chroma_base >> 1 : xd->mi_col;
  if (n_tx == cap_tx) txs = realloc(txs, sizeof(OracleTx) * (cap_tx = cap_tx ? 2 * cap_tx : 1 << 14));
  OracleTx* t = &txs[n_tx++];
  memset(t, 0, sizeof(*t));
  t->row = (int16_t)(base_r + blk_row), t->col = (int16_t)(base_c + blk_col);
  t->plane = (uint8_t)plane, t->tx_size = (uint8_t)tx;
  t->eob = d->eob_data[plane][offset[plane] / (TX_SIZE_W_MIN * TX_SIZE_H_MIN)].eob;
  if (plane == 0) t->tx_type = cm->mi_params.tx_type_map[(xd->mi_row + blk_row) * stride + xd->mi_col + blk_col];
  else t->cctx = cm->mi_params.cctx_type_map[(c->mi_row_chroma_base + (blk_row << 1)) * stride + c->mi_col_chroma_base + (blk_col << 1)];
  if (t->eob) {
    const int nq = AVMMIN(tx_size_wide[tx], 32) * AVMMIN(tx_size_high[tx], 32);
    const tran_low_t* q = d->qcoeff_block[plane] + offset[plane];
    const tran_low_t* dq = d->dqcoeff_block_copy[plane] + offset[plane];
    for (int i = 0; i < nq; i++) {
      if (!q[i]) continue;
      if (n_coef == cap_coef) coefs = realloc(coefs, sizeof(OracleCoef) * (cap_coef = cap_coef ? 2 * cap_coef : 1 << 16));
      OracleCoef* o = &coefs[n_coef++];
      o->pos = (uint16_t)i, o->level = q[i], o->dq = dq[i];
      t->n_coef++;
    }
  }
  offset[plane] += tx_size_wide[tx] * tx_size_high[tx];
}

// decode_token_recon_block()'s walk over one leaf's transform blocks (encoder.cpp's code_block() and code_inter_block()).
static void tx_leaf(AV2Decoder* pbi, MACROBLOCKD* xd, int mi_row, int mi_col) {
  AV2_COMMON* cm = &pbi->common;
  MB_MODE_INFO* m = cm->mi_params.mi_grid_base[mi_row * cm->mi_params.mi_stride + mi_col];
  const BLOCK_SIZE bsize = m->sb_type[0];
  const int inter = is_inter_block(m, SHARED_PART), cols = cm->mi_params.mi_cols;
  xd->mi = &m;
  xd->mi_row = mi_row, xd->mi_col = mi_col;
  xd->mb_to_right_edge = (cm->mi_params.mi_cols - mi_size_wide[bsize] - mi_col) * MI_SIZE * 8;
  xd->mb_to_bottom_edge = (cm->mi_params.mi_rows - mi_size_high[bsize] - mi_row) * MI_SIZE * 8;
  xd->is_chroma_ref = m->chroma_ref_info.is_chroma_ref;
  xd->tree_type = SHARED_PART;
  if (m->skip_txfm[0]) return;
  const TX_SIZE max_tx = max_txsize_rect_lookup[bsize];
  // Each max-size unit's partition.
  if (inter) {
    for (int r = 0; r < mi_size_high[bsize]; r += tx_size_high_unit[max_tx])
      for (int c = 0; c < mi_size_wide[bsize]; c += tx_size_wide_unit[max_tx])
        if (mi_row + r < cm->mi_params.mi_rows && mi_col + c < cols)
          tx_part[(mi_row + r) * cols + mi_col + c] = (unsigned char)m->tx_partition_type[av2_get_txb_size_index(bsize, r, c)];
  } else {
    for (int r = 0; r < mi_size_high[bsize]; r += tx_size_high_unit[max_tx])
      for (int c = 0; c < mi_size_wide[bsize]; c += tx_size_wide_unit[max_tx])
        if (mi_row + r < cm->mi_params.mi_rows && mi_col + c < cols) tx_part[(mi_row + r) * cols + mi_col + c] = (unsigned char)m->tx_partition_type[0];
  }
  const int max_w = max_block_wide(xd, bsize, 0), max_h = max_block_high(xd, bsize, 0);
  const int mu_w = AVMMIN(max_w, mi_size_wide[BLOCK_64X64]), mu_h = AVMMIN(max_h, mi_size_high[BLOCK_64X64]);
  const int mu128 = mi_size_wide[BLOCK_128X128];
  for (int row128 = 0; row128 < max_h; row128 += mu128)
    for (int col128 = 0; col128 < max_w; col128 += mu128)
      for (int row = row128; row < AVMMIN(row128 + mu128, max_h); row += mu_h)
        for (int col = col128; col < AVMMIN(col128 + mu128, max_w); col += mu_w)
          for (int plane = 0; plane < 3; plane++) {
            if (plane && !xd->is_chroma_ref) break;
            const struct macroblockd_plane* pd = &xd->plane[plane];
            const int ssx = pd->subsampling_x, ssy = pd->subsampling_y;
            const BLOCK_SIZE pb = get_mb_plane_block_size(xd, m, plane, ssx, ssy);
            const int unit_h = get_plane_tx_unit_height(xd, pb, plane, row, ssy), unit_w = get_plane_tx_unit_width(xd, pb, plane, col, ssx);
            const int high = max_block_high(xd, pb, plane), wide = max_block_wide(xd, pb, plane);
            if (inter || plane) {
              if (plane && skip_parsing_recon(row, col, ssx, ssy, 0)) continue;
              const TX_SIZE t = inter ? get_vartx_max_txsize(xd, pb, plane) : av2_get_tx_size(plane, xd);
              for (int br = row >> ssy; br < unit_h; br += tx_size_high_unit[t])
                for (int bc = col >> ssx; bc < unit_w; bc += tx_size_wide_unit[t]) {
                  if (br >= high || bc >= wide) continue;
                  const TX_PARTITION_TYPE p = plane ? TX_PARTITION_NONE : m->tx_partition_type[av2_get_txb_size_index(pb, br, bc)];
                  if (p == TX_PARTITION_NONE) {
                    put_tx(pbi, xd, plane, t, br, bc);
                    continue;
                  }
                  TXB_POS_INFO pos;
                  TX_SIZE subs[MAX_TX_PARTITIONS];
                  const int n = get_tx_partition_sizes(p, t, &pos, subs, xd->error_info);
                  for (int i = 0; i < n; i++)
                    if (br + pos.row_offset[i] < high && bc + pos.col_offset[i] < wide)
                      put_tx(pbi, xd, plane, subs[i], br + pos.row_offset[i], bc + pos.col_offset[i]);
                }
            } else {
              TXB_POS_INFO pos;
              TX_SIZE subs[MAX_TX_PARTITIONS];
              const int n = get_tx_partition_sizes(m->tx_partition_type[0], max_txsize_rect_lookup[pb], &pos, subs, xd->error_info);
              for (int i = 0; i < n; i++)
                if (row + pos.row_offset[i] < unit_h && col + pos.col_offset[i] < unit_w)
                  put_tx(pbi, xd, 0, subs[i], row + pos.row_offset[i], col + pos.col_offset[i]);
            }
          }
}

static void tx_walk(AV2Decoder* pbi, MACROBLOCKD* xd, const PARTITION_TREE* t, int mi_row, int mi_col, BLOCK_SIZE bsize) {
  const AV2_COMMON* cm = &pbi->common;
  if (mi_row >= cm->mi_params.mi_rows || mi_col >= cm->mi_params.mi_cols) return;
  const int hw = mi_size_wide[bsize] / 2, hh = mi_size_high[bsize] / 2;
  const BLOCK_SIZE sub = get_partition_subsize(bsize, t->partition);
  switch (t->partition) {
    case PARTITION_NONE: tx_leaf(pbi, xd, mi_row, mi_col); break;
    case PARTITION_HORZ:
      tx_walk(pbi, xd, t->sub_tree[0], mi_row, mi_col, sub);
      tx_walk(pbi, xd, t->sub_tree[1], mi_row + hh, mi_col, sub);
      break;
    case PARTITION_VERT:
      tx_walk(pbi, xd, t->sub_tree[0], mi_row, mi_col, sub);
      tx_walk(pbi, xd, t->sub_tree[1], mi_row, mi_col + hw, sub);
      break;
    default:
      for (int i = 0; i < 4; i++) tx_walk(pbi, xd, t->sub_tree[i], mi_row + (i >> 1) * hh, mi_col + (i & 1) * hw, sub);
  }
}

// After each superblock: its coefficient buffers hold its transform blocks, each plane's in turn.
static void on_sb(void* p, void* ctx) {
  (void)ctx;
  AV2Decoder* pbi = (AV2Decoder*)p;
  AV2_COMMON* cm = &pbi->common;
  static MACROBLOCKD xd;
  const SB_INFO* sbi = pbi->td.dcb.xd.sbi;
  const size_t cells = (size_t)cm->mi_params.mi_rows * cm->mi_params.mi_cols;
  if (sbi->mi_row == 0 && sbi->mi_col == 0) {
    tx_part = realloc(tx_part, cells);
    memset(tx_part, 0, cells);
  }
  xd = pbi->td.dcb.xd;
  offset[0] = offset[1] = offset[2] = 0;
  tx_walk(pbi, &xd, sbi->ptree_root[0], sbi->mi_row, sbi->mi_col, cm->sb_size);
  for (int i = 0; i < 3; i++)
    if (offset[i] != pbi->td.dcb.cb_offset[i]) {
      fprintf(stderr, "oracle_dump: plane %d: the walk took %u of %u at (%d, %d)\n", i, offset[i], pbi->td.dcb.cb_offset[i], sbi->mi_row, sbi->mi_col);
      exit(6);
    }
}

static void put_filters(const AV2_COMMON* cm) {
  static OracleFilters o;
  memset(&o, 0, sizeof(o));
  const struct loopfilter* lf = &cm->lf;
  o.lf_apply[0] = lf->apply_deblocking_filter[0], o.lf_apply[1] = lf->apply_deblocking_filter[1];
  o.lf_apply[2] = lf->apply_deblocking_filter_u, o.lf_apply[3] = lf->apply_deblocking_filter_v;
  o.lf_delta[0] = lf->delta_q_luma[0], o.lf_delta[1] = lf->delta_q_luma[1], o.lf_delta[2] = lf->delta_q_u, o.lf_delta[3] = lf->delta_q_v;
  const CdefInfo* c = &cm->cdef_info;
  o.cdef_enable = c->cdef_frame_enable, o.cdef_damping = c->cdef_damping, o.cdef_n = c->nb_cdef_strengths, o.cdef_on_skip = c->cdef_on_skip_txfm_frame_enable;
  for (int i = 0; i < c->nb_cdef_strengths; i++) o.cdef_y[i] = c->cdef_strengths[i], o.cdef_uv[i] = c->cdef_uv_strengths[i];
  const CommonModeInfoParams* mi = &cm->mi_params;
  const int nvfb = (mi->mi_rows + 15) / 16, nhfb = (mi->mi_cols + 15) / 16;
  o.cdef_units = nvfb * nhfb;
  const CcsoInfo* s = &cm->ccso_info;
  o.ccso_frame = s->ccso_frame_flag;
  const int log2 = get_ccso_unit_size_log2_adaptive_tile((AV2_COMMON*)cm, cm->mib_size_log2 + MI_SIZE_LOG2, CCSO_BLK_SIZE);
  const int fu = 1 << log2 >> MI_SIZE_LOG2, ccols = (mi->mi_cols + fu - 1) / fu, crows = (mi->mi_rows + fu - 1) / fu;
  for (int p = 0; p < 3; p++) {
    OracleCcso* q = &o.ccso[p];
    q->enable = s->ccso_enable[p], q->reuse = s->reuse_ccso[p], q->sb_reuse = s->sb_reuse_ccso[p], q->bo_only = s->ccso_bo_only[p];
    q->scale = s->scale_idx[p], q->quant = s->quant_idx[p], q->ext = s->ext_filter_support[p], q->edge_clf = s->edge_clf[p];
    q->band = s->max_band_log2[p], q->ref_idx = s->ccso_ref_idx[p];
    memcpy(o.ccso_offset[p], s->filter_offset[p], sizeof(o.ccso_offset[p]));
    o.ccso_units[p] = q->enable ? ccols * crows : 0;
    const RestorationInfo* r = &cm->rst_info[p];
    o.lr_type[p] = r->frame_restoration_type, o.lr_size[p] = r->restoration_unit_size, o.lr_frame_filters[p] = r->frame_filters_on;
    o.lr_temporal[p] = r->temporal_pred_flag, o.lr_ref[p] = r->rst_ref_pic_idx, o.lr_classes[p] = r->num_filter_classes;
    o.lr_units[p] = r->frame_restoration_type == RESTORE_NONE ? 0 : r->horz_units_per_frame * r->vert_units_per_frame;
  }
  o.wiener_bytes = sizeof(WienerNonsepInfo);
  fwrite(&o, sizeof(o), 1, out);
  for (int r = 0; r < nvfb; r++)
    for (int k = 0; k < nhfb; k++) {
      const int8_t v = mi->mi_grid_base[r * 16 * mi->mi_stride + k * 16]->cdef_strength;
      fwrite(&v, 1, 1, out);
    }
  for (int p = 0; p < 3; p++)
    for (int r = 0; r < crows && o.ccso_units[p]; r++)
      for (int k = 0; k < ccols; k++) {
        const MB_MODE_INFO* m = mi->mi_grid_base[r * fu * mi->mi_stride + k * fu];
        const unsigned char v = p == 0 ? m->ccso_blk_y : p == 1 ? m->ccso_blk_u : m->ccso_blk_v;
        fwrite(&v, 1, 1, out);
      }
  for (int p = 0; p < 3; p++) {
    const RestorationInfo* r = &cm->rst_info[p];
    if (!o.lr_units[p]) continue;
    fwrite(&r->frame_filters, sizeof(WienerNonsepInfo), 1, out);
    for (int i = 0; i < o.lr_units[p]; i++) {
      const unsigned char t = (unsigned char)r->unit_info[i].restoration_type;
      fwrite(&t, 1, 1, out);
    }
    for (int i = 0; i < o.lr_units[p]; i++)
      if (r->unit_info[i].restoration_type == RESTORE_WIENER_NONSEP) fwrite(&r->unit_info[i].wienerns_info, sizeof(WienerNonsepInfo), 1, out);
  }
}

static void on_frame(void* p, void* ctx) {
  (void)ctx;
  AV2Decoder* pbi = (AV2Decoder*)p;
  AV2_COMMON* cm = &pbi->common;
  OracleFrame f;
  memset(&f, 0, sizeof(f));
  const int key = cm->current_frame.frame_type == KEY_FRAME;
  f.key = key, f.display = cm->current_frame.display_order_hint, f.shown = cm->immediate_output_picture;
  f.refresh = cm->current_frame.refresh_frame_flags, f.qindex = cm->quant_params.base_qindex;
  f.n_refs = key ? 0 : cm->ref_frames_info.num_total_refs;
  for (int i = 0; i < f.n_refs; i++) f.ref_hint[i] = get_ref_frame_buf(cm, i)->display_order_hint;
  f.interp_filter = cm->features.interp_filter, f.reference_mode = cm->current_frame.reference_mode;
  f.skip_mode_flag = cm->current_frame.skip_mode_info.skip_mode_flag;
  f.skip_ref[0] = cm->current_frame.skip_mode_info.ref_frame_idx_0, f.skip_ref[1] = cm->current_frame.skip_mode_info.ref_frame_idx_1;
  f.max_drl_bits = cm->features.max_drl_bits, f.fr_mv_precision = cm->features.fr_mv_precision;
  f.use_pb_mv_precision = cm->features.use_pb_mv_precision, f.force_integer_mv = cm->features.cur_frame_force_integer_mv;
  f.tx_mode = cm->features.tx_mode, f.reduced_tx_set = cm->features.reduced_tx_set_used;
  f.delta_q_present = cm->delta_q_info.delta_q_present_flag, f.delta_q_res = cm->delta_q_info.delta_q_res;
  f.enable_bawp = cm->features.enable_bawp;
  for (int i = 0; i < f.n_refs; i++) {
    f.gm_type[i] = cm->global_motion[i].wmtype;
    for (int k = 0; k < 6; k++) f.gm_mat[i][k] = cm->global_motion[i].wmmat[k];
  }
  n_tree = n_blocks = 0;
  const int sb = mi_size_wide[cm->sb_size];
  for (int r = 0; r < cm->mi_params.mi_rows; r += sb)
    for (int c = 0; c < cm->mi_params.mi_cols; c += sb)
      walk(cm, av2_get_sb_info(cm, r, c)->ptree_root[0], r, c, cm->sb_size);
  f.n_tree = n_tree, f.n_blocks = n_blocks, f.n_tx = n_tx, f.n_coef = n_coef;
  f.mi_rows = cm->mi_params.mi_rows, f.mi_cols = cm->mi_params.mi_cols;
  const SequenceHeader* s = &cm->seq_params;
  f.reduced_tx_part_set = s->reduced_tx_part_set, f.enable_idtx_intra = s->enable_idtx_intra, f.enable_inter_ist = s->enable_inter_ist;
  f.enable_chroma_dctonly = s->enable_chroma_dctonly, f.enable_ist = s->enable_ist, f.enable_cctx = s->enable_cctx;
  f.lr_mask[0] = s->lr_tools_disable_mask[0], f.lr_mask[1] = s->lr_tools_disable_mask[1];
  f.df_par_bits_minus2 = s->df_par_bits_minus2, f.enable_cdef_on_skip_txfm = s->enable_cdef_on_skip_txfm;
  f.enable_ibp = s->enable_ibp, f.cfl_ds_filter_index = s->cfl_ds_filter_index;
  f.enable_cfl_intra = s->enable_cfl_intra, f.enable_mhccp = s->enable_mhccp, f.tmvp_sample_step = cm->tmvp_sample_step;
  fwrite(&f, sizeof(f), 1, out);
  fwrite(tree, 1, n_tree, out);
  fwrite(blocks, sizeof(OracleBlock), n_blocks, out);
  fwrite(tx_part, 1, (size_t)f.mi_rows * f.mi_cols, out);
  fwrite(txs, sizeof(OracleTx), n_tx, out);
  fwrite(coefs, sizeof(OracleCoef), n_coef, out);
  n_tx = n_coef = 0;
  put_filters(cm);
}

int main(int argc, char** argv) {
  if (argc != 3) return fprintf(stderr, "usage: oracle_dump <ivf> <out>\n"), 1;
  FILE* f = fopen(argv[1], "rb");
  out = fopen(argv[2], "wb");
  if (!f || !out) return fprintf(stderr, "oracle_dump: cannot open the files\n"), 1;
  unsigned char hdr[32];
  if (fread(hdr, 1, 32, f) != 32) return 1;
  const int32_t size[2] = {hdr[12] | hdr[13] << 8, hdr[14] | hdr[15] << 8};
  fwrite("CELV2OR1", 1, 8, out);
  fwrite(size, 4, 2, out);
  avm_codec_ctx_t codec;
  if (avm_codec_dec_init(&codec, avm_codec_av2_dx(), NULL, 0)) return 2;
  avm_inspect_init ii;
  memset(&ii, 0, sizeof(ii));
  ii.inspect_cb = on_frame;
  ii.inspect_sb_cb = on_sb;
  avm_codec_control(&codec, AV2_SET_INSPECTION_CALLBACK, &ii);
  unsigned char fh[12];
  while (fread(fh, 1, 12, f) == 12) {
    const unsigned n = fh[0] | fh[1] << 8 | fh[2] << 16 | (unsigned)fh[3] << 24;
    unsigned char* buf = malloc(n);
    if (fread(buf, 1, n, f) != n) return 3;
    if (avm_codec_decode(&codec, buf, n, NULL)) return fprintf(stderr, "decode: %s\n", avm_codec_error_detail(&codec)), 4;
    free(buf);
    avm_codec_iter_t it = NULL;
    while (avm_codec_get_frame(&codec, &it)) {
    }
  }
  fclose(out);
  return 0;
}
