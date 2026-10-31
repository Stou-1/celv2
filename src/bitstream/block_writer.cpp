// Portions of this file are derived from the AVM v1.0.0 decoder (av2/decoder),
// Copyright (c) 2021, Alliance for Open Media, under the BSD 3-Clause Clear License
// and the Alliance for Open Media Patent License 1.0: see third_party/avm/.

#include "bitstream/block_writer.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

extern "C" {
#include "av2/common/ccso.h"
#include "av2/common/cfl.h"
#include "av2/common/hr_coding.h"
#include "av2/common/entropymv.h"
#include "avm_dsp/binary_codes_writer.h"
#include "av2/common/mvref_common.h"
#include "av2/common/reconinter.h"
#include "av2/common/pred_common.h"
#include "av2/common/quant_common.h"
#include "av2/common/reconintra.h"
#include "av2/common/scan.h"
#include "av2/common/txb_common.h"
}

namespace {

// Joint luma mode index (mode + angle delta) of each celv2 intra mode, angle delta 0.
// Index into AV2's joint luma mode list: DC, SMOOTH, SMOOTH_V, SMOOTH_H, PAETH, then the
// directional modes (D45, D67, V, D113, D135, D157, H, D203) x angle delta -3..3; celv2 uses delta 0.
int joint_y_mode(IntraMode m) {
  auto dir = [](int k) { return NON_DIRECTIONAL_MODES_COUNT + k * TOTAL_ANGLE_DELTA_COUNT + MAX_ANGLE_DELTA; };
  switch (m) {
    case IntraMode::DC: return 0;
    case IntraMode::Smooth: return 1;
    case IntraMode::SmoothV: return 2;
    case IntraMode::SmoothH: return 3;
    case IntraMode::Paeth: return 4;
    case IntraMode::D45: return dir(0);
    case IntraMode::D67: return dir(1);
    case IntraMode::V: return dir(2);
    case IntraMode::D113: return dir(3);
    case IntraMode::D135: return dir(4);
    case IntraMode::D157: return dir(5);
    case IntraMode::H: return dir(6);
    case IntraMode::D203: return dir(7);
  }
  throw std::runtime_error("unsupported intra mode");
}

UV_PREDICTION_MODE uv_mode_of(IntraMode m) {
  switch (m) {
    case IntraMode::DC: return UV_DC_PRED;
    case IntraMode::Smooth: return UV_SMOOTH_PRED;
    case IntraMode::SmoothV: return UV_SMOOTH_V_PRED;
    case IntraMode::SmoothH: return UV_SMOOTH_H_PRED;
    case IntraMode::Paeth: return UV_PAETH_PRED;
    case IntraMode::D45: return UV_D45_PRED;
    case IntraMode::D67: return UV_D67_PRED;
    case IntraMode::V: return UV_V_PRED;
    case IntraMode::D113: return UV_D113_PRED;
    case IntraMode::D135: return UV_D135_PRED;
    case IntraMode::D157: return UV_D157_PRED;
    case IntraMode::H: return UV_H_PRED;
    case IntraMode::D203: return UV_D203_PRED;
  }
  throw std::runtime_error("unsupported intra mode");
}

PARTITION_TYPE fallback_partition(const bool* allowed, PARTITION_TYPE wanted) {
  if (allowed[wanted]) return wanted;
  const PARTITION_TYPE order[] = {PARTITION_SPLIT, PARTITION_HORZ, PARTITION_VERT, PARTITION_NONE};
  for (PARTITION_TYPE p : order)
    if (allowed[p]) return p;
  throw std::runtime_error("no basic partition type allowed");
}

}  // namespace

PARTITION_TYPE BlockWriter::partition(int mi_row, int mi_col, BLOCK_SIZE bsize,
                                      const PARTITION_TREE* ptree, PARTITION_TYPE wanted) {
  AV2_COMMON* const cm = st_.cm();
  MACROBLOCKD* const xd = xd_;
  FRAME_CONTEXT* ec_ctx = xd->tile_ctx;
  const int plane = xd->tree_type == CHROMA_PART;
  const int ssx = cm->seq_params.subsampling_x;
  const int ssy = cm->seq_params.subsampling_y;
  PARTITION_TYPE derived = av2_get_normative_forced_partition_type(
      &cm->mi_params, xd->tree_type, ssx, ssy, mi_row, mi_col, bsize, nullptr);
  bool allowed[ALL_PARTITION_TYPES];
  init_allowed_partitions_for_signaling(
      allowed, cm, xd->tree_type, (ptree->parent ? ptree->parent->region_type : INTRA_REGION),
      mi_row, mi_col, ssx, ssy, bsize, &ptree->chroma_ref_info);
  if (derived != PARTITION_INVALID && allowed[derived]) return derived;
  derived = only_allowed_partition(allowed);
  if (derived != PARTITION_INVALID) return derived;

  const PARTITION_TYPE p = fallback_partition(allowed, wanted);
  bool implied_do_split;
  if (!is_do_split_implied(allowed, &implied_do_split)) {
    const int ctx = partition_plane_context(xd, mi_row, mi_col, bsize, 0, SPLIT_CTX_MODE);
    w_->symbol(p != PARTITION_NONE, ec_ctx->do_split_cdf[plane][ctx], 2);
  }
  if (p == PARTITION_NONE) return PARTITION_NONE;

  if (allowed[PARTITION_SPLIT]) {
    const int ctx = partition_plane_context(xd, mi_row, mi_col, bsize, 0, SQUARE_SPLIT_CTX_MODE);
    w_->symbol(p == PARTITION_SPLIT, ec_ctx->do_square_split_cdf[plane][ctx], 2);
    if (p == PARTITION_SPLIT) return PARTITION_SPLIT;
  }

  RECT_PART_TYPE rect = rect_type_implied_by_bsize(bsize, xd->tree_type);
  if (rect == RECT_INVALID) rect = only_allowed_rect_type(allowed);
  if (rect == RECT_INVALID) {
    rect = p == PARTITION_VERT ? VERT : HORZ;
    const int ctx = partition_plane_context(xd, mi_row, mi_col, bsize, 0, RECT_TYPE_CTX_MODE);
    w_->symbol(rect, ec_ctx->rect_type_cdf[plane][ctx], NUM_RECT_PARTS);
  }
  bool implied_do_ext;
  if (!is_do_ext_partition_implied(allowed, rect, &implied_do_ext) || implied_do_ext)
    throw std::runtime_error("extended partitions are disabled but were not implied off");
  return rect == HORZ ? PARTITION_HORZ : PARTITION_VERT;
}

void BlockWriter::intra_frame_mode_info(const BlockDecision& d) {
  AV2_COMMON* const cm = st_.cm();
  MACROBLOCKD* const xd = xd_;
  MB_MODE_INFO* const mbmi = xd->mi[0];
  FRAME_CONTEXT* ec_ctx = xd->tile_ctx;
  const BLOCK_SIZE bsize = mbmi->sb_type[xd->tree_type == CHROMA_PART];

  mbmi->skip_mode = 0;
  if (xd->tree_type != CHROMA_PART) mbmi->morph_pred = 0;
  mbmi->motion_mode = SIMPLE_TRANSLATION;
  mbmi->refinemv_flag = 0;
  mbmi->skip_txfm[xd->tree_type == CHROMA_PART] = 0;  // no IntraBC -> no skip flag
  if (xd->tree_type != CHROMA_PART) mbmi->segment_id = 0;
  mbmi->seg_id_predicted = 0;
  if (cm->seq_params.enable_ccso && xd->tree_type != CHROMA_PART) ccso();
  if (xd->tree_type != CHROMA_PART) delta_qindex();
  mbmi->current_qindex = xd->current_base_qindex;
  const int seg_qindex = av2_get_qindex(&cm->seg, mbmi->segment_id, xd->current_base_qindex,
                                        cm->seq_params.bit_depth);
  get_qindex_with_offsets(cm, seg_qindex, mbmi->final_qindex_dc, mbmi->final_qindex_ac);
  mbmi->ref_frame[0] = INTRA_FRAME;
  mbmi->ref_frame[1] = NONE_FRAME;
  if (xd->tree_type != CHROMA_PART) mbmi->palette_mode_info.palette_size[0] = 0;
  mbmi->palette_mode_info.palette_size[1] = 0;
  if (xd->tree_type != CHROMA_PART) mbmi->use_intra_dip = 0;

  intra_modes(d);
  if (cm->seq_params.enable_refmvbank) decide_rmb_unit_update_count(cm, xd, mbmi);  // av2_read_mode_info()
}

namespace {
// Bits of symbol s under an AVM inverse CDF (avm_od_ec_encode_cdf_q15's probability).
double symbol_bits(const avm_cdf_prob* icdf, int s) {
  const int hi = s > 0 ? icdf[s - 1] : CDF_PROB_TOP, p = hi - icdf[s];
  return std::log2(static_cast<double>(CDF_PROB_TOP) / (p > 0 ? p : 1));
}
}  // namespace

// Bits of an MV difference component in 1/8 pels at quarter-pel precision (celv2::mvd_bits).
static int mvd_comp_bits(int d) {
  d = std::abs(d) / 2;
  int b = 1;
  for (; d; d >>= 1) b += 2;
  return b;
}

// The weight of a mode bit in BlockWriter::RdCost's units (squared quantizer steps, as TCQ weighs a
// level bit): the inter mode checks compare residual quantization costs plus this per bit.
// 0.025 / 0.05 / 0.1: -12.7 / -11.7 / -5.0% BD-rate on anime A 1080p, +1.3 / -1.7 / -2.1% on a
// close-up 1080p anime (luma and chroma costs, no NEAR bias).
constexpr double kRdBitWeight = 0.05;


double BlockWriter::luma_mode_bits(int mode_idx, int ctx) const {
  FRAME_CONTEXT* const ec_ctx = xd_->tile_ctx;
  if (mode_idx >= FIRST_MODE_COUNT) {
    const int rel = mode_idx - FIRST_MODE_COUNT;
    return symbol_bits(ec_ctx->y_mode_set_cdf, 1 + rel / SECOND_MODE_COUNT) + 4;
  }
  const int first = mode_idx < LUMA_INTRA_MODE_INDEX_COUNT - 1 ? mode_idx : LUMA_INTRA_MODE_INDEX_COUNT - 1;
  double bits = symbol_bits(ec_ctx->y_mode_set_cdf, 0) + symbol_bits(ec_ctx->y_mode_idx_cdf[ctx], first);
  if (first == LUMA_INTRA_MODE_INDEX_COUNT - 1)
    bits += symbol_bits(ec_ctx->y_mode_idx_offset_cdf[ctx], mode_idx - first);
  return bits;
}

// The 13 modes at angle delta 0, then the other deltas of the best directional mode.
int BlockWriter::search_luma_mode(int ctx) {
  MB_MODE_INFO* const mbmi = xd_->mi[0];
  int index_of[LUMA_MODE_COUNT];
  for (int i = 0; i < LUMA_MODE_COUNT; i++) index_of[mbmi->y_intra_mode_list[i]] = i;
  double best_cost = 1e300;
  int best = 0, best_dir = -1;
  double best_dir_cost = 1e300;
  auto consider = [&](int joint) {
    const double c = static_cast<double>(intra_cost_(joint)) * 256 + lambda_q8_ * luma_mode_bits(index_of[joint], ctx);
    if (c < best_cost) best_cost = c, best = joint;
    return c;
  };
  for (int j = 0; j < NON_DIRECTIONAL_MODES_COUNT; j++) consider(j);
  for (int k = 0; k < 8; k++) {
    const int joint = NON_DIRECTIONAL_MODES_COUNT + k * TOTAL_ANGLE_DELTA_COUNT + MAX_ANGLE_DELTA;
    const double c = consider(joint);
    if (c < best_dir_cost) best_dir_cost = c, best_dir = k;
  }
  for (int delta = -MAX_ANGLE_DELTA; delta <= MAX_ANGLE_DELTA; delta++)
    if (delta) consider(NON_DIRECTIONAL_MODES_COUNT + best_dir * TOTAL_ANGLE_DELTA_COUNT + MAX_ANGLE_DELTA + delta);
  return best;
}

// read_delta_qindex(): at the first block of a superblock (unless it is the whole superblock and
// skipped), the step from the qindex so far to the superblock's (CodecState::sb_qindex()).
void BlockWriter::delta_qindex() {
  AV2_COMMON* const cm = st_.cm();
  MACROBLOCKD* const xd = xd_;
  if (!cm->delta_q_info.delta_q_present_flag) return;
  const MB_MODE_INFO* const mbmi = xd->mi[0];
  if ((xd->mi_col & (cm->mib_size - 1)) || (xd->mi_row & (cm->mib_size - 1))) return;
  if (mbmi->sb_type[xd->tree_type == CHROMA_PART] == cm->sb_size && mbmi->skip_txfm[xd->tree_type == CHROMA_PART]) return;
  const int res = cm->delta_q_info.delta_q_res;
  const int diff = st_.sb_qindex(xd->mi_row, xd->mi_col) - xd->current_base_qindex;
  if (diff % res) throw std::logic_error("delta_qindex: superblock qindex off the delta_q_res grid");
  const int d = diff / res;
  w_->delta_q(st_.sb_qindex(xd->mi_row, xd->mi_col), res, d < 0 ? -d : d);
  xd->current_base_qindex += d * res;  // within 1..MAXQ: sb_qindex() is
}

// av2_read_pb_mv_precision() for a NEWMV block: the most probable precision's flag, else its step
// down. Sets the block's precision.
void BlockWriter::mv_precision(MvSubpelPrecision p) {
  AV2_COMMON* const cm = st_.cm();
  MACROBLOCKD* const xd = xd_;
  MB_MODE_INFO* const mbmi = xd->mi[0];
  const BLOCK_SIZE bsize = mbmi->sb_type[xd->tree_type == CHROMA_PART];
  set_precision_set(cm, xd, mbmi, bsize, mbmi->ref_mv_idx);
  set_most_probable_mv_precision(cm, mbmi, bsize);
  mbmi->pb_mv_precision = p;
  const int mpp = p == mbmi->most_probable_pb_mv_precision;
  w_->symbol(mpp, xd->tile_ctx->pb_mv_mpp_flag_cdf[av2_get_mpp_flag_context(cm, xd)], 2);
  if (!mpp)
    w_->symbol(av2_get_pb_mv_precision_index(mbmi),
                     xd->tile_ctx->pb_mv_precision_cdf[av2_get_pb_mv_precision_down_context(cm, xd)]
                                                      [mbmi->max_mv_precision - MV_PRECISION_HALF_PEL],
                     av2_mv_precision_sets[mbmi->mb_precision_set].num_precisions - 1);
}

// What mv_precision(p) would write, in bits (the block's mode and DRL index set).
double BlockWriter::mv_precision_bits(MvSubpelPrecision p) {
  AV2_COMMON* const cm = st_.cm();
  MACROBLOCKD* const xd = xd_;
  MB_MODE_INFO* const mbmi = xd->mi[0];
  const BLOCK_SIZE bsize = mbmi->sb_type[xd->tree_type == CHROMA_PART];
  const MvSubpelPrecision keep = mbmi->pb_mv_precision;
  set_precision_set(cm, xd, mbmi, bsize, mbmi->ref_mv_idx);
  set_most_probable_mv_precision(cm, mbmi, bsize);
  mbmi->pb_mv_precision = p;
  const int mpp = p == mbmi->most_probable_pb_mv_precision;
  double bits = symbol_bits(xd->tile_ctx->pb_mv_mpp_flag_cdf[av2_get_mpp_flag_context(cm, xd)], mpp);
  if (!mpp)
    bits += symbol_bits(xd->tile_ctx->pb_mv_precision_cdf[av2_get_pb_mv_precision_down_context(cm, xd)]
                                                          [mbmi->max_mv_precision - MV_PRECISION_HALF_PEL],
                        av2_get_pb_mv_precision_index(mbmi));
  mbmi->pb_mv_precision = keep;
  return bits;
}

void BlockWriter::ccso() {
  AV2_COMMON* const cm = st_.cm();
  MACROBLOCKD* const xd = xd_;
  if (cm->features.coded_lossless) return;
  const CommonModeInfoParams* const mi = &cm->mi_params;
  const int log2 = get_ccso_unit_size_log2_adaptive_tile(cm, cm->mib_size_log2 + MI_SIZE_LOG2, CCSO_BLK_SIZE);
  const int mask = (1 << (log2 - MI_SIZE_LOG2)) - 1;
  const int mi_row = xd->mi_row, mi_col = xd->mi_col;
  MB_MODE_INFO* const first = mi->mi_grid_base[(mi_row & ~mask) * mi->mi_stride + (mi_col & ~mask)];
  const bool unit_start = !(mi_row & mask) && !(mi_col & mask);
  for (int p = 0; p < 3; p++) {
    if (!cm->ccso_info.ccso_enable[p]) continue;
    // The plane's flag of a block (bit fields: no pointer to member).
    auto get = [p](const MB_MODE_INFO* m) { return p == 0 ? m->ccso_blk_y : p == 1 ? m->ccso_blk_u : m->ccso_blk_v; };
    auto set = [p](MB_MODE_INFO* m, int v) { (p == 0 ? void(m->ccso_blk_y = v) : p == 1 ? void(m->ccso_blk_u = v) : void(m->ccso_blk_v = v)); };
    if (unit_start) {
      constexpr int blk_idc = 1;
      w_->symbol(blk_idc, xd->tile_ctx->ccso_cdf[p][av2_get_ccso_context(cm, xd, p)], 2);
      (p == 0 ? xd->ccso_blk_y : p == 1 ? xd->ccso_blk_u : xd->ccso_blk_v) = blk_idc;
      set(first, blk_idc);
      // span_ccso()
      const BLOCK_SIZE bsize = xd->mi[0]->sb_type[PLANE_TYPE_Y];
      const int f = 1 << log2 >> MI_SIZE_LOG2, nhfb = (mi->mi_cols + f - 1) / f;
      for (int r = mi_row; r < mi_row + mi_size_high[bsize]; r += f)
        for (int c = mi_col; c < mi_col + mi_size_wide[bsize]; c += f)
          cm->cur_frame->ccso_info.sb_filter_control[p][(r / f) * nhfb + c / f] = blk_idc;
    } else if (av2_check_ccso_mbmi_inside_tile(cm, xd, xd->mi[0])) {
      set(mi->mi_grid_base[mi_row * mi->mi_stride + mi_col], get(first));
    }
  }
}

// Luma/chroma intra mode syntax shared by read_intra_frame_mode_info() (keyframes) and
// read_intra_block_mode_info() (intra blocks in inter frames).
void BlockWriter::intra_modes(const BlockDecision& d) {
  AV2_COMMON* const cm = st_.cm();
  MACROBLOCKD* const xd = xd_;
  MB_MODE_INFO* const mbmi = xd->mi[0];
  FRAME_CONTEXT* ec_ctx = xd->tile_ctx;
  const BLOCK_SIZE bsize = mbmi->sb_type[xd->tree_type == CHROMA_PART];
  if (xd->tree_type != CHROMA_PART) {
    mbmi->use_dpcm_y = 0;
    mbmi->dpcm_mode_y = 0;
    // read_intra_luma_mode()
    get_y_intra_mode_set(mbmi, xd);
    const int context = get_y_mode_idx_ctx(xd);
    mbmi->fsc_mode[xd->tree_type == CHROMA_PART] = 0;
    mbmi->mrl_index = 0;
    mbmi->multi_line_mrl = 0;
    const int joint = intra_cost_ ? search_luma_mode(context) : joint_y_mode(d.intra);
    int mode_idx = -1;
    for (int i = 0; i < LUMA_MODE_COUNT; i++)
      if (mbmi->y_intra_mode_list[i] == joint) mode_idx = i;
    if (mode_idx < 0) throw std::runtime_error("luma mode not in the mode list");
    if (mode_idx < FIRST_MODE_COUNT) {
      w_->symbol(0, ec_ctx->y_mode_set_cdf, INTRA_MODE_SETS);
      const int first = mode_idx < LUMA_INTRA_MODE_INDEX_COUNT - 1 ? mode_idx
                                                                   : LUMA_INTRA_MODE_INDEX_COUNT - 1;
      w_->symbol(first, ec_ctx->y_mode_idx_cdf[context], LUMA_INTRA_MODE_INDEX_COUNT);
      if (first == LUMA_INTRA_MODE_INDEX_COUNT - 1)
        w_->symbol(mode_idx - first, ec_ctx->y_mode_idx_offset_cdf[context],
                         LUMA_INTRA_MODE_OFFSET_COUNT);
    } else {
      const int rel = mode_idx - FIRST_MODE_COUNT;
      w_->symbol(1 + rel / SECOND_MODE_COUNT, ec_ctx->y_mode_set_cdf, INTRA_MODE_SETS);
      w_->literal(rel % SECOND_MODE_COUNT, 4);
    }
    mbmi->joint_y_mode_delta_angle = joint;
    av2_set_y_mode_and_delta_angle(joint, mbmi);
    mbmi->y_mode_idx = mode_idx;
    // read_fsc_mode(): celv2 codes no intra block with forward skip coding.
    mbmi->fsc_mode[xd->tree_type == CHROMA_PART] = 0;
    if (allow_fsc_intra(cm, bsize, mbmi))
      w_->symbol(0,
                       get_fsc_mode_cdf(xd, bsize, frame_is_intra_only(cm) || mbmi->region_type == INTRA_REGION),
                       FSC_MODES);
    // read_mrl_index() / read_multi_line_mrl(): the reference line of a directional mode (1-3 above
    // and left of the nearest), alone or averaged with line 0, by the search's cost.
    mbmi->mrl_index = 0;
    mbmi->multi_line_mrl = 0;
    if (cm->seq_params.enable_mrls && av2_is_directional_mode(mbmi->mode)) {
      avm_cdf_prob* const line_cdf = ec_ctx->mrl_index_cdf[get_mrl_index_ctx(xd->neighbors[0], xd->neighbors[1])];
      avm_cdf_prob* const multi_cdf =
          ec_ctx->multi_line_mrl_cdf[get_multi_line_mrl_index_ctx(xd->neighbors[0], xd->neighbors[1])];
      if (intra_cost_) {
        double best = 1e300;
        int best_line = 0, best_multi = 0;
        for (int line = 0; line < MRL_LINE_NUMBER; line++)
          for (int multi = 0; multi < (line ? 2 : 1); multi++) {
            mbmi->mrl_index = static_cast<uint8_t>(line), mbmi->multi_line_mrl = multi;
            const double bits = symbol_bits(line_cdf, line) + (line ? symbol_bits(multi_cdf, multi) : 0);
            const double c = static_cast<double>(intra_cost_(joint)) * 256 + lambda_q8_ * bits;
            if (c < best) best = c, best_line = line, best_multi = multi;
          }
        mbmi->mrl_index = static_cast<uint8_t>(best_line), mbmi->multi_line_mrl = best_multi;
        av2_set_y_mode_and_delta_angle(joint, mbmi);  // intra_cost_ set it too; keep the chosen one
      }
      w_->symbol(mbmi->mrl_index, line_cdf, MRL_LINE_NUMBER);
      if (mbmi->mrl_index) w_->symbol(mbmi->multi_line_mrl, multi_cdf, 2);
    }
  }

  if (xd->tree_type != LUMA_PART) {
    if (!cm->seq_params.monochrome && xd->is_chroma_ref) {
      // read_intra_uv_mode() with CfL/MHCCP not allowed.
      if (is_cfl_allowed(cm->seq_params.enable_cfl_intra, xd) || is_mhccp_allowed(cm, xd))
        throw std::runtime_error("CfL is disabled but reported allowed");
      get_uv_intra_mode_set(mbmi);
      // The luma mode's chroma counterpart (the enums share their first 13 values).
      const UV_PREDICTION_MODE uv = intra_cost_ ? static_cast<UV_PREDICTION_MODE>(mbmi->mode) : uv_mode_of(d.intra);
      int idx = -1;
      for (int i = 0; i < UV_INTRA_MODES - 1; i++)
        if (mbmi->uv_intra_mode_list[i] == uv) idx = i;
      if (idx < 0) throw std::runtime_error("chroma mode not in the mode list");
      const int context = av2_is_directional_mode(mbmi->mode) ? 1 : 0;
      const int first = idx < CHROMA_INTRA_MODE_INDEX_COUNT - 1 ? idx
                                                                : CHROMA_INTRA_MODE_INDEX_COUNT - 1;
      w_->symbol(first, ec_ctx->uv_mode_cdf[context], CHROMA_INTRA_MODE_INDEX_COUNT);
      if (first == CHROMA_INTRA_MODE_INDEX_COUNT - 1) w_->literal(idx - first, 3);
      mbmi->uv_mode = uv;
      mbmi->angle_delta[PLANE_TYPE_UV] =
          mbmi->uv_mode == mbmi->mode ? mbmi->angle_delta[PLANE_TYPE_Y] : 0;
      mbmi->use_dpcm_uv = 0;
      mbmi->dpcm_mode_uv = 0;
    } else {
      mbmi->uv_mode = UV_DC_PRED;
    }
    xd->cfl.store_y = store_cfl_required(cm, xd);
  } else {
    mbmi->uv_mode = UV_DC_PRED;
  }
  if (xd->tree_type != CHROMA_PART) mbmi->palette_mode_info.palette_size[0] = 0;
  mbmi->palette_mode_info.palette_size[1] = 0;
  if (av2_allow_palette(PLANE_TYPE_Y, cm->features.allow_screen_content_tools, bsize))
    throw std::runtime_error("palette is not supported");
  if (xd->tree_type != CHROMA_PART) mbmi->use_intra_dip = 0;  // DIP disabled
}


// decodemv.c read_inter_frame_mode_info(); intra blocks go through read_intra_block_mode_info().
void BlockWriter::inter_frame_mode_info(const BlockDecision& d) {
  AV2_COMMON* const cm = st_.cm();
  MACROBLOCKD* const xd = xd_;
  MB_MODE_INFO* const mbmi = xd->mi[0];
  FRAME_CONTEXT* ec_ctx = xd->tile_ctx;
  const BLOCK_SIZE bsize = mbmi->sb_type[PLANE_TYPE_Y];
  mbmi->mv[0].as_int = 0;
  mbmi->mv[1].as_int = 0;
  xd->submi[0]->mv[0].as_int = xd->submi[0]->mv[1].as_int = 0;
  span_submv(cm, xd->submi, xd->mi_row, xd->mi_col, bsize, 0);
  span_submv(cm, xd->submi, xd->mi_row, xd->mi_col, bsize, 1);
  set_default_max_mv_precision(mbmi, xd->sbi->sb_mv_precision);
  set_mv_precision(mbmi, mbmi->max_mv_precision);
  set_default_precision_set(cm, mbmi, bsize);
  set_most_probable_mv_precision(cm, mbmi, bsize);
  mbmi->bawp_flag[0] = 0;
  mbmi->bawp_flag[1] = 0;
  mbmi->refinemv_flag = 0;
  if (xd->tree_type != CHROMA_PART) mbmi->segment_id = 0;  // segmentation disabled
  mbmi->skip_mode = 0;
  mbmi->fsc_mode[xd->tree_type == CHROMA_PART] = 0;
  mbmi->cwp_idx = CWP_EQUAL;
  mbmi->warp_ref_idx = 0;
  mbmi->max_num_warp_candidates = 0;
  mbmi->warpmv_with_mvd_flag = 0;
  if (xd->tree_type != CHROMA_PART) {
    mbmi->use_intrabc[0] = 0;
    mbmi->use_intrabc[1] = 0;
    mbmi->use_intra_dip = 0;
    mbmi->morph_pred = 0;
  }
  // read_skip_mode(), decided once the fields above (which the prediction reads) are reset
  int skip_idx = -1;
  if (is_skip_mode_allowed(cm, xd)) {
    if (d.inter && rd_cost_) skip_idx = choose_skip_mode(d);
    w_->symbol(skip_idx >= 0, ec_ctx->skip_mode_cdfs[av2_get_skip_mode_context(xd)], 2);
  }
  mbmi->skip_mode = skip_idx >= 0;
  // read_is_inter_block()
  bool inter = d.inter;
  if (mbmi->skip_mode) {
    inter = true;
  } else if (mbmi->sb_type[PLANE_TYPE_Y] == BLOCK_4X4 || mbmi->region_type == INTRA_REGION) {
    inter = false;
  } else if (mbmi->tree_type == SHARED_PART && mbmi->region_type == MIXED_INTER_INTRA_REGION &&
             mbmi->chroma_ref_info.offset_started) {
    inter = true;
  } else {
    const int ctx = av2_get_intra_inter_context(xd);
    w_->symbol(inter, ec_ctx->intra_inter_cdf[ctx], 2);
  }
  // Where AV2 forces the choice: forced intra uses DC, forced inter codes NEARMV index 0.
  BlockDecision dd = d;
  if (!inter && d.inter) dd = BlockDecision{d.w4, d.h4, false, IntraMode::DC, 0, 0};
  const bool force_near = inter && !d.inter;
  if (inter) {
    // read_skip_txfm(): celv2 always codes the residual (all-zero blocks via the txb skip flags).
    w_->symbol(0, ec_ctx->skip_txfm_cdfs[av2_get_skip_txfm_context(xd)], 2);
  }
  mbmi->skip_txfm[xd->tree_type == CHROMA_PART] = 0;
  if (cm->seq_params.enable_ccso && xd->tree_type != CHROMA_PART) ccso();  // after read_cdef()
  if (xd->tree_type != CHROMA_PART) delta_qindex();
  mbmi->six_param_warp_model_flag = 0;
  mbmi->warp_precision_idx = 0;
  mbmi->warp_inter_intra = 0;
  mbmi->current_qindex = xd->current_base_qindex;
  const int seg_qindex = av2_get_qindex(&cm->seg, mbmi->segment_id, xd->current_base_qindex,
                                        cm->seq_params.bit_depth);
  get_qindex_with_offsets(cm, seg_qindex, mbmi->final_qindex_dc, mbmi->final_qindex_ac);
  if (inter) {
    inter_block_mode_info(dd, force_near, skip_idx);
    // av2_read_mode_info()
    if (cm->seq_params.enable_refmvbank) av2_update_ref_mv_bank(cm, xd, 1, mbmi);
    av2_update_warp_param_bank(cm, xd, 0, mbmi);
    return;
  }
  // read_intra_block_mode_info()
  mbmi->ref_frame[0] = INTRA_FRAME;
  mbmi->ref_frame[1] = NONE_FRAME;
  set_default_max_mv_precision(mbmi, xd->sbi->sb_mv_precision);
  set_mv_precision(mbmi, mbmi->max_mv_precision);
  set_default_precision_set(cm, mbmi, bsize);
  set_most_probable_mv_precision(cm, mbmi, bsize);
  mbmi->bawp_flag[0] = 0;
  mbmi->bawp_flag[1] = 0;
  mbmi->refinemv_flag = 0;
  mbmi->motion_mode = SIMPLE_TRANSLATION;
  intra_modes(dd);
  if (cm->seq_params.enable_refmvbank) decide_rmb_unit_update_count(cm, xd, mbmi);  // av2_read_mode_info()
}

void BlockWriter::pre_txb(int plane, TX_SIZE tx_size) {
  MACROBLOCKD* const xd = xd_;
  MB_MODE_INFO* const mbmi = xd->mi[0];
  if (is_inter_block(mbmi, xd->tree_type)) {
    const int txb_idx = get_tx_partition_idx(mbmi, plane);
    mbmi->is_wide_angle[plane > 0][txb_idx] = 0;
    mbmi->mapped_intra_mode[plane > 0][txb_idx] = DC_PRED;
  } else {
    // The decoder passes uv_mode as a PREDICTION_MODE value unchanged.
    const PREDICTION_MODE mode =
        static_cast<PREDICTION_MODE>(plane == PLANE_TYPE_Y ? mbmi->mode : mbmi->uv_mode);
    const int angle_delta = mbmi->angle_delta[plane != AVM_PLANE_Y] * ANGLE_STEP;
    wide_angle_mapping(mbmi, angle_delta, tx_size, mode, plane);
  }
}

// decodetxb.c decode_eob()
void BlockWriter::eob_symbols(int plane, TX_SIZE tx_size, int eob) {
  MACROBLOCKD* const xd = xd_;
  FRAME_CONTEXT* const ec_ctx = xd->tile_ctx;
  const int is_inter = is_inter_block(xd->mi[0], xd->tree_type);
  const int pl_ctx = get_eob_plane_ctx(plane, is_inter);
  int eob_pt = 1;
  while (eob_pt + 1 < 12 && av2_eob_group_start[eob_pt + 1] <= eob) eob_pt++;
  const int eob_extra = eob - av2_eob_group_start[eob_pt];
  const int sym = eob_pt - 1;
  const int ms = txsize_log2_minus4[tx_size];
  switch (ms) {
    case 0: w_->symbol(sym, ec_ctx->eob_flag_cdf16[pl_ctx], EOB_MAX_SYMS - 6); break;
    case 1: w_->symbol(sym, ec_ctx->eob_flag_cdf32[pl_ctx], EOB_MAX_SYMS - 5); break;
    case 2: w_->symbol(sym, ec_ctx->eob_flag_cdf64[pl_ctx], EOB_MAX_SYMS - 4); break;
    case 3: w_->symbol(sym, ec_ctx->eob_flag_cdf128[pl_ctx], EOB_MAX_SYMS - 3); break;
    default: {
      avm_cdf_prob* cdf = ms == 4   ? ec_ctx->eob_flag_cdf256[pl_ctx]
                          : ms == 5 ? ec_ctx->eob_flag_cdf512[pl_ctx]
                                    : ec_ctx->eob_flag_cdf1024[pl_ctx];
      const int low = AVMMIN(sym, EOB_PT_INDEX_COUNT - 1);
      w_->symbol(low, cdf, EOB_MAX_SYMS - 3);
      if (low == EOB_PT_INDEX_COUNT - 1) w_->literal(sym - low, ms == 4 ? 1 : 2);
      break;
    }
  }
  const int bits = av2_eob_offset_bits[eob_pt];
  if (bits > 0) {
    w_->symbol((eob_extra >> (bits - 1)) & 1, ec_ctx->eob_extra_cdf, 2);
    w_->literal(eob_extra & ((1 << (bits - 1)) - 1), bits - 1);
  }
}

namespace {

void write_low_range(SymbolLog* w, avm_cdf_prob* cdf, int br) {  // br = level above the base
  for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
    const int k = AVMMIN(br - idx, BR_CDF_SIZE - 1);
    w->symbol(k, cdf, BR_CDF_SIZE);
    if (k < BR_CDF_SIZE - 1) break;
  }
}

void write_exp_golomb(SymbolLog* w, int level, int k) {
  const int x = level + (1 << k);
  const int length = get_msb(x) + 1;
  w->literal(0, length - 1 - k);
  w->literal(x, length);
}

// Inverse of decodetxb.c read_adaptive_hr()
void write_adaptive_hr(SymbolLog* w, int level, int ctx) {
  const int m = get_adaptive_param(ctx);
  const int k = m + 1, cmax = AVMMIN(m + 4, 6);
  const int q = level >> m;
  if (q >= cmax) {
    w->literal(0, cmax);
    write_exp_golomb(w, level - (cmax << m), k);
  } else {
    w->literal(0, q);
    w->literal(1, 1);
    w->literal(level & ((1 << m) - 1), m);
  }
}

}  // namespace

// decodetxb.c av2_read_coeffs_txb_skip(): levels forward from bob (the first nonzero one, in scan
// order) to the end of the block, then signs and high ranges in the same order; no TCQ.
int BlockWriter::skip_levels_and_signs(int plane, TX_SIZE tx_size, TX_TYPE tx_type, const tran_low_t* qcoeff,
                                       int bob) {
  (void)plane;
  FRAME_CONTEXT* const ec_ctx = xd_->tile_ctx;
  const int bwl = get_txb_bwl(tx_size);
  const int width = get_txb_wide(tx_size);
  const int height = get_txb_high(tx_size);
  const int max_eob = av2_get_max_eob(tx_size);
  const int16_t* const scan = get_scan(tx_size, tx_type)->scan;
  const int size_ctx = AVMMIN(get_txsize_entropy_ctx(tx_size), TX_16X16);
  uint8_t levels_buf[TX_PAD_2D];
  int8_t signs_buf[TX_PAD_2D];
  std::memset(levels_buf, 0, sizeof(levels_buf));
  std::memset(signs_buf, 0, sizeof(signs_buf));
  uint8_t* const levels = set_levels(levels_buf, width);
  int8_t* const signs = set_signs(signs_buf, width);
  auto low_range = [&](int pos, int a) {
    if (a > NUM_BASE_LEVELS)
      write_low_range(w_, ec_ctx->coeff_br_cdf_idtx[size_ctx][get_br_ctx_skip(levels, pos, bwl)],
                      a - 1 - NUM_BASE_LEVELS);
    levels[get_padded_idx_left(pos, bwl)] = static_cast<uint8_t>(AVMMIN(a, MAX_BASE_BR_RANGE));
  };
  if (max_eob > 1) {
    const int pos = scan[bob];
    const int a = std::abs(qcoeff[pos]);
    w_->symbol(AVMMIN(a, NUM_BASE_LEVELS + 1) - 1,
                     ec_ctx->coeff_base_bob_cdf[size_ctx][get_lower_levels_ctx_bob(bwl, height, bob)], 3);
    low_range(pos, a);
    for (int c = bob + 1; c < max_eob; c++) {  // read_coeffs_forward_2d()
      const int p = scan[c];
      const int b = std::abs(qcoeff[p]);
      w_->symbol(AVMMIN(b, NUM_BASE_LEVELS + 1),
                       ec_ctx->coeff_base_cdf_idtx[size_ctx][get_upper_levels_ctx_2d(levels, p, bwl)], 4);
      low_range(p, b);
    }
  }
  int cul_level = 0, dc_val = 0, hr_level_avg = 0;
  for (int c = bob; c < max_eob; c++) {
    const int pos = scan[c];
    const int a = std::abs(qcoeff[pos]);
    if (!a) continue;
    const int sign = qcoeff[pos] < 0;
    w_->symbol(sign, ec_ctx->idtx_sign_cdf[size_ctx][get_sign_ctx_skip(signs, levels, pos, bwl)], 2);
    signs[get_padded_idx_left(pos, bwl)] = sign ? -1 : 1;
    if (a >= MAX_BASE_BR_RANGE) {  // read_high_range() without TCQ
      const int hr = a - MAX_BASE_BR_RANGE;
      write_adaptive_hr(w_, hr, hr_level_avg);
      hr_level_avg = (hr_level_avg + hr) >> 1;
    }
    if (c == 0) dc_val = sign ? -a : a;
    cul_level += a;
  }
  cul_level = AVMMIN(COEFF_CONTEXT_MASK, cul_level);
  set_dc_sign(&cul_level, dc_val);
  return cul_level;
}

// decodetxb.c av2_read_coeffs_txb() with IST and parity hiding off.
int BlockWriter::levels_and_signs(int plane, TX_SIZE tx_size, const TXB_CTX& txb_ctx,
                                  const tran_low_t* qcoeff, int eob, int blk_row, int blk_col) {
  AV2_COMMON* const cm = st_.cm();
  MACROBLOCKD* const xd = xd_;
  FRAME_CONTEXT* const ec_ctx = xd->tile_ctx;
  MB_MODE_INFO* const mbmi = xd->mi[0];
  const TX_SIZE txs_ctx = get_txsize_entropy_ctx(tx_size);
  const PLANE_TYPE plane_type = get_plane_type(plane);
  const int bwl = get_txb_bwl(tx_size);
  const int width = get_txb_wide(tx_size);
  const int height = get_txb_high(tx_size);
  uint8_t levels_buf[TX_PAD_2D];
  std::memset(levels_buf, 0, sizeof(levels_buf));
  uint8_t* const levels = set_levels(levels_buf, width);
  const TX_TYPE tx_type = av2_get_tx_type(xd, plane_type, blk_row, blk_col, tx_size,
                                          is_reduced_tx_set_used(cm, plane_type));
  const TX_CLASS tx_class = tx_type_to_class[get_primary_tx_type(tx_type)];
  const int16_t* const scan = get_scan(tx_size, tx_type)->scan;
  if (plane == AVM_PLANE_Y &&
      (is_inter_block(mbmi, xd->tree_type) ? (eob > 3 && cm->seq_params.enable_inter_ist)
                                           : (eob != 1 && cm->seq_params.enable_ist)))
    throw std::runtime_error("IST is disabled but signalled");
  const int tcq_mode =
      tcq_enable(cm->features.tcq_mode, xd->lossless[mbmi->segment_id], plane, tx_class);
  int state = tcq_init_state(tcq_mode);
  auto full_level = [&](int c) { return std::abs(static_cast<int>(qcoeff[scan[c]])); };
  // The level the base and range symbols code: with TCQ one from max_br - 1 up codes its parity
  // (max_br - 1 or max_br) and the rest as a high range in steps of 2 (read_high_range()).
  auto abs_level = [&](int c) {
    const int a = full_level(c);
    if (!tcq_mode) return a;
    const int pos = scan[c], row = pos >> bwl;
    const int max_br = get_lf_limits(row, pos - (row << bwl), tx_class, plane) ? LF_MAX_BASE_BR_RANGE : MAX_BASE_BR_RANGE;
    return a >= max_br - 1 ? max_br - 1 + ((a - max_br + 1) & 1) : a;
  };

  {  // last coefficient (scan index eob - 1)
    const int c = eob - 1;
    const int pos = scan[c];
    const int coeff_ctx = get_lower_levels_ctx_eob(bwl, height, c);
    const int a = abs_level(c);
    const int row = pos >> bwl;
    const int col = pos - (row << bwl);
    const int limits = get_lf_limits(row, col, tx_class, plane);
    int level;
    if (plane > 0) {
      if (limits) {
        level = AVMMIN(a, LF_BASE_SYMBOLS - 1);
        w_->symbol(level - 1, ec_ctx->coeff_base_lf_eob_uv_cdf[coeff_ctx],
                         LF_BASE_SYMBOLS - 1);
      } else {
        level = AVMMIN(a, 3);
        w_->symbol(level - 1, ec_ctx->coeff_base_eob_uv_cdf[coeff_ctx], 3);
        if (level > NUM_BASE_LEVELS) {
          write_low_range(w_, ec_ctx->coeff_br_uv_cdf[0], a - 1 - NUM_BASE_LEVELS);
          level = AVMMIN(a, MAX_BASE_BR_RANGE);
        }
      }
    } else {
      if (limits) {
        level = AVMMIN(a, LF_BASE_SYMBOLS - 1);
        w_->symbol(level - 1, ec_ctx->coeff_base_lf_eob_cdf[txs_ctx][coeff_ctx],
                         LF_BASE_SYMBOLS - 1);
        if (level > LF_NUM_BASE_LEVELS) {
          const int br_ctx = get_br_ctx_lf_eob(pos, tx_class);
          write_low_range(w_, ec_ctx->coeff_br_lf_cdf[br_ctx], a - 1 - LF_NUM_BASE_LEVELS);
          level = AVMMIN(a, LF_MAX_BASE_BR_RANGE);
        }
      } else {
        level = AVMMIN(a, 3);
        w_->symbol(level - 1, ec_ctx->coeff_base_eob_cdf[txs_ctx][coeff_ctx], 3);
        if (level > NUM_BASE_LEVELS) {
          write_low_range(w_, ec_ctx->coeff_br_cdf[0], a - 1 - NUM_BASE_LEVELS);
          level = AVMMIN(a, MAX_BASE_BR_RANGE);
        }
      }
    }
    levels[get_padded_idx(pos, bwl)] = static_cast<uint8_t>(level);
    state = tcq_next_state(state, level);
  }

  // Remaining coefficients in reverse scan order: read_coeffs_reverse_2d() for 1..eob-2 when the
  // class is 2D, then read_coeffs_reverse() for the rest (always for DC).
  auto code_one = [&](int c, bool use_2d) {
    const int pos = scan[c];
    const int a = abs_level(c);
    const int row = pos >> bwl;
    const int col = pos - (row << bwl);
    const int limits = get_lf_limits(row, col, use_2d ? TX_CLASS_2D : tx_class, plane);
    const int q_i = tcq_quant(state);
    int level;
    if (plane > 0) {
      if (limits) {
        const int ctx = use_2d ? get_lower_levels_ctx_lf_2d_chroma(levels, pos, bwl, plane)
                               : get_lower_levels_lf_ctx_chroma(levels, pos, bwl, tx_class, plane);
        level = AVMMIN(a, LF_BASE_SYMBOLS - 1);
        w_->symbol(level, ec_ctx->coeff_base_lf_uv_cdf[ctx], LF_BASE_SYMBOLS);
      } else {
        const int ctx = use_2d ? get_lower_levels_ctx_2d_chroma(levels, pos, bwl, plane)
                               : get_lower_levels_ctx_chroma(levels, pos, bwl, tx_class, plane);
        level = AVMMIN(a, 3);
        w_->symbol(level, ec_ctx->coeff_base_uv_cdf[ctx], 4);
        if (level > NUM_BASE_LEVELS) {
          const int br_ctx = use_2d ? get_br_ctx_2d_chroma(levels, pos, bwl)
                                    : get_br_ctx_chroma(levels, pos, bwl, tx_class);
          write_low_range(w_, ec_ctx->coeff_br_uv_cdf[br_ctx], a - 1 - NUM_BASE_LEVELS);
          level = AVMMIN(a, MAX_BASE_BR_RANGE);
        }
      }
    } else {
      if (limits) {
        const int ctx = use_2d ? get_lower_levels_ctx_lf_2d(levels, pos, bwl)
                               : get_lower_levels_lf_ctx(levels, pos, bwl, tx_class);
        level = AVMMIN(a, LF_BASE_SYMBOLS - 1);
        w_->symbol(level, ec_ctx->coeff_base_lf_cdf[txs_ctx][ctx][q_i],
                         LF_BASE_SYMBOLS);
        if (level > LF_NUM_BASE_LEVELS) {
          const int br_ctx = use_2d ? get_br_lf_ctx_2d(levels, pos, bwl)
                                    : get_br_lf_ctx(levels, pos, bwl, tx_class);
          write_low_range(w_, ec_ctx->coeff_br_lf_cdf[br_ctx], a - 1 - LF_NUM_BASE_LEVELS);
          level = AVMMIN(a, LF_MAX_BASE_BR_RANGE);
        }
      } else {
        const int ctx = use_2d ? get_lower_levels_ctx_2d(levels, pos, bwl, plane)
                               : get_lower_levels_ctx(levels, pos, bwl, tx_class, plane);
        level = AVMMIN(a, 3);
        w_->symbol(level, ec_ctx->coeff_base_cdf[txs_ctx][ctx][q_i], 4);
        if (level > NUM_BASE_LEVELS) {
          const int br_ctx = use_2d ? get_br_ctx_2d(levels, pos, bwl)
                                    : get_br_ctx(levels, pos, bwl, tx_class);
          write_low_range(w_, ec_ctx->coeff_br_cdf[br_ctx], a - 1 - NUM_BASE_LEVELS);
          level = AVMMIN(a, MAX_BASE_BR_RANGE);
        }
      }
    }
    levels[get_padded_idx(pos, bwl)] = static_cast<uint8_t>(level);
    state = tcq_next_state(state, level);
  };
  if (eob > 1) {
    for (int c = eob - 2; c >= 1; --c) code_one(c, tx_class == TX_CLASS_2D);
    code_one(0, false);
  }

  // Signs and high ranges.
  int cul_level = 0, dc_val = 0, hr_level_avg = 0;
  for (int c = eob - 1; c >= 0; --c) {
    const int pos = scan[c];
    int level = levels[get_padded_idx(pos, bwl)];
    if (plane == AVM_PLANE_U) xd->tmp_sign[pos] = 0;
    if (!level) continue;
    const int a = full_level(c);
    const int sign = qcoeff[pos] < 0;
    const int row = pos >> bwl;
    const int col = pos - (row << bwl);
    const bool dc_2dtx = c == 0;
    const bool dc_hor = col == 0 && tx_class == TX_CLASS_HORIZ;
    const bool dc_ver = row == 0 && tx_class == TX_CLASS_VERT;
    if (dc_2dtx || dc_hor || dc_ver) {
      const int dc_sign_ctx = dc_2dtx ? txb_ctx.dc_sign_ctx : 0;
      if (plane == AVM_PLANE_Y)
        w_->symbol(sign, ec_ctx->dc_sign_cdf[plane_type][0][dc_sign_ctx], 2);
      else
        w_->literal(sign, 1);
      if (plane == AVM_PLANE_U) xd->tmp_sign[pos] = sign ? 2 : 1;
    } else {
      w_->bit(sign);
    }
    // read_high_range()
    const int lf = get_lf_limits(row, col, tx_class, plane);
    const int max_br =
        lf ? (plane == 0 ? LF_MAX_BASE_BR_RANGE : LF_NUM_BASE_LEVELS + 1) : MAX_BASE_BR_RANGE;
    if (tcq_mode ? level >= max_br - 1 : level >= max_br) {
      const int hr = tcq_mode ? (a - level) >> 1 : a - level;
      write_adaptive_hr(w_, hr, hr_level_avg);
      level += tcq_mode ? hr << 1 : hr;
      hr_level_avg = (hr_level_avg + hr) >> 1;
    }
    if (c == 0) dc_val = sign ? -level : level;
    cul_level += level;
  }
  cul_level = AVMMIN(COEFF_CONTEXT_MASK, cul_level);
  set_dc_sign(&cul_level, dc_val);
  return cul_level;
}

int BlockWriter::coeffs(int plane, int blk_row, int blk_col, TX_SIZE tx_size,
                        const tran_low_t* qcoeff, int eob, TX_TYPE tx_type) {
  AV2_COMMON* const cm = st_.cm();
  MACROBLOCKD* const xd = xd_;
  MB_MODE_INFO* const mbmi = xd->mi[0];
  FRAME_CONTEXT* const ec_ctx = xd->tile_ctx;
  struct macroblockd_plane* const pd = &xd->plane[plane];
  const BLOCK_SIZE plane_bsize =
      get_mb_plane_block_size(xd, mbmi, plane, pd->subsampling_x, pd->subsampling_y);
  const PLANE_TYPE plane_type = get_plane_type(plane);
  const int is_inter = is_inter_block(mbmi, xd->tree_type);
  // av2_read_coeffs_txb_facade()
  TXB_CTX txb_ctx;
  get_txb_ctx(plane_bsize, tx_size, plane, pd->above_entropy_context + blk_col,
              pd->left_entropy_context + blk_row, &txb_ctx,
              mbmi->fsc_mode[xd->tree_type == CHROMA_PART] && cm->seq_params.enable_fsc);
  // av2_read_sig_txtype()
  const int all_zero = eob == 0;
  const TX_SIZE txs_ctx = get_txsize_entropy_ctx(tx_size);
  int txb_skip_ctx = txb_ctx.txb_skip_ctx;
  if (plane == AVM_PLANE_Y || plane == AVM_PLANE_U) {
    const int pred_mode_ctx = (is_inter || mbmi->fsc_mode[xd->tree_type == CHROMA_PART]) ? 1 : 0;
    w_->symbol(all_zero, ec_ctx->txb_skip_cdf[pred_mode_ctx][txs_ctx][txb_skip_ctx], 2);
  } else {
    txb_skip_ctx += (xd->eob_u_flag ? V_TXB_SKIP_CONTEXT_OFFSET : 0);
    w_->symbol(all_zero, ec_ctx->v_txb_skip_cdf[txb_skip_ctx], 2);
  }
  if (plane == AVM_PLANE_U) xd->eob_u_flag = all_zero ? 0 : 1;
  pre_txb(plane, tx_size);
  int cul_level = 0;
  if (all_zero) {
    if (plane == 0) xd->tx_type_map[blk_row * xd->tx_type_map_stride + blk_col] = DCT_DCT;
    av2_update_txk_skip_array(cm, xd->mi_row, xd->mi_col, xd->tree_type, &mbmi->chroma_ref_info,
                              plane, blk_row, blk_col, tx_size);
  } else {
    // Inter IDTX with forward skip coding: the eob symbols code the span from the first nonzero
    // level (in scan order) to the end of the block.
    const bool fsc = plane == AVM_PLANE_Y && use_inter_fsc(cm, PLANE_TYPE_Y, tx_type, is_inter);
    const int16_t* const scan = get_scan(tx_size, tx_type)->scan;
    int bob = 0;
    while (fsc && qcoeff[scan[bob]] == 0) bob++;
    if (fsc) eob = av2_get_max_eob(tx_size) - bob;
    eob_symbols(plane, tx_size, eob);
    // av2_read_tx_type(): luma only; intra with reduced_tx_set_used == 2 implies DCT_DCT.
    if (plane == PLANE_TYPE_Y) {
      xd->tx_type_map[blk_row * xd->tx_type_map_stride + blk_col] = is_inter ? tx_type : DCT_DCT;
      const bool dc_skip = !is_inter && eob == 1;
      if (is_inter)
        inter_tx_type(tx_size, eob, tx_type);
      else if (!dc_skip && cm->features.reduced_tx_set_used != 2 &&
               get_ext_tx_types(tx_size, 0, cm->features.reduced_tx_set_used) > 1)
        intra_tx_type(tx_size, tx_type);
      else if (tx_type != DCT_DCT)
        throw std::runtime_error("an intra transform type that cannot be coded");
      xd->tx_type_map[blk_row * xd->tx_type_map_stride + blk_col] = tx_type;
    }
    if (plane == AVM_PLANE_U && is_cctx_allowed(cm, xd))
      throw std::runtime_error("CCTX is disabled but reported allowed");
    cul_level = fsc ? skip_levels_and_signs(plane, tx_size, tx_type, qcoeff, bob)
                    : levels_and_signs(plane, tx_size, txb_ctx, qcoeff, eob, blk_row, blk_col);
  }
  av2_set_entropy_contexts(xd, pd, plane, plane_bsize, tx_size, cul_level, blk_col, blk_row);
  if (is_inter && plane == 0) {
    const TX_TYPE t = av2_get_tx_type(xd, plane_type, blk_row, blk_col, tx_size,
                                      is_reduced_tx_set_used(cm, plane_type));
    update_txk_array(xd, blk_row, blk_col, tx_size, t);
  }
  return cul_level;
}
namespace {

// Inverse of decodemv.c read_tu_quasi_uniform().
void write_tu_quasi_uniform(SymbolLog* w, nmv_context* ctx, int value, int max_coded_value,
                            int max_trunc_unary_value) {
  const int max_idx_bits = AVMMIN(max_coded_value, max_trunc_unary_value);
  for (int bit_idx = 0; bit_idx < max_idx_bits; ++bit_idx) {
    const int ci = bit_idx < NUM_CTX_COL_MV_GTX ? bit_idx : NUM_CTX_COL_MV_GTX - 1;
    const int bit = value > bit_idx;
    w->symbol(bit, ctx->col_mv_greater_flags_cdf[ci], 2);
    if (!bit) break;
  }
  if (max_coded_value > max_trunc_unary_value && value >= max_trunc_unary_value)
    w->quniform(max_coded_value - max_trunc_unary_value + 1,
                                 value - max_trunc_unary_value);
}

// Inverse of decodemv.c read_mv() (no adaptive MVD): magnitudes of mv_diff (1/8 pel, multiples
// of the precision step). Signs are written by the caller, as in assign_mv().
void write_mv_magnitude(SymbolLog* w, MV mv_diff, nmv_context* ctx, MvSubpelPrecision precision) {
  const int start_lsb = MV_PRECISION_ONE_EIGHTH_PEL - precision;
  const int row = std::abs(mv_diff.row) >> start_lsb, col = std::abs(mv_diff.col) >> start_lsb;
  const int num_mv_class = get_default_num_shell_class(precision);
  int c0, c1;
  split_num_shell_class(num_mv_class, &c0, &c1);
  const int shell_index = row + col;
  int shell_class = 0;
  while (shell_index >= (2 << shell_class)) shell_class++;
  if (shell_class >= num_mv_class) throw std::runtime_error("motion vector difference too large");
  const int shell_set = shell_class >= c0;
  w->symbol(shell_set, ctx->joint_shell_set_cdf, 2);
  if (shell_set) {
    if (precision == MV_PRECISION_ONE_EIGHTH_PEL) throw std::runtime_error("1/8-pel MVD unsupported");
    w->symbol(shell_class - c0, ctx->joint_shell_class_cdf_1[precision], c1);
  } else {
    w->symbol(shell_class, ctx->joint_shell_class_cdf_0[precision], c0);
  }
  const int base = shell_class == 0 ? 0 : (1 << shell_class);
  const int offset = shell_index - base;
  if (shell_class < 2) {
    w->symbol(offset, ctx->shell_offset_low_class_cdf[shell_class], 2);
  } else if (shell_class == 2) {  // read_truncated_unary_mvd(), max 3
    for (int bit_idx = 0; bit_idx < 3; ++bit_idx) {
      const int bit = offset > bit_idx;
      if (bit_idx)
        w->literal(bit, 1);
      else
        w->symbol(bit, ctx->shell_offset_class2_cdf, 2);
      if (!bit) break;
    }
  } else {
    for (int i = 0; i < shell_class; ++i)
      w->symbol((offset >> i) & 1, ctx->shell_offset_other_class_cdf[0][i], 2);
  }
  if (shell_index > 0) {
    const int max_pair = shell_index >> 1;
    const int pair = AVMMIN(col, row);
    if (max_pair > 0) write_tu_quasi_uniform(w, ctx, pair, max_pair, MAX_COL_TRUNCATED_UNARY_VAL);
    const bool skip_col_bit = pair == max_pair && shell_index % 2 == 0;
    if (!skip_col_bit) {
      const int ci = shell_class < NUM_CTX_COL_MV_INDEX ? shell_class : NUM_CTX_COL_MV_INDEX - 1;
      w->symbol(col != pair, ctx->col_mv_index_cdf[ci], 2);
    }
  }
}

// decodemv.c mv_clamp_to_integer()
void mv_clamp_to_integer(MV* mv) {
  if (mv->row < MV_LOW + 1 || mv->row > MV_UPP - 1)
    mv->row = static_cast<MV_COMP_DATA_TYPE>(clamp(mv->row, MV_LOW + 8, MV_UPP - 8));
  if (mv->col < MV_LOW + 1 || mv->col > MV_UPP - 1)
    mv->col = static_cast<MV_COMP_DATA_TYPE>(clamp(mv->col, MV_LOW + 8, MV_UPP - 8));
}

}  // namespace

bool BlockWriter::compound_codable(int r0, int r1) const {
  const RefFramesInfo& info = st_.cm()->ref_frames_info;
  return info.num_same_ref_compound == 0 && r0 < r1 && r1 < info.num_total_refs &&
         r0 <= std::min(RANKED_REF0_TO_PRUNE - 1, info.num_total_refs - 2);
}

// decodemv.c read_compound_ref() run with the wanted answers.
void BlockWriter::compound_ref(int r0, int r1) {
  const RefFramesInfo* const info = &st_.cm()->ref_frames_info;
  const int n_refs = info->num_total_refs;
  MV_REFERENCE_FRAME got[2] = {NONE_FRAME, NONE_FRAME};
  int n_bits = 0;
  for (int i = 0; i < n_refs + n_bits - 2 && n_bits < 2; i++) {
    const int bit_type = n_bits == 0 ? -1 : av2_get_compound_ref_bit_type(info, got[0], i);
    const bool implicit = n_bits == 0 && (i >= RANKED_REF0_TO_PRUNE - 1 || i >= n_refs - 2);
    const int bit = i == (n_bits == 0 ? r0 : r1);
    if (implicit) {
      if (!bit) throw std::runtime_error("compound reference pair not codable");
    } else {
      w_->symbol(bit, av2_get_pred_cdf_compound_ref(xd_, i, n_bits, bit_type, n_refs), 2);
    }
    if (bit) got[n_bits++] = static_cast<MV_REFERENCE_FRAME>(i);
  }
  if (n_bits < 2) got[1] = static_cast<MV_REFERENCE_FRAME>(n_refs - 1);
  if (n_bits < 1) got[0] = static_cast<MV_REFERENCE_FRAME>(n_refs - 2);
  if (got[0] != r0 || got[1] != r1) throw std::runtime_error("compound reference pair not codable");
}

void BlockWriter::compound_mvs(const MV want[2], const int_mv ref_mv[2]) {
  MB_MODE_INFO* const mbmi = xd_->mi[0];
  const MvSubpelPrecision precision = mbmi->pb_mv_precision;
  if (precision < MV_PRECISION_HALF_PEL) throw std::runtime_error("integer MV precision unsupported");
  if (is_mvd_sign_derive_allowed(st_.cm(), xd_, mbmi)) throw std::runtime_error("MVD sign derivation unsupported");
  const int step = 1 << (MV_PRECISION_ONE_EIGHTH_PEL - precision);
  MV diff[2];
  for (int r = 0; r < 2; r++) {
    diff[r].row = static_cast<int16_t>((want[r].row - ref_mv[r].as_mv.row) / step * step);
    diff[r].col = static_cast<int16_t>((want[r].col - ref_mv[r].as_mv.col) / step * step);
    write_mv_magnitude(w_, diff[r], &xd_->tile_ctx->nmvc, precision);
  }
  for (int r = 0; r < 2; r++) {
    if (diff[r].row) w_->literal(diff[r].row < 0, 1);
    if (diff[r].col) w_->literal(diff[r].col < 0, 1);
    mbmi->mv[r].as_mv.row = static_cast<int16_t>(ref_mv[r].as_mv.row + diff[r].row);
    mbmi->mv[r].as_mv.col = static_cast<int16_t>(ref_mv[r].as_mv.col + diff[r].col);
    mv_clamp_to_integer(&mbmi->mv[r].as_mv);
  }
}

// decodemv.c read_inter_block_mode_info() for one reference, translation only, NEARMV/NEWMV.
// With force_near (AV2 forces this block to be inter) NEARMV with DRL index 0 is coded.
// av2_build_one_bawp_inter_predictor()'s condition, which the decoder rejects a stream for failing:
// in each plane, the reference's BAWP_REF_LINES rows above and columns left of the block at its MV
// lie inside the frame. A pixel of slack on every side: a NEWMV block's MV is its difference rounded
// after this is decided.
static bool bawp_template_inside(const MACROBLOCKD* xd, BLOCK_SIZE bsize, MV mv) {
  constexpr int kSlack = 1;
  for (int p = 0; p < (xd->is_chroma_ref ? 3 : 1); p++) {
    const struct macroblockd_plane* const pd = &xd->plane[p];
    const int x = (xd->mi_col * MI_SIZE) >> pd->subsampling_x, y = (xd->mi_row * MI_SIZE) >> pd->subsampling_y;
    const int dx = GET_MV_RAWPEL(mv.col) >> pd->subsampling_x, dy = GET_MV_RAWPEL(mv.row) >> pd->subsampling_y;
    const int w = std::min(block_size_wide[bsize] >> pd->subsampling_x, pd->dst.width - x);
    const int h = std::min(block_size_high[bsize] >> pd->subsampling_y, pd->dst.height - y);
    if (w <= 0 || h <= 0 || x + dx - BAWP_REF_LINES - kSlack < 0 || y + dy - BAWP_REF_LINES - kSlack < 0 ||
        x + w + dx + kSlack > pd->dst.width || y + h + dy + kSlack > pd->dst.height)
      return false;
  }
  return true;
}

void BlockWriter::inter_block_mode_info(const BlockDecision& d, bool force_near, int skip_idx) {
  AV2_COMMON* const cm = st_.cm();
  MACROBLOCKD* const xd = xd_;
  MB_MODE_INFO* const mbmi = xd->mi[0];
  FRAME_CONTEXT* const ec_ctx = xd->tile_ctx;
  const BLOCK_SIZE bsize = mbmi->sb_type[PLANE_TYPE_Y];
  SB_INFO* sbi = xd->sbi;
  int_mv ref_mvs[MODE_CTX_REF_FRAMES][MAX_MV_REF_CANDIDATES] = {};
  int16_t inter_mode_ctx[MODE_CTX_REF_FRAMES];
  int pts0[SAMPLES_ARRAY_SIZE], pts0_inref[SAMPLES_ARRAY_SIZE];

  mbmi->uv_mode = UV_DC_PRED;
  mbmi->palette_mode_info.palette_size[0] = 0;
  mbmi->palette_mode_info.palette_size[1] = 0;
  mbmi->fsc_mode[PLANE_TYPE_Y] = 0;
  mbmi->fsc_mode[PLANE_TYPE_UV] = 0;
  mbmi->use_intrabc[0] = 0;
  mbmi->use_intrabc[1] = 0;
  mbmi->morph_pred = 0;
  set_default_max_mv_precision(mbmi, sbi->sb_mv_precision);
  set_mv_precision(mbmi, mbmi->max_mv_precision);
  set_default_precision_set(cm, mbmi, bsize);
  set_most_probable_mv_precision(cm, mbmi, bsize);
  mbmi->bawp_flag[0] = 0;
  mbmi->bawp_flag[1] = 0;
  mbmi->refinemv_flag = 0;
  mbmi->num_proj_ref[0] = 0;
  mbmi->num_proj_ref[1] = 0;
  mbmi->wm_params[0].invalid = 1;
  mbmi->wm_params[1].invalid = 1;
  // The MV checks below predict before read_mb_interp_filter() sets this; the frame filter is
  // what a fractional MV is predicted with (left as it was, the field held what this block's
  // position had in the previous frame coded by this state).
  mbmi->interp_fltr = cm->features.interp_filter;
  av2_collect_neighbors_ref_counts(xd);

  if (skip_idx >= 0) {
    skip_mode_info(skip_idx);
    inter_block_tail();
    return;
  }
  // read_ref_frames() -> read_single_ref(): no TIP, single reference mode. A reference the frame
  // does not have yet (just after a keyframe) becomes the farthest one it has.
  if (is_tip_allowed(cm, xd)) throw std::runtime_error("TIP is disabled but reported allowed");
  // read_block_reference_mode(): a compound block averages two references (NEW_NEWMV only); a
  // pair the syntax cannot code falls back to single prediction from d.ref.
  const int n_refs = cm->ref_frames_info.num_total_refs;
  const int ref = force_near ? 0 : std::min<int>(d.ref, n_refs - 1);
  const int r0 = std::min<int>(d.ref, d.ref2), r1 = std::max<int>(d.ref, d.ref2);
  const bool comp_signalled =
      is_comp_ref_allowed(bsize) && cm->current_frame.reference_mode == REFERENCE_MODE_SELECT;
  const bool compound = comp_signalled && !force_near && d.inter && d.ref2 != BlockDecision::kSingleRef &&
                        compound_codable(r0, r1);
  if (comp_signalled)
    w_->symbol(compound, xd->tile_ctx->comp_inter_cdf[av2_get_reference_mode_context(cm, xd)], 2);
  if (compound) {
    compound_ref(r0, r1);
    mbmi->ref_frame[0] = static_cast<MV_REFERENCE_FRAME>(r0);
    mbmi->ref_frame[1] = static_cast<MV_REFERENCE_FRAME>(r1);
  } else {
    for (int i = 0; i < n_refs - 1; i++) {
      w_->symbol(ref == i, av2_get_pred_cdf_single_ref(xd, i, n_refs), 2);
      if (ref == i) break;
    }
    mbmi->ref_frame[0] = static_cast<MV_REFERENCE_FRAME>(ref);
    mbmi->ref_frame[1] = NONE_FRAME;
  }
  const MV_REFERENCE_FRAME ref_frame = av2_ref_frame_type(mbmi->ref_frame);
  av2_initialize_warp_wrl_list(xd->warp_param_stack, xd->valid_num_warp_candidates);
  av2_find_mode_ctx(cm, xd, inter_mode_ctx, ref_frame);
  mbmi->ref_mv_idx[0] = 0;
  mbmi->ref_mv_idx[1] = 0;
  mbmi->cwp_idx = CWP_EQUAL;
  mbmi->jmvd_scale_mode = 0;
  mbmi->warp_ref_idx = 0;
  mbmi->max_num_warp_candidates = 0;
  mbmi->warpmv_with_mvd_flag = 0;
  mbmi->motion_mode = SIMPLE_TRANSLATION;
  mbmi->six_param_warp_model_flag = 0;
  mbmi->warp_precision_idx = 0;
  mbmi->warp_inter_intra = 0;

  if (compound) {
    // NEW_NEWMV: read_inter_compound_mode() (not joint, non-joint type NEW_NEWMV), one DRL index
    // for the pair, assign_mv() with both differences. Masked / weighted compound, optical flow
    // and MV refinement are off in the sequence, so the average is implied.
    // d.mv belongs to d.ref, which may be the higher of the pair (ref_frame[1]).
    const MV m = MV{d.mv_row, d.mv_col}, m2 = MV{d.mv2_row, d.mv2_col};
    const MV want[2] = {d.ref == r0 ? m : m2, d.ref == r0 ? m2 : m};
    mbmi->mode = NEW_NEWMV;
    av2_find_mv_refs(cm, xd, mbmi, ref_frame, ref_mv_count_, xd->ref_mv_stack, xd->weight, ref_mvs,
                     nullptr, xd->warp_param_stack, MAX_WARP_REF_CANDIDATES,
                     xd->valid_num_warp_candidates);
    const int max_drl_bits = cm->features.max_drl_bits;
    int idx = 0, best_dist = INT_MAX;
    for (int i = 0; i <= max_drl_bits; i++) {
      const MV a = xd->ref_mv_stack[ref_frame][i].this_mv.as_mv, b = xd->ref_mv_stack[ref_frame][i].comp_mv.as_mv;
      const int dist = std::abs(a.row - want[0].row) + std::abs(a.col - want[0].col) + std::abs(b.row - want[1].row) +
                       std::abs(b.col - want[1].col);
      if (dist < best_dist) best_dist = dist, idx = i;
    }
    if (is_new_nearmv_pred_mode_disallowed(mbmi)) throw std::runtime_error("same-reference compound unsupported");
    // The predictors (this first av2_find_mv_refs() pass matches the decoder's after the mode).
    const int_mv ref_mv[2] = {xd->ref_mv_stack[ref_frame][idx].this_mv, xd->ref_mv_stack[ref_frame][idx].comp_mv};
    const int step = 1 << (MV_PRECISION_ONE_EIGHTH_PEL - mbmi->pb_mv_precision);
    // The pair's MVs, as coded, moved one MV step (one of them, one component) at a time while that
    // lowers the RD cost with the differences' bits. 33 frames 1080p q100-190: anime B
    // Y -0.19%, live action Y -0.66%, V -0.63% BD-rate.
    MV got[2] = {want[0], want[1]};
    // JOINT_NEWMV: one difference coded, for the farther reference (base); the other's projected
    // from it by the references' distances and scaled by one of five modes.
    const int base = get_joint_mvd_base_ref_list(cm, mbmi);
    bool joint = false;
    int jscale = 0;
    MV jdiff = kZeroMv;
    auto joint_pair = [&](MV diff, int s, MV* m) {  // assign_mv()'s pair
      const int first = cm->ref_frame_relative_dist[mbmi->ref_frame[base]];
      int sec = cm->ref_frame_relative_dist[mbmi->ref_frame[1 - base]];
      if (!is_ref_frame_same_side(cm, mbmi)) sec = -sec;
      MV other = kZeroMv;
      get_mv_projection(&other, diff, sec, first);
      scale_other_mvd(&other, s, JOINT_NEWMV, 0);
      m[base].row = ref_mv[base].as_mv.row + diff.row, m[base].col = ref_mv[base].as_mv.col + diff.col;
      m[1 - base].row = ref_mv[1 - base].as_mv.row + other.row, m[1 - base].col = ref_mv[1 - base].as_mv.col + other.col;
      mv_clamp_to_integer(&m[0]);
      mv_clamp_to_integer(&m[1]);
    };
    if (rd_cost_) {
      for (int r = 0; r < 2; r++) {
        got[r].row = ref_mv[r].as_mv.row + (want[r].row - ref_mv[r].as_mv.row) / step * step;
        got[r].col = ref_mv[r].as_mv.col + (want[r].col - ref_mv[r].as_mv.col) / step * step;
      }
      auto cost = [&](const MV* m, double bits, double bound) {
        bits *= kRdBitWeight;
        mbmi->mv[0].as_mv = m[0], mbmi->mv[1].as_mv = m[1];
        return rd_cost_(bound - bits) + bits;
      };
      auto pair_bits = [&](const MV* m) {
        double bits = 0;
        for (int r = 0; r < 2; r++)
          bits += mvd_comp_bits(m[r].row - ref_mv[r].as_mv.row) + mvd_comp_bits(m[r].col - ref_mv[r].as_mv.col);
        return bits;
      };
      double best = cost(got, pair_bits(got), 1e300);
      // Pairs tried before lost to best then, and best only falls: they are not tried again.
      MV tried[6 * 8 + 1][2] = {{got[0], got[1]}};
      int ntried = 1;
      for (int it = 0; it < 6; it++) {
        MV next[2] = {got[0], got[1]};
        for (int s = 0; s < 8; s++) {
          MV t[2] = {got[0], got[1]};
          auto& comp = s & 2 ? t[s >> 2].col : t[s >> 2].row;
          comp += s & 1 ? -step : step;
          if (std::any_of(tried, tried + ntried, [&](const MV* p) { return std::memcmp(p, t, sizeof(t)) == 0; })) continue;
          tried[ntried][0] = t[0], tried[ntried][1] = t[1], ntried++;
          const double j = cost(t, pair_bits(t), best);
          if (j < best) best = j, next[0] = t[0], next[1] = t[1];
        }
        if (std::memcmp(next, got, sizeof(got)) == 0) break;
        got[0] = next[0], got[1] = next[1];
      }
      // The joint symbols against NEW_NEWMV's; from the refined base difference, the best scale
      // mode, then one step on one component at a time while that helps.
      const int16_t mctx = av2_mode_context_analyzer(inter_mode_ctx, mbmi->ref_frame);
      avm_cdf_prob* const joint_cdf =
          ec_ctx->inter_compound_mode_is_joint_cdf[get_inter_compound_mode_is_joint_context(cm, mbmi)];
      const double mode_bits = symbol_bits(joint_cdf, 1) - symbol_bits(joint_cdf, 0) -
                               symbol_bits(ec_ctx->inter_compound_mode_non_joint_type_cdf[mctx],
                                           INTER_COMPOUND_OFFSET(NEW_NEWMV));
      // kJointBias: extra bits on the joint pair, whose derived MV the luma cost judges but not
      // chroma. 33 frames q100-190, 0 / 2 / 6: anime B Y -0.46 / -0.34 / -0.35%, V -0.57 /
      // -0.24 / -0.19%; live action Y +0.04 / +0.05 / +0.03%, V +0.63 / +0.07 / +0.27% BD-rate.
      constexpr double kJointBias = 2;
      auto joint_cost = [&](MV diff, int s, double bound) {
        MV m[2];
        joint_pair(diff, s, m);
        const double bits = kJointBias + mode_bits + symbol_bits(ec_ctx->jmvd_scale_mode_cdf, s) + mvd_comp_bits(diff.row) +
                            mvd_comp_bits(diff.col);
        return cost(m, bits, bound);
      };
      mbmi->mode = JOINT_NEWMV;
      MV diff{got[base].row - ref_mv[base].as_mv.row, got[base].col - ref_mv[base].as_mv.col};
      double jbest = 1e300;
      int js = 0;
      for (int s = 0; s < JOINT_NEWMV_SCALE_FACTOR_CNT; s++) {
        const double j = joint_cost(diff, s, std::min(jbest, best));
        if (j < jbest) jbest = j, js = s;
      }
      for (int it = 0; it < 6; it++) {
        MV next = diff;
        for (int s = 0; s < 4; s++) {
          MV t = diff;
          auto& comp = s & 2 ? t.col : t.row;
          comp += s & 1 ? -step : step;
          const double j = joint_cost(t, js, std::min(jbest, best));
          if (j < jbest) jbest = j, next = t;
        }
        if (next.row == diff.row && next.col == diff.col) break;
        diff = next;
      }
      mbmi->mode = NEW_NEWMV;
      if (jbest < best) joint = true, jscale = js, jdiff = diff;
    }
    mbmi->mode = joint ? JOINT_NEWMV : NEW_NEWMV;
    const int16_t mode_ctx = av2_mode_context_analyzer(inter_mode_ctx, mbmi->ref_frame);
    w_->symbol(joint, ec_ctx->inter_compound_mode_is_joint_cdf[get_inter_compound_mode_is_joint_context(cm, mbmi)],
                     NUM_OPTIONS_IS_JOINT);
    if (!joint)
      w_->symbol(INTER_COMPOUND_OFFSET(NEW_NEWMV), ec_ctx->inter_compound_mode_non_joint_type_cdf[mode_ctx],
                       NUM_OPTIONS_NON_JOINT_TYPE);
    if (cm->features.opfl_refine_type == REFINE_SWITCHABLE) throw std::runtime_error("optical flow unsupported");
    mbmi->use_amvd = 0;
    if (cm->seq_params.enable_adaptive_mvd)  // use_amvd 0: the differences coded in full
      w_->symbol(0, ec_ctx->amvd_mode_cdf[amvd_mode_to_index(mbmi->mode)][get_amvd_context(xd)], 2);
    av2_find_mv_refs(cm, xd, mbmi, ref_frame, ref_mv_count_, xd->ref_mv_stack, xd->weight, ref_mvs,
                     nullptr, xd->warp_param_stack, MAX_WARP_REF_CANDIDATES,
                     xd->valid_num_warp_candidates);
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
    // read_jmvd_scale_mode()
    mbmi->jmvd_scale_mode = joint ? jscale : 0;
    if (joint) w_->symbol(jscale, ec_ctx->jmvd_scale_mode_cdf, JOINT_NEWMV_SCALE_FACTOR_CNT);
    // read_drl_idx()
    const int16_t pristine = av2_mode_context_pristine(inter_mode_ctx, mbmi->ref_frame);
    for (int i = 0; i < max_drl_bits; ++i) {
      const int bit = idx > i;
      w_->symbol(bit, av2_get_drl_cdf(mbmi, ec_ctx, pristine, i), 2);
      if (!bit) break;
    }
    mbmi->ref_mv_idx[0] = idx;
    set_mv_precision(mbmi, mbmi->max_mv_precision);
    if (enable_adaptive_mvd_resolution(cm, mbmi)) throw std::runtime_error("compound adaptive MVD unsupported");
    if (is_pb_mv_precision_active(cm, mbmi, bsize)) mv_precision(mbmi->max_mv_precision);  // the frame's
    if (xd->ref_mv_stack[ref_frame][idx].this_mv.as_int != ref_mv[0].as_int ||
        xd->ref_mv_stack[ref_frame][idx].comp_mv.as_int != ref_mv[1].as_int)
      throw std::runtime_error("compound predictors changed with the mode");
    if (joint) {
      // assign_mv(): the base reference's difference, then its signs.
      if (is_mvd_sign_derive_allowed(cm, xd, mbmi)) throw std::runtime_error("MVD sign derivation unsupported");
      write_mv_magnitude(w_, jdiff, &ec_ctx->nmvc, mbmi->pb_mv_precision);
      if (jdiff.row) w_->literal(jdiff.row < 0, 1);
      if (jdiff.col) w_->literal(jdiff.col < 0, 1);
      MV m[2];
      joint_pair(jdiff, jscale, m);
      mbmi->mv[0].as_mv = m[0], mbmi->mv[1].as_mv = m[1];
    } else {
      compound_mvs(got, ref_mv);
    }
    mbmi->interinter_comp.type = COMPOUND_AVERAGE;
  } else {
    // Choose NEARMV when a DRL candidate equals the wanted MV. The candidates are taken from a
    // first av2_find_mv_refs() pass; the decoder's pass after the mode symbol is redone below.
    MV wanted;
    wanted.row = d.mv_row;
    wanted.col = d.mv_col;
    mbmi->mode = NEWMV;
    av2_find_mv_refs(cm, xd, mbmi, ref_frame, ref_mv_count_, xd->ref_mv_stack, xd->weight, ref_mvs,
                     nullptr, xd->warp_param_stack, MAX_WARP_REF_CANDIDATES,
                     xd->valid_num_warp_candidates);
    const int max_drl_bits = cm->features.max_drl_bits;
    int near_idx = force_near ? 0 : -1, new_idx = 0, best_dist = INT_MAX;
    for (int i = 0; i <= max_drl_bits && !force_near; i++) {
      const MV c = xd->ref_mv_stack[ref_frame][i].this_mv.as_mv;
      if (c.row == wanted.row && c.col == wanted.col) {
        near_idx = i;
        break;
      }
      // By L1 distance: by difference plus DRL bits instead, on top of the compound refinement,
      // 33 frames q100-190 gave live action Y -0.20%, anime B Y +0.10%, chroma +0.4%.
      const int dist = std::abs(c.row - wanted.row) + std::abs(c.col - wanted.col);
      if (dist < best_dist) best_dist = dist, new_idx = i;
    }
    // GLOBALMV: the frame's model for this reference (warped prediction where it is ROTZOOM or
    // AFFINE), no DRL index or difference.
    const WarpedMotionParams& gm = cm->global_motion[ref_frame];
    const MV gmv =
        get_warp_motion_vector(xd, &gm, cm->features.fr_mv_precision, bsize, xd->mi_col, xd->mi_row).as_mv;
    bool global = false, amvd = false;
    MvSubpelPrecision new_prec = mbmi->max_mv_precision;  // NEWMV's coded precision (--flexmv)
    if (near_idx < 0 && rd_cost_) {
      const auto comp_bits = mvd_comp_bits;
      const MV c = xd->ref_mv_stack[ref_frame][new_idx].this_mv.as_mv;
      // Each distinct MV's cost once (the stack repeats MVs); a cost cut at its bound only loses,
      // and the bounds only fall (best falls, a later index costs more bits).
      std::pair<MV, double> seen[MAX_REF_MV_STACK_SIZE + 1];
      int nseen = 0;
      auto rd = [&](MV mv, double bound) {
        for (int k = 0; k < nseen; k++)
          if (seen[k].first.row == mv.row && seen[k].first.col == mv.col) return seen[k].second;
        const int_mv keep = mbmi->mv[0];
        mbmi->mv[0].as_mv = mv;
        const double v = rd_cost_(bound);
        mbmi->mv[0] = keep;
        seen[nseen++] = {mv, v};
        return v;
      };
      double best =
          rd(wanted, 1e300) + kRdBitWeight * (comp_bits(wanted.row - c.row) + comp_bits(wanted.col - c.col) + new_idx + 1);
      {
        // The NEWMV MV, as coded (rounded to the MV precision), moved one MV step at a time while
        // that lowers the RD cost: the decision MV's search saw neither this RD cost nor these bits.
        // 33 frames 1080p q100-190: anime B Y -0.38%, live action Y -0.59% BD-rate.
        const int step = 1 << (MV_PRECISION_ONE_EIGHTH_PEL - mbmi->max_mv_precision);
        MV m;
        m.row = static_cast<int16_t>(c.row + (wanted.row - c.row) / step * step);
        m.col = static_cast<int16_t>(c.col + (wanted.col - c.col) / step * step);
        const int_mv keep = mbmi->mv[0];
        // MVs tried before lost to best then, and best only falls: they are not tried again.
        MV tried[6 * 4];
        int ntried = 0;
        for (int it = 0; it < 6; it++) {
          MV next = m;
          for (int s = 0; s < 4; s++) {
            MV t = m;
            const int sign = s & 1 ? -1 : 1;
            if (s < 2) t.row = static_cast<int16_t>(t.row + sign * step);
            else t.col = static_cast<int16_t>(t.col + sign * step);
            if (std::any_of(tried, tried + ntried, [&](const MV& p) { return p.row == t.row && p.col == t.col; })) continue;
            tried[ntried++] = t;
            const double bits = kRdBitWeight * (comp_bits(t.row - c.row) + comp_bits(t.col - c.col) + new_idx + 1);
            mbmi->mv[0].as_mv = t;
            const double j = rd_cost_(best - bits) + bits;
            if (j < best) best = j, next = t;
          }
          if (next.row == m.row && next.col == m.col) break;
          m = next;
        }
        mbmi->mv[0] = keep;
        wanted = m;
      }
      if (cm->seq_params.enable_flex_mvres && cm->features.use_pb_mv_precision &&
          mbmi->max_mv_precision >= MV_PRECISION_HALF_PEL) {
        // A coarser precision (1, 4, 8 pel) when its MV's RD cost with fewer difference bits wins:
        // the predictor rounded to that precision, plus the difference rounded to its steps.
        mbmi->ref_mv_idx[0] = new_idx;
        best += kRdBitWeight * mv_precision_bits(new_prec);
        for (MvSubpelPrecision p : {MV_PRECISION_ONE_PEL, MV_PRECISION_FOUR_PEL, MV_PRECISION_8_PEL}) {
          const int step = 1 << (MV_PRECISION_ONE_EIGHTH_PEL - p);
          MV base = c;
          lower_mv_precision(&base, p);
          auto round = [&](int d) { return static_cast<int16_t>((d >= 0 ? (d + step / 2) / step : -((-d + step / 2) / step)) * step); };
          const MV diff{round(wanted.row - base.row), round(wanted.col - base.col)};
          const MV t{static_cast<int16_t>(base.row + diff.row), static_cast<int16_t>(base.col + diff.col)};
          // the difference's bits at that precision: mvd_comp_bits() counts quarter-pel steps
          const double bits = kRdBitWeight * (comp_bits(diff.row / (step / 2)) + comp_bits(diff.col / (step / 2)) + new_idx + 1 +
                                              mv_precision_bits(p));
          const double j = rd(t, best - bits) + bits;
          if (j < best) best = j, new_prec = p, wanted = t;
        }
      }
      if (cm->seq_params.enable_adaptive_mvd) {
        // Adaptive MVD: each difference component one of 0, +-2, 4, 6, 8, 16, 32, 64, 128 (1/8 pel),
        // coded as a joint type, an index per nonzero component and its sign. From the components
        // nearest the NEWMV MV's, one index step on one component at a time while that helps.
        avm_cdf_prob* const flag_cdf = ec_ctx->amvd_mode_cdf[amvd_mode_to_index(NEWMV)][get_amvd_context(xd)];
        const double flag_bits = symbol_bits(flag_cdf, 1) - symbol_bits(flag_cdf, 0);
        auto nearest = [](int d) {
          int best_i = 0;
          for (int i = 1; i <= MAX_AMVD_INDEX; i++)
            if (std::abs(std::abs(d) - amvd_index_to_mvd[i]) < std::abs(std::abs(d) - amvd_index_to_mvd[best_i])) best_i = i;
          return d < 0 ? -best_i : best_i;
        };
        auto amvd_bits = [&](int ir, int ic) {
          double b = flag_bits + new_idx + 1 +
                     symbol_bits(ec_ctx->nmvc.amvd_joints_cdf, (ir ? MV_JOINT_HZVNZ : 0) | (ic ? MV_JOINT_HNZVZ : 0));
          if (ir) b += symbol_bits(ec_ctx->nmvc.comps[0].amvd_indices_cdf, std::abs(ir) - 1) + 1;
          if (ic) b += symbol_bits(ec_ctx->nmvc.comps[1].amvd_indices_cdf, std::abs(ic) - 1) + 1;
          return b;
        };
        const int_mv keep = mbmi->mv[0];
        double abest = 1e300;
        int ir = nearest(wanted.row - c.row), ic = nearest(wanted.col - c.col);
        auto trial = [&](int r, int q) {
          const double bits = kRdBitWeight * amvd_bits(r, q);
          mbmi->mv[0].as_mv.row = static_cast<int16_t>(c.row + get_mvd_from_amvd_index(r));
          mbmi->mv[0].as_mv.col = static_cast<int16_t>(c.col + get_mvd_from_amvd_index(q));
          return rd_cost_(std::min(abest, best) - bits) + bits;
        };
        abest = trial(ir, ic);
        for (int it = 0; it < 6; it++) {
          int nr = ir, nc = ic;
          for (int s = 0; s < 4; s++) {
            const int r = ir + (s == 0) - (s == 1), q = ic + (s == 2) - (s == 3);
            if (std::abs(r) > MAX_AMVD_INDEX || std::abs(q) > MAX_AMVD_INDEX) continue;
            const double j = trial(r, q);
            if (j < abest) abest = j, nr = r, nc = q;
          }
          if (nr == ir && nc == ic) break;
          ir = nr, ic = nc;
        }
        mbmi->mv[0] = keep;
        if (abest < best) {
          best = abest, amvd = true, new_prec = mbmi->max_mv_precision;
          wanted.row = static_cast<int16_t>(c.row + get_mvd_from_amvd_index(ir));
          wanted.col = static_cast<int16_t>(c.col + get_mvd_from_amvd_index(ic));
        }
      }
      for (int i = 0; i <= max_drl_bits; i++) {
        // kNearBias: extra bits on a candidate, whose cost here leaves out what the decision MV's
        // search saw (the block's own transform partition). With kDecisionBits 8, 4 bits against
        // 2: -12.1% / -11.4% BD-rate on anime A 1080p. Exact copies of the decision MV skip this.
        constexpr int kNearBias = 4;
        const double bits = kRdBitWeight * (i + 1 + kNearBias);
        const double j = rd(xd->ref_mv_stack[ref_frame][i].this_mv.as_mv, best - bits) + bits;
        if (j < best) best = j, near_idx = i;
      }
      // Blocks of 16 and up: the zoom still is -6.6% then, -6.4% from 8 (the warp is barely off a
      // translation across a small block, which NEARMV codes as well).
      constexpr int kGlobalMin = 16;
      if (gm.wmtype > TRANSLATION && !gm.invalid &&
          AVMMIN(block_size_wide[bsize], block_size_high[bsize]) >= kGlobalMin) {
        // kGlobalBits: GLOBALMV's charge over the RD cost. 2 / 6 / 12: -7.2 / -6.6 / -6.4% BD-rate on a
        // 1080p anime still zooming in by 0.2% a frame, +1.4 / +0.3 / +0.1% on anime A 1080p (no zoom:
        // there the model is noise around a pan).
        constexpr int kGlobalBits = 12;
        const double bits = kRdBitWeight * kGlobalBits;
        const int_mv keep = mbmi->mv[0];
        mbmi->mode = GLOBALMV;  // av2_build_inter_predictors() warps a GLOBALMV block
        mbmi->mv[0].as_mv = gmv;
        const double j = rd_cost_(best - bits) + bits;
        mbmi->mode = NEWMV;
        mbmi->mv[0] = keep;
        if (j < best) best = j, global = true, near_idx = -1;
      }
    }
    const PREDICTION_MODE mode = global ? GLOBALMV : near_idx >= 0 ? NEARMV : NEWMV;
    mbmi->mode = mode;
    // BAWP (fades, light changes): the NEARMV / NEWMV prediction scaled and offset to fit the
    // reconstructed rows above and columns left of the block against the reference's there
    // (implicit: no parameters coded).
    bool bawp = false;
    if (rd_cost_ && !force_near && cm->features.enable_bawp && (mode == NEARMV || mode == NEWMV) &&
        av2_allow_bawp(cm, mbmi, xd->mi_row, xd->mi_col)) {
      // kBawpBits: the flag's charge over the luma RD cost. 2 / 4 / 8: -2.5 / -2.3 / -2.1% BD-rate on a
      // 1080p anime fade to black, -0.9 / -0.8 / -0.4% on anime A, +0.05 / -0.13 / +0.01% on anime
      // B.
      constexpr int kBawpBits = 4;
      const MV mv = mode == NEARMV ? xd->ref_mv_stack[ref_frame][near_idx].this_mv.as_mv : wanted;
      if (bawp_template_inside(xd, bsize, mv)) {
        const int_mv keep = mbmi->mv[0];
        mbmi->mv[0].as_mv = mv;
        const double plain = rd_cost_(1e300), bits = kRdBitWeight * kBawpBits;
        mbmi->bawp_flag[0] = 1;
        bawp = rd_cost_(plain - bits) + bits < plain;
        mbmi->bawp_flag[0] = 0;
        mbmi->mv[0] = keep;
      }
    }

    // read_inter_mode()
    if (is_warpmv_mode_allowed(cm, mbmi, bsize)) throw std::runtime_error("warp modes unsupported");
    const int16_t mode_ctx = av2_mode_context_analyzer(inter_mode_ctx, mbmi->ref_frame);
    w_->symbol(mode - SINGLE_INTER_MODE_START,
                     ec_ctx->inter_single_mode_cdf[inter_single_mode_ctx(mode_ctx)],
                     INTER_SINGLE_MODES);
    mbmi->use_amvd = amvd && mode == NEWMV;
    if (cm->seq_params.enable_adaptive_mvd && allow_amvd_mode(mode))
      w_->symbol(mbmi->use_amvd, ec_ctx->amvd_mode_cdf[amvd_mode_to_index(mode)][get_amvd_context(xd)], 2);
    av2_find_mv_refs(cm, xd, mbmi, ref_frame, ref_mv_count_, xd->ref_mv_stack, xd->weight, ref_mvs,
                     nullptr, xd->warp_param_stack, MAX_WARP_REF_CANDIDATES,
                     xd->valid_num_warp_candidates);
    // bawp_flag luma (implicit: explicit_bawp_flag 0), then chroma with it.
    mbmi->bawp_flag[0] = mbmi->bawp_flag[1] = 0;
    if (cm->features.enable_bawp && av2_allow_bawp(cm, mbmi, xd->mi_row, xd->mi_col)) {
      w_->symbol(bawp, ec_ctx->bawp_cdf[0], 2);
      if (bawp && av2_allow_explicit_bawp(mbmi))
        w_->symbol(0, ec_ctx->explicit_bawp_cdf[mode == NEARMV ? 0 : mbmi->use_amvd ? 1 : 2], 2);
      mbmi->bawp_flag[0] = bawp;
    } else if (bawp) {
      throw std::runtime_error("BAWP not allowed");
    }
    if (!cm->seq_params.monochrome && xd->is_chroma_ref && mbmi->bawp_flag[0]) {
      w_->symbol(1, ec_ctx->bawp_cdf[1], 2);
      mbmi->bawp_flag[1] = 1;
    }
    xd->block_ref_scale_factors[0] = get_ref_scale_factors_const(cm, mbmi->ref_frame[0]);
    if (is_motion_variation_allowed_bsize(bsize, xd->mi_row, xd->mi_col) &&
        !is_tip_ref_frame(mbmi->ref_frame[0]))
      mbmi->num_proj_ref[0] = av2_findSamples(cm, xd, pts0, pts0_inref, 0);
    // read_motion_mode(): translation only.
    mbmi->max_num_warp_candidates = 0;
    mbmi->six_param_warp_model_flag = 0;
    mbmi->warp_precision_idx = 0;
    if (motion_mode_allowed(cm, xd, xd->ref_mv_stack[mbmi->ref_frame[0]], mbmi) !=
        (1 << SIMPLE_TRANSLATION))
      throw std::runtime_error("only translation is supported");
    mbmi->use_wedge_interintra = 0;
    mbmi->motion_mode = SIMPLE_TRANSLATION;
    if (allow_warpmv_with_mvd_coding(cm, mbmi)) throw std::runtime_error("warp MVD unsupported");
    mbmi->warpmv_with_mvd_flag = 0;
    mbmi->jmvd_scale_mode = 0;  // not a joint MVD mode
    // read_drl_idx()
    const int idx = mode == NEARMV ? near_idx : mode == NEWMV ? new_idx : 0;
    if (have_drl_index(mbmi->mode)) {
      const int16_t pristine = av2_mode_context_pristine(inter_mode_ctx, mbmi->ref_frame);
      for (int i = 0; i < max_drl_bits; ++i) {
        const int bit = idx > i;
        w_->symbol(bit, av2_get_drl_cdf(mbmi, ec_ctx, pristine, i), 2);
        if (!bit) break;
      }
      mbmi->ref_mv_idx[0] = idx;
    } else if (idx != 0) {
      throw std::runtime_error("mode without DRL index");
    }
    set_mv_precision(mbmi, mbmi->max_mv_precision);
    if (is_pb_mv_precision_active(cm, mbmi, bsize)) mv_precision(new_prec);
    if (enable_adaptive_mvd_resolution(cm, mbmi)) set_amvd_mv_precision(mbmi, mbmi->max_mv_precision);

    // assign_mv()
    const int_mv ref_mv = xd->ref_mv_stack[ref_frame][get_ref_mv_idx(mbmi, 0)].this_mv;
    if (mode == GLOBALMV) {
      mbmi->mv[0].as_mv = gmv;
    } else if (mode == NEARMV) {
      mbmi->mv[0].as_int = ref_mv.as_int;
    } else if (mbmi->use_amvd) {
      // read_vq_amvd(): the joint type, each nonzero component's index, then the signs.
      const MV diff{static_cast<int16_t>(wanted.row - ref_mv.as_mv.row), static_cast<int16_t>(wanted.col - ref_mv.as_mv.col)};
      const int ir = get_amvd_index_from_mvd(diff.row), ic = get_amvd_index_from_mvd(diff.col);
      if (std::abs(ir) > MAX_AMVD_INDEX || std::abs(ic) > MAX_AMVD_INDEX) throw std::runtime_error("difference not in the AMVD set");
      w_->symbol((ir ? MV_JOINT_HZVNZ : 0) | (ic ? MV_JOINT_HNZVZ : 0), ec_ctx->nmvc.amvd_joints_cdf, MV_JOINTS);
      if (ir) w_->symbol(std::abs(ir) - 1, ec_ctx->nmvc.comps[0].amvd_indices_cdf, MAX_AMVD_INDEX);
      if (ic) w_->symbol(std::abs(ic) - 1, ec_ctx->nmvc.comps[1].amvd_indices_cdf, MAX_AMVD_INDEX);
      if (is_mvd_sign_derive_allowed(cm, xd, mbmi)) throw std::runtime_error("MVD sign derivation unsupported");
      if (diff.row) w_->literal(diff.row < 0, 1);
      if (diff.col) w_->literal(diff.col < 0, 1);
      mbmi->mv[0].as_mv = wanted;
    } else {
      const MvSubpelPrecision precision = mbmi->pb_mv_precision;
      const int step = 1 << (MV_PRECISION_ONE_EIGHTH_PEL - precision);
      MV base = ref_mv.as_mv;  // below half pel the predictor is rounded to the precision first
      if (precision < MV_PRECISION_HALF_PEL) lower_mv_precision(&base, precision);
      MV diff;  // rounded to the MV precision (the decision MV may move by less than a step)
      diff.row = static_cast<int16_t>((wanted.row - base.row) / step * step);
      diff.col = static_cast<int16_t>((wanted.col - base.col) / step * step);
      write_mv_magnitude(w_, diff, &ec_ctx->nmvc, precision);
      if (is_mvd_sign_derive_allowed(cm, xd, mbmi)) throw std::runtime_error("MVD sign derivation unsupported");
      if (diff.row) w_->literal(diff.row < 0, 1);
      if (diff.col) w_->literal(diff.col < 0, 1);
      mbmi->mv[0].as_mv.row = static_cast<int16_t>(base.row + diff.row);
      mbmi->mv[0].as_mv.col = static_cast<int16_t>(base.col + diff.col);
    }
    mv_clamp_to_integer(&mbmi->mv[0].as_mv);
  }
  inter_block_tail();
}

// read_inter_block_mode_info() after the MVs: no warp inter-intra, refinement or masked compound.
void BlockWriter::inter_block_tail() {
  AV2_COMMON* const cm = st_.cm();
  MACROBLOCKD* const xd = xd_;
  MB_MODE_INFO* const mbmi = xd->mi[0];
  mbmi->warp_inter_intra = 0;
  if (allow_warp_inter_intra(mbmi)) throw std::runtime_error("warp inter-intra unsupported");
  // read_refinemv_flag()
  mbmi->refinemv_flag = get_default_refinemv_flag(cm, mbmi);
  if (switchable_refinemv_flag(cm, mbmi)) throw std::runtime_error("refinemv unsupported");
  mbmi->comp_group_idx = 0;
  mbmi->interinter_comp.type = COMPOUND_AVERAGE;
  mbmi->cwp_idx = CWP_EQUAL;
  // read_mb_interp_filter(): frame filter, not switchable.
  if (!av2_is_interp_needed(cm, xd))
    set_default_interp_filters(mbmi, cm, xd, cm->features.interp_filter);
  else
    mbmi->interp_fltr = cm->features.interp_filter;
  if (xd->tree_type != LUMA_PART) xd->cfl.store_y = store_cfl_required(cm, xd);
}

// read_inter_block_mode_info() for a skip-mode block: the frame's skip-mode reference pair,
// NEAR_NEARMV with a DRL index into the skip-mode candidate list (read_drl_idx() before
// av2_find_mv_refs(), as the decoder does).
void BlockWriter::skip_mode_info(int skip_idx) {
  AV2_COMMON* const cm = st_.cm();
  MACROBLOCKD* const xd = xd_;
  MB_MODE_INFO* const mbmi = xd->mi[0];
  FRAME_CONTEXT* const ec_ctx = xd->tile_ctx;
  int_mv ref_mvs[MODE_CTX_REF_FRAMES][MAX_MV_REF_CANDIDATES] = {};
  int16_t inter_mode_ctx[MODE_CTX_REF_FRAMES];
  set_skip_mode_ref_frame(cm, xd, mbmi->ref_frame);
  const MV_REFERENCE_FRAME ref_frame = av2_ref_frame_type(mbmi->ref_frame);
  av2_initialize_warp_wrl_list(xd->warp_param_stack, xd->valid_num_warp_candidates);
  av2_find_mode_ctx(cm, xd, inter_mode_ctx, ref_frame);
  mbmi->ref_mv_idx[0] = 0;
  mbmi->ref_mv_idx[1] = 0;
  mbmi->cwp_idx = CWP_EQUAL;
  mbmi->jmvd_scale_mode = 0;
  mbmi->warp_ref_idx = 0;
  mbmi->max_num_warp_candidates = 0;
  mbmi->warpmv_with_mvd_flag = 0;
  mbmi->motion_mode = SIMPLE_TRANSLATION;
  mbmi->six_param_warp_model_flag = 0;
  mbmi->warp_precision_idx = 0;
  mbmi->warp_inter_intra = 0;
  mbmi->mode = NEAR_NEARMV;
  const int max_drl_bits = cm->features.max_drl_bits;
  const int16_t pristine = av2_mode_context_pristine(inter_mode_ctx, mbmi->ref_frame);
  for (int i = 0; i < max_drl_bits; ++i) {
    const int bit = skip_idx > i;
    w_->symbol(bit, av2_get_drl_cdf(mbmi, ec_ctx, pristine, i), 2);
    if (!bit) break;
  }
  mbmi->ref_mv_idx[0] = skip_idx;
  av2_find_mv_refs(cm, xd, mbmi, ref_frame, ref_mv_count_, xd->ref_mv_stack, xd->weight, ref_mvs, nullptr,
                   nullptr, 0, nullptr);
  if (!has_second_ref(mbmi)) throw std::runtime_error("skip mode without two references");
  // assign_mv(), NEAR_NEARMV
  mbmi->mv[0] = xd->skip_mvp_candidate_list.ref_mv_stack[skip_idx].this_mv;
  mbmi->mv[1] = xd->skip_mvp_candidate_list.ref_mv_stack[skip_idx].comp_mv;
  mv_clamp_to_integer(&mbmi->mv[0].as_mv);
  mv_clamp_to_integer(&mbmi->mv[1].as_mv);
  for (int r = 0; r < 2; r++) xd->block_ref_scale_factors[r] = get_ref_scale_factors_const(cm, mbmi->ref_frame[r]);
}

int BlockWriter::choose_skip_mode(const BlockDecision& d) {
  AV2_COMMON* const cm = st_.cm();
  MACROBLOCKD* const xd = xd_;
  MB_MODE_INFO* const mbmi = xd->mi[0];
  const MB_MODE_INFO saved = *mbmi;
  // State av2_find_mv_refs() reads, as read_inter_block_mode_info() sets it before.
  av2_collect_neighbors_ref_counts(xd);
  set_default_max_mv_precision(mbmi, xd->sbi->sb_mv_precision);
  set_mv_precision(mbmi, mbmi->max_mv_precision);
  mbmi->bawp_flag[0] = mbmi->bawp_flag[1] = 0;
  mbmi->refinemv_flag = 0;
  mbmi->jmvd_scale_mode = 0;
  mbmi->use_amvd = 0;
  mbmi->comp_group_idx = 0;
  // The decision's prediction: its pair where the syntax codes one, else its first reference.
  const int n_refs = cm->ref_frames_info.num_total_refs;
  const int r0 = std::min<int>(d.ref, d.ref2), r1 = std::max<int>(d.ref, d.ref2);
  const bool compound = is_comp_ref_allowed(mbmi->sb_type[PLANE_TYPE_Y]) &&
                        cm->current_frame.reference_mode == REFERENCE_MODE_SELECT &&
                        d.ref2 != BlockDecision::kSingleRef && compound_codable(r0, r1);
  const MV m = MV{d.mv_row, d.mv_col}, m2 = MV{d.mv2_row, d.mv2_col};
  mbmi->skip_mode = 0;
  mbmi->motion_mode = SIMPLE_TRANSLATION;
  mbmi->interinter_comp.type = COMPOUND_AVERAGE;
  mbmi->interp_fltr = cm->features.interp_filter;
  mbmi->num_proj_ref[0] = mbmi->num_proj_ref[1] = 0;
  mbmi->wm_params[0].invalid = mbmi->wm_params[1].invalid = 1;
  if (compound) {
    mbmi->mode = NEW_NEWMV;
    mbmi->ref_frame[0] = static_cast<MV_REFERENCE_FRAME>(r0);
    mbmi->ref_frame[1] = static_cast<MV_REFERENCE_FRAME>(r1);
    mbmi->mv[0].as_mv = d.ref == r0 ? m : m2;
    mbmi->mv[1].as_mv = d.ref == r0 ? m2 : m;
  } else {
    mbmi->mode = NEWMV;
    mbmi->ref_frame[0] = static_cast<MV_REFERENCE_FRAME>(std::min<int>(d.ref, n_refs - 1));
    mbmi->ref_frame[1] = NONE_FRAME;
    mbmi->mv[0].as_mv = m;
  }
  // kDecisionBits: what coding the decision's references and MVs costs over a skip-mode index.
  // 8 and 12 (with a NEAR bias of 2) gave the same mean BD-rate over anime A, two more 1080p anime
  // sources and the anime clips; 4 lost 10% in bytes on anime A at +0.1 dB.
  constexpr int kDecisionBits = 8;
  double best = rd_cost_(1e300) + kRdBitWeight * kDecisionBits;
  int skip_idx = -1;
  // Skip-mode candidates.
  int_mv ref_mvs[MODE_CTX_REF_FRAMES][MAX_MV_REF_CANDIDATES] = {};
  mbmi->skip_mode = 1;
  mbmi->mode = NEAR_NEARMV;
  mbmi->ref_mv_idx[0] = mbmi->ref_mv_idx[1] = 0;
  set_skip_mode_ref_frame(cm, xd, mbmi->ref_frame);
  av2_find_mv_refs(cm, xd, mbmi, av2_ref_frame_type(mbmi->ref_frame), ref_mv_count_, xd->ref_mv_stack, xd->weight,
                   ref_mvs, nullptr, nullptr, 0, nullptr);
  const int n = std::min<int>(xd->skip_mvp_candidate_list.ref_mv_count, cm->features.max_drl_bits + 1);
  for (int i = 0; i < n && has_second_ref(mbmi); i++) {
    mbmi->mv[0] = xd->skip_mvp_candidate_list.ref_mv_stack[i].this_mv;
    mbmi->mv[1] = xd->skip_mvp_candidate_list.ref_mv_stack[i].comp_mv;
    mv_clamp_to_integer(&mbmi->mv[0].as_mv);
    mv_clamp_to_integer(&mbmi->mv[1].as_mv);
    bool repeat = false;  // an earlier candidate with the same pair costs fewer index bits
    for (int k = 0; k < i && !repeat; k++) {
      MV a = xd->skip_mvp_candidate_list.ref_mv_stack[k].this_mv.as_mv, b = xd->skip_mvp_candidate_list.ref_mv_stack[k].comp_mv.as_mv;
      mv_clamp_to_integer(&a);
      mv_clamp_to_integer(&b);
      repeat = a.row == mbmi->mv[0].as_mv.row && a.col == mbmi->mv[0].as_mv.col && b.row == mbmi->mv[1].as_mv.row &&
               b.col == mbmi->mv[1].as_mv.col;
    }
    if (repeat) continue;
    const double bits = kRdBitWeight * (i + 1);
    const double c = rd_cost_(best - bits) + bits;
    if (c < best) best = c, skip_idx = i;
  }
  *mbmi = saved;
  return skip_idx;
}

// decodeframe.c read_tx_partition() (forward skip coding off, so is_fsc is 0).
void BlockWriter::tx_partition(TX_SIZE max_tx, int blk_row, int blk_col, TX_PARTITION_TYPE p) {
  MACROBLOCKD* const xd = xd_;
  MB_MODE_INFO* const mbmi = xd->mi[0];
  FRAME_CONTEXT* const ec_ctx = xd->tile_ctx;
  const BLOCK_SIZE bsize = mbmi->sb_type[PLANE_TYPE_Y];
  const int is_inter = is_inter_block(mbmi, xd->tree_type);
  if (is_inter && (blk_row >= max_block_high(xd, bsize, 0) || blk_col >= max_block_wide(xd, bsize, 0))) return;
  const int allow_horz = allow_tx_horz_split(bsize, max_tx), allow_vert = allow_tx_vert_split(bsize, max_tx);
  const int h_and_v = get_vert_and_horz_group(bsize), h_or_v = get_vert_or_horz_group(bsize);
  if (!allow_horz && !allow_vert) {
    if (p != TX_PARTITION_NONE) throw std::runtime_error("transform partition not allowed");
  } else {
    w_->symbol(p != TX_PARTITION_NONE,
                     ec_ctx->txfm_do_partition_cdf[0][is_inter][size_to_tx_part_group_lookup[bsize]], 2);
    if (p != TX_PARTITION_NONE) {
      if (allow_horz && allow_vert) {
        w_->symbol(p - 1, ec_ctx->txfm_4way_partition_type_cdf[0][is_inter][h_and_v], TX_PARTITION_TYPE_NUM);
      } else {
        const bool four = p == TX_PARTITION_HORZ4 || p == TX_PARTITION_VERT4;
        const TX_PARTITION_TYPE two = allow_horz ? TX_PARTITION_HORZ : TX_PARTITION_VERT;
        if (p != (four ? (allow_horz ? TX_PARTITION_HORZ4 : TX_PARTITION_VERT4) : two) ||
            (four && (!h_or_v || xd->reduced_tx_part_set)))
          throw std::runtime_error("transform partition not codable here");
        if (h_or_v && !xd->reduced_tx_part_set)
          w_->symbol(four, ec_ctx->txfm_2or3_way_partition_type_cdf[0][is_inter][h_or_v - 1], 2);
      }
    }
  }
  TX_SIZE sub_txs[MAX_TX_PARTITIONS] = {};
  const int n = get_tx_partition_sizes(p, max_tx, &mbmi->txb_pos, sub_txs, xd->error_info);
  mbmi->tx_size = sub_txs[n - 1];
  mbmi->tx_partition_type[is_inter ? av2_get_txb_size_index(bsize, blk_row, blk_col) : 0] = p;
}

// decodemv.c av2_read_tx_type(), intra luma (reduced_tx_set_used != 2, a set with a choice).
void BlockWriter::intra_tx_type(TX_SIZE tx_size, TX_TYPE tx_type) {
  AV2_COMMON* const cm = st_.cm();
  FRAME_CONTEXT* const ec_ctx = xd_->tile_ctx;
  MB_MODE_INFO* const mbmi = xd_->mi[0];
  const int reduced = cm->features.reduced_tx_set_used;
  const TxSetType set = av2_get_ext_tx_set_type(tx_size, 0, reduced);
  const int eset = get_ext_tx_set(tx_size, 0, reduced);
  const TX_SIZE square = txsize_sqr_map[tx_size];
  if (set != EXT_TX_SET_LONG_SIDE_64 && set != EXT_TX_SET_LONG_SIDE_32) {
    const PREDICTION_MODE intra_mode = get_intra_mode(mbmi, PLANE_TYPE_Y);
    const int size_info = av2_size_class[tx_size];
    const int n = reduced ? av2_num_reduced_tx_set[reduced - 1] : av2_num_ext_tx_set_intra[set];
    int idx = -1;
    for (int i = 0; i < n && idx < 0; i++)
      if (av2_tx_idx_to_type(i, set, intra_mode, size_info) == tx_type) idx = i;
    if (idx < 0) throw std::runtime_error("transform type not in the intra transform set");
    w_->symbol(idx, ec_ctx->intra_ext_tx_cdf[eset + (reduced ? 1 : 0)][square], n);
  } else {
    // is_long_side_dct (coded for a 32-sample long side), then the short side's index.
    int idx = -1, long_dct = 1;
    for (long_dct = 1; long_dct >= 0 && idx < 0; long_dct -= idx < 0)
      for (int i = 0; i < 4 && idx < 0; i++)
        if (get_txtype_from_idx_for_large_txfm(tx_size, set, i, long_dct) == tx_type) idx = i;
    if (idx < 0) throw std::runtime_error("transform type not in the long-side transform set");
    if (txsize_sqr_up_map[tx_size] == TX_32X32) w_->symbol(long_dct, ec_ctx->tx_ext_32_cdf[0], 2);
    else if (!long_dct) throw std::runtime_error("a 64-sample long side is DCT only");
    w_->symbol(idx, ec_ctx->intra_ext_tx_short_side_cdf[square], 4);
  }
}

// decodemv.c av2_read_tx_type(), inter luma: codes tx_type (DCT_DCT where the set has no choice).
void BlockWriter::inter_tx_type(TX_SIZE tx_size, int eob, TX_TYPE tx_type) {
  AV2_COMMON* const cm = st_.cm();
  FRAME_CONTEXT* const ec_ctx = xd_->tile_ctx;
  const int reduced = cm->features.reduced_tx_set_used;
  if (get_ext_tx_types(tx_size, 1, reduced) <= 1) {
    if (tx_type != DCT_DCT) throw std::runtime_error("a transform type the set cannot code");
    return;
  }
  const TxSetType set = av2_get_ext_tx_set_type(tx_size, 1, reduced);
  const int eset = get_ext_tx_set(tx_size, 1, reduced);
  const TX_SIZE square = txsize_sqr_map[tx_size];
  const int eob_tx_ctx = get_lp2tx_ctx(tx_size, get_txb_bwl(tx_size), eob);
  if (set != EXT_TX_SET_LONG_SIDE_64 && set != EXT_TX_SET_LONG_SIDE_32) {
    int idx = -1;
    for (int i = 0; i < av2_num_ext_tx_set[set]; i++)
      if (av2_ext_tx_inv[set][i] == tx_type) idx = i;
    if (idx < 0) throw std::runtime_error("transform type not in the inter transform set");
    if (eset == 1 || eset == 2) {
      const int in_offset = idx >= INTER_TX_TYPE_INDEX_COUNT;
      w_->symbol(in_offset, ec_ctx->inter_tx_type_set[eset - 1][eob_tx_ctx][square], 2);
      if (!in_offset)
        w_->symbol(idx, ec_ctx->inter_tx_type_idx[eset - 1][eob_tx_ctx],
                         INTER_TX_TYPE_INDEX_COUNT);
      else if (eset == 1)
        w_->symbol(idx - INTER_TX_TYPE_INDEX_COUNT,
                         ec_ctx->inter_tx_type_offset_1[eob_tx_ctx], INTER_TX_TYPE_OFFSET1_COUNT);
      else
        w_->symbol(idx - INTER_TX_TYPE_INDEX_COUNT,
                         ec_ctx->inter_tx_type_offset_2[eob_tx_ctx], INTER_TX_TYPE_OFFSET2_COUNT);
    } else {
      w_->symbol(idx, ec_ctx->inter_ext_tx_cdf[eset][eob_tx_ctx][square],
                       av2_num_ext_tx_set[set]);
    }
  } else {
    if (txsize_sqr_up_map[tx_size] == TX_32X32) w_->symbol(1, ec_ctx->tx_ext_32_cdf[1], 2);
    int idx = -1;
    for (int i = 3; i >= 0; i--)
      if (get_txtype_from_idx_for_large_txfm(tx_size, set, i, true) == tx_type) idx = i;
    if (idx < 0) throw std::runtime_error("transform type not in the long-side transform set");
    w_->symbol(idx, ec_ctx->inter_ext_tx_short_side_cdf[eob_tx_ctx][square], 4);
  }
}
