// The file oracle_dump writes and celv2_oracle reads: plain data, no AVM headers (the two link different
// AVM builds). "CELV2OR1", width, height (int32), then per coded frame: an OracleFrame, its partition tree
// (n_tree bytes, superblocks in raster order, each preorder: 0 NONE, 1 HORZ, 2 VERT, 3 SPLIT, 255 outside
// the frame), its leaves (n_blocks OracleBlocks, in coding order), its transform partitions (one byte per mi,
// mi_rows x mi_cols, at each max-size transform unit's top left), its transform blocks (n_tx OracleTx, each
// plane's in coding order), their nonzero coefficients (n_coef OracleCoef, each OracleTx's in turn), and its filters:
// an OracleFilters, cdef_units signed bytes (each 64x64's strength index), per plane ccso_units bytes (each CCSO
// unit's flag), then per plane with restoration on: wiener_bytes of its frame filter, lr_units bytes (each unit's type)
// and wiener_bytes for each unit of type WIENER_NONSEP (AVM's WienerNonsepInfo, as it is).
#pragma once
#include <stdint.h>

#pragma pack(push, 1)
typedef struct OracleFrame {
  int32_t key, display, shown, refresh, qindex;  // X_L0's list; shown 0: hidden until its show-existing
  int32_t n_refs, ref_hint[7];                   // the references' display order hints, in AV2's ranking
  int32_t interp_filter, reference_mode, skip_mode_flag, skip_ref[2];
  int32_t max_drl_bits, fr_mv_precision, use_pb_mv_precision, force_integer_mv;
  int32_t tx_mode, reduced_tx_set, delta_q_present, delta_q_res, enable_bawp;
  int32_t gm_type[7], gm_mat[7][6];
  int32_t n_tree, n_blocks, mi_rows, mi_cols, n_tx, n_coef;
  // The sequence's transform tools (L3 on).
  int32_t reduced_tx_part_set, enable_idtx_intra, enable_inter_ist, enable_chroma_dctonly, enable_ist, enable_cctx;
  // Its filter tools (L4 on) and what L6 needs of its prediction.
  int32_t lr_mask[2], df_par_bits_minus2, enable_cdef_on_skip_txfm, enable_ibp, cfl_ds_filter_index, enable_cfl_intra, enable_mhccp, tmvp_sample_step;
} OracleFrame;

typedef struct OracleCcso {
  int32_t enable, reuse, sb_reuse, bo_only, scale, quant, ext, edge_clf, band, ref_idx;
} OracleCcso;

typedef struct OracleFilters {
  int32_t lf_apply[4], lf_delta[4];  // luma vertical, horizontal, U, V
  int32_t cdef_enable, cdef_damping, cdef_n, cdef_on_skip, cdef_y[16], cdef_uv[16], cdef_units;
  int32_t ccso_frame, ccso_units[3];
  OracleCcso ccso[3];
  int8_t ccso_offset[3][64 * 16];
  int32_t lr_type[3], lr_size[3], lr_frame_filters[3], lr_temporal[3], lr_ref[3], lr_classes[3], lr_units[3];
  int32_t wiener_bytes;
} OracleFilters;

typedef struct OracleBlock {
  int16_t mi_row, mi_col;
  uint8_t w4, h4;
  uint8_t inter, skip_mode, skip_txfm;
  int16_t qindex;  // current_qindex
  // Intra
  uint8_t y_mode, joint, uv_mode, mrl, multi_mrl, cfl_idx, cfl_alpha_idx, mh_dir;
  int8_t angle_y, angle_uv, cfl_alpha_signs;
  // Inter (mode: PREDICTION_MODE, as y_mode)
  uint8_t ref_mv_idx[2], precision, amvd, interp, motion_mode, bawp[2], comp_type, comp_group, cwp;
  int8_t ref[2];
  int16_t mv[2][2];  // [list][row, col], 1/8 pel
  uint8_t fsc;
} OracleBlock;

typedef struct OracleTx {
  int16_t row, col;  // the block's top left in the plane, 4-sample units
  uint8_t plane, tx_size, cctx;  // cctx: of the U and V blocks at this place
  uint16_t eob, tx_type;  // tx_type: luma's, the secondary transform included
  uint16_t n_coef;
} OracleTx;

typedef struct OracleCoef {
  uint16_t pos;  // AVM layout
  int32_t level, dq;  // signed
} OracleCoef;
#pragma pack(pop)
