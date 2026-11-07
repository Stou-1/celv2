// Block-level symbol writer: each method mirrors one AVM decoder reader (decodeframe.c,
// decodemv.c, decodetxb.c), writing the symbol the reader would read and making the same
// state updates the reader makes.
#pragma once
#include <cstdint>
#include <functional>

#include "bitstream/symbol_log.h"
#include "gpu/decision_map.h"
#include "state/codec_state.h"

extern "C" {
#include "av2/common/txb_common.h"
}

class BlockWriter {
 public:
  // xd: the block state of the tile being written (CodecState::xd(t)).
  BlockWriter(CodecState& st, MACROBLOCKD* xd, SymbolLog* w) : st_(st), xd_(xd), w_(w) {}
  SymbolLog* log() { return w_; }

  // read_partition(): codes `wanted` when the syntax allows it; where the partition is forced or
  // `wanted` is not allowed, codes the normative/nearest allowed one. Returns what was coded.
  PARTITION_TYPE partition(int mi_row, int mi_col, BLOCK_SIZE bsize, const PARTITION_TREE* ptree,
                           PARTITION_TYPE wanted);
  // SATD of the luma prediction of the current block with AV2 joint luma mode `joint` (mode and
  // angle delta), predicted by the encoder from the reconstruction.
  using IntraCost = std::function<int64_t(int joint)>;
  // With a cost, intra blocks take the joint mode of least SATD * 256 + lambda_q8 * bits (the
  // bits from the current CDFs) instead of the decision's mode; nullptr goes back to the decision.
  void set_intra_search(IntraCost cost, int64_t lambda_q8) { intra_cost_ = std::move(cost), lambda_q8_ = lambda_q8; }
  void set_intra_lambda(int64_t lambda_q8) { lambda_q8_ = lambda_q8; }  // per superblock (delta qindex)
  // The current block's luma with joint mode `joint` (and mbmi's reference line) as the coder would
  // quantize it (squared error in quantizer steps plus quantize()'s weight per level bit). With it,
  // the SATD search's best few joint modes, each with every reference line where it is directional,
  // are re-scored by it plus the RdCost weight per mode / line bit.
  using IntraRd = std::function<double(int joint)>;
  void set_intra_rd(IntraRd rd) { intra_rd_ = std::move(rd); }
  // SATD of both chroma predictions of the current intra block with chroma mode uv. With a cost (and
  // the luma search on), the chroma mode is the one of least SATD * 256 + lambda_q8 * bits instead of
  // the luma mode's counterpart.
  using UvCost = std::function<int64_t(UV_PREDICTION_MODE uv)>;
  void set_uv_search(UvCost cost) { uv_cost_ = std::move(cost); }
  // --cfl-rd: the chroma shortlist (the SATD's 4 best modes and the CfL / MHCCP types) re-scored by
  // this quantized U + V cost plus kRdBitWeight per mode bit.
  using UvRd = std::function<double(UV_PREDICTION_MODE uv)>;
  void set_uv_rd(UvRd rd) { uv_rd_ = std::move(rd); }
  // --screen: the current block's luma palette candidate, 2-8 colours ascending into colors and its
  // colour index map (plane block width, edges repeated as the decoder does) into
  // xd->plane[0].color_index_map; returns the colour count, 0 for none. With it (and the RD search),
  // where the frame allows palettes the candidate competes with the joint modes as DC_PRED.
  using PaletteSource = std::function<int(uint16_t* colors)>;
  void set_palette_source(PaletteSource s) { palette_src_ = std::move(s); }
  // decode_color_map_tokens(): a palette block's colour index map, after its mode info.
  void palette_tokens();
  // --screen IntraBC: the current block's candidate block vectors (1/8 pel, integer, valid), at most
  // max into dvs, and the RD cost of one (its luma copy's quantization cost, as IntraRd's). With them,
  // where the frame and block allow IntraBC, an intra decision's block takes the candidate of least
  // cost plus kRdBitWeight per bit when it beats the decided mode's IntraRd.
  using IntraBcSearch = std::function<int(MV* dvs, int max)>;
  using IntraBcRd = std::function<double(MV dv)>;
  void set_intrabc_search(IntraBcSearch s, IntraBcRd rd) { ibc_search_ = std::move(s), ibc_rd_ = std::move(rd); }
  // decode_partition() at a superblock root: the flag of each restoration unit whose top left lies
  // in it, deferred to CodecState::lr_flags() (indexed by unit).
  void lr_units(int mi_row, int mi_col, BLOCK_SIZE bsize);
  // RD cost of the current block's prediction as its mode info stands (one reference or two): its
  // luma residual's quantization cost at the largest transform size, in squared quantizer steps
  // plus weighted level bits. Past `bound` it may stop early, returning a value above the bound.
  using RdCost = std::function<double(double bound)>;
  // With a cost, an inter block's mode is checked at coding time by that cost plus kRdBitWeight per
  // mode bit: a single-reference block whose MV no MV-stack candidate repeats takes a candidate
  // (NEARMV) when it beats the decision MV with its estimated difference bits, and where the frame
  // allows skip mode a skip-mode candidate replaces the decision when it beats it; nullptr codes
  // the decision.
  void set_inter_search(RdCost rd) { rd_cost_ = std::move(rd); }
  // --warp: the decision engine's MV (1/8 pel) of reference r for the 8x8 cell at luma (x, y), false
  // where it has none. Read only inside the current block (its rows are decided).
  using MotionField = std::function<bool(int r, int x, int y, MV* mv)>;
  void set_motion_field(MotionField f) { field_ = std::move(f); }
  // The CfL / MHCCP estimates' SATD factor: 1.2 from the source luma, 1 from the reconstruction (--cfl-rd).
  void set_cfl_factor(double f) { cfl_factor_ = f; }
  // read_intra_frame_mode_info() for a keyframe block (xd->mi[0] already set up).
  void intra_frame_mode_info(const BlockDecision& d);
  // read_inter_frame_mode_info() for a block of an inter frame.
  void inter_frame_mode_info(const BlockDecision& d);
  // read_tx_partition(): the luma transform partition of the max-size transform unit at (blk_row,
  // blk_col) (in 4x4 units; intra: 0, 0); sets mbmi's partition, txb_pos and tx_size as the decoder.
  void tx_partition(TX_SIZE max_tx, int blk_row, int blk_col, TX_PARTITION_TYPE p);
  // State the decoder sets in av2_read_sig_txtype() before prediction (wide-angle mapping).
  void pre_txb(int plane, TX_SIZE tx_size);
  // av2_read_coeffs_txb_facade(): all-zero flag, eob, tx type, levels, signs, context updates.
  // qcoeff is in AVM layout, eob counts in the tx type's scan. tx_type: inter luma's chosen type
  // (coded when eob > 0); elsewhere DCT_DCT (or what the decoder derives). Returns the decoder's
  // cul_level. cctx: the U block's cross-chroma rotation (coded after its type where CCTX is allowed).
  int coeffs(int plane, int blk_row, int blk_col, TX_SIZE tx_size, const tran_low_t* qcoeff,
             int eob, TX_TYPE tx_type = DCT_DCT, CctxType cctx = CCTX_NONE);

 private:
  void intra_modes(const BlockDecision& d);
  // decodemv.c read_ccso(): every CCSO unit's flag on, at its first block.
  void ccso();
  void delta_qindex();
  void mv_precision(MvSubpelPrecision p);
  double mv_precision_bits(MvSubpelPrecision p);
  // Bits of coding the luma mode at index mode_idx of mbmi->y_intra_mode_list (context ctx).
  double luma_mode_bits(int mode_idx, int ctx) const;
  // The joint luma mode; with intra_rd_ also the reference line in rd_mrl_ / rd_multi_.
  int search_luma_mode(int ctx);
  // use_intrabc where the frame and the block allow IntraBC: ibc_ (choose_intrabc()'s) or 0.
  void use_intrabc(BLOCK_SIZE bsize);
  // Whether the intra decision d's block takes IntraBC (ibc_, with its vector and how it is coded).
  bool choose_intrabc(const BlockDecision& d, BLOCK_SIZE bsize);
  // read_intrabc_info() of an IntraBC block (the mode info set as the decoder sets it).
  void intrabc_info();
  // write_palette_mode_info() / write_palette_colors_y() (mbmi's palette, mode DC_PRED).
  void palette_mode_info();
  // Bits of the palette colours (cache flags, then the rest delta coded); with write, written too.
  double palette_colors(bool write);
  // The colour index map's bits in its cheaper direction (where both are allowed); with write, written.
  double color_map(bool write);
  double map_pass(int direction, bool write);
  // skip_idx >= 0: skip mode with that DRL index (decided by choose_skip_mode()).
  void inter_block_mode_info(const BlockDecision& d, bool force_near, int skip_idx);
  // The skip-mode DRL index to code the inter decision d with, or -1 (see set_inter_search).
  int choose_skip_mode(const BlockDecision& d);
  void skip_mode_info(int skip_idx);
  void inter_block_tail();
  // read_compound_ref() for references r0 < r1 (BlockWriter::compound_codable).
  void compound_ref(int r0, int r1);
  bool compound_codable(int r0, int r1) const;
  // read_mv() magnitudes of both differences, then their signs (assign_mv's order).
  void compound_mvs(const MV want[2], const int_mv ref_mv[2]);
#ifdef CELV2_ORACLE  // tools/oracle/block_import.inc
  long long block_area() const;
  void import_single(const BlockDecision& d, MV_REFERENCE_FRAME ref_frame, MV gmv, MV wanted, int& near_idx, int& new_idx,
                     bool& global, bool& amvd, MvSubpelPrecision& new_prec);
  void import_bawp(const OracleBlock& b);
  void import_magnitude(MV diff);
  void import_compound(const BlockDecision& d, int r0, MV_REFERENCE_FRAME ref_frame, int16_t* inter_mode_ctx,
                       int_mv (*ref_mvs)[MAX_MV_REF_CANDIDATES], int* pts0, int* pts0_inref);
  const OracleBlock* imp_ = nullptr;  // the current block's, for inter_block_tail()
  void import_cdef();
  void import_lr_unit(int plane, int unit);

 public:
  // X_ORACLE_QCHECK, after coeffs(): the bits of a transform block's flag, end of block, levels and signs on fixed CDFs.
  double oracle_bits(int plane, int blk_row, int blk_col, TX_SIZE tx, const tran_low_t* q, int eob, FRAME_CONTEXT& fc, const TXB_CTX& ctx);

 private:
#endif
  void inter_tx_type(TX_SIZE tx_size, int eob, TX_TYPE tx_type);
  void intra_tx_type(TX_SIZE tx_size, TX_TYPE tx_type);
  // The secondary transform (IST) of a luma block, after its primary type (tx_type: both).
  void sec_tx_type(TX_SIZE tx_size, int eob, TX_TYPE tx_type);
  void eob_symbols(int plane, TX_SIZE tx_size, int eob);
  int skip_levels_and_signs(int plane, TX_SIZE tx_size, TX_TYPE tx_type, const tran_low_t* qcoeff, int bob);
  int levels_and_signs(int plane, TX_SIZE tx_size, const TXB_CTX& txb_ctx,
                       const tran_low_t* qcoeff, int eob, int blk_row, int blk_col);

  CodecState& st_;
  MACROBLOCKD* xd_;
  SymbolLog* w_;
  uint8_t ref_mv_count_[MODE_CTX_REF_FRAMES] = {};  // DecoderCodingBlock::ref_mv_count
  IntraCost intra_cost_;
  UvCost uv_cost_;
  UvRd uv_rd_;
  RdCost rd_cost_;
  MotionField field_;
  // A warp-delta block's choice: warp_ref_idx k, warp_precision_idx p, steps d on the zoom and
  // rotation terms (index 0 only), the coded MV.
  struct WarpChoice {
    int k = 0, p = 0, d[2] = {0, 0};
    MV mv{};
  };
  bool choose_warp(int ref, MV c, MV wanted, int new_idx, int16_t mode_ctx, double* best, WarpChoice* out);
  int64_t lambda_q8_ = 0;
  double cfl_factor_ = 1.2;
  IntraRd intra_rd_;
  PaletteSource palette_src_;
  IntraBcSearch ibc_search_;
  IntraBcRd ibc_rd_;
  struct IntraBc {
    bool on = false;
    MV dv{};
    int mode = 0, drl = 0;  // intrabc_mode (1: the predictor itself), intrabc_drl_idx
    MV ref{};               // the predictor at drl, at integer precision
  } ibc_;
  int rd_mrl_ = 0, rd_multi_ = 0;
};
