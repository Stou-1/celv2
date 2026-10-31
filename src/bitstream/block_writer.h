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
  // cul_level.
  int coeffs(int plane, int blk_row, int blk_col, TX_SIZE tx_size, const tran_low_t* qcoeff,
             int eob, TX_TYPE tx_type = DCT_DCT);

 private:
  void intra_modes(const BlockDecision& d);
  // decodemv.c read_ccso(): every CCSO unit's flag on, at its first block.
  void ccso();
  void delta_qindex();
  void mv_precision(MvSubpelPrecision p);
  double mv_precision_bits(MvSubpelPrecision p);
  // Bits of coding the luma mode at index mode_idx of mbmi->y_intra_mode_list (context ctx).
  double luma_mode_bits(int mode_idx, int ctx) const;
  int search_luma_mode(int ctx);
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
  void inter_tx_type(TX_SIZE tx_size, int eob, TX_TYPE tx_type);
  void intra_tx_type(TX_SIZE tx_size, TX_TYPE tx_type);
  void eob_symbols(int plane, TX_SIZE tx_size, int eob);
  int skip_levels_and_signs(int plane, TX_SIZE tx_size, TX_TYPE tx_type, const tran_low_t* qcoeff, int bob);
  int levels_and_signs(int plane, TX_SIZE tx_size, const TXB_CTX& txb_ctx,
                       const tran_low_t* qcoeff, int eob, int blk_row, int blk_col);

  CodecState& st_;
  MACROBLOCKD* xd_;
  SymbolLog* w_;
  uint8_t ref_mv_count_[MODE_CTX_REF_FRAMES] = {};  // DecoderCodingBlock::ref_mv_count
  IntraCost intra_cost_;
  RdCost rd_cost_;
  int64_t lambda_q8_ = 0;
};
