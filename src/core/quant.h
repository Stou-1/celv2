// Quantization (encoder side: a simple rate-distortion choice of each level) and the normative
// dequantization (same formula as decodetxb.c av2_read_coeffs_txb with TCQ and quant matrices off).
#pragma once

extern "C" {
#include "av2/common/av2_common_int.h"
}

// coeff/qcoeff/dqcoeff use the AVM coefficient layout of tx (width min(w,32), height min(h,32)).
// Returns eob in `tx_type`'s scan order (0 if all zero). cost (optional): the squared error in
// quantizer steps plus the RDOQ weight per estimated bit, to compare transform types of a block.
// tcq: the frame codes this block with trellis coded quantization (tcq_enable()): levels by a
// search over its states, dqcoeff dequantized along them.
// qindex: as coded (10 bits: 8-bit scale + 48); bd: the bit depth, for the decoder's clamp of dqcoeff.
// psy (--psy-trellis): each AC coefficient also costs psy * | |e| - |l| |, e and its reconstruction l
// in steps, so small levels stop dropping to 0 as readily.
int quantize(const tran_low_t* coeff, tran_low_t* qcoeff, tran_low_t* dqcoeff, TX_SIZE tx,
             int qindex, int plane, TX_TYPE tx_type = DCT_DCT, double* cost = nullptr, bool tcq = false,
             int bd = 8, double psy = 0);
// quantize()'s cost without writing levels or dequantizing: to compare ways of coding a residual.
// eob (optional) gets quantize()'s end of block.
double quantize_cost(const tran_low_t* coeff, TX_SIZE tx, int qindex, TX_TYPE tx_type = DCT_DCT, int* eob = nullptr,
                     bool tcq = false, double psy = 0);
// --rd-zero on this thread's frame: zeroing residual by RD (docs/superpowers/specs/2026-11-06-rd-zeroing-design.md). TCQ takes any
// end of block above the threshold or all zero in one pass of the trellis, the cost holding the end's term; the plain path
// weighs a block's levels against none.
extern thread_local bool g_rd_zero;
// The one pass's oracle, for the tests: J of the trellis forced to this end of block, its levels and their dequantization
// into qcoeff, dqcoeff (end 0: all zero). -1 where the coefficient at the end is under the threshold of a level.
double tcq_end_cost(const tran_low_t* coeff, tran_low_t* qcoeff, tran_low_t* dqcoeff, TX_SIZE tx, int qindex, TX_TYPE tx_type, int end, double psy);
// RDOQ weight of a bit in quantize()'s costs, in squared quantizer steps (0.1 / 0.2 / 0.3 / 0.4 gave
// -5.6 / -7.1 / -3.8 / +0.6% BD-rate at qindex 30-75 on anime clips, and 0.2 gave -2.9% at qindex 45-180).
constexpr double kBitWeight = 0.2;
// The trellis's weight of a bit, in squared steps (anime clips, against TCQ off: 0.2 / 0.1 / 0.05 /
// 0.03 / 0 gave +0.1 / -4.2 / -5.5 / -5.2 / -0.2% BD-rate at qindex 45-180, -0.5 / -7.5 / -10.6 /
// -11.1 / -6.7% at 30-75; a 1080p anime -1.0 / -4.0 / -4.2 / -3.3 / 0%).
constexpr double kTcqBitWeight = 0.05;
// Rough bits of a coefficient level l >= 0, sign included (what quantize() weighs).
double coeff_level_bits(int l);
// A 64-point transform codes only the low 32 frequencies (fwd_dct() leaves out the rest): their
// energy is lost, as squared error in quantizer steps like quantize_cost()'s (0 below 64). res: the
// residual fwd_dct() transformed with `scale` into coeff.
double dropped_cost(const int16_t* res, int stride, TX_SIZE tx, double scale, const tran_low_t* coeff, int qindex);
// The decisions' lambda * 256 (rounded): 0.002 * qstep^2 * (qstep(120) / qstep)^0.8, qstep = AVM AC
// quantizer step in 8-bit pixels (qindex on the 8-bit scale).
// Costs are SATD * 256 + lambda_q8 * bits.
int64_t lambda_q8(int qindex);
// Normative dequantization of one level (dqv = DC or AC dequant step).
tran_low_t dequant_coeff(int level, int dqv, TX_SIZE tx, int bd = 8);
// av2_{dc,ac}_quant_QTX() differ by bit depth only in how far they clamp qindex (one table serves
// every depth): the widest clamp, so a coded qindex of any depth reads its own step.
constexpr avm_bit_depth_t kQDepth = AVM_BITS_12;
