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
int quantize(const tran_low_t* coeff, tran_low_t* qcoeff, tran_low_t* dqcoeff, TX_SIZE tx,
             int qindex, int plane, TX_TYPE tx_type = DCT_DCT, double* cost = nullptr, bool tcq = false,
             int bd = 8);
// quantize()'s cost without writing levels or dequantizing: to compare ways of coding a residual.
// eob (optional) gets quantize()'s end of block.
double quantize_cost(const tran_low_t* coeff, TX_SIZE tx, int qindex, TX_TYPE tx_type = DCT_DCT, int* eob = nullptr,
                     bool tcq = false);
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
