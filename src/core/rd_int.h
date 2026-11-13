// The RD pass's transform and quantizers in integers (C17 step 1, docs/superpowers/specs/2026-11-09-gpu-rd-pass-design.md),
// so a GPU port can give the same costs. Units, per superblock qindex:
// - dqv: av2_ac_quant_QTX(qindex, 0, 0, kQDepth), the step with 3 fractional bits (CelV2 sets no DC delta: the DC
//   step is the same);
// - J: 1/(1024 dqv^2) of a squared quantizer step in every transform size;
// - bits: 1/512 bit (AVM's cost unit).
// A trial's J is its distortion plus lam x its bits, lam = llround(2 w dqv^2) for a weight w in squared steps a bit.
#pragma once
#include <cstdint>

extern "C" {
#include "av2/common/enums.h"
}

namespace celv2::rdi {

constexpr int64_t kInf = int64_t{1} << 62;  // a block with no choice; sums stop there
inline int64_t add(int64_t a, int64_t b) { return a >= kInf - b ? kInf : a + b; }  // a, b in [0, kInf]

// lam for a weight of `w` squared steps a bit at qindex.
int64_t lam_of(int qindex, double w);
// llround(512 x bits).
inline int64_t bits512(double bits) { return static_cast<int64_t>(bits * 512 + (bits >= 0 ? 0.5 : -0.5)); }

// The coder's scale over AVM's integer gain, per coefficient (AVM's layout), Q14: fwd_dct()'s `scale` (an orthonormal
// DCT's times it; Encoder::luma_scale(): 1 over the inverse transform's DC gain) against AVM's matrices' row norms,
// which are off by up to 1% a pass on AC rows and 0.6% in all on 2:1 sizes, with the sqrt(2) of 2:1 sizes in it.
// Made on the CPU (gains_of()), uploaded as a table to a GPU. AVM's own gains, and rounding again after AVM's sqrt(2)
// step, cost 0.3-1.4% BD-rate on AIR, AM and chika (global PSNR-Y).
struct Gains {
  alignas(32) int g14[1024];
};
void gains_of(TX_SIZE t, double scale, Gains* g);  // scale > 0

// The low min(w, 32) x min(h, 32) coefficients of res's DCT2 in AVM's layout, as av2_fwd_txfm2d() makes them (AVM's
// matrices; AVM's total shift, less of it in the first pass), then to g's scale, rounded once (null: AVM's own scale
// and sqrt(2) step). *dropped: the squares of what a 64-point transform leaves out (0 below 64) at the same scale, each
// output by its rows' norms, as an orthonormal DCT's (fwd_dct()'s Parseval sum). res: stride `stride`, |r| < 2^8.
void fdct(const int16_t* res, int stride, TX_SIZE t, const Gains* g, int32_t* coeff, int64_t* dropped);
// The same by plain matrix products (the tests' reference).
void fdct_ref(const int16_t* res, int stride, TX_SIZE t, const Gains* g, int32_t* coeff, int64_t* dropped);

// A transform's cost: distortion in J against the decoder's dequantization (the dropped coefficients included) and
// the bits of its levels and end of block.
struct TxCost {
  int64_t d = 0, bits = 0;
};

// Quantized as the coder does, the RD pass's way: trellis coded quantization when tcq (the Viterbi over the 8
// states as tcq_levels(), kTcqBitWeight a bit), else quantize()'s levels and trailing-1 trim (kBitWeight); rd_zero:
// --rd-zero's whole-block choice. levels (signed) and dq (the decoder's dequantization), AVM's layout, when given.
TxCost quant_cost(const int32_t* coeff, int64_t dropped, TX_SIZE t, int qindex, bool tcq, bool rd_zero,
                  int32_t* levels = nullptr, int32_t* dq = nullptr);
// TCQ's levels by a plain scalar Viterbi (the tests' reference for quant_cost()'s vector one); returns the end of
// block.
int tcq_ref(const int32_t* coeff, TX_SIZE t, int qindex, bool rd_zero, int32_t* levels);
int tcq_vec(const int32_t* coeff, TX_SIZE t, int qindex, bool rd_zero, int32_t* levels);

// The sums of a transform's cost from its levels (end of block `end` in scan order).
TxCost sums(const int32_t* coeff, int64_t dropped, const int32_t* levels, int end, TX_SIZE t, int qindex, bool tcq,
            int32_t* dq = nullptr);

}  // namespace celv2::rdi
