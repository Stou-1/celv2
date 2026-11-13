// The real-cost trellis (C1 Q1, docs/superpowers/specs/2026-11-06-real-cost-trellis-design.md, row Qc): a transform block's
// levels by J = squared error + lam x the bits of the symbols BlockWriter::coeffs() writes for them, on given CDFs and
// contexts, in TCQ's eight states (or the plain path's one). Fixed point: bits in 1/512 (rounded per symbol), J in int64.
// The oracle's double version (tools/oracle/real_trellis.inc, X_ORACLE_Q) proved the rates against the written symbols;
// tests/test_real_trellis.cpp holds this one to it.
#pragma once
#include <cstdint>

extern "C" {
#include "av2/common/entropymode.h"
#include "av2/common/enums.h"
}

namespace celv2::real {

// A transform block about to be coded, with what its symbols' contexts need.
struct Block {
  const FRAME_CONTEXT* fc;  // the CDFs the rates are read from
  TX_SIZE tx;
  TX_TYPE type;
  int plane, pl_ctx, dc_sign_ctx, qindex, bd;  // pl_ctx: get_eob_plane_ctx(); qindex: CelV2's (kQDepth tables)
  bool tcq;
  int32_t flag[2];  // the all-zero flag's bits x 512: coded, all zero
  int64_t lam;      // a bit's weight: J = 2^20 e^2 + lam x bits512, e in coefficient units (lam_of())
};

// lam for `weight` squared DC steps a bit (Qa's: avmenc_weight()).
int64_t lam_of(int qindex, TX_SIZE tx, double weight);

// avmenc's weight of a bit in squared DC steps: RDMULT_FROM_Q2_NUM / _DEN, plane_rd_mult, rshift 2 and RDCOST's scale.
inline double avmenc_weight(bool inter, int plane) {
  static constexpr int plane_mult[2][2] = {{17, 13}, {16, 10}};
  return 84.0 / 32 * plane_mult[inter][plane > 0] / 4 / 128;
}

// llround(512 x -log2) of symbol v's probability in an AVM inverse CDF.
int32_t bits512(const avm_cdf_prob* cdf, int v);

// Levels of coeff (AVM's layout, the coded min(w,32) x min(h,32)) into q (signed) and their dequantization into dq (zeroed
// first, tx_size_2d[tx] values, clamped as the decoder), the end of block returned; *bits512: the bits of what it chose,
// the all-zero flag included.
int levels(const Block& b, const tran_low_t* coeff, tran_low_t* q, tran_low_t* dq, int64_t* bits512);

}  // namespace celv2::real
