// celv2::real::levels() (fixed point, a window of each path's levels) against the oracle's double version
// (tools/oracle/real_trellis.inc), whose rates X_ORACLE_QCHECK proved equal to the written symbols' bits.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "check.h"
#include "core/quant.h"
#include "core/real_trellis.h"

extern "C" {
#include "av2/common/av2_common_int.h"
#include "av2/common/entropy.h"
#include "av2/common/hr_coding.h"
#include "av2/common/idct.h"
#include "av2/common/quant_common.h"
#include "av2/common/scan.h"
#include "av2/common/txb_common.h"
}

#include "../tools/oracle/real_trellis.inc"  // namespace ::real, the double reference

namespace {

uint32_t g_rng = 12345;
int rnd(int lo, int hi) {  // [lo, hi]
  g_rng = g_rng * 1664525u + 1013904223u;
  return lo + static_cast<int>((g_rng >> 8) % static_cast<uint32_t>(hi - lo + 1));
}

struct Case {
  TX_SIZE tx;
  TX_TYPE type;
  int plane, qindex, dc_sign_ctx;
  bool tcq;
  std::vector<tran_low_t> coeff;
};

Case random_case(bool golomb) {
  static const TX_TYPE kTypes[] = {DCT_DCT, DCT_DCT, DCT_DCT, IDTX, V_DCT, H_DCT, ADST_DCT};
  Case k;
  k.tx = static_cast<TX_SIZE>(rnd(0, TX_SIZES_ALL - 1));
  k.plane = rnd(0, 2);
  k.type = k.plane ? DCT_DCT : kTypes[rnd(0, 6)];
  k.qindex = rnd(60, 255);
  k.dc_sign_ctx = rnd(0, 2);
  k.tcq = rnd(0, 3) != 0 && k.plane == 0 && tx_type_to_class[k.type] == TX_CLASS_2D;  // AVM's TCQ: luma 2D only
  const int n = get_txb_wide(k.tx) * get_txb_high(k.tx);
  const double step = av2_ac_quant_QTX(k.qindex, 0, 0, kQDepth) / static_cast<double>(1 << (QUANT_TABLE_BITS + av2_get_tx_scale(k.tx)));
  const SCAN_ORDER* const so = get_scan(k.tx, k.type);
  k.coeff.assign(1024, 0);
  const int last = rnd(0, n - 1), big = rnd(1, 4);
  for (int c = 0; c <= last; c++) {
    if (rnd(0, 3) == 0) continue;
    const double mag = step * (rnd(0, 1000) / 1000.0) * (c < 4 ? 6 * big : big) * (golomb ? 64 : 1);
    k.coeff[so->scan[c]] = static_cast<tran_low_t>(std::lround(mag)) * (rnd(0, 1) ? -1 : 1);
  }
  return k;
}

// Runs both on k; returns false when the levels differ, *rate_gap the bits' difference in 1/512.
bool same(const FRAME_CONTEXT* fc, const Case& k, int64_t* rate_gap, int* end) {
  const double weight = celv2::real::avmenc_weight(false, k.plane);
  const TX_SIZE txs_ctx = get_txsize_entropy_ctx(k.tx);
  const avm_cdf_prob* const flag = fc->txb_skip_cdf[0][txs_ctx][0];
  const int pl_ctx = get_eob_plane_ctx(k.plane, 0);
  const ::real::Block bd{fc, k.tx, k.type, k.plane, pl_ctx, k.dc_sign_ctx, k.qindex, 8, k.tcq, {::real::bits(flag, 0), ::real::bits(flag, 1)}, weight};
  const celv2::real::Block bi{fc, k.tx, k.type, k.plane, pl_ctx, k.dc_sign_ctx, k.qindex, 8, k.tcq,
                              {celv2::real::bits512(flag, 0), celv2::real::bits512(flag, 1)}, celv2::real::lam_of(k.qindex, k.tx, weight)};
  tran_low_t q1[1024], q2[1024], dq1[MAX_TX_SQUARE] = {}, dq2[MAX_TX_SQUARE] = {};
  double r1 = 0;
  int64_t r2 = 0;
  const int e1 = ::real::levels(bd, k.coeff.data(), q1, dq1, &r1);
  const int e2 = celv2::real::levels(bi, k.coeff.data(), q2, dq2, &r2);
  *rate_gap = std::llabs(r2 - std::llround(r1 * 512));
  *end = e1;
  const int n = get_txb_wide(k.tx) * get_txb_high(k.tx);
  return e1 == e2 && std::equal(q1, q1 + n, q2) && std::equal(dq1, dq1 + n, dq2);  // the oracle zeroes the coded n only
}

int matches_double(bool golomb) {
  auto cm = std::make_unique<AV2_COMMON>();
  std::memset(cm.get(), 0, sizeof(AV2_COMMON));
  auto fc = std::make_unique<FRAME_CONTEXT>();
  cm->fc = fc.get();
  int differ = 0, rate_off = 0, coded = 0;
  constexpr int kTrials = 20000;
  for (int t = 0; t < kTrials; t++) {
    cm->quant_params.base_qindex = rnd(0, 255);
    av2_default_coef_probs(cm.get());
    const Case k = random_case(golomb);
    int64_t gap = 0;
    int end = 0;
    if (!same(fc.get(), k, &gap, &end)) {
      differ++;
      continue;
    }
    coded += end > 0;
    // Each symbol's bits rounded once: at most half a 1/512 a symbol, a few symbols a level.
    if (gap > 2 * (end + 4)) rate_off++;
  }
  std::printf("  %s: %d of %d blocks differ (J ties within the rounding), %d coded, %d rates off\n", golomb ? "x64" : "plain",
              differ, kTrials, coded, rate_off);
  CHECK(differ * 200 < kTrials);  // under 0.5%
  CHECK(coded * 4 > kTrials);
  CHECK_EQ(rate_off, 0);
  return 0;
}

int plain_levels() { return matches_double(false); }
int golomb_levels() { return matches_double(true); }

}  // namespace

int main() {
  int fails = 0;
  fails += RUN(plain_levels);
  fails += RUN(golomb_levels);
  if (fails) std::printf("%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
