#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "check.h"
#include "core/rd_int.h"

extern "C" {
#include "av2/common/common_data.h"
#include "av2/common/idct.h"
#include "av2/common/txb_common.h"
#include "av2/common/quant_common.h"
#include "av2/encoder/hybrid_fwd_txfm.h"
#include "config/av2_rtcd.h"
}
#include "core/quant.h"

namespace {

uint32_t g_rng = 12345;
int rnd(int lo, int hi) {  // [lo, hi]
  g_rng = g_rng * 1664525u + 1013904223u;
  return lo + static_cast<int>((g_rng >> 8) % static_cast<uint32_t>(hi - lo + 1));
}

// Residual patterns: random of a few ranges, flat at the limits, a checkerboard and ramps (the worst cases for the
// first pass's range).
void pattern(int k, int w, int h, int16_t* r) {
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      int v;
      switch (k % 7) {
        case 0: v = rnd(-255, 255); break;
        case 1: v = rnd(-8, 8); break;
        case 2: v = 255; break;
        case 3: v = -255; break;
        case 4: v = ((x + y) & 1) ? 255 : -255; break;
        case 5: v = (x * 510) / std::max(1, w - 1) - 255; break;
        default: v = rnd(0, 3) == 0 ? rnd(-255, 255) : 0; break;
      }
      r[y * w + x] = static_cast<int16_t>(v);
    }
}

// fdct() (AVM's AVX2 butterflies and the upper halves) equals plain matrix products, every size.
int fdct_matches_products() {
  int16_t r[64 * 64];
  int32_t a[1024], b[1024];
  for (int t = 0; t < TX_SIZES_ALL; t++) {
    const int w = tx_size_wide[t], h = tx_size_high[t], n = std::min(w, 32) * std::min(h, 32);
    for (int k = 0; k < 28; k++) {
      pattern(k, w, h, r);
      int64_t da = -1, db = -2;
      celv2::rdi::fdct(r, w, static_cast<TX_SIZE>(t), nullptr, a, &da);
      celv2::rdi::fdct_ref(r, w, static_cast<TX_SIZE>(t), nullptr, b, &db);
      int bad = 0;
      for (int i = 0; i < n; i++) bad += a[i] != b[i];
      if (bad || da != db) std::fprintf(stderr, "size %d (%dx%d) pattern %d: %d coefficients differ, dropped %lld / %lld\n", t, w, h, k, bad, static_cast<long long>(da), static_cast<long long>(db));
      CHECK_EQ(bad, 0);
      CHECK_EQ(da, db);
      if (w < 64 && h < 64) CHECK_EQ(da, 0);
    }
  }
  return 0;
}

// fdct()'s kept coefficients are within 3 of AVM's own forward transform (DCT_DCT, 8 bits) at every size.
int fdct_matches_avm() {
  av2_rtcd();
  int16_t r[64 * 64];
  int32_t a[1024];
  tran_low_t b[4096];
  for (int t = 0; t < TX_SIZES_ALL; t++) {
    const int w = tx_size_wide[t], h = tx_size_high[t], n = std::min(w, 32) * std::min(h, 32);
    for (int k = 0; k < 14; k++) {
      pattern(k, w, h, r);
      int64_t d;
      celv2::rdi::fdct(r, w, static_cast<TX_SIZE>(t), nullptr, a, &d);
      TxfmParam p{};
      p.tx_type = DCT_DCT, p.tx_size = static_cast<TX_SIZE>(t), p.bd = 8;
      av2_fwd_txfm(r, b, w, &p);
      int worst = 0;
      for (int i = 0; i < n; i++) worst = std::max(worst, std::abs(a[i] - b[i]));
      if (worst > 3) std::fprintf(stderr, "size %d (%dx%d) pattern %d: %d from AVM's\n", t, w, h, k, worst);
      CHECK(worst <= 3);  // only the first pass's rounding differs (its shift moved to the second)
    }
  }
  return 0;
}

// The dropped energy of 64-point transforms, at an orthonormal DCT's scale (gains for scale 1), is within 2% of an
// exact double DCT's, plus a thousandth of the block's energy (the kept coefficients' rounding).
int dropped_near_exact() {
  const double kPi = 3.14159265358979323846;
  int16_t r[64 * 64];
  int32_t c[1024];
  static celv2::rdi::Gains g;
  for (TX_SIZE t : {TX_64X64, TX_32X64, TX_64X32, TX_16X64, TX_64X16}) {
    const int w = tx_size_wide[t], h = tx_size_high[t];
    celv2::rdi::gains_of(t, 8.0, &g);  // an orthonormal DCT times 8: large enough to round finely
    for (int k = 0; k < 7; k++) {
      pattern(k, w, h, r);
      int64_t d;
      celv2::rdi::fdct(r, w, t, &g, c, &d);
      double all = 0, kept = 0;
      for (int i = 0; i < w * h; i++) all += static_cast<double>(r[i]) * r[i];
      for (int v = 0; v < 32 && v < h; v++)
        for (int u = 0; u < 32 && u < w; u++) {
          double s = 0;
          for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
              s += r[y * w + x] * std::cos(kPi * (2 * y + 1) * v / (2 * h)) * std::cos(kPi * (2 * x + 1) * u / (2 * w));
          s *= std::sqrt((v ? 2.0 : 1.0) / h) * std::sqrt((u ? 2.0 : 1.0) / w);
          kept += s * s;
        }
      const double exact = 64 * (all - kept);
      const double tol = 0.02 * exact + 0.001 * 64 * all + 64;
      if (std::abs(d - exact) > tol) std::fprintf(stderr, "size %d pattern %d: dropped %lld, exact %.0f\n", t, k, static_cast<long long>(d), exact);
      CHECK(std::abs(d - exact) <= tol);
    }
  }
  return 0;
}
// A flat block keeps only DC, and the dropped part of a 64-point one with no high frequencies is 0.
int flat_is_dc() {
  int16_t r[64 * 64];
  int32_t c[1024];
  for (TX_SIZE t : {TX_8X8, TX_16X32, TX_64X64, TX_64X32}) {
    const int w = tx_size_wide[t], h = tx_size_high[t], n = std::min(w, 32) * std::min(h, 32);
    for (int i = 0; i < w * h; i++) r[i] = 40;
    int64_t d = -1;
    celv2::rdi::fdct(r, w, t, nullptr, c, &d);
    CHECK(c[0] > 0);
    for (int i = 1; i < n; i++) CHECK_EQ(c[i], 0);
    CHECK_EQ(d, 0);
  }
  return 0;
}

// The vector Viterbi equals the scalar one: levels and end of block, with and without --rd-zero, on transforms of
// the patterns and on coefficients at the thresholds (ties).
int tcq_vec_matches_ref() {
  int16_t r[64 * 64];
  int32_t c[1024], la[1024], lb[1024];
  long long blocks = 0;
  for (int t = 0; t < TX_SIZES_ALL; t++) {
    const TX_SIZE tx = static_cast<TX_SIZE>(t);
    const int w = tx_size_wide[t], h = tx_size_high[t], n = std::min(w, 32) * std::min(h, 32);
    for (int qi : {1, 40, 100, 130, 190, 255}) {
      const int dqv = av2_ac_quant_QTX(qi, 0, 0, kQDepth), sh = 4 + av2_get_tx_scale(tx);
      for (int k = 0; k < 24; k++) {
        if (k < 14) {
          pattern(k, w, h, r);
          int64_t d;
          celv2::rdi::fdct(r, w, tx, nullptr, c, &d);
        } else {  // on and about the levels' boundaries: X2 = m dqv + small, m random
          for (int i = 0; i < n; i++) {
            const int m = rnd(0, 6) == 0 ? rnd(0, 40) : rnd(0, 3);
            const int64_t x2 = static_cast<int64_t>(m) * dqv + rnd(-1, 1);
            const int v = static_cast<int>(std::max<int64_t>(0, x2) >> sh);
            c[i] = rnd(0, 1) ? v : -v;
          }
        }
        for (int z = 0; z < 2; z++) {
          const int ea = celv2::rdi::tcq_vec(c, tx, qi, z != 0, la);
          const int eb = celv2::rdi::tcq_ref(c, tx, qi, z != 0, lb);
          int bad = 0;
          for (int i = 0; i < n; i++) bad += la[i] != lb[i];
          if (bad || ea != eb) std::fprintf(stderr, "size %d q %d case %d rd_zero %d: end %d / %d, %d levels differ\n", t, qi, k, z, ea, eb, bad);
          CHECK_EQ(ea, eb);
          CHECK_EQ(bad, 0);
          blocks++;
        }
      }
    }
  }
  std::printf("tcq: %lld blocks equal\n", blocks);
  return 0;
}

// All-zero coefficients cost the all-zero flag only; a lone DC level is coded with its end of block.
int costs_of_simple_blocks() {
  int32_t c[1024] = {};
  for (int tcq = 0; tcq < 2; tcq++) {
    const celv2::rdi::TxCost z = celv2::rdi::quant_cost(c, 0, TX_8X8, 120, tcq != 0, false);
    CHECK_EQ(z.d, 0);
    CHECK_EQ(z.bits, 512);
    c[0] = 2000;
    int32_t lv[1024];
    const celv2::rdi::TxCost one = celv2::rdi::quant_cost(c, 0, TX_8X8, 120, tcq != 0, false, lv);
    CHECK(lv[0] > 0);
    CHECK(one.bits > 512);
    c[0] = 0;
  }
  return 0;
}

}  // namespace

int main() {
  int fails = 0;
  fails += RUN(fdct_matches_products);
  fails += RUN(fdct_matches_avm);
  fails += RUN(dropped_near_exact);
  fails += RUN(flat_is_dc);
  fails += RUN(tcq_vec_matches_ref);
  fails += RUN(costs_of_simple_blocks);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
