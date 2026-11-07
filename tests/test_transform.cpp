// fwd_dct against the direct double-precision DCT-II (the original implementation) at every
// transform size, on random and extreme residuals: each coefficient within +/-1. And TCQ's
// cost-only path (quantize_cost) against quantize()'s cost and end of block.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "check.h"
#include "core/quant.h"
extern "C" {
#include "av2/common/idct.h"
#include "av2/common/quant_common.h"
#include "av2/common/scan.h"
}
#include "core/transform.h"

static void reference(const int16_t* r, int w, int h, double scale, std::vector<long>& out) {
  const double pi = 3.14159265358979323846;
  const int cw = w > 32 ? 32 : w, ch = h > 32 ? 32 : h;
  auto basis = [&](int n, int k, int i) {
    return (k == 0 ? std::sqrt(1.0 / n) : std::sqrt(2.0 / n)) * std::cos(pi * (2 * i + 1) * k / (2.0 * n));
  };
  out.assign(static_cast<size_t>(cw) * ch, 0);
  for (int v = 0; v < ch; v++)
    for (int u = 0; u < cw; u++) {
      double s = 0;
      for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) s += basis(h, v, y) * basis(w, u, x) * r[y * w + x];
      out[v * cw + u] = std::lround(s * scale);
    }
}

// Run in a fresh process: threads released together call fwd_dct on a cold basis cache (as tile
// threads do on a process's first multi-tile frame); every result must equal a sequential rerun.
static int race_child() {
  const TX_SIZE txs[] = {TX_64X64, TX_32X32, TX_16X16, TX_8X8, TX_4X4};
  std::vector<int16_t> r(64 * 64);
  for (size_t i = 0; i < r.size(); i++) r[i] = static_cast<int16_t>((i * 37) % 511 - 255);
  constexpr int kThreads = 16;
  std::vector<std::vector<tran_low_t>> got(kThreads, std::vector<tran_low_t>(5 * 1024));
  std::atomic<int> ready{0};
  std::vector<std::thread> pool;
  for (int t = 0; t < kThreads; t++)
    pool.emplace_back([&, t] {
      ready++;
      while (ready < kThreads) {
      }
      for (int i = 0; i < 5; i++) fwd_dct(r.data(), 64, txs[i], 1.0, &got[t][i * 1024]);
    });
  for (auto& th : pool) th.join();
  std::vector<tran_low_t> want(5 * 1024);
  for (int i = 0; i < 5; i++) fwd_dct(r.data(), 64, txs[i], 1.0, &want[i * 1024]);
  for (int t = 0; t < kThreads; t++)
    if (got[t] != want) return 1;
  return 0;
}

static int cold_cache_threads_agree(const char* self) {
  int bad = 0;
  for (int i = 0; i < 40; i++)  // cmd.exe strips one pair of outer quotes, hence the extra pair
    bad += std::system(("\"\"" + std::string(self) + "\" --race-child\"").c_str()) != 0;
  if (bad) std::fprintf(stderr, "cold-cache race: %d of 40 processes disagreed\n", bad);
  CHECK_EQ(bad, 0);
  return 0;
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--race-child") return race_child();
  CHECK_EQ(cold_cache_threads_agree(argv[0]), 0);
  uint32_t seed = 12345;
  auto rnd = [&seed]() {
    seed = seed * 1664525u + 1013904223u;
    return static_cast<int>(seed >> 8);
  };
  int worst = 0;
  for (int tx = 0; tx < TX_SIZES_ALL; tx++) {
    const int w = tx_size_wide[tx], h = tx_size_high[tx];
    const double scale = 0.5 + tx * 0.07;
    std::vector<int16_t> r(static_cast<size_t>(w) * h);
    std::vector<long> ref;
    tran_low_t out[1024];
    for (int trial = 0; trial < 4; trial++) {
      for (auto& v : r)
        v = static_cast<int16_t>(trial == 0 ? 255 : trial == 1 ? (rnd() & 1 ? 255 : -255)
                                                               : rnd() % 511 - 255);
      fwd_dct(r.data(), w, static_cast<TX_SIZE>(tx), scale, out);
      reference(r.data(), w, h, scale, ref);
      for (size_t i = 0; i < ref.size(); i++) {
        const int d = std::abs(static_cast<int>(out[i] - ref[i]));
        if (d > worst) worst = d;
        if (d > 1) std::fprintf(stderr, "tx %d trial %d pos %zu: %d vs %ld\n", tx, trial, i, out[i], ref[i]);
        CHECK(d <= 1);
      }
    }
  }
  // --rd-zero: a lone coefficient 0.6 steps at scan position 1 is coded off, none on (J 0.50 against 0.41), and the plain
  // path's one at 1 step; cost-only agrees. Larger ones stay.
  for (const bool tcq : {true, false}) {
    tran_low_t coeff[64] = {}, q[64], dq[64];
    const double step = av2_ac_quant_QTX(130, 0, 0, kQDepth) / static_cast<double>(1 << (QUANT_TABLE_BITS + av2_get_tx_scale(TX_8X8)));
    coeff[get_scan(TX_8X8, DCT_DCT)->scan[1]] = static_cast<tran_low_t>((tcq ? 0.6 : 1.0) * step);
    CHECK(quantize(coeff, q, dq, TX_8X8, 130, 0, DCT_DCT, nullptr, tcq) > 0);
    g_rd_zero = true;
    int eob = -1;
    double full = 0;
    CHECK_EQ(quantize(coeff, q, dq, TX_8X8, 130, 0, DCT_DCT, &full, tcq), 0);
    CHECK_EQ(quantize_cost(coeff, TX_8X8, 130, DCT_DCT, &eob, tcq), full);
    CHECK_EQ(eob, 0);
    coeff[get_scan(TX_8X8, DCT_DCT)->scan[1]] = static_cast<tran_low_t>((tcq ? 0.9 : 1.2) * step);
    CHECK(quantize(coeff, q, dq, TX_8X8, 130, 0, DCT_DCT, nullptr, tcq) > 0);
    g_rd_zero = false;
  }
  // --rd-zero's one pass against reruns of the trellis: its levels, dequantization and J are the least over forcing each end
  // above the threshold and all zero. Cost-only with eob null is the same J, with eob given the same end. Where two ends
  // come within rounding of each other only J is compared. Inside one end paths can tie exactly (a coefficient 0.75 steps
  // is as near in Q0 as in Q1), so levels are compared by what they cost, walked along the states as the decoder does.
  int shorter = 0, zeroed = 0;
  for (int tx = 0; tx < TX_SIZES_ALL; tx++) {
    const int n = std::min<int>(tx_size_wide[tx], 32) * std::min<int>(tx_size_high[tx], 32);
    const double scale = static_cast<double>(1 << (QUANT_TABLE_BITS + av2_get_tx_scale(static_cast<TX_SIZE>(tx))));
    tran_low_t coeff[1024], q[1024], dq[1024], fq[1024], fdq[1024], bq[1024], bdq[1024];
    for (int trial = 0; trial < 8; trial++)
      for (int qindex : {40, 130, 220})
        for (double psy : {0.0, 2.0}) {
          const double step = av2_ac_quant_QTX(qindex, 0, 0, kQDepth) / scale;
          for (int i = 0; i < n; i++) {
            const int r = rnd();
            const double steps = (r >> 8) % 1000 * (trial < 4 ? 0.0012 : 0.004);  // mostly under one step, or a few
            coeff[i] = trial == 0 || r % (2 + 3 * trial) ? 0 : static_cast<tran_low_t>((r & 1 ? steps : -steps) * step);
          }
          double least = 1e300, next = 1e300;
          int prep = 0, best = 0;
          for (int end = n; end >= 0; end--) {
            const double j = tcq_end_cost(coeff, fq, fdq, static_cast<TX_SIZE>(tx), qindex, DCT_DCT, end, psy);
            if (j < 0) continue;
            if (!prep) prep = end;
            if (j < least) {
              next = least, least = j, best = end;
              std::copy(fq, fq + n, bq), std::copy(fdq, fdq + n, bdq);
            } else {
              next = std::min(next, j);
            }
          }
          g_rd_zero = true;
          double got = 0;
          const int e = quantize(coeff, q, dq, static_cast<TX_SIZE>(tx), qindex, 0, DCT_DCT, &got, true, 8, psy);
          CHECK(std::abs(got - least) < 1e-9);
          if (next - least > 1e-9) {
            CHECK_EQ(e, best);
            const SCAN_ORDER* const so = get_scan(static_cast<TX_SIZE>(tx), DCT_DCT);
            bool dequantized = true;
            auto path_j = [&](const tran_low_t* l, const tran_low_t* d) {
              static const uint8_t next_state[8][2] = {{0, 4}, {4, 0}, {1, 5}, {5, 1}, {6, 2}, {2, 6}, {7, 3}, {3, 7}};
              double j = 0;
              int state = 0;
              for (int c = e - 1; c >= 0; c--) {
                const int pos = so->scan[c], level = std::abs(l[pos]);
                const double s = pos ? step : av2_dc_quant_QTX(qindex, 0, 0, kQDepth) / scale;
                const double rec = !level ? 0 : state & 2 ? level - 0.5 : level, miss = std::abs(std::abs(coeff[pos]) / s - rec);
                j += miss * miss + (pos ? psy * miss : 0) + kTcqBitWeight * coeff_level_bits(level);
                dequantized = dequantized && std::abs(std::abs(d[pos]) - rec * s) <= 1;
                state = next_state[state][level & 1];
              }
              return j;
            };
            CHECK(std::abs(path_j(q, dq) - path_j(bq, bdq)) < 1e-9);
            CHECK(dequantized);
          }
          CHECK_EQ(quantize_cost(coeff, static_cast<TX_SIZE>(tx), qindex, DCT_DCT, nullptr, true, psy), got);
          int eob = -1;
          quantize_cost(coeff, static_cast<TX_SIZE>(tx), qindex, DCT_DCT, &eob, true, psy);
          CHECK_EQ(eob, e);
          g_rd_zero = false;
          shorter += e && e < prep, zeroed += !e && prep;
        }
  }
  CHECK(shorter > 0 && zeroed > 0);
  // TCQ's cost-only path against quantize(): coefficients mostly small (many zeros past the end of
  // block), some large, at a few qindex.
  for (int tx = 0; tx < TX_SIZES_ALL; tx++) {
    const int n = std::min<int>(tx_size_wide[tx], 32) * std::min<int>(tx_size_high[tx], 32);
    tran_low_t coeff[1024], q[1024], dq[1024];
    for (int trial = 0; trial < 6; trial++)
      for (int qindex : {40, 130, 220}) {
        for (int i = 0; i < n; i++) {
          const int r = rnd();
          coeff[i] = trial == 0 ? 0 : r % 7 == 0 ? r % 4001 - 2000 : r % 5 == 0 ? r % 81 - 40 : 0;
        }
        double full = 0;
        int eob = -1;
        const int e = quantize(coeff, q, dq, static_cast<TX_SIZE>(tx), qindex, 0, DCT_DCT, &full, true);
        const double fast = quantize_cost(coeff, static_cast<TX_SIZE>(tx), qindex, DCT_DCT, &eob, true);
        CHECK_EQ(eob, e);
        CHECK_EQ(fast, full);
      }
  }
  std::printf("test_transform ok (max diff %d)\n", worst);
  return 0;
}
