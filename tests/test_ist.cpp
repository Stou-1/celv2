// fwd_ist against AVM's normative inverse (av2_inv_stxfm): for random secondary coefficients s,
// fwd_ist(inv(s)) gives s back with an error under 3% of |s| (the 7-bit kernels are only about
// orthonormal; a wrong coefficient placement gives ~100%) at every tx size, primary type, kernel set,
// type and transposing mode, and the energy it reports dropped is under 3% of the input's.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "check.h"
#include "core/ist.h"

extern "C" {
#include "av2/common/av2_txfm.h"
#include "av2/common/idct.h"
#include "av2/common/scan.h"
#include "config/av2_rtcd.h"
}

static int round_trip() {
  av2_rtcd();
  av2_init_stxfm_kernels();
  uint32_t seed = 1;
  auto rnd = [&] { return static_cast<int>((seed = seed * 1103515245u + 12345u) >> 8); };
  double worst = 0;
  int cases = 0;
  for (int t = 0; t < TX_SIZES_ALL; t++) {
    const TX_SIZE tx = static_cast<TX_SIZE>(t);
    const int w = tx_size_wide[tx], h = tx_size_high[tx];
    const int n = (w > 32 ? 32 : w) * (h > 32 ? 32 : h);
    for (TX_TYPE primary : {DCT_DCT, ADST_ADST}) {
      if (primary == ADST_ADST && (w > 16 || h > 16)) continue;  // no ADST above 16
      const int cls = ist_size_class(tx, primary);
      const int sets = cls == 3 ? IST_REDUCED_SET_SIZE : IST_SET_SIZE;
      for (int set = 0; set < sets; set++)
        for (int stx = 1; stx <= 3; stx++)
          for (PREDICTION_MODE mode : {DC_PRED, H_PRED, D67_PRED, V_PRED}) {
            const int full_set = set + (primary == ADST_ADST ? IST_SET_SIZE : 0);
            tran_low_t c[1024] = {}, s[64] = {};
            const int16_t* scan = get_scan(tx, primary)->scan;
            for (int i = 0; i < ist_out_count(cls); i++) c[scan[i]] = s[i] = rnd() % 801 - 400;
            TxfmParam p{};
            p.tx_type = primary, p.sec_tx_set = full_set, p.sec_tx_type = stx, p.intra_mode = mode;
            p.tx_size = tx, p.bd = 10;
            av2_inv_stxfm(c, &p);
            double in = 0;
            for (int i = 0; i < n; i++) in += static_cast<double>(c[i]) * c[i];
            const double dropped = fwd_ist(c, tx, primary, mode, full_set, stx, 10);
            if (dropped > 0.03 * in + 64)
              std::fprintf(stderr, "tx %d primary %d set %d stx %d mode %d: dropped %.4f of %.0f\n", t, primary,
                           full_set, stx, mode, dropped / in, in);
            CHECK(dropped <= 0.03 * in + 64);
            double err = 0, norm = 0;
            for (int i = 0; i < ist_out_count(cls); i++) {
              const double d = c[scan[i]] - s[i];
              err += d * d, norm += static_cast<double>(s[i]) * s[i];
            }
            const double rel = std::sqrt(err / norm);
            worst = rel > worst ? rel : worst;
            if (rel >= 0.03)
              std::fprintf(stderr, "tx %d primary %d set %d stx %d mode %d: error %.3f\n", t, primary, full_set,
                           stx, mode, rel);
            CHECK(rel < 0.03);
            for (int i = ist_out_count(cls); i < n; i++) CHECK_EQ(c[scan[i]], 0);
            cases++;
          }
    }
  }
  std::printf("test_ist ok (%d cases, worst error %.4f)\n", cases, worst);
  return 0;
}

int main() { return run_test("round_trip", round_trip); }
