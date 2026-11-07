#include "core/ist.h"

#include <cstring>

extern "C" {
#include "av2/common/av2_txfm.h"
#include "av2/common/scan.h"
#include "av2/common/secondary_tx.h"
}

int ist_size_class(TX_SIZE tx, TX_TYPE primary) {
  const int w = tx_size_wide[tx], h = tx_size_high[tx];
  return w == 8 && h == 8 && primary == DCT_DCT ? 1 : w >= 8 && h >= 8 ? (primary == DCT_DCT ? 2 : 3) : 0;
}

int ist_out_count(int size_class) {
  return size_class == 0 ? IST_4x4_HEIGHT : size_class == 1 ? IST_8x8_HEIGHT_RED
                                          : size_class == 3 ? IST_ADST_NZ_CNT : IST_8x8_HEIGHT;
}

// hybrid_fwd_txfm.c av2_fwd_stxfm() and av2_fwd_txfm2d.c fwd_stxfm_c(), on the int32 copies of
// the kernels (av2_init_stxfm_kernels(), called by CodecState).
double fwd_ist(tran_low_t* coeff, TX_SIZE tx, TX_TYPE primary, PREDICTION_MODE mode, int set, int stx, int bd) {
  const int width = tx_size_wide[tx] <= 32 ? tx_size_wide[tx] : 32;
  const int height = tx_size_high[tx] <= 32 ? tx_size_high[tx] : 32;
  const int log2width = tx_size_wide_log2[tx];
  const bool big = width >= 8 && height >= 8;
  const PREDICTION_MODE m = AVMMIN(mode, SMOOTH_H_PRED);
  const bool transpose = m == H_PRED || m == D157_PRED || m == D67_PRED || m == SMOOTH_H_PRED;
  const int16_t* const in_scan = big ? (transpose ? stx_scan_orders_transpose_8x8 : stx_scan_orders_8x8)[log2width - 2]
                                     : (transpose ? stx_scan_orders_transpose_4x4 : stx_scan_orders_4x4)[log2width - 2];
  const int in_n = big ? IST_8x8_WIDTH : IST_4x4_WIDTH;
  const int16_t* const map = big ? coeff8x8_mapping[set * 3 + stx - 1] : nullptr;
  int32_t in[IST_8x8_WIDTH];
  for (int i = 0; i < in_n; i++) in[i] = coeff[in_scan[big ? map[i] : i]];
  const int cls = ist_size_class(tx, primary);
  const int out_n = ist_out_count(cls);
  const int32_t* kernel = big ? ist_8x8_kernel_int32[set][stx - 1][0] : ist_4x4_kernel_int32[set][stx - 1][0];
  double before = 0, after = 0;
  for (int i = 0; i < width * height; i++) before += static_cast<double>(coeff[i]) * coeff[i];
  std::memset(coeff, 0, sizeof(tran_low_t) * width * height);
  const int16_t* const scan = get_scan(tx, primary)->scan;
  const int lim = 1 << (7 + bd);  // clamp_value(x, 8 + bd)
  for (int j = 0; j < out_n; j++, kernel += in_n) {
    int64_t s = 0;
    for (int i = 0; i < in_n; i++) s += static_cast<int64_t>(in[i]) * kernel[i];
    int v = static_cast<int>(s >= 0 ? (s + 64) >> 7 : -((-s + 64) >> 7));
    v = v < -lim ? -lim : v > lim - 1 ? lim - 1 : v;
    coeff[scan[j]] = v;
    after += static_cast<double>(v) * v;
  }
  return before > after ? before - after : 0;
}
