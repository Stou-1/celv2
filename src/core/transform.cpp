#include "core/transform.h"

#include <array>
#include <cmath>
#include <immintrin.h>
#include <stdexcept>
#include <vector>

extern "C" {
#include "av2/common/av2_txfm.h"
#include "av2/common/common_data.h"
#include "av2/common/txb_common.h"
}

namespace {

// Lee's odd-half factors 1 / (2 cos(pi (2i+1) / (2n))), i < n/2, for n = 2..64, by log2 n.
const std::array<std::array<float, 32>, 7> kLee = [] {
  std::array<std::array<float, 32>, 7> t{};
  const double pi = 3.14159265358979323846;
  for (int log2 = 1; log2 < 7; log2++) {
    const int sz = 1 << log2;
    for (int i = 0; i < sz / 2; i++) t[log2][i] = static_cast<float>(0.5 / std::cos(pi * (2 * i + 1) / (2.0 * sz)));
  }
  return t;
}();

constexpr int log2_of(int n) { return n <= 1 ? 0 : 1 + log2_of(n / 2); }

// Unnormalised DCT-II X[k] = sum_i x[i] cos(pi (2i+1) k / (2n)) of `lanes` signals at once, in
// place: x[i * lanes + l] is sample i of signal l. Lee's recursion: the even outputs are the
// half-length DCT of a[i] = x[i] + x[n-1-i]; the odd ones are X[2m+1] = C[m] + C[m+1], C being the
// half-length DCT of (x[i] - x[n-1-i]) / (2 cos(pi (2i+1) / (2n))). Every step loops over the
// lanes, so it vectorises; the size is a template argument so the small ones unroll. tmp holds
// 2 * n * lanes floats.
template <int N>
void dct_lanes(float* x, int lanes, float* tmp) {
  if constexpr (N == 2) {  // X[0] = x0 + x1, X[1] = (x0 - x1) / (2 cos(pi/4)); saves the n = 1 calls
    const float f = kLee[1][0];
    for (int l = 0; l < lanes; l++) {
      const float a = x[l], b = x[lanes + l];
      x[l] = a + b;
      x[lanes + l] = (a - b) * f;
    }
  } else {
    constexpr int h = N / 2;
    const float* f = kLee[log2_of(N)].data();
    float* a = tmp;
    float* c = tmp + h * lanes;
    for (int i = 0; i < h; i++) {
      const float* p = x + i * lanes;
      const float* q = x + (N - 1 - i) * lanes;
      float* pa = a + i * lanes;
      float* pc = c + i * lanes;
      const float fi = f[i];
      for (int l = 0; l < lanes; l++) {
        pa[l] = p[l] + q[l];
        pc[l] = (p[l] - q[l]) * fi;
      }
    }
    float* const next = tmp + N * lanes;
    dct_lanes<h>(a, lanes, next);
    dct_lanes<h>(c, lanes, next);
    for (int m = 0; m < h; m++) {
      const float* pa = a + m * lanes;
      const float* c0 = c + m * lanes;
      float* e = x + 2 * m * lanes;
      float* o = e + lanes;
      for (int l = 0; l < lanes; l++) e[l] = pa[l];
      if (m + 1 < h) {
        const float* c1 = c0 + lanes;
        for (int l = 0; l < lanes; l++) o[l] = c0[l] + c1[l];
      } else {
        for (int l = 0; l < lanes; l++) o[l] = c0[l];
      }
    }
  }
}

void dct_lanes(float* x, int n, int lanes, float* tmp) {
  switch (n) {
    case 4: return dct_lanes<4>(x, lanes, tmp);
    case 8: return dct_lanes<8>(x, lanes, tmp);
    case 16: return dct_lanes<16>(x, lanes, tmp);
    case 32: return dct_lanes<32>(x, lanes, tmp);
    case 64: return dct_lanes<64>(x, lanes, tmp);
  }
  throw std::invalid_argument("dct_lanes: size");
}

// d[c * dst_stride + r] = s[r * src_stride + c] for an 8x8 block.
void transpose8(const float* s, int src_stride, float* d, int dst_stride) {
  __m256 r[8], t[8];
  for (int i = 0; i < 8; i++) r[i] = _mm256_loadu_ps(s + i * src_stride);
  for (int i = 0; i < 8; i += 2) t[i] = _mm256_unpacklo_ps(r[i], r[i + 1]), t[i + 1] = _mm256_unpackhi_ps(r[i], r[i + 1]);
  for (int i = 0; i < 8; i += 4) {
    r[i] = _mm256_shuffle_ps(t[i], t[i + 2], 0x44), r[i + 1] = _mm256_shuffle_ps(t[i], t[i + 2], 0xee);
    r[i + 2] = _mm256_shuffle_ps(t[i + 1], t[i + 3], 0x44), r[i + 3] = _mm256_shuffle_ps(t[i + 1], t[i + 3], 0xee);
  }
  for (int i = 0; i < 4; i++) {
    _mm256_storeu_ps(d + i * dst_stride, _mm256_permute2f128_ps(r[i], r[i + 4], 0x20));
    _mm256_storeu_ps(d + (i + 4) * dst_stride, _mm256_permute2f128_ps(r[i], r[i + 4], 0x31));
  }
}

// d[c * dst_stride + r] = s[r * src_stride + c] for an h x w block, 8x8 blocks where both divide.
void transpose(const float* s, int src_stride, float* d, int dst_stride, int h, int w) {
  if (h % 8 == 0 && w % 8 == 0) {
    for (int r = 0; r < h; r += 8)
      for (int c = 0; c < w; c += 8) transpose8(s + r * src_stride + c, src_stride, d + c * dst_stride + r, dst_stride);
    return;
  }
  for (int r = 0; r < h; r++)
    for (int c = 0; c < w; c++) d[c * dst_stride + r] = s[r * src_stride + c];
}

}  // namespace

void fwd_dct(const int16_t* residual, int stride, TX_SIZE tx, double scale, tran_low_t* out) {
  const int w = tx_size_wide[tx], h = tx_size_high[tx];
  const int cw = w > 32 ? 32 : w, ch = h > 32 ? 32 : h;
  alignas(32) float rows[64 * 64];  // [x][y]: sample x of row y
  alignas(32) float tmp[2 * 64 * 64];
  float* const in = tmp;  // [y][x], before the transpose
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) in[y * w + x] = static_cast<float>(residual[y * stride + x]);
  transpose(in, w, rows, h, h, w);
  dct_lanes(rows, w, h, tmp);  // rows[u * h + y]: horizontal frequency u of row y
  alignas(32) float cols[64 * 32];  // [y][u]: the kept horizontal frequencies, row by row
  transpose(rows, h, cols, cw, cw, h);
  dct_lanes(cols, h, cw, tmp);  // cols[v * cw + u]
  // Orthonormal scale: the horizontal factor per column, the vertical one (with the AVM coefficient
  // scale) per row.
  alignas(32) float ku[32];
  ku[0] = static_cast<float>(std::sqrt(1.0 / w));
  for (int u = 1; u < cw; u++) ku[u] = static_cast<float>(std::sqrt(2.0 / w));
  const float kv0 = static_cast<float>(scale * std::sqrt(1.0 / h));
  const float kv = static_cast<float>(scale * std::sqrt(2.0 / h));
  for (int v = 0; v < ch; v++) {
    const float k = v ? kv : kv0;
    for (int u = 0; u < cw; u++) {  // round half away from zero, like lround (a call per value)
      const float x = cols[v * cw + u] * ku[u] * k;
      out[v * cw + u] = static_cast<tran_low_t>(x + (x >= 0 ? 0.5f : -0.5f));
    }
  }
}

namespace {

// AVM's forward kernel of one 1D type and size (rows: output frequency, columns: input sample).
const int* fwd_kernel(int kind, int n) {
  switch (kind) {
    case DCT2: return n == 4 ? tx_kernel_dct2_size4[FWD_TXFM][0] : n == 8 ? tx_kernel_dct2_size8[FWD_TXFM][0]
                    : n == 16 ? tx_kernel_dct2_size16[FWD_TXFM][0] : tx_kernel_dct2_size32[FWD_TXFM][0];
    case DST7: return n == 4 ? tx_kernel_adst_size4[FWD_TXFM][0] : n == 8 ? tx_kernel_adst_size8[FWD_TXFM][0]
                    : tx_kernel_adst_size16[FWD_TXFM][0];
    case DCT8: return n == 4 ? tx_kernel_fdst_size4[FWD_TXFM][0] : n == 8 ? tx_kernel_fdst_size8[FWD_TXFM][0]
                    : tx_kernel_fdst_size16[FWD_TXFM][0];
    default: return nullptr;  // IDT: a scale
  }
}

// One pass over `line` vectors of n samples (sample k of vector i at src[i + k * line]); vector i's
// coefficients go to dst[i * n ..]. fwd_transform_1d_c with no skipped lines.
void fwd_1d(const int* src, int* dst, int shift, int line, int n, int kind) {
  const int offset = shift > 0 ? 1 << (shift - 1) : 0;  // int sums, as fwd_txfm_*_c
  const int* const mat = fwd_kernel(kind, n);
  const int idt = n == 4 ? 128 : n == 8 ? 181 : n == 16 ? 256 : 362;  // fwd_txfm_idtx_size*_c
  for (int i = 0; i < line; i++)
    for (int j = 0; j < n; j++) {
      int s = 0;
      if (!mat)
        s = src[i + j * line] * idt;
      else
        for (int k = 0; k < n; k++) s += src[i + k * line] * mat[j * n + k];
      dst[i * n + j] = static_cast<int>((s + offset) >> shift);
    }
}

}  // namespace

void fwd_transform(const int16_t* residual, int stride, TX_SIZE tx, TX_TYPE type, tran_low_t* out) {
  const int w = tx_size_wide[tx], h = tx_size_high[tx];
  int in[32 * 32], buf[32 * 32];
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) in[y * w + x] = residual[y * stride + x];
  fwd_1d(in, buf, fwd_tx_shift[tx][0], w, h, g_ver_tx_type[type]);   // columns
  fwd_1d(buf, in, fwd_tx_shift[tx][1], h, w, g_hor_tx_type[type]);   // rows: in[v * w + u]
  const bool sqrt2 = (tx_size_wide_log2[tx] + tx_size_high_log2[tx]) & 1;
  for (int i = 0; i < w * h; i++)
    out[i] = sqrt2 ? round_shift(static_cast<int64_t>(in[i]) * NewSqrt2, NewSqrt2Bits) : in[i];
}

TX_TYPE most_compact(const int16_t* residual, int stride, TX_SIZE tx, const TX_TYPE* types, int n, tran_low_t* out) {
  const int w = tx_size_wide[tx], h = tx_size_high[tx];
  const bool sqrt2 = (tx_size_wide_log2[tx] + tx_size_high_log2[tx]) & 1;
  int in[16 * 16], cols[4][16 * 16], row[16 * 16];
  bool have[4] = {};
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) in[y * w + x] = residual[y * stride + x];
  int64_t least = INT64_MAX;
  TX_TYPE best = types[0];
  for (int k = 0; k < n; k++) {
    const int v = g_ver_tx_type[types[k]];  // DCT2, IDT, DST7 or DCT8
    if (!have[v]) fwd_1d(in, cols[v], fwd_tx_shift[tx][0], w, h, v), have[v] = true;
    fwd_1d(cols[v], row, fwd_tx_shift[tx][1], h, w, g_hor_tx_type[types[k]]);
    int64_t l1 = 0;
    for (int i = 0; i < w * h; i++) l1 += std::abs(row[i]);  // the sqrt(2) step scales every type alike
    if (l1 < least) {
      least = l1, best = types[k];
      for (int i = 0; i < w * h; i++)
        out[i] = sqrt2 ? round_shift(static_cast<int64_t>(row[i]) * NewSqrt2, NewSqrt2Bits) : row[i];
    }
  }
  return best;
}