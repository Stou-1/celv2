// fwd_dct()'s AVX-512 path for sizes with both sides >= 16: 16 signals per 512-bit vector, each signal's whole DCT
// held in vectors (no pass through memory per butterfly level), the int16 converts, transposes and the final rounding
// fused around it. Every lane gets transform.cpp's float operations in the same order (no FMA: MSVC contracts none
// under /fp:precise, and intrinsics are never fused), so the values are the same. Intrinsics only, the TU keeps the
// build's /arch:AVX2: no inline function from a shared header is compiled with AVX-512 for the linker to pick.
#include <cmath>
#include <immintrin.h>

#include "core/transform.h"

const float* lee_table();  // transform.cpp: Lee's factors, row log2 n of 32

namespace {

constexpr int log2_of(int n) { return n <= 1 ? 0 : 1 + log2_of(n / 2); }

// The first K outputs of transform.cpp's dct_lanes<N> on 16 lanes, in place in v[0..N). The outputs past K are not
// computed; none of the first K depends on them, so theirs are the same operations.
template <int N, int K>
__forceinline void dct16(__m512* v, const float* lee) {
  if constexpr (N == 2) {
    const __m512 a = v[0], b = v[1];
    v[0] = _mm512_add_ps(a, b);
    if constexpr (K > 1) v[1] = _mm512_mul_ps(_mm512_sub_ps(a, b), _mm512_set1_ps(lee[32]));
  } else {
    constexpr int h = N / 2;
    constexpr int ka = (K + 1) / 2;                              // X[2m] = A[m], 2m < K
    constexpr int kc = K < 2 ? 0 : K / 2 + 1 > h ? h : K / 2 + 1;  // X[2m+1] = C[m] + C[m+1], 2m+1 < K
    constexpr int lg = log2_of(N);  // a constant: MSVC called log2_of at run time
    const float* const f = lee + 32 * lg;
    __m512 a[h], c[h];
    for (int i = 0; i < h; i++) {
      a[i] = _mm512_add_ps(v[i], v[N - 1 - i]);
      if constexpr (kc > 0) c[i] = _mm512_mul_ps(_mm512_sub_ps(v[i], v[N - 1 - i]), _mm512_set1_ps(f[i]));
    }
    dct16<h, ka>(a, lee);
    if constexpr (kc > 0) dct16<h, kc>(c, lee);
    for (int m = 0; m < h; m++) {
      if (2 * m < K) v[2 * m] = a[m];
      if (2 * m + 1 < K) v[2 * m + 1] = m + 1 < h ? _mm512_add_ps(c[m], c[m + 1]) : c[m];
    }
  }
}

// r[i] lane j <- r[j] lane i.
__forceinline void transpose16(__m512* r) {
  __m512 t[16];
  for (int i = 0; i < 16; i += 2) t[i] = _mm512_unpacklo_ps(r[i], r[i + 1]), t[i + 1] = _mm512_unpackhi_ps(r[i], r[i + 1]);
  for (int i = 0; i < 16; i += 4) {  // r[i + j], 128-bit lane k: column 4k + j of rows i..i+3
    r[i] = _mm512_shuffle_ps(t[i], t[i + 2], 0x44), r[i + 1] = _mm512_shuffle_ps(t[i], t[i + 2], 0xee);
    r[i + 2] = _mm512_shuffle_ps(t[i + 1], t[i + 3], 0x44), r[i + 3] = _mm512_shuffle_ps(t[i + 1], t[i + 3], 0xee);
  }
  for (int j = 0; j < 4; j++) {
    const __m512 a0 = _mm512_shuffle_f32x4(r[j], r[4 + j], 0x88), a1 = _mm512_shuffle_f32x4(r[j], r[4 + j], 0xdd);
    const __m512 b0 = _mm512_shuffle_f32x4(r[8 + j], r[12 + j], 0x88), b1 = _mm512_shuffle_f32x4(r[8 + j], r[12 + j], 0xdd);
    t[j] = _mm512_shuffle_f32x4(a0, b0, 0x88), t[8 + j] = _mm512_shuffle_f32x4(a0, b0, 0xdd);
    t[4 + j] = _mm512_shuffle_f32x4(a1, b1, 0x88), t[12 + j] = _mm512_shuffle_f32x4(a1, b1, 0xdd);
  }
  for (int i = 0; i < 16; i++) r[i] = t[i];
}

template <int W, int H>
void fwd(const int16_t* residual, int stride, double scale, tran_low_t* out) {
  constexpr int cw = W > 32 ? 32 : W, ch = H > 32 ? 32 : H;
  const float* const lee = lee_table();
  alignas(64) float cols[H * cw];  // [y][u]: the kept horizontal frequencies, row by row
  __m512 v[W > H ? W : H], r[16];
  for (int y0 = 0; y0 < H; y0 += 16) {  // rows y0..y0+15, one per lane
    for (int x0 = 0; x0 < W; x0 += 16) {
      for (int j = 0; j < 16; j++)  // exact: int16 to float
        r[j] = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(residual + (y0 + j) * stride + x0))));
      transpose16(r);
      for (int i = 0; i < 16; i++) v[x0 + i] = r[i];
    }
    dct16<W, cw>(v, lee);  // v[u]: horizontal frequency u
    for (int u0 = 0; u0 < cw; u0 += 16) {
      for (int i = 0; i < 16; i++) r[i] = v[u0 + i];
      transpose16(r);
      for (int j = 0; j < 16; j++) _mm512_store_ps(cols + (y0 + j) * cw + u0, r[j]);
    }
  }
  // transform.cpp's scale and rounding: (x ku[u]) k, plus +-0.5 by x >= 0, truncated.
  const float k0 = static_cast<float>(std::sqrt(1.0 / W)), k1 = static_cast<float>(std::sqrt(2.0 / W));
  const float kv0 = static_cast<float>(scale * std::sqrt(1.0 / H)), kv = static_cast<float>(scale * std::sqrt(2.0 / H));
  for (int u0 = 0; u0 < cw; u0 += 16) {  // columns u0..u0+15, one per lane
    for (int y = 0; y < H; y++) v[y] = _mm512_load_ps(cols + y * cw + u0);
    dct16<H, ch>(v, lee);  // v[v]: vertical frequency
    const __m512 ku = u0 ? _mm512_set1_ps(k1) : _mm512_mask_blend_ps(1, _mm512_set1_ps(k1), _mm512_set1_ps(k0));
    for (int i = 0; i < ch; i++) {
      const __m512 x = _mm512_mul_ps(_mm512_mul_ps(v[i], ku), _mm512_set1_ps(i ? kv : kv0));
      const __m512 half = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_GE_OQ),
                                               _mm512_set1_ps(-0.5f), _mm512_set1_ps(0.5f));
      _mm512_storeu_si512(out + i * cw + u0, _mm512_cvttps_epi32(_mm512_add_ps(x, half)));
    }
  }
}

}  // namespace

bool fwd_dct_avx512(const int16_t* residual, int stride, TX_SIZE tx, double scale, tran_low_t* out) {
  switch (tx) {
    case TX_16X16: return fwd<16, 16>(residual, stride, scale, out), true;
    case TX_32X32: return fwd<32, 32>(residual, stride, scale, out), true;
    case TX_64X64: return fwd<64, 64>(residual, stride, scale, out), true;
    case TX_16X32: return fwd<16, 32>(residual, stride, scale, out), true;
    case TX_32X16: return fwd<32, 16>(residual, stride, scale, out), true;
    case TX_32X64: return fwd<32, 64>(residual, stride, scale, out), true;
    case TX_64X32: return fwd<64, 32>(residual, stride, scale, out), true;
    case TX_16X64: return fwd<16, 64>(residual, stride, scale, out), true;
    case TX_64X16: return fwd<64, 16>(residual, stride, scale, out), true;
    default: return false;
  }
}
