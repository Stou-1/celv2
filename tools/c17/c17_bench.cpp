// C17 step 0b (docs/superpowers/specs/2026-11-09-gpu-rd-pass-design.md), throwaway: the RD pass's transforms on
// residuals dumped from real encodes (CELV2_TX_DUMP), timed three ways:
//   float  - today's fwd_dct() (Lee's butterflies in float);
//   avm    - AVM's own AVX2 integer DCT2 (fwd_transform_1d_avx2: int32, even-odd butterflies), both passes;
//   madd16 - AVM's DCT2 matrices as full products with _mm256_madd_epi16, both passes in 16 bits: the first pass
//            shifted by log2(h) - 1 so its outputs fit, the second by the rest of AVM's total.
// madd16 is checked exactly against a scalar product with the same shifts, and its drift from fwd_dct() reported.
// The projection: ns a call x the encode's uncached calls (<dump>.counts) / frames = ms a frame on one thread.
// usage: c17_bench <dump> <frames> [rounds]
#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/quant.h"
#include "core/rd_int.h"
#include "core/transform.h"

extern "C" {
#include "av2/common/av2_txfm.h"
#include "av2/common/common_data.h"
#include "av2/common/txb_common.h"
#include "av2/common/idct.h"
#include "av2/common/quant_common.h"
#include "av2/common/scan.h"
void fwd_transform_1d_avx2(const int* src, int* dst, int shift, int line, int skip_line, int zero_line,
                           const int tx_type_index, const int size_index);
}

namespace {

struct Rec {
  int t, tcq, qindex;
  double scale;
  size_t at;
};

int log2i(int n) { return n <= 1 ? 0 : 1 + log2i(n / 2); }

const int* dct_matrix(int n) {
  return n == 4 ? tx_kernel_dct2_size4[FWD_TXFM][0] : n == 8 ? tx_kernel_dct2_size8[FWD_TXFM][0]
       : n == 16 ? tx_kernel_dct2_size16[FWD_TXFM][0] : n == 32 ? tx_kernel_dct2_size32[FWD_TXFM][0]
       : tx_kernel_dct2_size64[FWD_TXFM][0];
}

// pairs[n][j * n/2 + kp]: m[j][2kp] and m[j][2kp+1] as two int16 in one int32 (madd's operand).
std::vector<int32_t> g_pairs[7];
void make_pairs() {
  for (int l = 2; l <= 6; l++) {
    const int n = 1 << l;
    const int* m = dct_matrix(n);
    g_pairs[l].resize(n * n / 2);
    for (int j = 0; j < n; j++)
      for (int kp = 0; kp < n / 2; kp++)
        g_pairs[l][j * n / 2 + kp] =
            static_cast<int32_t>((static_cast<uint32_t>(static_cast<uint16_t>(m[j * n + 2 * kp]))) |
                                 (static_cast<uint32_t>(static_cast<uint16_t>(m[j * n + 2 * kp + 1])) << 16));
  }
}

// One pass: out[j][l] = (sum_k m[j][k] in[k][l] + round) >> shift for j < nout, l < L. in: n rows of int16 (stride is);
// out: int16 (Out32 false) or int32, rows of stride os.
template <bool Out32>
void pass(const int16_t* in, int is, int n, int nout, int L, int shift, void* out, int os) {
  const int32_t* const pr = g_pairs[log2i(n)].data();
  const int np = n / 2;
  const __m256i off = _mm256_set1_epi32(shift > 0 ? 1 << (shift - 1) : 0);
  const __m128i sh = _mm_cvtsi32_si128(shift);
  alignas(32) __m256i P[2 * 32];
  if (L >= 16) {
    for (int c = 0; c < L; c += 16) {
      for (int kp = 0; kp < np; kp++) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + 2 * kp * is + c));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + (2 * kp + 1) * is + c));
        P[2 * kp] = _mm256_unpacklo_epi16(a, b), P[2 * kp + 1] = _mm256_unpackhi_epi16(a, b);
      }
      for (int j = 0; j < nout; j++) {
        __m256i lo = _mm256_setzero_si256(), hi = _mm256_setzero_si256();
        const int32_t* q = pr + j * np;
        for (int kp = 0; kp < np; kp++) {
          const __m256i m = _mm256_set1_epi32(q[kp]);
          lo = _mm256_add_epi32(lo, _mm256_madd_epi16(P[2 * kp], m));
          hi = _mm256_add_epi32(hi, _mm256_madd_epi16(P[2 * kp + 1], m));
        }
        lo = _mm256_sra_epi32(_mm256_add_epi32(lo, off), sh), hi = _mm256_sra_epi32(_mm256_add_epi32(hi, off), sh);
        if constexpr (Out32) {
          int32_t* o = static_cast<int32_t*>(out) + j * os + c;
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(o), _mm256_permute2x128_si256(lo, hi, 0x20));
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(o + 8), _mm256_permute2x128_si256(lo, hi, 0x31));
        } else {
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(static_cast<int16_t*>(out) + j * os + c),
                              _mm256_packs_epi32(lo, hi));
        }
      }
    }
  } else if (L == 8) {
    for (int kp = 0; kp < np; kp++) {
      const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + 2 * kp * is));
      const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + (2 * kp + 1) * is));
      P[kp] = _mm256_set_m128i(_mm_unpackhi_epi16(a, b), _mm_unpacklo_epi16(a, b));
    }
    for (int j = 0; j < nout; j++) {
      __m256i acc = _mm256_setzero_si256();
      const int32_t* q = pr + j * np;
      for (int kp = 0; kp < np; kp++) acc = _mm256_add_epi32(acc, _mm256_madd_epi16(P[kp], _mm256_set1_epi32(q[kp])));
      acc = _mm256_sra_epi32(_mm256_add_epi32(acc, off), sh);
      if constexpr (Out32) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(static_cast<int32_t*>(out) + j * os), acc);
      } else {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(static_cast<int16_t*>(out) + j * os),
                         _mm_packs_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1)));
      }
    }
  } else {  // L == 4: two pairs of rows a vector
    for (int kp = 0; kp < np; kp += 2) {
      const __m128i a0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(in + 2 * kp * is));
      const __m128i b0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(in + (2 * kp + 1) * is));
      const __m128i a1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(in + (2 * kp + 2) * is));
      const __m128i b1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(in + (2 * kp + 3) * is));
      P[kp / 2] = _mm256_set_m128i(_mm_unpacklo_epi16(a1, b1), _mm_unpacklo_epi16(a0, b0));
    }
    for (int j = 0; j < nout; j++) {
      __m256i acc = _mm256_setzero_si256();
      const int32_t* q = pr + j * np;
      for (int kp = 0; kp < np; kp += 2)
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(P[kp / 2], _mm256_setr_epi32(q[kp], q[kp], q[kp], q[kp], q[kp + 1],
                                                                                   q[kp + 1], q[kp + 1], q[kp + 1])));
      __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
      s = _mm_sra_epi32(_mm_add_epi32(s, _mm256_castsi256_si128(off)), sh);
      if constexpr (Out32) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(static_cast<int32_t*>(out) + j * os), s);
      } else {
        _mm_storel_epi64(reinterpret_cast<__m128i*>(static_cast<int16_t*>(out) + j * os), _mm_packs_epi32(s, s));
      }
    }
  }
}

// d[c][r] = s[r][c], int16, rows x cols both multiples of 4.
void transpose16(const int16_t* s, int ss, int16_t* d, int ds, int rows, int cols) {
  if (rows % 8 == 0 && cols % 8 == 0) {
    for (int r = 0; r < rows; r += 8)
      for (int c = 0; c < cols; c += 8) {
        __m128i x[8];
        for (int i = 0; i < 8; i++) x[i] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + (r + i) * ss + c));
        const __m128i a0 = _mm_unpacklo_epi16(x[0], x[1]), a1 = _mm_unpackhi_epi16(x[0], x[1]);
        const __m128i a2 = _mm_unpacklo_epi16(x[2], x[3]), a3 = _mm_unpackhi_epi16(x[2], x[3]);
        const __m128i a4 = _mm_unpacklo_epi16(x[4], x[5]), a5 = _mm_unpackhi_epi16(x[4], x[5]);
        const __m128i a6 = _mm_unpacklo_epi16(x[6], x[7]), a7 = _mm_unpackhi_epi16(x[6], x[7]);
        const __m128i b0 = _mm_unpacklo_epi32(a0, a2), b1 = _mm_unpackhi_epi32(a0, a2);
        const __m128i b2 = _mm_unpacklo_epi32(a1, a3), b3 = _mm_unpackhi_epi32(a1, a3);
        const __m128i b4 = _mm_unpacklo_epi32(a4, a6), b5 = _mm_unpackhi_epi32(a4, a6);
        const __m128i b6 = _mm_unpacklo_epi32(a5, a7), b7 = _mm_unpackhi_epi32(a5, a7);
        const __m128i y[8] = {_mm_unpacklo_epi64(b0, b4), _mm_unpackhi_epi64(b0, b4), _mm_unpacklo_epi64(b1, b5),
                              _mm_unpackhi_epi64(b1, b5), _mm_unpacklo_epi64(b2, b6), _mm_unpackhi_epi64(b2, b6),
                              _mm_unpacklo_epi64(b3, b7), _mm_unpackhi_epi64(b3, b7)};
        for (int i = 0; i < 8; i++) _mm_storeu_si128(reinterpret_cast<__m128i*>(d + (c + i) * ds + r), y[i]);
      }
    return;
  }
  for (int r = 0; r < rows; r += 4)
    for (int c = 0; c < cols; c += 4) {
      __m128i x[4];
      for (int i = 0; i < 4; i++) x[i] = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(s + (r + i) * ss + c));
      const __m128i a0 = _mm_unpacklo_epi16(x[0], x[1]), a1 = _mm_unpacklo_epi16(x[2], x[3]);
      const __m128i b0 = _mm_unpacklo_epi32(a0, a1), b1 = _mm_unpackhi_epi32(a0, a1);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(d + c * ds + r), b0);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(d + (c + 1) * ds + r), _mm_srli_si128(b0, 8));
      _mm_storel_epi64(reinterpret_cast<__m128i*>(d + (c + 2) * ds + r), b1);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(d + (c + 3) * ds + r), _mm_srli_si128(b1, 8));
    }
}

// int32, the same (8x8 blocks where both divide, else 4x4).
void transpose32(const int32_t* s, int ss, int32_t* d, int ds, int rows, int cols) {
  if (rows % 8 == 0 && cols % 8 == 0) {
    for (int r = 0; r < rows; r += 8)
      for (int c = 0; c < cols; c += 8) {
        __m256 x[8], t[8];
        for (int i = 0; i < 8; i++) x[i] = _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + (r + i) * ss + c)));
        for (int i = 0; i < 8; i += 2) t[i] = _mm256_unpacklo_ps(x[i], x[i + 1]), t[i + 1] = _mm256_unpackhi_ps(x[i], x[i + 1]);
        for (int i = 0; i < 8; i += 4) {
          x[i] = _mm256_shuffle_ps(t[i], t[i + 2], 0x44), x[i + 1] = _mm256_shuffle_ps(t[i], t[i + 2], 0xee);
          x[i + 2] = _mm256_shuffle_ps(t[i + 1], t[i + 3], 0x44), x[i + 3] = _mm256_shuffle_ps(t[i + 1], t[i + 3], 0xee);
        }
        for (int i = 0; i < 4; i++) {
          _mm256_storeu_ps(reinterpret_cast<float*>(d + (c + i) * ds + r), _mm256_permute2f128_ps(x[i], x[i + 4], 0x20));
          _mm256_storeu_ps(reinterpret_cast<float*>(d + (c + i + 4) * ds + r), _mm256_permute2f128_ps(x[i], x[i + 4], 0x31));
        }
      }
    return;
  }
  for (int r = 0; r < rows; r += 4)
    for (int c = 0; c < cols; c += 4) {
      __m128 r0 = _mm_castsi128_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(s + r * ss + c)));
      __m128 r1 = _mm_castsi128_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(s + (r + 1) * ss + c)));
      __m128 r2 = _mm_castsi128_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(s + (r + 2) * ss + c)));
      __m128 r3 = _mm_castsi128_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(s + (r + 3) * ss + c)));
      _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
      _mm_storeu_ps(reinterpret_cast<float*>(d + c * ds + r), r0);
      _mm_storeu_ps(reinterpret_cast<float*>(d + (c + 1) * ds + r), r1);
      _mm_storeu_ps(reinterpret_cast<float*>(d + (c + 2) * ds + r), r2);
      _mm_storeu_ps(reinterpret_cast<float*>(d + (c + 3) * ds + r), r3);
    }
}

void shifts(TX_SIZE t, int* s1, int* s2) {
  const int h = tx_size_high[t];
  *s1 = std::max(0, log2i(h) - 1);  // 8 bits: log2(h) + bd - 9
  *s2 = fwd_tx_shift[t][0] + fwd_tx_shift[t][1] - *s1;
}
bool has_sqrt2(TX_SIZE t) { return (tx_size_wide_log2[t] + tx_size_high_log2[t]) & 1; }

// madd16: out in AVM's layout (pos = v * cw + u), cw = min(w, 32).
void fwd_madd16(const int16_t* res, int stride, TX_SIZE t, int32_t* out) {
  const int w = tx_size_wide[t], h = tx_size_high[t], cw = std::min(w, 32), ch = std::min(h, 32);
  int s1, s2;
  shifts(t, &s1, &s2);
  alignas(32) int16_t Y[32 * 64], T[64 * 32];
  alignas(32) int32_t Z[32 * 32];
  pass<false>(res, stride, h, ch, w, s1, Y, w);  // columns: Y[v][x]
  transpose16(Y, w, T, ch, ch, w);               // T[x][v]
  pass<true>(T, ch, w, cw, ch, s2, Z, ch);       // rows: Z[u][v]
  transpose32(Z, ch, out, cw, cw, ch);           // out[v][u]
  if (has_sqrt2(t)) {
    const __m256i k = _mm256_set1_epi32(NewSqrt2), r = _mm256_set1_epi32(1 << (NewSqrt2Bits - 1));
    for (int i = 0; i < cw * ch; i += 8) {
      const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(out + i));
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i),
                          _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(x, k), r), NewSqrt2Bits));
    }
  }
}

// The same arithmetic, scalar (madd16's reference).
void fwd_ref(const int16_t* res, int stride, TX_SIZE t, int32_t* out) {
  const int w = tx_size_wide[t], h = tx_size_high[t], cw = std::min(w, 32), ch = std::min(h, 32);
  int s1, s2;
  shifts(t, &s1, &s2);
  std::vector<int> Y(ch * w);
  const int* mv = dct_matrix(h);
  const int* mh = dct_matrix(w);
  for (int v = 0; v < ch; v++)
    for (int x = 0; x < w; x++) {
      int s = 0;
      for (int k = 0; k < h; k++) s += mv[v * h + k] * res[k * stride + x];
      Y[v * w + x] = (s + (s1 ? 1 << (s1 - 1) : 0)) >> s1;
    }
  for (int v = 0; v < ch; v++)
    for (int u = 0; u < cw; u++) {
      int s = 0;
      for (int k = 0; k < w; k++) s += mh[u * w + k] * Y[v * w + k];
      int x = (s + (1 << (s2 - 1))) >> s2;
      if (has_sqrt2(t)) x = static_cast<int>((static_cast<int64_t>(x) * NewSqrt2 + (1 << (NewSqrt2Bits - 1))) >> NewSqrt2Bits);
      out[v * cw + u] = x;
    }
}

// AVM's own AVX2 DCT2, both passes (av2_fwd_txfm2d's order and shifts), the low 32 x 32 kept.
void fwd_avm(const int16_t* res, int stride, TX_SIZE t, int32_t* out) {
  const int w = tx_size_wide[t], h = tx_size_high[t];
  alignas(32) int a[64 * 64], b[64 * 64];
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) a[y * w + x] = res[y * stride + x];
  const int sw = w > 32 ? w - 32 : 0, sh = h > 32 ? h - 32 : 0;
  fwd_transform_1d_avx2(a, b, fwd_tx_shift[t][0], w, 0, sh, 0, tx_size_high_log2[t] - 2);
  fwd_transform_1d_avx2(b, a, fwd_tx_shift[t][1], h, sh, sw, 0, tx_size_wide_log2[t] - 2);
  const int cw = std::min(w, 32), ch = std::min(h, 32);
  const bool s2 = has_sqrt2(t);
  for (int v = 0; v < ch; v++)
    for (int u = 0; u < cw; u++) {
      const int x = a[v * w + u];
      out[v * cw + u] = s2 ? static_cast<int>((static_cast<int64_t>(x) * NewSqrt2 + (1 << (NewSqrt2Bits - 1))) >> NewSqrt2Bits) : x;
    }
}

double now_ns() {
  return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// tcq_levels() (quant.cpp) in integers, for timing: magnitudes in 1/256 steps by a multiply ((|c| m_q) >> 16, m_q =
// 2^(24 + qshift) / q), squared errors in 1/65536 squared steps, bits in 1/512 bit weighted by 0.05, path costs int64
// with a clamped infinity of 2^60, the same vectors, blends and tie rules on int64. Returns eob; levels into q.
std::vector<int> g_bits_q9;
int64_t g_wb[256];
int g_first_q8;
void int_tables() {
  g_bits_q9.resize(4096);
  for (int l = 0; l < 4096; l++) g_bits_q9[l] = static_cast<int>(std::lround(coeff_level_bits(l) * 512));
  for (int l = 0; l < 256; l++) g_wb[l] = (static_cast<int64_t>(g_bits_q9[l]) * 3277) >> 9;
  g_first_q8 = static_cast<int>(std::lround((0.5 + kTcqBitWeight * (coeff_level_bits(1) - coeff_level_bits(0)) / 2) * 256));
}
int64_t wb_of(int l) { return l < 256 ? g_wb[l] : (static_cast<int64_t>(g_bits_q9[std::min(l, 4095)]) * 3277) >> 9; }

int int_tcq(const int32_t* coeff, int32_t* q, TX_SIZE tx, int qindex) {
  static const uint8_t next[8][2] = {{0, 4}, {4, 0}, {1, 5}, {5, 1}, {6, 2}, {2, 6}, {7, 3}, {3, 7}};
  const int dcq = av2_dc_quant_QTX(qindex, 0, 0, kQDepth), acq = av2_ac_quant_QTX(qindex, 0, 0, kQDepth);
  const int n = std::min(tx_size_wide[tx], 32) * std::min(tx_size_high[tx], 32);
  const int qshift = QUANT_TABLE_BITS + av2_get_tx_scale(tx);
  const SCAN_ORDER* so = get_scan(tx, DCT_DCT);
  const uint32_t m_ac = static_cast<uint32_t>((1ull << (24 + qshift)) / acq), m_dc = static_cast<uint32_t>((1ull << (24 + qshift)) / dcq);
  const int max_dc = 0xffffff / (2 * dcq), max_ac = 0xffffff / (2 * acq);
  alignas(32) int32_t e_of[1024];
  int eob = 0;
  const __m256i vm = _mm256_set1_epi64x(m_ac);
  const __m128i first4 = _mm_set1_epi32(g_first_q8 - 1);
  for (int pos = 0; pos < n; pos += 4) {
    const __m128i a = _mm_abs_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(coeff + pos)));
    const __m256i p = _mm256_srli_epi64(_mm256_mul_epu32(_mm256_cvtepu32_epi64(a), vm), 16);
    const __m128i e = _mm_castps_si128(_mm256_castps256_ps128(_mm256_castsi256_ps(
        _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0)))));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(e_of + pos), e);
    for (int m = _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(e, first4))) & (pos ? 15 : 14); m; m &= m - 1)
      eob = std::max(eob, so->iscan[pos + _tzcnt_u32(m)] + 1);
  }
  e_of[0] = static_cast<int32_t>((static_cast<uint64_t>(std::abs(coeff[0])) * m_dc) >> 16);
  if (e_of[0] >= g_first_q8) eob = std::max(eob, 1);
  std::fill(q, q + n, 0);
  if (!eob) return 0;
  constexpr int64_t kInf = 1ll << 60;
  alignas(32) static thread_local int64_t step[1024][8];
  const __m256i inf = _mm256_set1_epi64x(kInf);
  __m256i lo = _mm256_setr_epi64x(0, kInf, kInf, kInf), hi = inf;
  const __m256i from_lo = _mm256_setr_epi64x(0ll << 32, 1ll << 32, 2ll << 32, 3ll << 32);
  const __m256i from_hi = _mm256_setr_epi64x(4ll << 32, 5ll << 32, 6ll << 32, 7ll << 32);
  const __m256i second = _mm256_setr_epi64x(0, -1, 0, -1);
  for (int c = eob - 1; c >= 0; c--) {
    const int pos = so->scan[c];
    const int e = e_of[pos], maxl = pos ? max_ac : max_dc;
    const bool last = c == eob - 1;
    int64_t ec[2][2], oc[2];
    int64_t el[2][2], ol[2];
    for (int q1 = 0; q1 < 2; q1++) {
      int ne = 0;
      ec[q1][0] = ec[q1][1] = oc[q1] = kInf, el[q1][0] = el[q1][1] = ol[q1] = 0;
      const int base = q1 ? (e + 128) >> 8 : e >> 8;
      const int cands[3] = {base, base + 1, 0};
      int prev = -1;
      for (int k = 0; k < 3; k++) {
        const int l = std::min(cands[k], maxl);
        if ((l == 0 && last) || (k == 2 && base == 0) || l == prev) continue;
        prev = l;
        const int64_t rec = l == 0 ? 0 : q1 ? (static_cast<int64_t>(l) << 8) - 128 : static_cast<int64_t>(l) << 8;
        const int64_t j = (e - rec) * (e - rec) + wb_of(l);
        if (l & 1) oc[q1] = j, ol[q1] = l;
        else ec[q1][ne] = j, el[q1][ne++] = l;
      }
    }
    auto per = [](int64_t a, int64_t b) { return _mm256_setr_epi64x(a, a, b, b); };
    const __m256i e0 = per(ec[0][0], ec[1][0]), e1 = per(ec[0][1], ec[1][1]), ov = per(oc[0], oc[1]);
    const __m256i l0 = per(el[0][0], el[1][0]), l1 = per(el[0][1], el[1][1]), olv = per(ol[0], ol[1]);
    __m256i f[2], g[2];
    for (int h = 0; h < 2; h++) {
      const __m256i p = h ? hi : lo, from = h ? from_hi : from_lo;
      const __m256i c0 = _mm256_add_epi64(p, e0), c1 = _mm256_add_epi64(p, e1), odd = _mm256_add_epi64(p, ov);
      const __m256i lt = _mm256_cmpgt_epi64(c0, c1);
      const __m256i even = _mm256_blendv_epi8(c0, c1, lt);
      const __m256i even_step = _mm256_blendv_epi8(_mm256_or_si256(l0, from), _mm256_or_si256(l1, from), lt);
      const __m256i other = _mm256_shuffle_epi32(odd, 0x4e);  // swap the 64-bit halves of each 128
      const __m256i other_step = _mm256_shuffle_epi32(_mm256_or_si256(olv, from), 0x4e);
      const __m256i olt = _mm256_cmpgt_epi64(even, other), ole = _mm256_xor_si256(_mm256_cmpgt_epi64(other, even), _mm256_set1_epi64x(-1));
      const __m256i take = _mm256_blendv_epi8(olt, ole, second);
      const __m256i v = _mm256_blendv_epi8(even, other, take);
      f[h] = _mm256_blendv_epi8(v, inf, _mm256_cmpgt_epi64(v, inf));
      g[h] = _mm256_blendv_epi8(even_step, other_step, take);
    }
    const __m256i a = _mm256_permute4x64_epi64(f[0], 0b11011000), d = _mm256_permute4x64_epi64(f[1], 0b10001101);
    lo = _mm256_permute2x128_si256(a, d, 0x20), hi = _mm256_permute2x128_si256(a, d, 0x31);
    const __m256i ga = _mm256_permute4x64_epi64(g[0], 0b11011000), gd = _mm256_permute4x64_epi64(g[1], 0b10001101);
    _mm256_store_si256(reinterpret_cast<__m256i*>(step[c]), _mm256_permute2x128_si256(ga, gd, 0x20));
    _mm256_store_si256(reinterpret_cast<__m256i*>(step[c] + 4), _mm256_permute2x128_si256(ga, gd, 0x31));
  }
  alignas(32) int64_t path[8];
  _mm256_store_si256(reinterpret_cast<__m256i*>(path), lo);
  _mm256_store_si256(reinterpret_cast<__m256i*>(path + 4), hi);
  int s = 0;
  for (int k = 1; k < 8; k++)
    if (path[k] < path[s]) s = k;
  for (int c = 0; c < eob; c++) {
    const int pos = so->scan[c];
    const int l = static_cast<int16_t>(step[c][s]);
    q[pos] = coeff[pos] < 0 ? -l : l;
    s = static_cast<int>(step[c][s] >> 32);
  }
  int state = 0;  // the decoder's walk (the dequantization's state), as tcq_levels()
  for (int c = eob - 1; c >= 0; c--) state = next[state][std::abs(q[so->scan[c]]) & 1];
  return eob + (state & 0);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: c17_bench <dump> <frames> [rounds]\n");
    return 2;
  }
  const int frames = std::atoi(argv[2]);
  const int rounds = argc > 3 ? std::atoi(argv[3]) : 5;
  make_pairs();
  std::vector<int16_t> pool;
  std::vector<Rec> recs;
  {
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) return std::perror(argv[1]), 1;
    for (;;) {
      uint8_t head[2];
      int16_t q;
      double sc;
      if (std::fread(head, 1, 2, f) != 2 || std::fread(&q, 2, 1, f) != 1 || std::fread(&sc, 8, 1, f) != 1) break;
      const int n = tx_size_wide[head[0]] * tx_size_high[head[0]];
      const size_t at = pool.size();
      pool.resize(at + n);
      if (std::fread(pool.data() + at, 2, n, f) != static_cast<size_t>(n)) break;
      recs.push_back({head[0], head[1], q, sc, at});
    }
    std::fclose(f);
  }
  unsigned long long calls[TX_SIZES_ALL] = {}, unc[TX_SIZES_ALL] = {};
  unsigned every = 0;
  if (FILE* c = std::fopen((std::string(argv[1]) + ".counts").c_str(), "r")) {
    if (std::fscanf(c, "every %u", &every) != 1) every = 0;
    int t;
    unsigned long long a, b;
    while (std::fscanf(c, "%d %llu %llu", &t, &a, &b) == 3) calls[t] = a, unc[t] = b;
    std::fclose(c);
  }
  std::printf("%zu records, %d frames, sampled 1 in %u\n", recs.size(), frames, every);

  // cmp: today's double cost (fwd_dct with the dumped scale, quantize(), the squares in steps and the bits) against
  // rdi's integer one in the same units, per size: the ratios of the summed distortion and bits.
  if (argc > 3 && std::string(argv[3]) == "cmp") {
    const bool rz = argc > 4 && std::atoi(argv[4]) != 0;
    g_rd_zero = rz;
    for (int t = 0; t < TX_SIZES_ALL; t++) {
      double dsum_o = 0, dsum_n = 0, bsum_o = 0, bsum_n = 0;
      long long cnt = 0, eob_ne = 0, same_q_ne = 0;
      double xy = 0, xx = 0, ad = 0, drop_o = 0, drop_n = 0;
      for (const Rec& r : recs) {
        if (r.t != t) continue;
        const TX_SIZE tx = static_cast<TX_SIZE>(t);
        const int w = tx_size_wide[tx], n = std::min(w, 32) * std::min(int{tx_size_high[tx]}, 32);
        const int16_t* res = pool.data() + r.at;
        alignas(32) int32_t co[1024], q[1024], dq[1024], ci[1024], qi[1024];
        fwd_dct(res, w, tx, r.scale, co);
        const int eob = quantize(co, q, dq, tx, r.qindex, 0, DCT_DCT, nullptr, r.tcq != 0, 8, 0);
        const double step = av2_ac_quant_QTX(r.qindex, 0, 0, kQDepth) / static_cast<double>(1 << (QUANT_TABLE_BITS + av2_get_tx_scale(tx)));
        double d = 0, b = eob ? 2 + 1.5 * std::log2(static_cast<double>(eob)) : 1;
        const SCAN_ORDER* so = get_scan(tx, DCT_DCT);
        for (int k = 0; k < n; k++) {
          const double e = (co[k] - dq[k]) / step;
          d += e * e;
          if (so->iscan[k] < eob) b += coeff_level_bits(std::abs(q[k]));
        }
        const double dc_old = dropped_cost(res, w, tx, r.scale, co, r.qindex);
        d += dc_old;
        int64_t dr;
        static celv2::rdi::Gains g14;
        celv2::rdi::gains_of(tx, r.scale, &g14);
        celv2::rdi::fdct(res, w, tx, &g14, ci, &dr);
        { const double st = av2_ac_quant_QTX(r.qindex, 0, 0, kQDepth) / static_cast<double>(1 << (QUANT_TABLE_BITS + av2_get_tx_scale(tx))); drop_o += dc_old, drop_n += dr / (st * st); }
        const celv2::rdi::TxCost c = celv2::rdi::quant_cost(ci, dr, tx, r.qindex, r.tcq != 0, rz, qi);
        const double dqv = av2_ac_quant_QTX(r.qindex, 0, 0, kQDepth);
        dsum_o += d, dsum_n += c.d / (1024 * dqv * dqv), bsum_o += b, bsum_n += c.bits / 512.0;
        for (int k = 0; k < n; k++) xy += static_cast<double>(ci[k]) * co[k], xx += static_cast<double>(co[k]) * co[k], ad += std::abs(ci[k] - co[k]);
        int e2 = 0;
        for (int k = 0; k < n; k++)
          if (qi[k]) e2 = std::max(e2, so->iscan[k] + 1);
        eob_ne += e2 != eob;
        {  // the same coefficients (fwd_dct()'s) through both quantizers
          const celv2::rdi::TxCost c2 = celv2::rdi::quant_cost(co, 0, tx, r.qindex, r.tcq != 0, rz, qi);
          (void)c2;
          int lv_ne = 0;
          for (int k = 0; k < n; k++) lv_ne += qi[k] != q[k];
          same_q_ne += lv_ne != 0;
        }
        cnt++;
      }
      if (cnt)
        std::printf("%2dx%-3d %8lld  d new/old %.4f  bits new/old %.4f  eob differs %.1f%%, levels differ on the same coefficients %.2f%%  int/float slope %.5f  mean |diff| %.3f  dropped new/old %.4f (old share of d %.3f)\n", tx_size_wide[t],
                    tx_size_high[t], cnt, dsum_n / dsum_o, bsum_n / bsum_o, 100.0 * eob_ne / cnt, 100.0 * same_q_ne / cnt, xy / xx, ad / (cnt * (std::min(tx_size_wide[t], 32) * std::min(int{tx_size_high[t]}, 32))), drop_o > 0 ? drop_n / drop_o : 0.0, drop_o / dsum_o);
    }
    return 0;
  }

  // Exactness of madd16 against its scalar reference, and its drift from fwd_dct().
  long long mism = 0, n_c = 0;
  double drift = 0, mag = 0;
  for (size_t i = 0; i < recs.size(); i += 7) {
    const Rec& r = recs[i];
    const TX_SIZE t = static_cast<TX_SIZE>(r.t);
    alignas(32) int32_t a[1024], b[1024], f[1024];
    fwd_madd16(pool.data() + r.at, tx_size_wide[t], t, a);
    fwd_ref(pool.data() + r.at, tx_size_wide[t], t, b);
    fwd_dct(pool.data() + r.at, tx_size_wide[t], t, r.scale, f);
    const int n = std::min(tx_size_wide[t], 32) * std::min(tx_size_high[t], 32);
    for (int k = 0; k < n; k++) {
      mism += a[k] != b[k];
      drift += std::abs(a[k] - f[k]), mag += std::abs(f[k]);
    }
    n_c += n;
  }
  std::printf("madd16 against its reference: %lld of %lld coefficients differ; |madd16 - fwd_dct| / |fwd_dct| = %.4f\n",
              mism, n_c, mag ? drift / mag : 0.0);

  std::printf("%-8s %9s %11s %9s %9s %9s %9s %9s %9s\n", "size", "records", "uncached/f", "float ns", "avm ns",
              "madd ns", "float ms", "avm ms", "madd ms");
  double tot[3] = {};
  alignas(32) int32_t out[1024];
  volatile int32_t sink = 0;
  for (int t = 0; t < TX_SIZES_ALL; t++) {
    std::vector<const Rec*> mine;
    for (const Rec& r : recs)
      if (r.t == t) mine.push_back(&r);
    if (mine.empty()) continue;
    const TX_SIZE tx = static_cast<TX_SIZE>(t);
    const int w = tx_size_wide[tx];
    double best[3] = {1e300, 1e300, 1e300};
    for (int round = 0; round < rounds; round++)
      for (int m = 0; m < 3; m++) {
        const double t0 = now_ns();
        for (const Rec* r : mine) {
          const int16_t* res = pool.data() + r->at;
          if (m == 0) fwd_dct(res, w, tx, r->scale, out);
          else if (m == 1) fwd_avm(res, w, tx, out);
          else { int64_t dd; celv2::rdi::fdct(res, w, tx, nullptr, out, &dd); }
          sink = sink + out[0];
        }
        best[m] = std::min(best[m], (now_ns() - t0) / mine.size());
      }
    const double per_frame = frames ? static_cast<double>(unc[t]) / frames : 0;
    double ms[3];
    for (int m = 0; m < 3; m++) ms[m] = best[m] * per_frame / 1e6, tot[m] += ms[m];
    std::printf("%2dx%-5d %9zu %11.0f %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f\n", w, tx_size_high[tx], mine.size(), per_frame,
                best[0], best[1], best[2], ms[0], ms[1], ms[2]);
  }
  std::printf("total ms a frame (one thread, uncached calls): float %.1f, avm %.1f (%.2fx), madd16 %.1f (%.2fx)\n",
              tot[0], tot[1], tot[1] / tot[0], tot[2], tot[2] / tot[0]);

  // Today's quantize() (double TCQ) on fwd_dct()'s coefficients: its share beside the transform, and its mean end of
  // block (c17_gpu's should be near it).
  std::printf("%-8s %9s %9s %9s %9s %9s %9s %9s\n", "size", "quant ns", "int ns", "quant ms", "int ms", "mean eob",
              "avm eob", "int eob");
  double qtot = 0, itot = 0;
  int_tables();
  for (int t = 0; t < TX_SIZES_ALL; t++) {
    std::vector<const Rec*> mine;
    for (const Rec& r : recs)
      if (r.t == t) mine.push_back(&r);
    if (mine.empty()) continue;
    const TX_SIZE tx = static_cast<TX_SIZE>(t);
    const int w = tx_size_wide[tx], nc = std::min(w, 32) * std::min(tx_size_high[tx], 32);
    std::vector<int32_t> co(mine.size() * nc), ca(mine.size() * nc);
    for (size_t i = 0; i < mine.size(); i++) {
      fwd_dct(pool.data() + mine[i]->at, w, tx, mine[i]->scale, co.data() + i * nc);
      fwd_avm(pool.data() + mine[i]->at, w, tx, ca.data() + i * nc);
    }
    alignas(32) int32_t q[1024], dq[1024];
    double best = 1e300, eobs = 0, eobs_avm = 0;
    for (int round = 0; round < rounds; round++) {
      double e = 0;
      const double t0 = now_ns();
      for (size_t i = 0; i < mine.size(); i++)
        e += quantize(co.data() + i * nc, q, dq, tx, mine[i]->qindex, 0, DCT_DCT, nullptr, mine[i]->tcq != 0, 8, 0);
      best = std::min(best, (now_ns() - t0) / mine.size());
      eobs = e;
    }
    for (size_t i = 0; i < mine.size(); i++)
      eobs_avm += quantize(ca.data() + i * nc, q, dq, tx, mine[i]->qindex, 0, DCT_DCT, nullptr, mine[i]->tcq != 0, 8, 0);
    double ibest = 1e300, ieobs = 0;
    for (int round = 0; round < rounds; round++) {
      double e = 0;
      const double t0 = now_ns();
      for (size_t i = 0; i < mine.size(); i++) e += celv2::rdi::tcq_vec(ca.data() + i * nc, tx, mine[i]->qindex, true, q);
      ibest = std::min(ibest, (now_ns() - t0) / mine.size());
      ieobs = e;
    }
    const double per_frame = frames ? static_cast<double>(unc[t]) / frames : 0;
    const double ms = best * per_frame / 1e6, ims = ibest * per_frame / 1e6;
    qtot += ms, itot += ims;
    std::printf("%2dx%-5d %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f\n", w, tx_size_high[tx], best, ibest, ms, ims,
                eobs / mine.size(), eobs_avm / mine.size(), ieobs / mine.size());
  }
  std::printf("quantize() total ms a frame (one thread, uncached calls): double %.1f, integer %.1f (%.2fx)\n", qtot, itot,
              itot / qtot);
  return 0;
}
