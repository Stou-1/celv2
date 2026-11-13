#include "core/rd_int.h"

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

#include "core/profile.h"
#include "core/quant.h"

extern "C" {
#include "av2/common/av2_txfm.h"
#include "av2/common/common_data.h"
#include "av2/common/idct.h"
#include "av2/common/quant_common.h"
#include "av2/common/scan.h"
#include "av2/common/txb_common.h"
void fwd_transform_1d_avx2(const int* src, int* dst, int shift, int line, int skip_line, int zero_line,
                           const int tx_type_index, const int size_index);
}

namespace celv2::rdi {

namespace {

constexpr int64_t kInfV = int64_t{1} << 61;  // the Viterbi's: a missing candidate, an unreachable state

// Per qindex: the step, the max levels, the weights and TCQ's first-level threshold; x / (2 dqv) as a multiply,
// exact for x < 2^26 (m = ceil(2^(26 + l) / d), l = ceil(log2 d)).
struct Q {
  int dqv = 1, maxl_tcq = 0, maxl_plain = 0;
  int64_t lam_tcq = 0, lam_plain = 0, first = 0;
  uint64_t m = 0;
  int s = 0;
  int div2(int64_t x) const { return static_cast<int>((static_cast<uint64_t>(x) * m) >> s); }
};

int64_t lb512_of(int l) { return bits512(coeff_level_bits(l)); }

const std::array<int64_t, 4096>& lb_table() {
  static const std::array<int64_t, 4096> t = [] {
    std::array<int64_t, 4096> a{};
    for (int l = 0; l < 4096; l++) a[l] = lb512_of(l);
    return a;
  }();
  return t;
}
inline int64_t lb512(int l) { return l < 4096 ? lb_table()[l] : lb512_of(l); }

// The end of block's bits (and the all-zero flag's, 1 bit, at 0), as the pass's 2 + 1.5 log2(eob).
const std::array<int64_t, 1025>& eb_table() {
  static const std::array<int64_t, 1025> t = [] {
    std::array<int64_t, 1025> a{};
    a[0] = 512;
    for (int e = 1; e <= 1024; e++) a[e] = bits512(2 + 1.5 * std::log2(static_cast<double>(e)));
    return a;
  }();
  return t;
}

const Q& q_of(int qindex) {
  static const std::array<Q, 256> t = [] {
    std::array<Q, 256> a{};
    for (int qi = 0; qi < 256; qi++) {
      Q& q = a[qi];
      q.dqv = av2_ac_quant_QTX(qi, 0, 0, kQDepth);
      q.maxl_tcq = 0xffffff / (2 * q.dqv), q.maxl_plain = 0xffffff / q.dqv;
      q.lam_tcq = lam_of(qi, kTcqBitWeight), q.lam_plain = lam_of(qi, kBitWeight);
      // Level 1 under Q0 costs no more than 0: (X2 - 2 dqv)^2 << 8 + lam b(1) <= X2^2 << 8 + lam b(0).
      const int64_t num = q.lam_tcq * (lb512_of(1) - lb512_of(0)), den = 1024 * static_cast<int64_t>(q.dqv);
      q.first = q.dqv + (num + den - 1) / den;
      const int d = 2 * q.dqv;
      int l = 0;
      while ((1 << l) < d) l++;
      q.s = 26 + l;
      q.m = ((uint64_t{1} << q.s) + d - 1) / d;
    }
    return a;
  }();
  return t[std::clamp(qindex, 0, 255)];
}

const int* dct_matrix(int n) {
  return n == 4 ? tx_kernel_dct2_size4[FWD_TXFM][0] : n == 8 ? tx_kernel_dct2_size8[FWD_TXFM][0]
       : n == 16 ? tx_kernel_dct2_size16[FWD_TXFM][0] : n == 32 ? tx_kernel_dct2_size32[FWD_TXFM][0]
       : tx_kernel_dct2_size64[FWD_TXFM][0];
}

// The passes' shifts: AVM's total (fwd_tx_shift), but as little as possible in the first pass, so its rounding
// stays below the second's. The int32 sums bound it: |r| < 2^8 and sum |m| <= 64 N give 2^20 h w over 2^s1, so s1 =
// log2(h w) - 10 above 1024 samples, 0 below (AVM's own first-pass shifts cost +0.24% BD-rate, AIR +0.91%).
int shift1(TX_SIZE t) { return std::max(0, tx_size_wide_log2[t] + tx_size_high_log2[t] - 10); }
int shift2(TX_SIZE t) { return fwd_tx_shift[t][0] + fwd_tx_shift[t][1] - shift1(t); }

inline int32_t sqrt2_step(int64_t x) { return static_cast<int32_t>((x * NewSqrt2 + (1 << (NewSqrt2Bits - 1))) >> NewSqrt2Bits); }

// --- the quantizers ---

const SCAN_ORDER* scan_of(TX_SIZE t) { return get_scan(t, DCT_DCT); }
int coeffs_of(TX_SIZE t) { return std::min(int{tx_size_wide[t]}, 32) * std::min(int{tx_size_high[t]}, 32); }

// X2 = |c| << (4 + txs): 2 dqv times the coefficient in steps.
inline int x2_shift(TX_SIZE t) { return 4 + av2_get_tx_scale(t); }

// A coefficient's TCQ candidates per quantizer, as tcq_cands(): the nearest levels and zero (the last must keep a
// level), by parity in the order tried, each with its search term; a clamped repeat is left out.
struct Cands {
  int64_t ec[2][2], oc[2];  // costs (kInfV: none)
  int el[2][2], ol[2];      // levels
  int ne[2], no[2];
};
inline void cands_of(int64_t X, const Q& q, bool last, Cands* t) {
  for (int q1 = 0; q1 < 2; q1++) {
    t->ne[q1] = t->no[q1] = 0;
    t->ec[q1][0] = t->ec[q1][1] = t->oc[q1] = kInfV, t->el[q1][0] = t->el[q1][1] = t->ol[q1] = 0;
    const int base = q.div2(q1 ? X + q.dqv : X);
    const int cs[3] = {base, base + 1, 0};
    int prev = -1;
    for (int k = 0; k < 3; k++) {
      const int l = std::min(cs[k], q.maxl_tcq);
      if ((l == 0 && last) || (k == 2 && base == 0) || l == prev) continue;
      prev = l;
      const int64_t e = X - (l == 0 ? 0 : static_cast<int64_t>(2 * l - q1) * q.dqv);
      const int64_t j = ((e * e) << 8) + q.lam_tcq * lb512(l);
      if (l & 1) t->oc[q1] = j, t->ol[q1] = l, t->no[q1] = 1;
      else t->ec[q1][t->ne[q1]] = j, t->el[q1][t->ne[q1]++] = l;
    }
  }
}

int prep(const int32_t* coeff, int n, TX_SIZE t, const Q& q, const SCAN_ORDER* so, int32_t* x2) {
  const __m128i sh = _mm_cvtsi32_si128(x2_shift(t)), thr = _mm_set1_epi32(static_cast<int>(q.first) - 1);
  int eob = 0;
  for (int pos = 0; pos < n; pos += 4) {
    const __m128i a = _mm_sll_epi32(_mm_abs_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(coeff + pos))), sh);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(x2 + pos), a);
    for (int m = _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(a, thr))); m; m &= m - 1)
      eob = std::max(eob, so->iscan[pos + _tzcnt_u32(m)] + 1);
  }
  return eob;
}

constexpr int64_t kStart = int64_t{8} << 32;  // a step's previous state 8: the path starts here (--rd-zero)

int plain_levels(const int32_t* coeff, TX_SIZE t, int qindex, bool rd_zero, int32_t* levels) {
  const Q& q = q_of(qindex);
  const int n = coeffs_of(t), sh = x2_shift(t);
  const SCAN_ORDER* const so = scan_of(t);
  int64_t X[1024];
  for (int pos = 0; pos < n; pos++) {
    X[pos] = static_cast<int64_t>(std::abs(coeff[pos])) << sh;
    const int hi = q.div2(X[pos] + q.dqv);
    int level = 0;
    int64_t best = INT64_MAX;
    for (int l : {hi, hi - 1, 0}) {
      if (l < 0) continue;
      const int64_t e = X[pos] - 2 * static_cast<int64_t>(l) * q.dqv;
      const int64_t j = ((e * e) << 8) + q.lam_plain * lb512(l);
      if (j < best) best = j, level = l;
    }
    level = std::min(level, q.maxl_plain);
    levels[pos] = coeff[pos] < 0 ? -level : level;
  }
  int last = n - 1;
  while (last >= 0 && !levels[so->scan[last]]) last--;
  while (last >= 0 && std::abs(levels[so->scan[last]]) == 1) {
    int prev = last - 1;
    while (prev >= 0 && !levels[so->scan[prev]]) prev--;
    const int64_t x = X[so->scan[last]], e1 = x - 2 * static_cast<int64_t>(q.dqv);
    const int64_t dj = ((x * x - e1 * e1) << 8) - q.lam_plain * (lb512(1) + lb512(0) * (last - prev - 1) + 512);
    if (dj >= 0) break;
    levels[so->scan[last]] = 0;
    last = prev;
  }
  if (rd_zero && last >= 0) {  // zero_wins()
    const int eob = last + 1;
    int64_t with = q.lam_plain * eb_table()[eob], none = q.lam_plain * 512;
    for (int pos = 0; pos < n; pos++) {
      const int l = std::abs(levels[pos]);
      const int64_t e = X[pos] - 2 * static_cast<int64_t>(l) * q.dqv;
      with = add(with, ((e * e) << 8) + (so->iscan[pos] < eob ? q.lam_plain * lb512(l) : 0));
      none = add(none, (X[pos] * X[pos]) << 8);
    }
    if (none < with) {
      std::fill(levels, levels + n, 0);
      last = -1;
    }
  }
  return last + 1;
}

}  // namespace

int64_t lam_of(int qindex, double w) {
  const double d = av2_ac_quant_QTX(std::clamp(qindex, 0, 255), 0, 0, kQDepth);
  return std::llround(2 * w * d * d);
}

namespace {

constexpr int kFrac = 4;  // with gains: the bits the second pass keeps below the coefficient's unit

// The 64-point DCT2's outputs 32-63, which AVM's AVX2 form leaves out: dst[m * 64 + k] for the first `count` lines,
// as fwd_txfm_dct2_size64_c computes them (its first butterfly, then each output's sum). src[n * line + m]: sample n
// of line m.
void dct64_high(const int* src, int* dst, int shift, int line, int count) {
  const int* const M = tx_kernel_dct2_size64[FWD_TXFM][0];
  const int off = shift > 0 ? 1 << (shift - 1) : 0;
  int m = 0;
  for (; m + 8 <= count; m += 8) {
    __m256i a[32], b[32];
    for (int n = 0; n < 32; n++) {
      const __m256i x0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + n * line + m));
      const __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + (63 - n) * line + m));
      a[n] = _mm256_add_epi32(x0, x1), b[n] = _mm256_sub_epi32(x0, x1);
    }
    alignas(32) int out[32][8];
    const __m256i v_off = _mm256_set1_epi32(off);
    const __m128i sh = _mm_cvtsi32_si128(shift);
    for (int k = 32; k < 64; k++) {
      const __m256i* const v = (k & 1) ? b : a;
      __m256i s = v_off;
      for (int n = 0; n < 32; n++) s = _mm256_add_epi32(s, _mm256_mullo_epi32(v[n], _mm256_set1_epi32(M[k * 64 + n])));
      _mm256_store_si256(reinterpret_cast<__m256i*>(out[k - 32]), _mm256_sra_epi32(s, sh));
    }
    for (int i = 0; i < 8; i++)
      for (int k = 0; k < 32; k++) dst[(m + i) * 64 + 32 + k] = out[k][i];
  }
  for (; m < count; m++) {
    int a[32], b[32];
    for (int n = 0; n < 32; n++) a[n] = src[n * line + m] + src[(63 - n) * line + m], b[n] = src[n * line + m] - src[(63 - n) * line + m];
    for (int k = 32; k < 64; k++) {
      const int* const v = (k & 1) ? b : a;
      int s = off;
      for (int n = 0; n < 32; n++) s += M[k * 64 + n] * v[n];
      dst[m * 64 + k] = s >> shift;
    }
  }
}

// The same for the first pass of a 64-high block, from the residual itself (|r| < 2^8, so the butterfly's sums fit
// 16 bits): pairs of taps by _mm256_madd_epi16. dst[x * 64 + k], k 32-63, for the w columns.
void dct64_high_cols(const int16_t* res, int stride, int w, int shift, int* dst) {
  const int* const M = tx_kernel_dct2_size64[FWD_TXFM][0];
  const int off = shift > 0 ? 1 << (shift - 1) : 0;
  int x = 0;
  for (; x + 8 <= w; x += 8) {
    __m128i a[32], b[32];
    for (int n = 0; n < 32; n++) {
      const __m128i r0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(res + n * stride + x));
      const __m128i r1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(res + (63 - n) * stride + x));
      a[n] = _mm_add_epi16(r0, r1), b[n] = _mm_sub_epi16(r0, r1);
    }
    __m256i A[16], B[16];  // taps 2j and 2j + 1 side by side: columns 0-3 in the low half, 4-7 in the high
    for (int j = 0; j < 16; j++) {
      A[j] = _mm256_set_m128i(_mm_unpackhi_epi16(a[2 * j], a[2 * j + 1]), _mm_unpacklo_epi16(a[2 * j], a[2 * j + 1]));
      B[j] = _mm256_set_m128i(_mm_unpackhi_epi16(b[2 * j], b[2 * j + 1]), _mm_unpacklo_epi16(b[2 * j], b[2 * j + 1]));
    }
    alignas(32) int out[32][8];
    const __m256i v_off = _mm256_set1_epi32(off);
    const __m128i sh = _mm_cvtsi32_si128(shift);
    for (int k = 32; k < 64; k++) {
      const __m256i* const v = (k & 1) ? B : A;
      const int* const row = M + k * 64;
      __m256i s = v_off;
      for (int j = 0; j < 16; j++) {
        const int pair = static_cast<int>(static_cast<uint32_t>(static_cast<uint16_t>(row[2 * j])) |
                                          (static_cast<uint32_t>(static_cast<uint16_t>(row[2 * j + 1])) << 16));
        s = _mm256_add_epi32(s, _mm256_madd_epi16(v[j], _mm256_set1_epi32(pair)));
      }
      _mm256_store_si256(reinterpret_cast<__m256i*>(out[k - 32]), _mm256_sra_epi32(s, sh));
    }
    for (int i = 0; i < 8; i++)
      for (int k = 0; k < 32; k++) dst[(x + i) * 64 + 32 + k] = out[k][i];
  }
  for (; x < w; x++) {
    int a[32], b[32];
    for (int n = 0; n < 32; n++) {
      const int r0 = res[n * stride + x], r1 = res[(63 - n) * stride + x];
      a[n] = r0 + r1, b[n] = r0 - r1;
    }
    for (int k = 32; k < 64; k++) {
      const int* const v = (k & 1) ? b : a;
      int s = off;
      for (int n = 0; n < 32; n++) s += M[k * 64 + n] * v[n];
      dst[x * 64 + k] = s >> shift;
    }
  }
}

// Per row k of AVM's N-point DCT2, 2^14 x 64^2 N / |m_k|^2: an output's square times it is an orthonormal DCT's
// (times 64^2 N). The rows are off by up to 1%.
const int* norm_w2(int n) {
  static const std::array<std::array<int, 64>, 7> t = [] {
    std::array<std::array<int, 64>, 7> a{};
    for (int l = 2; l <= 6; l++) {
      const int N = 1 << l;
      const int* const m = dct_matrix(N);
      for (int k = 0; k < N; k++) {
        double e = 0;
        for (int i = 0; i < N; i++) e += static_cast<double>(m[k * N + i]) * m[k * N + i];
        a[l][k] = static_cast<int>(std::lround(16384.0 * 4096 * N / e));
      }
    }
    return a;
  }();
  return t[n == 4 ? 2 : n == 8 ? 3 : n == 16 ? 4 : n == 32 ? 5 : 6].data();
}

// A 64-point transform's dropped energy at an orthonormal DCT's scale times 4096^2 h w / 2^(2 total shift) (AVM's DC
// gain, no sqrt(2) step), each output by its rows' norm_w2():
// - rows 32-63 of a 64-high block whole, by Parseval over the row transform: the first pass's squares (b[x * 64 + v]
//   at shift1()) times 4096 w over 2^(2 shift2());
// - the kept rows' outputs 32-63 of a 64-wide one (a[v * 64 + u] at shift2()), exactly.
// 64x64's came out 7% below the float scoring's with the kept energy subtracted (AVM's rows are not quite
// orthogonal), and AM and chika lost 0.3%.
int64_t dropped_direct(const int* a, const int* b, TX_SIZE t) {
  const int w = tx_size_wide[t], h = tx_size_high[t], ch = std::min(h, 32);
  int64_t e = 0;
  if (h == 64) {
    const int* const nv = norm_w2(64);
    int64_t rows = 0;
    for (int v = 32; v < 64; v++) {
      int64_t r = 0;
      for (int x = 0; x < w; x++) r += static_cast<int64_t>(b[x * 64 + v]) * b[x * 64 + v];
      rows += (r * nv[v]) >> 14;
    }
    e += rows >> (2 * shift2(t) - 12 - tx_size_wide_log2[t]);
  }
  if (w == 64) {
    const int* const nv = norm_w2(h);
    const int* const nu = norm_w2(64);
    for (int v = 0; v < ch; v++)
      for (int u = 32; u < 64; u++) {
        const int64_t z = a[v * 64 + u];
        e += ((z * z) * ((static_cast<int64_t>(nv[v]) * nu[u]) >> 14)) >> 14;
      }
  }
  return e;
}

// x k / 2^14 rounded, exact for 0 <= x < 2^47.
int64_t mul_q14(int64_t x, int k) { return (x >> 14) * k + (((x & 16383) * k + 8192) >> 14); }

// AVM's DCT2 of res as av2_fwd_txfm2d() runs it (a 64-point pass keeps its low 32 outputs; the second pass skips the
// rows the first left out), with shift1() and shift2(): the low 32 x 32 into coeff, AVM's layout. frac > 0 (with
// gains): the second pass keeps `frac` more bits and no sqrt(2) step, for apply() to round once. *raw:
// dropped_direct() (0 below 64 points).
void fdct_avm(const int16_t* res, int stride, TX_SIZE t, int32_t* coeff, int frac, int64_t* raw) {
  const int w = tx_size_wide[t], h = tx_size_high[t], cw = std::min(w, 32), ch = std::min(h, 32);
  alignas(32) int a[64 * 64], b[64 * 64];
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) a[y * w + x] = res[y * stride + x];
  fwd_transform_1d_avx2(a, b, shift1(t), w, 0, h - ch, 0, tx_size_high_log2[t] - 2);  // b[x * h + v], v < ch
  if (h == 64) dct64_high_cols(res, stride, w, shift1(t), b);
  fwd_transform_1d_avx2(b, a, shift2(t) - frac, h, h - ch, w - cw, 0, tx_size_wide_log2[t] - 2);  // a[v * w + u], u < cw
  if (w == 64) dct64_high(b, a, shift2(t), h, ch);
  *raw = w == 64 || h == 64 ? dropped_direct(a, b, t) : 0;
  const bool s2 = !frac && ((tx_size_wide_log2[t] + tx_size_high_log2[t]) & 1);
  if (!s2 && w == cw) {
    std::memcpy(coeff, a, sizeof(int32_t) * cw * ch);
    return;
  }
  const __m256i k = _mm256_set1_epi64x(NewSqrt2), r = _mm256_set1_epi64x(1 << (NewSqrt2Bits - 1));
  for (int v = 0; v < ch; v++)
    for (int u = 0; u < cw; u += 4) {
      const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + v * w + u));
      if (!s2) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(coeff + v * cw + u), x);
        continue;
      }
      // x * 5793 is exact in 64 bits; AVM's round_shift on int64, its sign kept by hand (no 64-bit arithmetic shift).
      const __m256i p = _mm256_add_epi64(_mm256_mul_epi32(_mm256_cvtepi32_epi64(x), k), r);
      const __m256i neg = _mm256_cmpgt_epi64(_mm256_setzero_si256(), p);
      const __m256i q = _mm256_or_si256(_mm256_srli_epi64(p, NewSqrt2Bits), _mm256_slli_epi64(neg, 64 - NewSqrt2Bits));
      const __m256i lo = _mm256_permutevar8x32_epi32(q, _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(coeff + v * cw + u), _mm256_castsi256_si128(lo));
    }
}

// The same by plain matrix products, the dropped parts too.
void ref_avm(const int16_t* res, int stride, TX_SIZE t, int32_t* coeff, int frac, int64_t* raw) {
  const int w = tx_size_wide[t], h = tx_size_high[t], cw = std::min(w, 32), ch = std::min(h, 32);
  const int s0 = shift1(t), s1 = shift2(t);
  const int* const mv = dct_matrix(h);
  const int* const mh = dct_matrix(w);
  const bool s2 = !frac && ((tx_size_wide_log2[t] + tx_size_high_log2[t]) & 1);
  static thread_local int Y[64 * 64], A[64 * 64], B[64 * 64];
  for (int v = 0; v < h; v++)
    for (int x = 0; x < w; x++) {
      int64_t s = s0 > 0 ? 1 << (s0 - 1) : 0;
      for (int k = 0; k < h; k++) s += static_cast<int64_t>(mv[v * h + k]) * res[k * stride + x];
      Y[v * w + x] = static_cast<int>(s >> s0);
      if (v >= 32) B[x * 64 + v] = Y[v * w + x];
    }
  for (int v = 0; v < ch; v++)
    for (int u = 0; u < w; u++) {
      const int sh = u < 32 ? s1 - frac : s1;
      int64_t s = sh > 0 ? 1 << (sh - 1) : 0;
      for (int k = 0; k < w; k++) s += static_cast<int64_t>(mh[u * w + k]) * Y[v * w + k];
      const int z = static_cast<int>(s >> sh);
      if (u >= 32) A[v * 64 + u] = z;
      else if (u < cw) coeff[v * cw + u] = s2 ? sqrt2_step(z) : z;
    }
  *raw = w == 64 || h == 64 ? dropped_direct(A, B, t) : 0;
}

// To the coder's scale: each coefficient times its gain / 2^14 (from 1/16, kFrac, rounded once), the dropped energy
// times the DC gain's square (dropped_direct() is at AVM's DC gain without the sqrt(2) step).
void apply(const Gains* g, TX_SIZE t, int32_t* coeff, int64_t raw, int64_t* dropped) {
  const int n = coeffs_of(t);
  if (g) {  // 64-bit products (|c| < 2^21, gains < 2^15)
    const __m256i r = _mm256_set1_epi64x(int64_t{1} << (13 + kFrac));
    for (int i = 0; i < n; i += 4) {
      const __m256i c = _mm256_cvtepi32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i*>(coeff + i)));
      const __m256i k = _mm256_cvtepi32_epi64(_mm_load_si128(reinterpret_cast<const __m128i*>(g->g14 + i)));
      const __m256i p = _mm256_add_epi64(_mm256_mul_epi32(c, k), r);
      // An arithmetic shift of int64 lanes: AVX2 has none, so the sign by hand.
      const __m256i neg = _mm256_cmpgt_epi64(_mm256_setzero_si256(), p);
      const __m256i q = _mm256_or_si256(_mm256_srli_epi64(p, 14 + kFrac), _mm256_slli_epi64(neg, 64 - 14 - kFrac));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(coeff + i),
                       _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(q, _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0))));
    }
  }
  const bool s2 = (tx_size_wide_log2[t] + tx_size_high_log2[t]) & 1;
  const int k0 = g ? g->g14[0] : s2 ? (16384 * NewSqrt2 + 2048) / 4096 : 16384;
  *dropped = raw ? mul_q14(mul_q14(raw, k0), k0) : 0;
}

}  // namespace

void fdct(const int16_t* res, int stride, TX_SIZE t, const Gains* g, int32_t* coeff, int64_t* dropped) {
  celv2::HotScope prof(celv2::Stage::Transform);
  int64_t raw;
  fdct_avm(res, stride, t, coeff, g ? kFrac : 0, &raw);
  apply(g, t, coeff, raw, dropped);
}

void fdct_ref(const int16_t* res, int stride, TX_SIZE t, const Gains* g, int32_t* coeff, int64_t* dropped) {
  int64_t raw;
  ref_avm(res, stride, t, coeff, g ? kFrac : 0, &raw);
  if (g)
    for (int i = 0; i < coeffs_of(t); i++)
      coeff[i] = static_cast<int32_t>((static_cast<int64_t>(coeff[i]) * g->g14[i] + (int64_t{1} << (13 + kFrac))) >> (14 + kFrac));
  const bool s2 = (tx_size_wide_log2[t] + tx_size_high_log2[t]) & 1;
  const int k0 = g ? g->g14[0] : s2 ? (16384 * NewSqrt2 + 2048) / 4096 : 16384;
  *dropped = raw ? mul_q14(mul_q14(raw, k0), k0) : 0;
}

void gains_of(TX_SIZE t, double scale, Gains* g) {
  // Per frequency, AVM's gain over an orthonormal DCT: each pass's row k gives |m_k| (about 64 sqrt(N), off by up to
  // 1% on AC rows), over 2^(both shifts), sqrt(2) on 2:1 sizes.
  const int w = tx_size_wide[t], h = tx_size_high[t], cw = std::min(w, 32), ch = std::min(h, 32);
  const bool s2 = (tx_size_wide_log2[t] + tx_size_high_log2[t]) & 1;
  auto norm = [](int n, int k) {
    const int* const m = dct_matrix(n) + k * n;
    double e = 0;
    for (int i = 0; i < n; i++) e += static_cast<double>(m[i]) * m[i];
    return std::sqrt(e);
  };
  (void)s2;  // the sqrt(2) of 2:1 sizes is in the norms' product against scale, no step of its own
  const double base = 1.0 / (1 << (fwd_tx_shift[t][0] + fwd_tx_shift[t][1]));
  std::fill(std::begin(g->g14), std::end(g->g14), 1 << 14);
  for (int v = 0; v < ch; v++)
    for (int u = 0; u < cw; u++) g->g14[v * cw + u] = static_cast<int>(std::lround(16384 * scale / (norm(h, v) * norm(w, u) * base)));
}
int tcq_ref(const int32_t* coeff, TX_SIZE t, int qindex, bool rd_zero, int32_t* levels) {
  static const uint8_t next[8][2] = {{0, 4}, {4, 0}, {1, 5}, {5, 1}, {6, 2}, {2, 6}, {7, 3}, {3, 7}};
  const Q& q = q_of(qindex);
  const int n = coeffs_of(t), sh = x2_shift(t);
  const SCAN_ORDER* const so = scan_of(t);
  int64_t X[1024];
  int eob = 0;
  for (int pos = 0; pos < n; pos++) {
    X[pos] = static_cast<int64_t>(std::abs(coeff[pos])) << sh;
    if (X[pos] >= q.first) eob = std::max(eob, so->iscan[pos] + 1);
  }
  std::fill(levels, levels + n, 0);
  if (!eob) return 0;
  static thread_local int64_t step[1024][8];
  int64_t path[8];
  for (int s = 0; s < 8; s++) path[s] = s == 0 && !rd_zero ? 0 : kInfV;
  int64_t tail = 0;
  for (int c = eob - 1; c >= 0; c--) {
    const int pos = so->scan[c];
    Cands k;
    cands_of(X[pos], q, !rd_zero && c == eob - 1, &k);
    // Each next state from a pair (a, a + 1) of states with the same quantizer: the first of the pair's next
    // states takes a's even level or a + 1's odd one (the odd on a strict win), the second a + 1's even or a's odd
    // (the odd winning a tie). Within a state, the second even candidate on a strict win.
    int64_t np[8];
    for (int a = 0; a < 8; a += 2) {
      const int q1 = (a & 2) ? 1 : 0;
      for (int side = 0; side < 2; side++) {
        const int self = a + side, other = a + 1 - side;
        const int64_t c0 = path[self] + k.ec[q1][0], c1 = path[self] + k.ec[q1][1];
        const bool second_even = c1 < c0;
        const int64_t even = second_even ? c1 : c0;
        const int64_t odd = path[other] + k.oc[q1];
        const bool take = side ? odd <= even : odd < even;
        const int ns = next[self][0];  // == next[other][1]
        np[ns] = std::min(take ? odd : even, kInfV);
        step[c][ns] = take ? (k.ol[q1] | (static_cast<int64_t>(other) << 32))
                           : (k.el[q1][second_even ? 1 : 0] | (static_cast<int64_t>(self) << 32));
      }
    }
    std::copy(np, np + 8, path);
    if (rd_zero) {
      if (X[pos] >= q.first) {
        const int64_t s = tail + q.lam_tcq * eb_table()[c + 1];
        if (k.ne[0] && k.el[0][0] && s + k.ec[0][0] < path[0]) path[0] = s + k.ec[0][0], step[c][0] = k.el[0][0] | kStart;
        if (k.no[0] && s + k.oc[0] < path[4]) path[4] = s + k.oc[0], step[c][4] = k.ol[0] | kStart;
      }
      tail = std::min(tail + ((X[pos] * X[pos]) << 8), kInfV);
    }
  }
  int s = 0;
  for (int k = 1; k < 8; k++)
    if (path[k] < path[s]) s = k;
  if (rd_zero && tail + q.lam_tcq * 512 < path[s]) return 0;
  for (int c = 0; c < eob; c++) {
    const int pos = so->scan[c];
    const int l = static_cast<int16_t>(step[c][s]);
    levels[pos] = coeff[pos] < 0 ? -l : l;
    s = static_cast<int>(step[c][s] >> 32);
    if (s == 8) return c + 1;
  }
  return eob;
}

int tcq_vec(const int32_t* coeff, TX_SIZE t, int qindex, bool rd_zero, int32_t* levels) {
  const Q& q = q_of(qindex);
  const int n = coeffs_of(t);
  const SCAN_ORDER* const so = scan_of(t);
  alignas(32) int32_t x2[1024];
  const int eob = prep(coeff, n, t, q, so, x2);
  std::fill(levels, levels + n, 0);
  if (!eob) return 0;
  alignas(32) static thread_local int64_t step[1024][8];  // level | previous state << 32
  const __m256i inf = _mm256_set1_epi64x(kInfV);
  __m256i lo = _mm256_setr_epi64x(rd_zero ? kInfV : 0, kInfV, kInfV, kInfV), hi = inf;  // states 0-3, 4-7
  const __m256i from_lo = _mm256_setr_epi64x(0ll << 32, 1ll << 32, 2ll << 32, 3ll << 32);
  const __m256i from_hi = _mm256_setr_epi64x(4ll << 32, 5ll << 32, 6ll << 32, 7ll << 32);
  const __m256i second = _mm256_setr_epi64x(0, -1, 0, -1);  // a pair's second next state
  const __m256i ones = _mm256_set1_epi64x(-1);
  int64_t tail = 0;
  for (int c = eob - 1; c >= 0; c--) {
    const int pos = so->scan[c];
    const int64_t X = x2[pos];
    Cands k;
    cands_of(X, q, !rd_zero && c == eob - 1, &k);
    auto per = [](int64_t a, int64_t b) { return _mm256_setr_epi64x(a, a, b, b); };
    const __m256i e0 = per(k.ec[0][0], k.ec[1][0]), e1 = per(k.ec[0][1], k.ec[1][1]), ov = per(k.oc[0], k.oc[1]);
    const __m256i l0 = per(k.el[0][0], k.el[1][0]), l1 = per(k.el[0][1], k.el[1][1]), olv = per(k.ol[0], k.ol[1]);
    __m256i f[2], g[2];
    for (int h = 0; h < 2; h++) {
      const __m256i p = h ? hi : lo, from = h ? from_hi : from_lo;
      const __m256i c0 = _mm256_add_epi64(p, e0), c1 = _mm256_add_epi64(p, e1), odd = _mm256_add_epi64(p, ov);
      const __m256i lt = _mm256_cmpgt_epi64(c0, c1);  // c1 < c0
      const __m256i even = _mm256_blendv_epi8(c0, c1, lt);
      const __m256i even_step = _mm256_blendv_epi8(_mm256_or_si256(l0, from), _mm256_or_si256(l1, from), lt);
      const __m256i other = _mm256_shuffle_epi32(odd, 0x4e);  // the pair's other state's odd level
      const __m256i other_step = _mm256_shuffle_epi32(_mm256_or_si256(olv, from), 0x4e);
      const __m256i olt = _mm256_cmpgt_epi64(even, other), ole = _mm256_xor_si256(_mm256_cmpgt_epi64(other, even), ones);
      const __m256i take = _mm256_blendv_epi8(olt, ole, second);
      const __m256i v = _mm256_blendv_epi8(even, other, take);
      f[h] = _mm256_blendv_epi8(v, inf, _mm256_cmpgt_epi64(v, inf));
      g[h] = _mm256_blendv_epi8(even_step, other_step, take);
    }
    // f[0] holds the next states 0 4 1 5, f[1] 6 2 7 3: as 0 1 4 5 and 2 3 6 7, then halves.
    const __m256i a = _mm256_permute4x64_epi64(f[0], 0b11011000), d = _mm256_permute4x64_epi64(f[1], 0b10001101);
    lo = _mm256_permute2x128_si256(a, d, 0x20), hi = _mm256_permute2x128_si256(a, d, 0x31);
    const __m256i ga = _mm256_permute4x64_epi64(g[0], 0b11011000), gd = _mm256_permute4x64_epi64(g[1], 0b10001101);
    _mm256_store_si256(reinterpret_cast<__m256i*>(step[c]), _mm256_permute2x128_si256(ga, gd, 0x20));
    _mm256_store_si256(reinterpret_cast<__m256i*>(step[c] + 4), _mm256_permute2x128_si256(ga, gd, 0x31));
    if (rd_zero) {
      if (X >= q.first) {
        const int64_t s = tail + q.lam_tcq * eb_table()[c + 1];
        if (k.ne[0] && k.el[0][0] && s + k.ec[0][0] < _mm256_extract_epi64(lo, 0)) {
          lo = _mm256_blend_epi32(lo, _mm256_set1_epi64x(s + k.ec[0][0]), 0x03);
          step[c][0] = k.el[0][0] | kStart;
        }
        if (k.no[0] && s + k.oc[0] < _mm256_extract_epi64(hi, 0)) {
          hi = _mm256_blend_epi32(hi, _mm256_set1_epi64x(s + k.oc[0]), 0x03);
          step[c][4] = k.ol[0] | kStart;
        }
      }
      tail = std::min(tail + ((X * X) << 8), kInfV);
    }
  }
  alignas(32) int64_t path[8];
  _mm256_store_si256(reinterpret_cast<__m256i*>(path), lo);
  _mm256_store_si256(reinterpret_cast<__m256i*>(path + 4), hi);
  int s = 0;
  for (int k = 1; k < 8; k++)
    if (path[k] < path[s]) s = k;
  if (rd_zero && tail + q.lam_tcq * 512 < path[s]) return 0;  // all zero, the levels winning a tie
  for (int c = 0; c < eob; c++) {
    const int pos = so->scan[c];
    const int l = static_cast<int16_t>(step[c][s]);
    levels[pos] = coeff[pos] < 0 ? -l : l;
    s = static_cast<int>(step[c][s] >> 32);
    if (s == 8) return c + 1;
  }
  return eob;
}

TxCost sums(const int32_t* coeff, int64_t dropped, const int32_t* levels, int end, TX_SIZE t, int qindex, bool tcq,
            int32_t* dq_out) {
  static const uint8_t next[8][2] = {{0, 4}, {4, 0}, {1, 5}, {5, 1}, {6, 2}, {2, 6}, {7, 3}, {3, 7}};
  const Q& q = q_of(qindex);
  const int n = coeffs_of(t), txs = av2_get_tx_scale(t);
  const SCAN_ORDER* const so = scan_of(t);
  int32_t dq[1024];
  std::fill(dq, dq + n, 0);
  TxCost c;
  c.bits = eb_table()[end];
  constexpr int32_t kMax = (1 << 15) - 1, kMin = -(1 << 15);  // 8 bits: the decoder's clamp
  int state = 0;
  for (int i = end - 1; i >= 0; i--) {
    const int pos = so->scan[i];
    const int l = std::abs(levels[pos]);
    c.bits += lb512(l);
    if (!l) {
      if (tcq) state = next[state][0];
      continue;
    }
    if (tcq) {
      const int64_t q_idx = 2 * l - ((state & 2) ? 1 : 0);
      const int64_t dq_hp = q_idx * q.dqv & 0xffffff;
      const int32_t v = static_cast<int32_t>(ROUND_POWER_OF_TWO_64(dq_hp, QUANT_TABLE_BITS)) >> (txs + 1);
      dq[pos] = std::clamp(levels[pos] < 0 ? -v : v, kMin, kMax);
      state = next[state][l & 1];
    } else {
      dq[pos] = dequant_coeff(levels[pos], q.dqv, t, 8);
    }
  }
  int64_t e2 = 0;
  for (int pos = 0; pos < n; pos++) {
    const int64_t e = static_cast<int64_t>(coeff[pos]) - dq[pos];
    e2 += e * e;
  }
  e2 += dropped;
  const int sh = 16 + 2 * txs;
  c.d = e2 >= (kInf >> sh) ? kInf : e2 << sh;
  if (dq_out) std::copy(dq, dq + n, dq_out);
  return c;
}

TxCost quant_cost(const int32_t* coeff, int64_t dropped, TX_SIZE t, int qindex, bool tcq, bool rd_zero, int32_t* levels,
                  int32_t* dq) {
  celv2::HotScope prof(celv2::Stage::Quant);
  int32_t own[1024];
  int32_t* const lv = levels ? levels : own;
  const int end = tcq ? tcq_vec(coeff, t, qindex, rd_zero, lv) : plain_levels(coeff, t, qindex, rd_zero, lv);
  return sums(coeff, dropped, lv, end, t, qindex, tcq, dq);
}

}  // namespace celv2::rdi
