#include "core/wiener_sums.h"

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include "core/threads.h"

namespace celv2 {

namespace {

constexpr int N = kWienerSums;
constexpr int kChunk = 128;  // pixels a pass: a lane holds 16 products of at most 2^26 (12 bits), below 2^31
constexpr int kPairs = N * (N + 1) / 2;

// acc[p] += every pair i <= j of features f[0..k) (rows of stride kChunk, m16 samples, a multiple of 16)'s dot product,
// widened to 64 bits; f_i loaded once for four pairs (1.25 loads a madd, not 2).
void madd_pairs(const int16_t* f, int k, int m16, __m256i* acc) {
  auto ld = [&](int i, int q) { return _mm256_load_si256(reinterpret_cast<const __m256i*>(f + i * kChunk + q)); };
  auto widen = [](__m256i* d, __m256i s) {
    *d = _mm256_add_epi64(*d, _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(s)),
                                               _mm256_cvtepi32_epi64(_mm256_extracti128_si256(s, 1))));
  };
  for (int i = 0, p = 0; i < k; i++) {
    int j = i;
    for (; j + 4 <= k; j += 4, p += 4) {
      __m256i s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0;
      for (int q = 0; q < m16; q += 16) {
        const __m256i fi = ld(i, q);
        s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(fi, ld(j, q)));
        s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(fi, ld(j + 1, q)));
        s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(fi, ld(j + 2, q)));
        s3 = _mm256_add_epi32(s3, _mm256_madd_epi16(fi, ld(j + 3, q)));
      }
      widen(acc + p, s0), widen(acc + p + 1, s1), widen(acc + p + 2, s2), widen(acc + p + 3, s3);
    }
    for (; j < k; j++, p++) {
      __m256i s = _mm256_setzero_si256();
      for (int q = 0; q < m16; q += 16) s = _mm256_add_epi32(s, _mm256_madd_epi16(ld(i, q), ld(j, q)));
      widen(acc + p, s);
    }
  }
}

}  // namespace

// The features of kChunk pixels at a time as int16 rows (12-bit samples: at most 8190), then every pair's dot
// product by _mm256_madd_epi16 (16 products an instruction against 4 in doubles), its 32-bit lanes widened to 64
// bits after each chunk.
void wiener_rect_sums(const uint16_t* x, int st, const int16_t* luma, int lw, const uint16_t* s16, const uint8_t* s8,
                      int sst, int x0, int x1, int y0, int y1, const WienerTaps& t, int64_t* out) {
  std::fill(out, out + kPairs, int64_t{0});
  if (x1 <= x0 || y1 <= y0) return;
  alignas(32) int16_t f[N][kChunk];
  alignas(32) __m256i acc[kPairs];
  for (__m256i& a : acc) a = _mm256_setzero_si256();
  for (int r = y0; r < y1; r++)
    for (int c0 = x0; c0 < x1; c0 += kChunk) {
      const int m = std::min(kChunk, x1 - c0), m16 = (m + 15) & ~15;
      std::memset(f, 0, sizeof(f));
      const uint16_t* const o = x + static_cast<size_t>(r) * st + c0;
      const int16_t* const l = luma + static_cast<size_t>(r) * lw + c0;
      for (int q = 0; q < t.nx; q++) {
        int16_t* const d = f[t.px[q]];
        const int dx = t.dx[q];
        for (int c = 0; c < m; c++) d[c] = static_cast<int16_t>(o[c + dx] + o[c - dx] - 2 * o[c]);
      }
      for (int q = 0; q < t.nl; q++) {
        int16_t* const d = f[t.pl[q]];
        const int dl = t.dl[q];
        for (int c = 0; c < m; c++) d[c] = static_cast<int16_t>(l[c + dl] - l[c]);
      }
      const size_t at = static_cast<size_t>(r) * sst + c0;
      int16_t* const tg = f[kWienerFeatures];
      if (s16)
        for (int c = 0; c < m; c++) tg[c] = static_cast<int16_t>(s16[at + c] - o[c]);
      else
        for (int c = 0; c < m; c++) tg[c] = static_cast<int16_t>(s8[at + c] - o[c]);
      madd_pairs(f[0], N, m16, acc);
    }
  for (int p = 0; p < kPairs; p++) {
    alignas(32) int64_t v[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(v), acc[p]);
    out[p] = v[0] + v[1] + v[2] + v[3];
  }
}

void wiener_sums(const uint16_t* x, int st, const int16_t* luma, int lw, const uint16_t* s16, const uint8_t* s8,
                 int sst, int w, int h, const WienerTaps& t, int64_t sums[kWienerSums][kWienerSums], int bands) {
  std::memset(sums, 0, sizeof(int64_t) * N * N);
  const int rows = h - 4, n = w - 4;
  if (rows <= 0 || n <= 0) return;
  bands = std::max(1, std::min(bands, rows));
  std::vector<std::array<int64_t, kPairs>> part(bands);
  parallel(bands, [&](int k) {
    wiener_rect_sums(x, st, luma, lw, s16, s8, sst, 2, w - 2, 2 + rows * k / bands, 2 + rows * (k + 1) / bands, t, part[k].data());
  });
  int p = 0;
  for (int i = 0; i < N; i++)
    for (int j = i; j < N; j++, p++)
      for (const auto& q : part) sums[i][j] += q[p];
}

// A row's features for up to kChunk samples (vectorised), then each 4-sample group (one 4x4 block: one class) copied to
// its class's buffer; a full buffer's pairs go through _mm256_madd_epi16 as above.
void wiener_class_sums(const uint16_t* x, int st, const uint16_t* s16, const uint8_t* s8, int sst, int x0, int x1, int y0,
                       int y1, int ystep, const WienerTaps& t, const uint8_t* cls, int cst, const uint8_t* sub, int classes,
                       int64_t* sums) {
  const int k = t.nx + 1, pairs = k * (k + 1) / 2;
  std::fill(sums, sums + static_cast<size_t>(classes) * pairs, int64_t{0});
  if (x1 <= x0 || y1 <= y0) return;
  // Kept per thread: a fresh 150 KB a call (a unit) from 48 threads at once cost more than the sums.
  static thread_local std::vector<__m256i> mem;
  mem.assign(static_cast<size_t>(classes + 1) * k * kChunk / 16 + static_cast<size_t>(classes) * pairs, __m256i{});
  int16_t* const row = reinterpret_cast<int16_t*>(mem.data());  // k x kChunk
  int16_t* const buf = row + static_cast<size_t>(k) * kChunk;   // classes x k x kChunk
  __m256i* const acc = mem.data() + static_cast<size_t>(classes + 1) * k * kChunk / 16;  // classes x pairs
  int fill[16] = {};
  auto flush = [&](int c) {
    const int m16 = (fill[c] + 15) & ~15;
    int16_t* const f = buf + static_cast<size_t>(c) * k * kChunk;
    for (int i = 0; i < k; i++) std::fill(f + i * kChunk + fill[c], f + i * kChunk + m16, int16_t{0});
    madd_pairs(f, k, m16, acc + static_cast<size_t>(c) * pairs);
    fill[c] = 0;
  };
  for (int y = y0; y < y1; y += ystep)
    for (int c0 = x0; c0 < x1; c0 += kChunk) {
      const int m = std::min(kChunk, x1 - c0), m4 = (m + 3) & ~3;
      const uint16_t* const o = x + static_cast<size_t>(y) * st + c0;
      for (int q = 0; q < t.nx; q++) {  // in 16 bits (12-bit samples cannot overflow); reads up to 15 samples past m
        int16_t* const d = row + q * kChunk;
        const uint16_t *const a = o + t.dx[q], *const b = o - t.dx[q];
        for (int c = 0; c < m; c += 16) {
          const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(o + c));
          const __m256i s = _mm256_add_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + c)),
                                             _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + c)));
          _mm256_store_si256(reinterpret_cast<__m256i*>(d + c), _mm256_sub_epi16(s, _mm256_add_epi16(v, v)));
        }
      }
      int16_t* const tg = row + t.nx * kChunk;
      const size_t at = static_cast<size_t>(y) * sst + c0;
      if (s16)
        for (int c = 0; c < m; c++) tg[c] = static_cast<int16_t>(s16[at + c] - o[c]);
      else
        for (int c = 0; c < m; c++) tg[c] = static_cast<int16_t>(s8[at + c] - o[c]);
      for (int i = 0; i < k; i++) std::fill(row + i * kChunk + m, row + i * kChunk + m4, int16_t{0});
      const uint8_t* const cl = cls + static_cast<size_t>(y >> 2) * cst + (c0 >> 2);
      for (int g = 0; g < m;) {  // runs of groups of one class, one copy a feature
        const int c = sub[cl[g >> 2]], end = std::min(m, g + kChunk - fill[c]);
        int e = g + 4;
        while (e < end && sub[cl[e >> 2]] == c) e += 4;
        int16_t* const f = buf + static_cast<size_t>(c) * k * kChunk + fill[c];
        if (e - g == 4)  // a fixed-size copy (a variable one calls memcpy)
          for (int i = 0; i < k; i++) std::memcpy(f + i * kChunk, row + i * kChunk + g, 4 * sizeof(int16_t));
        else
          for (int i = 0; i < k; i++) std::memcpy(f + i * kChunk, row + i * kChunk + g, (e - g) * sizeof(int16_t));
        if ((fill[c] += e - g) == kChunk) flush(c);
        g = e;
      }
    }
  for (int c = 0; c < classes; c++) {
    if (fill[c]) flush(c);
    for (int p = 0; p < pairs; p++) {
      alignas(32) int64_t v[4];
      _mm256_store_si256(reinterpret_cast<__m256i*>(v), acc[static_cast<size_t>(c) * pairs + p]);
      sums[static_cast<size_t>(c) * pairs + p] = v[0] + v[1] + v[2] + v[3];
    }
  }
}

void wiener_class_sums_ref(const uint16_t* x, int st, const uint16_t* s16, const uint8_t* s8, int sst, int x0, int x1,
                           int y0, int y1, int ystep, const WienerTaps& t, const uint8_t* cls, int cst, const uint8_t* sub,
                           int classes, int64_t* sums) {
  const int k = t.nx + 1, pairs = k * (k + 1) / 2;
  std::fill(sums, sums + static_cast<size_t>(classes) * pairs, int64_t{0});
  for (int y = y0; y < y1; y += ystep)
    for (int c = x0; c < x1; c++) {
      int64_t f[kWienerSums] = {};
      const uint16_t* const o = x + static_cast<size_t>(y) * st + c;
      for (int q = 0; q < t.nx; q++) f[q] = o[t.dx[q]] + o[-t.dx[q]] - 2 * o[0];
      const size_t at = static_cast<size_t>(y) * sst + c;
      f[t.nx] = (s16 ? s16[at] : s8[at]) - o[0];
      int64_t* const s = sums + static_cast<size_t>(sub[cls[static_cast<size_t>(y >> 2) * cst + (c >> 2)]]) * pairs;
      for (int i = 0, p = 0; i < k; i++)
        for (int j = i; j < k; j++, p++) s[p] += f[i] * f[j];
    }
}

void wiener_sums_ref(const uint16_t* x, int st, const int16_t* luma, int lw, const uint16_t* s16, const uint8_t* s8,
                     int sst, int w, int h, const WienerTaps& t, int64_t sums[kWienerSums][kWienerSums]) {
  std::memset(sums, 0, sizeof(int64_t) * kWienerSums * kWienerSums);
  for (int r = 2; r < h - 2; r++)
    for (int c = 2; c < w - 2; c++) {
      int64_t f[kWienerSums] = {};
      const uint16_t* const o = x + static_cast<size_t>(r) * st + c;
      const int16_t* const l = luma + static_cast<size_t>(r) * lw + c;
      for (int k = 0; k < t.nx; k++) f[t.px[k]] = o[t.dx[k]] + o[-t.dx[k]] - 2 * o[0];
      for (int k = 0; k < t.nl; k++) f[t.pl[k]] = l[t.dl[k]] - l[0];
      const size_t at = static_cast<size_t>(r) * sst + c;
      f[kWienerFeatures] = (s16 ? s16[at] : s8[at]) - o[0];
      for (int i = 0; i < kWienerSums; i++)
        for (int j = i; j < kWienerSums; j++) sums[i][j] += f[i] * f[j];
    }
}

}  // namespace celv2
