#include "core/ccso_apply.h"

#include <algorithm>
#include <immintrin.h>
#include <stdexcept>

namespace celv2 {

void ccso_neighbours(int ext_filter_support, int s, int loc[2]) {
  // Around the sample x:  4 2 0 3 5 / 6 1 x 1 6 / 5 3 0 2 4
  static const int dy[7] = {-1, 0, -1, -1, -1, 1, 0}, dx[7] = {0, -1, -1, 1, -2, -2, 2};
  if (ext_filter_support < 0 || ext_filter_support > 6) throw std::invalid_argument("ext_filter_support must be 0..6");
  loc[0] = dy[ext_filter_support] * s + dx[ext_filter_support];
  loc[1] = -loc[0];
}

namespace {

enum Mode { kBandOnly, kEdge3, kEdge2 };  // bo_only; edge_clf 0 (classes 0..2); edge_clf 1 (0..1)

// 16 luma samples at p[k << SX].
template <int SX>
inline __m256i luma16(const uint16_t* p) {
  if (SX == 0) return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
  const __m256i low = _mm256_set1_epi32(0xffff);
  const __m256i a = _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)), low);
  const __m256i b = _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 16)), low);
  return _mm256_permute4x64_epi64(_mm256_packus_epi32(a, b), 0xd8);
}

template <int M>
inline int class_of(int d, int thr) {
  if (M == kEdge3) return d > thr ? 2 : d < -thr ? 0 : 1;
  return d < -thr ? 0 : 1;
}

template <int SX, int M>
void row_blocks(const CcsoPlane& p, int row, const int* table) {
  const int y0 = row << kCcsoBlockLog2, y1 = std::min(p.h, y0 + (1 << kCcsoBlockLog2));
  const int shift = p.bit_depth - p.max_band_log2, maxv = (1 << p.bit_depth) - 1;
  const bool one_band = p.max_band_log2 == 0;
  const __m256i thr = _mm256_set1_epi16(static_cast<int16_t>(p.thr)), nthr = _mm256_set1_epi16(static_cast<int16_t>(-p.thr));
  const __m256i one = _mm256_set1_epi16(1), zero = _mm256_setzero_si256(), vmax = _mm256_set1_epi16(static_cast<int16_t>(maxv));
  const __m128i vshift = _mm_cvtsi32_si128(shift);
  auto vclass = [&](__m256i d) {
    if (M == kEdge3) return _mm256_add_epi16(_mm256_sub_epi16(one, _mm256_cmpgt_epi16(d, thr)), _mm256_cmpgt_epi16(nthr, d));
    return _mm256_add_epi16(one, _mm256_cmpgt_epi16(nthr, d));
  };
  const uint8_t* on = p.on + static_cast<size_t>(row) * p.on_stride;
  for (int bx = 0; (bx << kCcsoBlockLog2) < p.w; bx++) {
    if (!on[bx]) continue;
    const int x0 = bx << kCcsoBlockLog2, x1 = std::min(p.w, x0 + (1 << kCcsoBlockLog2));
    for (int y = y0; y < y1; y++) {
      const uint16_t* const lr = p.luma + static_cast<ptrdiff_t>(y << p.sy) * p.luma_stride;
      uint16_t* const dr = p.dst + static_cast<ptrdiff_t>(y) * p.dst_stride;
      int x = x0;
      for (; x + 16 <= x1; x += 16) {
        const uint16_t* const l = lr + (x << SX);
        const __m256i c = luma16<SX>(l);
        __m256i idx = one_band ? zero : _mm256_slli_epi16(_mm256_srl_epi16(c, vshift), 4);
        if (M != kBandOnly) {
          const __m256i c0 = vclass(_mm256_sub_epi16(luma16<SX>(l + p.loc[0]), c));
          const __m256i c1 = vclass(_mm256_sub_epi16(luma16<SX>(l + p.loc[1]), c));
          idx = _mm256_add_epi16(idx, _mm256_add_epi16(_mm256_slli_epi16(c0, 2), c1));
        }
        const __m256i g0 = _mm256_i32gather_epi32(table, _mm256_cvtepu16_epi32(_mm256_castsi256_si128(idx)), 4);
        const __m256i g1 = _mm256_i32gather_epi32(table, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(idx, 1)), 4);
        const __m256i off = _mm256_permute4x64_epi64(_mm256_packs_epi32(g0, g1), 0xd8);
        __m256i* const d = reinterpret_cast<__m256i*>(dr + x);
        const __m256i v = _mm256_add_epi16(_mm256_loadu_si256(d), off);
        _mm256_storeu_si256(d, _mm256_min_epi16(_mm256_max_epi16(v, zero), vmax));
      }
      for (; x < x1; x++) {
        const uint16_t* const l = lr + (x << SX);
        int idx = one_band ? 0 : (l[0] >> shift) << 4;
        if (M != kBandOnly) idx += (class_of<M>(l[p.loc[0]] - l[0], p.thr) << 2) + class_of<M>(l[p.loc[1]] - l[0], p.thr);
        dr[x] = static_cast<uint16_t>(std::clamp(dr[x] + table[idx], 0, maxv));
      }
    }
  }
}

}  // namespace

void ccso_apply_row(const CcsoPlane& p, int row) {
  if (p.sx < 0 || p.sx > 1 || p.max_band_log2 < 0 || p.max_band_log2 > 6 || p.max_band_log2 > p.bit_depth)
    throw std::invalid_argument("ccso_apply_row: unsupported plane");
  int table[64 * 16];  // the offsets widened for the gathers
  const int n = 16 << p.max_band_log2;
  for (int k = 0; k < n; k++) table[k] = p.offset[k];
  const int m = p.bo_only ? kBandOnly : p.edge_clf ? kEdge2 : kEdge3;
  if (p.sx) {
    if (m == kBandOnly) row_blocks<1, kBandOnly>(p, row, table);
    else if (m == kEdge3) row_blocks<1, kEdge3>(p, row, table);
    else row_blocks<1, kEdge2>(p, row, table);
  } else {
    if (m == kBandOnly) row_blocks<0, kBandOnly>(p, row, table);
    else if (m == kEdge3) row_blocks<0, kEdge3>(p, row, table);
    else row_blocks<0, kEdge2>(p, row, table);
  }
}

}  // namespace celv2
