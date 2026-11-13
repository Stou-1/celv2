#include "core/ccso_apply.h"

#include <algorithm>
#include <immintrin.h>
#include <stdexcept>
#include <type_traits>

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

// The offset index (band << 4) + (class0 << 2) + class1 of the sample whose luma is at l: 16 at once, or one.
template <int SX, int M>
struct Classes {
  explicit Classes(const CcsoPlane& p)
      : p(p), shift(p.bit_depth - p.max_band_log2), one_band(p.max_band_log2 == 0),
        thr(_mm256_set1_epi16(static_cast<int16_t>(p.thr))), nthr(_mm256_set1_epi16(static_cast<int16_t>(-p.thr))),
        vshift(_mm_cvtsi32_si128(shift)) {}
  __m256i vclass(__m256i d) const {
    if (M == kEdge3) return _mm256_add_epi16(_mm256_sub_epi16(one, _mm256_cmpgt_epi16(d, thr)), _mm256_cmpgt_epi16(nthr, d));
    return _mm256_add_epi16(one, _mm256_cmpgt_epi16(nthr, d));
  }
  __m256i idx16(const uint16_t* l) const {
    const __m256i c = luma16<SX>(l);
    __m256i idx = one_band ? zero : _mm256_slli_epi16(_mm256_srl_epi16(c, vshift), 4);
    if (M != kBandOnly) {
      const __m256i c0 = vclass(_mm256_sub_epi16(luma16<SX>(l + p.loc[0]), c));
      const __m256i c1 = vclass(_mm256_sub_epi16(luma16<SX>(l + p.loc[1]), c));
      idx = _mm256_add_epi16(idx, _mm256_add_epi16(_mm256_slli_epi16(c0, 2), c1));
    }
    return idx;
  }
  int idx1(const uint16_t* l) const {
    int idx = one_band ? 0 : (l[0] >> shift) << 4;
    if (M != kBandOnly) idx += (class_of<M>(l[p.loc[0]] - l[0], p.thr) << 2) + class_of<M>(l[p.loc[1]] - l[0], p.thr);
    return idx;
  }
  const CcsoPlane& p;
  const int shift;
  const bool one_band;
  const __m256i thr, nthr, one = _mm256_set1_epi16(1), zero = _mm256_setzero_si256();
  const __m128i vshift;
};

template <int SX, int M>
void row_blocks(const CcsoPlane& p, int row, const int* table) {
  const Classes<SX, M> cl(p);
  const int y0 = row << kCcsoBlockLog2, y1 = std::min(p.h, y0 + (1 << kCcsoBlockLog2));
  const int maxv = (1 << p.bit_depth) - 1;
  const __m256i zero = _mm256_setzero_si256(), vmax = _mm256_set1_epi16(static_cast<int16_t>(maxv));
  const uint8_t* on = p.on + static_cast<size_t>(row) * p.on_stride;
  for (int bx = 0; (bx << kCcsoBlockLog2) < p.w; bx++) {
    if (!on[bx]) continue;
    const int x0 = bx << kCcsoBlockLog2, x1 = std::min(p.w, x0 + (1 << kCcsoBlockLog2));
    for (int y = y0; y < y1; y++) {
      const uint16_t* const lr = p.luma + static_cast<ptrdiff_t>(y << p.sy) * p.luma_stride;
      uint16_t* const dr = p.dst + static_cast<ptrdiff_t>(y) * p.dst_stride;
      int x = x0;
      for (; x + 16 <= x1; x += 16) {
        const __m256i idx = cl.idx16(lr + (x << SX));
        const __m256i g0 = _mm256_i32gather_epi32(table, _mm256_cvtepu16_epi32(_mm256_castsi256_si128(idx)), 4);
        const __m256i g1 = _mm256_i32gather_epi32(table, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(idx, 1)), 4);
        const __m256i off = _mm256_permute4x64_epi64(_mm256_packs_epi32(g0, g1), 0xd8);
        __m256i* const d = reinterpret_cast<__m256i*>(dr + x);
        const __m256i v = _mm256_add_epi16(_mm256_loadu_si256(d), off);
        _mm256_storeu_si256(d, _mm256_min_epi16(_mm256_max_epi16(v, zero), vmax));
      }
      for (; x < x1; x++) dr[x] = static_cast<uint16_t>(std::clamp(dr[x] + table[cl.idx1(lr + (x << SX))], 0, maxv));
    }
  }
}

template <int SX, int M>
void class_row(const CcsoPlane& p, int y, uint16_t* idx) {
  const Classes<SX, M> cl(p);
  const uint16_t* const lr = p.luma + static_cast<ptrdiff_t>(y << p.sy) * p.luma_stride;
  int x = 0;
  for (; x + 16 <= p.w; x += 16) _mm256_storeu_si256(reinterpret_cast<__m256i*>(idx + x), cl.idx16(lr + (x << SX)));
  for (; x < p.w; x++) idx[x] = static_cast<uint16_t>(cl.idx1(lr + (x << SX)));
}

// f(SX, M) for the plane's subsampling and classifier (as integral_constants).
template <class F>
void by_mode(const CcsoPlane& p, const F& f) {
  if (p.sx < 0 || p.sx > 1 || p.max_band_log2 < 0 || p.max_band_log2 > 6 || p.max_band_log2 > p.bit_depth)
    throw std::invalid_argument("ccso_apply_row: unsupported plane");
  using std::integral_constant;
  const int m = p.bo_only ? kBandOnly : p.edge_clf ? kEdge2 : kEdge3;
  if (p.sx) {
    if (m == kBandOnly) f(integral_constant<int, 1>(), integral_constant<int, kBandOnly>());
    else if (m == kEdge3) f(integral_constant<int, 1>(), integral_constant<int, kEdge3>());
    else f(integral_constant<int, 1>(), integral_constant<int, kEdge2>());
  } else {
    if (m == kBandOnly) f(integral_constant<int, 0>(), integral_constant<int, kBandOnly>());
    else if (m == kEdge3) f(integral_constant<int, 0>(), integral_constant<int, kEdge3>());
    else f(integral_constant<int, 0>(), integral_constant<int, kEdge2>());
  }
}

}  // namespace

void ccso_apply_row(const CcsoPlane& p, int row) {
  int table[64 * 16];  // the offsets widened for the gathers
  const int n = 16 << p.max_band_log2;
  for (int k = 0; k < n; k++) table[k] = p.offset[k];
  by_mode(p, [&](auto sx, auto m) { row_blocks<sx.value, m.value>(p, row, table); });
}

void ccso_class_row(const CcsoPlane& p, int y, uint16_t* idx) {
  by_mode(p, [&](auto sx, auto m) { class_row<sx.value, m.value>(p, y, idx); });
}

}  // namespace celv2
