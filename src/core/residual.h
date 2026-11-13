// A block's residual for the RD pass (the speed-first spec's step 3): source minus prediction, the same values as
// Plane::at() sample by sample. Inside the plane 8 or 4 samples at a time; at its right or bottom edge the
// clamped accessor, as before.
#pragma once
#include <immintrin.h>

#include <cstdint>
#include <cstring>

#include "gpu/intra_common.h"

namespace celv2 {

// res (stride w) = s's w x h block at (x, y) - pred (stride w). x, y >= 0; w a multiple of 4.
inline void block_residual(const Plane& s, int x, int y, int w, int h, const uint8_t* pred, int16_t* res) {
  if (x + w > s.w || y + h > s.h) {
    for (int i = 0; i < h; i++)
      for (int j = 0; j < w; j++) res[i * w + j] = static_cast<int16_t>(s.at(y + i, x + j) - pred[i * w + j]);
    return;
  }
  for (int i = 0; i < h; i++) {
    const uint8_t* const row = s.p + static_cast<size_t>(y + i) * s.stride + x;
    const uint8_t* const pr = pred + i * w;
    int16_t* const out = res + i * w;
    int j = 0;
    for (; j + 8 <= w; j += 8) {
      const __m128i a = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(row + j)));
      const __m128i b = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(pr + j)));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(out + j), _mm_sub_epi16(a, b));
    }
    if (j < w) {  // w == 4
      int32_t va, vb;
      std::memcpy(&va, row + j, 4), std::memcpy(&vb, pr + j, 4);
      const __m128i d = _mm_sub_epi16(_mm_cvtepu8_epi16(_mm_cvtsi32_si128(va)), _mm_cvtepu8_epi16(_mm_cvtsi32_si128(vb)));
      _mm_storel_epi64(reinterpret_cast<__m128i*>(out + j), d);
    }
  }
}

// p0 = (p0 + p1 + 1) >> 1 over n samples (n a multiple of 16): the compound average.
inline void average_into(uint8_t* p0, const uint8_t* p1, int n) {
  for (int k = 0; k < n; k += 16) {
    const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p0 + k));
    const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p1 + k));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p0 + k), _mm_avg_epu8(a, b));
  }
}

}  // namespace celv2
