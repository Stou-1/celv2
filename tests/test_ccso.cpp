#include <cstdio>
#include <vector>

#include "check.h"
#include "core/ccso_search.h"

// A plane w x h whose reconstruction is the source less `bias` where the luma is bright and
// unchanged where it is dark; the luma itself is padded as extend_ccso_border() pads it.
struct Case {
  int w = 64, h = 48, pad = CCSO_PADDING_SIZE;
  std::vector<uint16_t> luma, rec;
  std::vector<uint8_t> src;
  explicit Case(int bias) {
    const int stride = w + 2 * pad;
    luma.resize(static_cast<size_t>(stride) * (h + 2 * pad));
    rec.resize(static_cast<size_t>(w) * h);
    src.resize(static_cast<size_t>(w) * h);
    for (int y = -pad; y < h + pad; y++)
      for (int x = -pad; x < w + pad; x++) {
        const int cy = y < 0 ? 0 : y >= h ? h - 1 : y, cx = x < 0 ? 0 : x >= w ? w - 1 : x;
        luma[static_cast<size_t>(y + pad) * stride + x + pad] = static_cast<uint16_t>(cx < w / 2 ? 40 : 200 + (cy & 1));
      }
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++) {
        const int v = luma[static_cast<size_t>(y + pad) * stride + x + pad];
        src[static_cast<size_t>(y) * w + x] = static_cast<uint8_t>(v);
        rec[static_cast<size_t>(y) * w + x] = static_cast<uint16_t>(v >= 128 ? v - bias : v);
      }
  }
  CcsoChoice search() const {
    const int stride = w + 2 * pad;
    return ccso_search(luma.data() + pad * stride + pad, stride, rec.data(), w, src.data(), w, w, h, 0, 0, 1.0, 2);
  }
};

// The bright half's error is 3 levels: an offset of +3 there and none in the dark half.
static int band_bias_is_offset() {
  const CcsoChoice c = Case(3).search();
  CHECK(c.gain > 0);
  CHECK(c.max_band_log2 >= 1);
  const int top = (1 << c.max_band_log2) - 1;  // the band of luma 200
  int bright = 0, dark = 0;
  for (int k = 0; k < 16; k++) bright += c.offset[(top << 4) + k], dark += c.offset[k];
  CHECK(bright > 0);
  CHECK_EQ(dark, 0);
  return 0;
}

// No error: all offsets 0 and nothing gained.
static int exact_reconstruction_keeps_zero() {
  const CcsoChoice c = Case(0).search();
  CHECK_EQ(c.gain, 0.0);
  for (int k = 0; k < CCSO_BAND_NUM * 16; k++) CHECK_EQ(static_cast<int>(c.offset[k]), 0);
  return 0;
}

int main() {
  int fails = 0;
  fails += RUN(band_bias_is_offset);
  fails += RUN(exact_reconstruction_keeps_zero);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
