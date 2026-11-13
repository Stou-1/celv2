// The intra prediction of a whole block, for the RD pass (the speed-first spec's step 1, C18 (c)): the same values
// as intra_pred() and dc_value() (gpu/intra_common.h, which the shaders port sample by sample), computed from the
// block's edges read once instead of through the clamped accessor for every sample.
#pragma once
#include <cstdint>

#include "gpu/intra_common.h"

namespace celv2 {

// pred (stride w) = the w x h block at (x, y) predicted by `mode` (IntraMode order); w, h up to 64.
inline void intra_pred_block(const Plane& s, int x, int y, int w, int h, int mode, uint8_t* pred) {
  const int last = w + h - 1;
  int above_buf[64 + 64 + 1], left_buf[64 + 64 + 1];
  int* const a = above_buf + 1;  // a[k] = edge_above(k), k = -1 .. last
  int* const l = left_buf + 1;
  for (int k = -1; k <= last; k++) a[k] = edge_above(s, x, y, k), l[k] = edge_left(s, x, y, k);
  auto put = [&](auto f) {
    for (int i = 0; i < h; i++)
      for (int j = 0; j < w; j++) pred[i * w + j] = static_cast<uint8_t>(f(i, j));
  };
  switch (mode) {
    case 0: {  // DC
      const bool has_a = y > 0, has_l = x > 0;
      int sum = 0, n = 0;
      if (has_a)
        for (int j = 0; j < w; j++) sum += a[j], n++;
      if (has_l)
        for (int i = 0; i < h; i++) sum += l[i], n++;
      const int dc = n ? (sum + n / 2) / n : 128;
      return put([&](int, int) { return dc; });
    }
    case 1: return put([&](int, int j) { return a[j]; });  // V (edge_above is 128 at the top, as intra_pred)
    case 2: return put([&](int i, int) { return l[i]; });  // H
    case 3: {                                              // Smooth
      const int tr = a[w - 1], bl = l[h - 1], den = w + h + 2;
      return put([&](int i, int j) { return ((h - i) * a[j] + (i + 1) * bl + (w - j) * l[i] + (j + 1) * tr + den / 2) / den; });
    }
    case 4: return put([&](int i, int j) { return ((h - 1 - i) * a[j] + (i + 1) * l[h - 1] + h / 2) / h; });  // SmoothV
    case 5: return put([&](int i, int j) { return ((w - 1 - j) * l[i] + (j + 1) * a[w - 1] + w / 2) / w; });  // SmoothH
    case 6:                                                                                                    // Paeth
      return put([&](int i, int j) {
        const int av = a[j], lv = l[i], tl = a[-1], base = av + lv - tl;
        const int pa = base > av ? base - av : av - base, pl = base > lv ? base - lv : lv - base;
        const int ptl = base > tl ? base - tl : tl - base;
        return pl <= pa && pl <= ptl ? lv : pa <= ptl ? av : tl;
      });
    default: {  // directional, as dir_pred()
      const int angles[6] = {45, 67, 113, 135, 157, 203};
      const int angle = angles[mode - 7];
      if (angle < 90) {
        const int d = dr_derivative(angle);
        return put([&](int i, int j) {
          const int pos = (i + 1) * d, base = (pos >> 6) + j, shift = (pos & 63) >> 1;
          return (a[base < last ? base : last] * (32 - shift) + a[base + 1 < last ? base + 1 : last] * shift + 16) >> 5;
        });
      }
      if (angle > 180) {
        const int d = dr_derivative(270 - angle);
        return put([&](int i, int j) {
          const int pos = (j + 1) * d, base = (pos >> 6) + i, shift = (pos & 63) >> 1;
          return (l[base < last ? base : last] * (32 - shift) + l[base + 1 < last ? base + 1 : last] * shift + 16) >> 5;
        });
      }
      const int dx = dr_derivative(180 - angle), dy = dr_derivative(angle - 90);
      return put([&](int i, int j) {
        const int px = (j << 6) - (i + 1) * dx;
        const int bx = px >= 0 ? px >> 6 : -((-px + 63) >> 6);
        if (bx >= -1) {
          const int shift = (px - bx * 64) >> 1;
          return (a[bx] * (32 - shift) + a[bx + 1] * shift + 16) >> 5;
        }
        const int py = (i << 6) - (j + 1) * dy;
        const int by = py >= 0 ? py >> 6 : -((-py + 63) >> 6);
        const int shift = (py - by * 64) >> 1;
        const int b0 = by < -1 ? -1 : by, b1 = by + 1 < -1 ? -1 : by + 1;
        return (l[b0] * (32 - shift) + l[b1] * shift + 16) >> 5;
      });
    }
  }
}

}  // namespace celv2
