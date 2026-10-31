// Deterministic synthetic test pictures and PSNR, shared by the encoder tests.
#pragma once
#include <cmath>
#include <cstdint>

#include "io/y4m.h"

// Smooth shading, hard edges, stripes and noisy patches; (dx, dy) shifts the whole picture
// content by whole pixels (positive = content moves right/down), for motion tests.
inline Frame420 scene_frame(int w, int h, int dx = 0, int dy = 0, uint32_t seed = 1) {
  Frame420 f = make_frame(w, h);
  auto hash = [seed](int x, int y) {
    uint32_t v = static_cast<uint32_t>(x) * 73856093u ^ static_cast<uint32_t>(y) * 19349663u ^ seed;
    v ^= v >> 13;
    v *= 0x5bd1e995u;
    return static_cast<int>((v ^ (v >> 15)) & 0xff);
  };
  auto luma = [&](int x, int y) {
    int v = 90 + (x * 3 + y * 2) / 16 % 80;                        // slow shading
    if (((x + 1000) / 48 + (y + 1000) / 40) % 4 == 0) v = 200 - (y & 31);  // flat-ish tiles
    if (((x + 1000) / 23) % 7 == 0) v = (y / 3 % 2) ? 30 : 230;     // stripes / edges
    if (((x + 1000) / 64 + (y + 1000) / 64) % 5 == 2) v += hash(x, y) % 31 - 15;  // texture
    return v < 0 ? 0 : v > 255 ? 255 : v;
  };
  for (int r = 0; r < h; r++)
    for (int c = 0; c < w; c++) f.y[r * f.stride_y() + c] = static_cast<uint8_t>(luma(c - dx, r - dy));
  const int cw = (w + 1) / 2, ch = (h + 1) / 2;
  for (int r = 0; r < ch; r++)
    for (int c = 0; c < cw; c++) {
      const int x = 2 * c - dx, y = 2 * r - dy;
      f.u[r * f.stride_uv() + c] = static_cast<uint8_t>(110 + luma(x, y) / 8);
      f.v[r * f.stride_uv() + c] = static_cast<uint8_t>(140 - luma(x + 7, y) / 10);
    }
  pad_frame(f);
  return f;
}

inline double psnr_y(const Frame420& a, const Frame420& b) {
  double se = 0;
  for (int r = 0; r < a.height; r++)
    for (int c = 0; c < a.width; c++) {
      const double d = a.y[r * a.stride_y() + c] - b.y[r * b.stride_y() + c];
      se += d * d;
    }
  const double mse = se / (static_cast<double>(a.width) * a.height);
  return mse == 0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
}
