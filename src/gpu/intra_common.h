// Integer primitives of the intra decision rule, shared by the CUDA kernel and the CPU reference
// so both produce identical DecisionMaps. Reads the source through a clamped accessor (the
// padded luma plane, edge-replicated beyond it).
#pragma once
#include <cstdint>

#ifdef __CUDACC__
#define CELV2_HD __host__ __device__ __forceinline__
#else
#define CELV2_HD inline
#endif

namespace celv2 {

struct Plane {
  const uint8_t* p;
  int stride, w, h;  // padded size
  // A reference's level 0 also holds its 16 quarter-pel phases: phase (qy & 3) * 4 + (qx & 3) at
  // p + that * phase (0: no phases, at_q() only at whole pels).
  size_t phase = 0;
  CELV2_HD int at(int y, int x) const {
    y = y < 0 ? 0 : (y >= h ? h - 1 : y);
    x = x < 0 ? 0 : (x >= w ? w - 1 : x);
    return p[y * stride + x];
  }
  // The sample at (y + qy / 4, x + qx / 4), qy and qx in quarter pels.
  CELV2_HD int at_q(int y, int x, int qy, int qx) const {
    return Plane{p + static_cast<size_t>((qy & 3) * 4 + (qx & 3)) * phase, stride, w, h}.at(y + (qy >> 2), x + (qx >> 2));
  }
};

// Bits charged per choice (plan Task 6): DC=1, other modes=3, HORZ/VERT/SPLIT=2, NONE=0.
// The modes after the first four are charged 9: the decision's predictions of them are rough (no
// edge filter, source neighbours), and the encoder re-picks the mode of every intra block anyway
// (BlockWriter::set_intra_search); 3 / 6 / 9 gave +2.3 / -1.2 / -1.6% BD-rate.
CELV2_HD int mode_bits(int mode) { return mode == 0 ? 1 : mode < 4 ? 3 : 9; }
constexpr int kPartBits = 2;
// Largest block side the decision picks: AV2 codes only the low 32x32 coefficients of a 64-point
// transform, and an intra block is one transform in celv2, so 64-wide intra blocks would lose detail
// for good. Inter blocks go to 64: the encoder splits their transform where the 64-point one would
// drop too much (TileCoder::choose_tx_partition()). -3.0% BD-rate on anime A 1080p, -1.3% on live
// action, +0.5% on anime B, and faster (fewer blocks).
constexpr int kMaxBlockSize = 32;
constexpr int kMaxInterBlockSize = 64;
// IntraMode order: DC, V, H, Smooth, SmoothV, SmoothH, Paeth, D45, D67, D113, D135, D157, D203.
constexpr int kNumModes = 13;

// Edge samples of the block at (x, y) from the original: above(k) is the row above at column
// x + k, left(k) the column left at row y + k, both with k == -1 the top-left corner. An edge
// outside the frame (top or left border) reads 128.
CELV2_HD int edge_above(const Plane& s, int x, int y, int k) {
  if (y == 0) return 128;
  if (k < 0 && x == 0) return s.at(y - 1, x);
  return s.at(y - 1, x + k);
}
CELV2_HD int edge_left(const Plane& s, int x, int y, int k) {
  if (x == 0) return 128;
  if (k < 0 && y == 0) return s.at(y, x - 1);
  return s.at(y + k, x - 1);
}

// AV1/AV2 directional prediction (dr_prediction z1 / z2 / z3, no edge filter or upsampling) at
// angle 45, 67, 113, 135, 157 or 203 degrees; derivatives of 23 / 45 / 67 degrees: 151 / 64 / 27.
CELV2_HD int dr_derivative(int a) { return a == 23 ? 151 : a == 45 ? 64 : 27; }
CELV2_HD int dir_pred(const Plane& s, int x, int y, int w, int h, int angle, int i, int j) {
  const int last = w + h - 1;
  if (angle < 90) {  // z1: the row above (and above-right)
    const int pos = (i + 1) * dr_derivative(angle), base = (pos >> 6) + j, shift = (pos & 63) >> 1;
    const int a0 = edge_above(s, x, y, base < last ? base : last), a1 = edge_above(s, x, y, base + 1 < last ? base + 1 : last);
    return (a0 * (32 - shift) + a1 * shift + 16) >> 5;
  }
  if (angle > 180) {  // z3: the column left (and below-left)
    const int pos = (j + 1) * dr_derivative(270 - angle), base = (pos >> 6) + i, shift = (pos & 63) >> 1;
    const int l0 = edge_left(s, x, y, base < last ? base : last), l1 = edge_left(s, x, y, base + 1 < last ? base + 1 : last);
    return (l0 * (32 - shift) + l1 * shift + 16) >> 5;
  }
  // z2: the row above while the projection stays on it, else the column left
  const int px = (j << 6) - (i + 1) * dr_derivative(180 - angle);
  const int bx = px >= 0 ? px >> 6 : -((-px + 63) >> 6);
  if (bx >= -1) {
    const int shift = (px - bx * 64) >> 1;
    return (edge_above(s, x, y, bx) * (32 - shift) + edge_above(s, x, y, bx + 1) * shift + 16) >> 5;
  }
  const int py = (i << 6) - (j + 1) * dr_derivative(angle - 90);
  const int by = py >= 0 ? py >> 6 : -((-py + 63) >> 6);
  const int shift = (py - by * 64) >> 1;
  const int b0 = by < -1 ? -1 : by, b1 = by + 1 < -1 ? -1 : by + 1;
  return (edge_left(s, x, y, b0) * (32 - shift) + edge_left(s, x, y, b1) * shift + 16) >> 5;
}

// Prediction from original neighbours at block origin (x, y); the row above / column left is
// unavailable only at the frame's top / left edge (128 is used instead).
CELV2_HD int intra_pred(const Plane& s, int x, int y, int w, int h, int mode, int i, int j) {
  const bool has_a = y > 0, has_l = x > 0;
  if (mode == 1) return has_a ? s.at(y - 1, x + j) : 128;  // V
  if (mode == 2) return has_l ? s.at(y + i, x - 1) : 128;  // H
  if (mode == 4)  // SmoothV: above blended towards the bottom-left sample
    return ((h - 1 - i) * edge_above(s, x, y, j) + (i + 1) * edge_left(s, x, y, h - 1) + h / 2) / h;
  if (mode == 5)  // SmoothH: left blended towards the top-right sample
    return ((w - 1 - j) * edge_left(s, x, y, i) + (j + 1) * edge_above(s, x, y, w - 1) + w / 2) / w;
  if (mode == 6) {  // Paeth
    const int a = edge_above(s, x, y, j), l = edge_left(s, x, y, i), tl = edge_above(s, x, y, -1);
    const int base = a + l - tl;
    const int pa = base > a ? base - a : a - base, pl = base > l ? base - l : l - base;
    const int ptl = base > tl ? base - tl : tl - base;
    return pl <= pa && pl <= ptl ? l : pa <= ptl ? a : tl;
  }
  if (mode >= 7) {
    const int angles[6] = {45, 67, 113, 135, 157, 203};
    return dir_pred(s, x, y, w, h, angles[mode - 7], i, j);
  }
  if (mode == 3) {                                         // Smooth: bilinear blend
    const int a = has_a ? s.at(y - 1, x + j) : 128;
    const int l = has_l ? s.at(y + i, x - 1) : 128;
    const int tr = has_a ? s.at(y - 1, x + w - 1) : 128;
    const int bl = has_l ? s.at(y + h - 1, x - 1) : 128;
    const int den = w + h + 2;
    const int num = (h - i) * a + (i + 1) * bl + (w - j) * l + (j + 1) * tr;
    return (num + den / 2) / den;
  }
  return -1;  // DC handled by the caller (needs a sum)
}

CELV2_HD int dc_value(const Plane& s, int x, int y, int w, int h) {
  const bool has_a = y > 0, has_l = x > 0;
  int sum = 0, n = 0;
  if (has_a)
    for (int j = 0; j < w; j++) sum += s.at(y - 1, x + j), n++;
  if (has_l)
    for (int i = 0; i < h; i++) sum += s.at(y + i, x - 1), n++;
  return n ? (sum + n / 2) / n : 128;
}

// Sum of absolute 4x4 Hadamard coefficients of d[16] (row-major).
CELV2_HD int hadamard4x4(const int* d) {
  int t[16];
  for (int r = 0; r < 4; r++) {
    const int a0 = d[r * 4 + 0] + d[r * 4 + 1], a1 = d[r * 4 + 0] - d[r * 4 + 1];
    const int a2 = d[r * 4 + 2] + d[r * 4 + 3], a3 = d[r * 4 + 2] - d[r * 4 + 3];
    t[r * 4 + 0] = a0 + a2, t[r * 4 + 1] = a1 + a3, t[r * 4 + 2] = a0 - a2, t[r * 4 + 3] = a1 - a3;
  }
  int sum = 0;
  for (int c = 0; c < 4; c++) {
    const int a0 = t[c] + t[4 + c], a1 = t[c] - t[4 + c];
    const int a2 = t[8 + c] + t[12 + c], a3 = t[8 + c] - t[12 + c];
    const int v[4] = {a0 + a2, a1 + a3, a0 - a2, a1 - a3};
    for (int k = 0; k < 4; k++) sum += v[k] < 0 ? -v[k] : v[k];
  }
  return sum;
}

// SATD of one block against one intra mode.
CELV2_HD int64_t block_mode_satd(const Plane& s, int x, int y, int w, int h, int mode) {
  const int dc = mode == 0 ? dc_value(s, x, y, w, h) : 0;
  int64_t total = 0;
  for (int ty = 0; ty < h; ty += 4)
    for (int tx = 0; tx < w; tx += 4) {
      int d[16];
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
          const int p = mode == 0 ? dc : intra_pred(s, x, y, w, h, mode, ty + i, tx + j);
          d[i * 4 + j] = s.at(y + ty + i, x + tx + j) - p;
        }
      total += hadamard4x4(d);
    }
  return total;
}

}  // namespace celv2
