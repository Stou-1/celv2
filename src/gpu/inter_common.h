// Integer primitives of the motion search and inter cost, shared by the CUDA kernels and the CPU
// reference (plan Task 8): 3-level pyramid (2x2 average), +-8 (level 0: +-4) full search of 8x8 blocks per level
// (SAD plus a distance penalty, first minimum in raster order) around the best of 6 starts,
// integer-pel MVs, cur(x, y) ~ ref(x + mvx, y + mvy). Level 0 is then refined to quarter pels
// (me_refine), so the level-0 field and the inter costs use quarter-pel MVs.
#pragma once
#include "gpu/intra_common.h"

namespace celv2 {

constexpr int kSubpelPhases = 16;  // 4 x 4 quarter-pel positions
// AV2's EIGHTTAP_REGULAR taps (av2_sub_pel_filters_8) at quarter pel q (1/16-pel phase 4q).
CELV2_HD int subpel_tap(int q, int k) {
  const int t[4][8] = {{0, 0, 0, 128, 0, 0, 0, 0},
                       {0, 2, -14, 110, 38, -10, 2, 0},
                       {0, 2, -14, 76, 76, -14, 2, 0},
                       {0, 2, -10, 38, 110, -14, 2, 0}};
  return t[q][k];
}
// The sample of src at (y + qy / 4, x + qx / 4), qy and qx in 0..3: horizontal then vertical 8-tap,
// rounded once (AV2 rounds the intermediate too, a difference of at most 1).
CELV2_HD int subpel_px(const Plane& src, int y, int x, int qy, int qx) {
  int v = 0;
  for (int i = 0; i < 8; i++) {
    const int ty = subpel_tap(qy, i);
    if (ty == 0) continue;
    int hsum = 0;
    for (int j = 0; j < 8; j++) hsum += subpel_tap(qx, j) * src.at(y + i - 3, x + j - 3);
    v += ty * hsum;
  }
  v = (v + (1 << 13)) >> 14;
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

constexpr int kMeRange = 8;
// Level 0 (full resolution) searches +-4 around the coarser level's MV: 16K 1.52 -> 1.70 fps, BD-rate
// +0.0 / -0.2% on anime clips (+-2: 1.74 fps, +0.1 / -0.2%).
constexpr int kMeRange0 = 4;
constexpr int kMeLevels = 3;
constexpr int kInterBits = 3;  // bits charged for an inter (NEWMV) block
constexpr int kMaxRefs = 7;    // AV2's INTER_REFS_PER_FRAME
// Bits added to intra blocks of inter frames: their coded cost runs well above their SATD's (anime,
// against CDEF alone: 10 / 16 / 24 / 40 bits gave -13.8 / -13.8 / -13.2 / -12.1% BD-rate at qindex
// 45-180 and -6.7 / -8.0 / -9.1 / -10.2% at 30-75; scaling their SATD instead did worse).
constexpr int kIntraInInterBits = 32;
// Bits charged for an inter block from reference r: the index is coded as a truncated unary code.
CELV2_HD int inter_bits(int r) { return kInterBits + r; }
// SAD added per pel of distance from the search centre: keeps the field smooth where SAD is flat,
// so neighbouring blocks share MVs (tuned on anime clips: 2 and 8 were worse than 4).
constexpr int kMePenalty = 4;

// Inter MV candidates of the block (x, y, w, h): the level-0 field's MV of every 8x8 cell it covers
// (raster order), then of the 8 cells around it. A block whose own cells have a wrong MV can still
// take its neighbours' (on anime clips the neighbours were worth 9% of the bits at equal PSNR).
CELV2_HD int mv_cells(int x, int y, int w, int h) {
  return ((x + w - 1) / 8 - x / 8 + 1) * ((y + h - 1) / 8 - y / 8 + 1);
}
CELV2_HD int mv_candidates(int x, int y, int w, int h) { return mv_cells(x, y, w, h) + 8; }
// Field index of candidate k (cells clamped into the cols x rows field).
CELV2_HD int candidate_mv_index(int x, int y, int w, int h, int k, int cols, int rows) {
  const int cells = mv_cells(x, y, w, h), cw = (x + w - 1) / 8 - x / 8 + 1;
  int cx = x / 8 + k % cw, cy = y / 8 + k / cw;
  if (k >= cells) {
    const int n = k - cells;
    // left, above, right, below, then above-left, above-right, below-left, below-right
    const int l = x / 8 - 1, r = (x + w) / 8, t = y / 8 - 1, b = (y + h) / 8;
    cx = n == 0 || n == 4 || n == 6 ? l : n == 2 || n == 5 || n == 7 ? r : x / 8;
    cy = n == 1 || n == 4 || n == 5 ? t : n == 3 || n == 6 || n == 7 ? b : y / 8;
  }
  cx = cx < 0 ? 0 : cx >= cols ? cols - 1 : cx;
  cy = cy < 0 ? 0 : cy >= rows ? rows - 1 : cy;
  return cy * cols + cx;
}

// Whether candidate k of block (x, y, w, h) repeats the MV of an earlier one: it would cost the same,
// and the first strict minimum wins, so it needs no evaluation (anime: neighbours share MVs).
CELV2_HD bool repeated_candidate(const int* fx, const int* fy, int cols, int rows, int x, int y, int w, int h, int k) {
  const int i = candidate_mv_index(x, y, w, h, k, cols, rows);
  for (int j = 0; j < k; j++) {
    const int ij = candidate_mv_index(x, y, w, h, j, cols, rows);
    if (fx[ij] == fx[i] && fy[ij] == fy[i]) return true;
  }
  return false;
}

CELV2_HD int downsample_px(const Plane& s, int y, int x) {
  return (s.at(2 * y, 2 * x) + s.at(2 * y, 2 * x + 1) + s.at(2 * y + 1, 2 * x) +
          s.at(2 * y + 1, 2 * x + 1) + 2) >> 2;
}

CELV2_HD int sad8(const Plane& cur, int x, int y, const Plane& ref, int mvx, int mvy) {
  int sad = 0;
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++) {
      const int d = cur.at(y + i, x + j) - ref.at(y + i + mvy, x + j + mvx);
      sad += d < 0 ? -d : d;
    }
  return sad;
}

// Best MV of the 8x8 block (bx, by) of one pyramid level around the predictor (px, py).
CELV2_HD void me_search(const Plane& cur, const Plane& ref, int bx, int by, int px, int py, int range,
                       int* mvx, int* mvy) {
  int best = 0x7fffffff;
  *mvx = px, *mvy = py;
  for (int dy = -range; dy <= range; dy++)
    for (int dx = -range; dx <= range; dx++) {
      const int s = sad8(cur, bx * 8, by * 8, ref, px + dx, py + dy) +
                    kMePenalty * ((dx < 0 ? -dx : dx) + (dy < 0 ? -dy : dy));
      if (s < best) best = s, *mvx = px + dx, *mvy = py + dy;
    }
}

// SATD of the 8x8 block at (x, y) predicted from ref with a quarter-pel MV.
CELV2_HD int satd8_q(const Plane& cur, int x, int y, const Plane& ref, int qmx, int qmy) {
  int total = 0;
  for (int ty = 0; ty < 8; ty += 4)
    for (int tx = 0; tx < 8; tx += 4) {
      int d[16];
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
          d[i * 4 + j] = cur.at(y + ty + i, x + tx + j) - ref.at_q(y + ty + i, x + tx + j, qmy, qmx);
      total += hadamard4x4(d);
    }
  return total;
}

// Quarter-pel MV of level-0 block (bx, by) from its integer one (ref has its phases): the best by
// SATD (first strict minimum, the centre first) of it and its 8 half-pel neighbours, then of that
// and its 8 quarter-pel neighbours. A move must save kSubpelPenalty: neighbouring cells then share
// more MVs, which AV2 codes as NEARMV instead of a difference (16 / 32: -0.4 / -0.5% BD-rate).
constexpr int kSubpelPenalty = 32;
CELV2_HD void me_refine(const Plane& cur, const Plane& ref, int bx, int by, int* mvx, int* mvy) {
  int cx = *mvx * 4, cy = *mvy * 4;
  for (int step = 2; step >= 1; step--) {
    int best = satd8_q(cur, bx * 8, by * 8, ref, cx, cy), nx = cx, ny = cy;
    for (int dy = -1; dy <= 1; dy++)
      for (int dx = -1; dx <= 1; dx++) {
        if (dx == 0 && dy == 0) continue;
        const int s = satd8_q(cur, bx * 8, by * 8, ref, cx + dx * step, cy + dy * step) + kSubpelPenalty;
        if (s < best) best = s, nx = cx + dx * step, ny = cy + dy * step;
      }
    cx = nx, cy = ny;
  }
  *mvx = cx, *mvy = cy;
}

// Motion search of 8x8 block (bx, by) of one level. (pmx, pmy) is the coarser level's field of
// pcols x prows (nullptr at the top): the search centre is the best (by SAD) of its MV under this
// block, its 4 neighbours and zero, all scaled to this level.
CELV2_HD void me_cell(const Plane& cur, const Plane& ref, int bx, int by, const int* pmx, const int* pmy,
                     int pcols, int prows, int* mvx, int* mvy, bool level0 = false) {
  int px = 0, py = 0;
  if (pmx) {
    const int ox[5] = {0, -1, 1, 0, 0}, oy[5] = {0, 0, 0, -1, 1};
    int best = 0x7fffffff;
    for (int k = 0; k < 6; k++) {
      int cx = 0, cy = 0;  // k == 5: zero
      if (k < 5) {
        int qx = bx / 2 + ox[k], qy = by / 2 + oy[k];
        qx = qx < 0 ? 0 : qx >= pcols ? pcols - 1 : qx;
        qy = qy < 0 ? 0 : qy >= prows ? prows - 1 : qy;
        cx = 2 * pmx[qy * pcols + qx], cy = 2 * pmy[qy * pcols + qx];
      }
      const int s = sad8(cur, bx * 8, by * 8, ref, cx, cy);
      if (s < best) best = s, px = cx, py = cy;
    }
  }
  me_search(cur, ref, bx, by, px, py, level0 ? kMeRange0 : kMeRange, mvx, mvy);
}

// Compound candidates: the best single candidate of the two best references, averaged. AV2 codes a
// pair only when its nearer reference is among the first three and not the last
// (read_compound_ref). The decision's average has no rounding offsets, so it is charged well above
// its syntax (two MV differences). MVs in quarter pels.
CELV2_HD bool compound_pair_ok(int a, int b, int nrefs) {
  const int lo = a < b ? a : b, limit = nrefs - 2 < 2 ? nrefs - 2 : 2;
  return a != b && lo <= limit;
}
// 20 + the reference indices: 3 / 6 / 10 / 14 / 20 gave +1.2 / -1.2 / -1.9 / -1.9 / -2.2% BD-rate.
CELV2_HD int compound_bits(int a, int b) { return 20 + a + b; }
CELV2_HD int64_t block_compound_satd(const Plane& cur, const Plane& r0, int mx0, int my0, const Plane& r1, int mx1,
                                    int my1, int x, int y, int w, int h) {
  int64_t total = 0;
  for (int ty = 0; ty < h; ty += 4)
    for (int tx = 0; tx < w; tx += 4) {
      int d[16];
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
          const int yy = y + ty + i, xx = x + tx + j;
          const int p = (r0.at_q(yy, xx, my0, mx0) + r1.at_q(yy, xx, my1, mx1) + 1) >> 1;
          d[i * 4 + j] = cur.at(yy, xx) - p;
        }
      total += hadamard4x4(d);
    }
  return total;
}

// SATD of a block predicted from ref with a quarter-pel MV.
CELV2_HD int64_t block_inter_satd(const Plane& cur, const Plane& ref, int x, int y, int w, int h,
                                 int mvx, int mvy) {
  int64_t total = 0;
  for (int ty = 0; ty < h; ty += 4)
    for (int tx = 0; tx < w; tx += 4) {
      int d[16];
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
          d[i * 4 + j] = cur.at(y + ty + i, x + tx + j) - ref.at_q(y + ty + i, x + tx + j, mvy, mvx);
      total += hadamard4x4(d);
    }
  return total;
}

}  // namespace celv2
