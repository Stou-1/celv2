// gpu/intra_common.h and gpu/inter_common.h in Slang, line for line (the CPU reference is the C++).
// Planes are device addresses (buffer device address), as the CUDA kernels take pointers.
module common;

public struct Plane {
  public uint8_t* p;
  public int stride, w, h;  // padded size
  // A reference's level 0 also holds its 16 quarter-pel phases: phase (qy & 3) * 4 + (qx & 3) at
  // p + that * phase (0: no phases, at_q() only at whole pels). The top bits pick a VRAM-saving
  // layout (the speed budget spec's): kPhasesOnDemand, none kept, each interpolated when read (A);
  // kPhasesHalf, the 4 half-pel ones kept, (qy >> 1) * 2 + (qx >> 1) apart, the rest interpolated (B).
  // Every layout reads the same values.
  public uint64_t phase;
  public int at(int y, int x) {
    y = y < 0 ? 0 : (y >= h ? h - 1 : y);
    x = x < 0 ? 0 : (x >= w ? w - 1 : x);
    return int(p[y * stride + x]);
  }
  // The sample at (y + qy / 4, x + qx / 4), qy and qx in quarter pels.
  public int at_q(int y, int x, int qy, int qx) {
    const int fy = qy & 3, fx = qx & 3;
    Plane q = this;
    q.phase = 0;
    const uint64_t spacing = phase & kPhaseSpacing;
    if ((phase & kPhasesOnDemand) != 0 || ((phase & kPhasesHalf) != 0 && ((fy | fx) & 1) != 0)) {
      if ((fy | fx) == 0) return q.at(y + (qy >> 2), x + (qx >> 2));
      int yy = y + (qy >> 2), xx = x + (qx >> 2);  // the phase planes hold the clamped position's value
      yy = yy < 0 ? 0 : (yy >= h ? h - 1 : yy);
      xx = xx < 0 ? 0 : (xx >= w ? w - 1 : xx);
      return subpel_px(q, yy, xx, fy, fx);
    }
    if ((phase & kPhasesHalf) != 0) q.p = p + uint64_t((fy >> 1) * 2 + (fx >> 1)) * spacing;
    else q.p = p + uint64_t(fy * 4 + fx) * spacing;
    return q.at(y + (qy >> 2), x + (qx >> 2));
  }
};

public int hadamard4x4(int d[16]) {
  int t[16];
  for (int r = 0; r < 4; r++) {
    const int a0 = d[r * 4 + 0] + d[r * 4 + 1], a1 = d[r * 4 + 0] - d[r * 4 + 1];
    const int a2 = d[r * 4 + 2] + d[r * 4 + 3], a3 = d[r * 4 + 2] - d[r * 4 + 3];
    t[r * 4 + 0] = a0 + a2; t[r * 4 + 1] = a1 + a3; t[r * 4 + 2] = a0 - a2; t[r * 4 + 3] = a1 - a3;
  }
  int sum = 0;
  for (int c = 0; c < 4; c++) {
    const int a0 = t[c] + t[4 + c], a1 = t[c] - t[4 + c];
    const int a2 = t[8 + c] + t[12 + c], a3 = t[8 + c] - t[12 + c];
    const int v[4] = { a0 + a2, a1 + a3, a0 - a2, a1 - a3 };
    for (int k = 0; k < 4; k++) sum += v[k] < 0 ? -v[k] : v[k];
  }
  return sum;
}

public static const int kSubpelPhases = 16;
public static const uint64_t kPhasesOnDemand = 1ull << 63, kPhasesHalf = 1ull << 62;
public static const uint64_t kPhaseSpacing = (1ull << 62) - 1;
static const int kSubpelTaps[4][8] = { { 0, 0, 0, 128, 0, 0, 0, 0 },
                                       { 0, 2, -14, 110, 38, -10, 2, 0 },
                                       { 0, 2, -14, 76, 76, -14, 2, 0 },
                                       { 0, 2, -10, 38, 110, -14, 2, 0 } };
public int subpel_tap(int q, int k) { return kSubpelTaps[q][k]; }
public int subpel_px(Plane src, int y, int x, int qy, int qx) {
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

public static const int kMeRange = 8;
public static const int kMeRange0 = 8;
static const int kMePenalty = 4;
static const int kSubpelPenalty = 16;

public int downsample_px(Plane s, int y, int x) {
  return (s.at(2 * y, 2 * x) + s.at(2 * y, 2 * x + 1) + s.at(2 * y + 1, 2 * x) + s.at(2 * y + 1, 2 * x + 1) + 2) >> 2;
}

public int sad8(Plane cur, int x, int y, Plane ref, int mvx, int mvy) {
  int sad = 0;
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++) {
      const int d = cur.at(y + i, x + j) - ref.at(y + i + mvy, x + j + mvx);
      sad += d < 0 ? -d : d;
    }
  return sad;
}

public int me_search(Plane cur, Plane ref, int bx, int by, int px, int py, int range, out int mvx, out int mvy) {
  int best = 0x7fffffff;
  mvx = px; mvy = py;
  for (int dy = -range; dy <= range; dy++)
    for (int dx = -range; dx <= range; dx++) {
      const int s = sad8(cur, bx * 8, by * 8, ref, px + dx, py + dy) + kMePenalty * ((dx < 0 ? -dx : dx) + (dy < 0 ? -dy : dy));
      if (s < best) { best = s; mvx = px + dx; mvy = py + dy; }
    }
  return best;
}

public int satd8_q(Plane cur, int x, int y, Plane ref, int qmx, int qmy) {
  int total = 0;
  for (int ty = 0; ty < 8; ty += 4)
    for (int tx = 0; tx < 8; tx += 4) {
      int d[16];
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) d[i * 4 + j] = cur.at(y + ty + i, x + tx + j) - ref.at_q(y + ty + i, x + tx + j, qmy, qmx);
      total += hadamard4x4(d);
    }
  return total;
}

public void me_refine(Plane cur, Plane ref, int bx, int by, inout int mvx, inout int mvy) {
  int cx = mvx * 4, cy = mvy * 4;
  for (int step = 2; step >= 1; step--) {
    int best = satd8_q(cur, bx * 8, by * 8, ref, cx, cy), nx = cx, ny = cy;
    for (int dy = -1; dy <= 1; dy++)
      for (int dx = -1; dx <= 1; dx++) {
        if (dx == 0 && dy == 0) continue;
        const int s = satd8_q(cur, bx * 8, by * 8, ref, cx + dx * step, cy + dy * step) + kSubpelPenalty;
        if (s < best) { best = s; nx = cx + dx * step; ny = cy + dy * step; }
      }
    cx = nx; cy = ny;
  }
  mvx = cx; mvy = cy;
}

// pmx / pmy: the coarser level's field of pcols x prows (null at the top level).
public void me_cell(Plane cur, Plane ref, int bx, int by, int* pmx, int* pmy, int pcols, int prows, out int mvx,
                    out int mvy, bool level0) {
  int px = 0, py = 0, px2 = 0, py2 = 0;
  bool two = false;
  if (pmx != nullptr) {
    const int ox[5] = { 0, -1, 1, 0, 0 }, oy[5] = { 0, 0, 0, -1, 1 };
    int best = 0x7fffffff, second = 0x7fffffff;
    for (int k = 0; k < 6; k++) {
      int cx = 0, cy = 0;  // k == 5: zero
      if (k < 5) {
        int qx = bx / 2 + ox[k], qy = by / 2 + oy[k];
        qx = qx < 0 ? 0 : qx >= pcols ? pcols - 1 : qx;
        qy = qy < 0 ? 0 : qy >= prows ? prows - 1 : qy;
        cx = 2 * pmx[qy * pcols + qx]; cy = 2 * pmy[qy * pcols + qx];
      }
      if ((k > 0 && cx == px && cy == py) || (two && cx == px2 && cy == py2)) continue;  // each centre once
      const int s = sad8(cur, bx * 8, by * 8, ref, cx, cy);
      if (s < best) {
        if (k > 0) { second = best; px2 = px; py2 = py; two = true; }
        best = s; px = cx; py = cy;
      } else if (s < second) {
        second = s; px2 = cx; py2 = cy; two = true;
      }
    }
  }
  const int range = level0 ? kMeRange0 : kMeRange;
  const int cost = me_search(cur, ref, bx, by, px, py, range, mvx, mvy);
  if (two) {
    int mx2, my2;
    if (me_search(cur, ref, bx, by, px2, py2, range, mx2, my2) < cost) { mvx = mx2; mvy = my2; }
  }
}

// ---- gpu/intra_common.h: intra prediction from the source, block limits and bit charges.
public static const int kMaxBlockSize = 32;
public static const int kMaxInterBlockSize = 64;
public static const int kNumModes = 13;
public static const int kPartBits = 2;
public static const int kIntraInInterBits = 32;
public static const int kMaxRefs = 7;
public int mode_bits(int mode) { return mode == 0 ? 1 : mode < 4 ? 3 : 9; }
public int inter_bits(int r) { return 3 + r; }  // kInterBits + r

int edge_above(Plane s, int x, int y, int k) {
  if (y == 0) return 128;
  if (k < 0 && x == 0) return s.at(y - 1, x);
  return s.at(y - 1, x + k);
}
int edge_left(Plane s, int x, int y, int k) {
  if (x == 0) return 128;
  if (k < 0 && y == 0) return s.at(y, x - 1);
  return s.at(y + k, x - 1);
}
int dr_derivative(int a) { return a == 23 ? 151 : a == 45 ? 64 : 27; }
int dir_pred(Plane s, int x, int y, int w, int h, int angle, int i, int j) {
  const int last = w + h - 1;
  if (angle < 90) {
    const int pos = (i + 1) * dr_derivative(angle), base = (pos >> 6) + j, shift = (pos & 63) >> 1;
    const int a0 = edge_above(s, x, y, base < last ? base : last), a1 = edge_above(s, x, y, base + 1 < last ? base + 1 : last);
    return (a0 * (32 - shift) + a1 * shift + 16) >> 5;
  }
  if (angle > 180) {
    const int pos = (j + 1) * dr_derivative(270 - angle), base = (pos >> 6) + i, shift = (pos & 63) >> 1;
    const int l0 = edge_left(s, x, y, base < last ? base : last), l1 = edge_left(s, x, y, base + 1 < last ? base + 1 : last);
    return (l0 * (32 - shift) + l1 * shift + 16) >> 5;
  }
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
static const int kAngles[6] = { 45, 67, 113, 135, 157, 203 };
public int intra_pred(Plane s, int x, int y, int w, int h, int mode, int i, int j) {
  const bool has_a = y > 0, has_l = x > 0;
  if (mode == 1) return has_a ? s.at(y - 1, x + j) : 128;
  if (mode == 2) return has_l ? s.at(y + i, x - 1) : 128;
  if (mode == 4) return ((h - 1 - i) * edge_above(s, x, y, j) + (i + 1) * edge_left(s, x, y, h - 1) + h / 2) / h;
  if (mode == 5) return ((w - 1 - j) * edge_left(s, x, y, i) + (j + 1) * edge_above(s, x, y, w - 1) + w / 2) / w;
  if (mode == 6) {
    const int a = edge_above(s, x, y, j), l = edge_left(s, x, y, i), tl = edge_above(s, x, y, -1);
    const int base = a + l - tl;
    const int pa = base > a ? base - a : a - base, pl = base > l ? base - l : l - base;
    const int ptl = base > tl ? base - tl : tl - base;
    return pl <= pa && pl <= ptl ? l : pa <= ptl ? a : tl;
  }
  if (mode >= 7) return dir_pred(s, x, y, w, h, kAngles[mode - 7], i, j);
  if (mode == 3) {
    const int a = has_a ? s.at(y - 1, x + j) : 128;
    const int l = has_l ? s.at(y + i, x - 1) : 128;
    const int tr = has_a ? s.at(y - 1, x + w - 1) : 128;
    const int bl = has_l ? s.at(y + h - 1, x - 1) : 128;
    const int den = w + h + 2;
    const int num = (h - i) * a + (i + 1) * bl + (w - j) * l + (j + 1) * tr;
    return (num + den / 2) / den;
  }
  return -1;
}
public int dc_value(Plane s, int x, int y, int w, int h) {
  const bool has_a = y > 0, has_l = x > 0;
  int sum = 0, n = 0;
  if (has_a)
    for (int j = 0; j < w; j++) { sum += s.at(y - 1, x + j); n++; }
  if (has_l)
    for (int i = 0; i < h; i++) { sum += s.at(y + i, x - 1); n++; }
  return n != 0 ? (sum + n / 2) / n : 128;
}

// ---- gpu/inter_common.h: inter candidates and compound costs.
public int mv_cells(int x, int y, int w, int h) { return ((x + w - 1) / 8 - x / 8 + 1) * ((y + h - 1) / 8 - y / 8 + 1); }
public int mv_candidates(int x, int y, int w, int h) { return mv_cells(x, y, w, h) + 8; }
public int candidate_mv_index(int x, int y, int w, int h, int k, int cols, int rows) {
  const int cells = mv_cells(x, y, w, h), cw = (x + w - 1) / 8 - x / 8 + 1;
  int cx = x / 8 + k % cw, cy = y / 8 + k / cw;
  if (k >= cells) {
    const int n = k - cells;
    const int l = x / 8 - 1, r = (x + w) / 8, t = y / 8 - 1, b = (y + h) / 8;
    cx = n == 0 || n == 4 || n == 6 ? l : n == 2 || n == 5 || n == 7 ? r : x / 8;
    cy = n == 1 || n == 4 || n == 5 ? t : n == 3 || n == 6 || n == 7 ? b : y / 8;
  }
  cx = cx < 0 ? 0 : cx >= cols ? cols - 1 : cx;
  cy = cy < 0 ? 0 : cy >= rows ? rows - 1 : cy;
  return cy * cols + cx;
}
public bool repeated_candidate(int* fx, int* fy, int cols, int rows, int x, int y, int w, int h, int k) {
  const int i = candidate_mv_index(x, y, w, h, k, cols, rows);
  for (int j = 0; j < k; j++) {
    const int ij = candidate_mv_index(x, y, w, h, j, cols, rows);
    if (fx[ij] == fx[i] && fy[ij] == fy[i]) return true;
  }
  return false;
}
public bool compound_pair_ok(int a, int b, int nrefs) {
  const int lo = a < b ? a : b, limit = nrefs - 2 < 2 ? nrefs - 2 : 2;
  return a != b && lo <= limit;
}
public int compound_bits(int a, int b) { return 20 + a + b; }
public int64_t block_compound_satd(Plane cur, Plane r0, int mx0, int my0, Plane r1, int mx1, int my1, int x, int y, int w, int h) {
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

// ---- gpu/sb_blocks.h: the 425 candidate blocks of a 64x64 superblock.
public static const int kSbSquares = 85;
public static const int kSbBlocks = kSbSquares * 5;
public int sb_level_offset(int level) { return ((1 << (2 * level)) - 1) / 3; }
public int sb_square_id(int level, int qx, int qy) { return sb_level_offset(level) + qy * (1 << level) + qx; }
public void sb_block_geom(int b, out int x, out int y, out int w, out int h) {
  const int sq = b / 5, kind = b % 5;
  int level = 0;
  while (level < 3 && sq >= sb_level_offset(level + 1)) level++;
  const int idx = sq - sb_level_offset(level);
  const int n = 1 << level, size = 64 >> level;
  const int sx = (idx % n) * size, sy = (idx / n) * size, half = size / 2;
  x = sx; y = sy; w = size; h = size;
  if (kind == 1 || kind == 2) { h = half; y = sy + (kind == 2 ? half : 0); }
  if (kind == 3 || kind == 4) { w = half; x = sx + (kind == 4 ? half : 0); }
}

// ---- The cost / finalize kernels' shared inputs (vk/vk_kernels.cpp mirrors these layouts).
public static const int kInterSlots = 4400;
public static const int kNumIntra = kSbBlocks * kNumModes;
// gpu/kernels.cu's DevRefs.
public struct KRefs {
  public int n;
  public Plane p[7];
  public int* mvx[7];
  public int* mvy[7];
  public int mv_cols, mv_rows;
};
// gpu/kernels.cu's task tables (inter_tasks_kernel), built on the host: inter_first[kSbBlocks + 1],
// inter_b / inter_k[kInterSlots], and the chunks, 4 ints each (block, first task, count, lanes).
// Pointers, not arrays: a 200 KB struct type crashed NVIDIA's shader compiler.
public struct Tables {
  public int n_intra_chunks, n_inter_chunks;
  public int* inter_first;
  public int* inter_b;
  public int* inter_k;
  public int* intra_chunks;
  public int* inter_chunks;
};
