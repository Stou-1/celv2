// The candidate blocks of a 64x64 decision superblock, shared by the GPU kernels, the CPU engine and
// the CPU RD pass: the 85 squares from 64x64 down to 8x8 (level by level, raster order within a
// level), each whole and as its top, bottom, left and right half. Block b = square * 5 + kind.
#pragma once
#include "gpu/intra_common.h"

namespace celv2 {

constexpr int kSbSquares = 85;
constexpr int kSbBlocks = kSbSquares * 5;

CELV2_HD int sb_level_offset(int level) { return ((1 << (2 * level)) - 1) / 3; }  // 0, 1, 5, 21

CELV2_HD int sb_square_id(int level, int qx, int qy) { return sb_level_offset(level) + qy * (1 << level) + qx; }

// Geometry of block b relative to the superblock origin.
CELV2_HD void sb_block_geom(int b, int* x, int* y, int* w, int* h) {
  const int sq = b / 5, kind = b % 5;
  int level = 0;
  while (level < 3 && sq >= sb_level_offset(level + 1)) level++;
  const int idx = sq - sb_level_offset(level);
  const int n = 1 << level, size = 64 >> level;
  const int sx = (idx % n) * size, sy = (idx / n) * size, half = size / 2;
  *x = sx, *y = sy, *w = size, *h = size;
  if (kind == 1 || kind == 2) *h = half, *y = sy + (kind == 2 ? half : 0);
  if (kind == 3 || kind == 4) *w = half, *x = sx + (kind == 4 ? half : 0);
}

// The block at (x, y), w x h relative to the superblock origin (a square or a half of one).
CELV2_HD int sb_block_index(int x, int y, int w, int h) {
  const int size = w > h ? w : h;
  int level = 0;
  while ((64 >> level) > size) level++;
  const int sx = x / size * size, sy = y / size * size;
  const int kind = w == h ? 0 : w > h ? (y == sy ? 1 : 2) : (x == sx ? 3 : 4);
  return sb_square_id(level, sx / size, sy / size) * 5 + kind;
}

}  // namespace celv2
