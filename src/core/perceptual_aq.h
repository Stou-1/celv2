// --paq: perceptual adaptive quantization. A 16x16 importance map of the source luma (variance, edge
// coherence, darkness) as qindex offsets, folded into the per-superblock offsets of FrameSpec::sb_dq
// (docs/superpowers/specs/2026-11-04-perceptual-aq-design.md).
#pragma once
#include <cstdint>
#include <vector>

#include "io/y4m.h"

namespace celv2 {

// Per 16x16 block, raster; cols = ceil(width / 16), rows = ceil(height / 16): the offset in qindex, and
// the measures psy-rd reads: log2(variance + 1), edge coherence C (0..1) and periodicity P (0..1; 0
// where not computed: blocks not coarsened, or with C >= 0.5), and the lag where P peaked (the period
// vector, dy >= 0; (0, 0) where not computed), along which IntraBC's stage B1 looks for copies.
struct AqMap {
  int cols = 0, rows = 0;
  std::vector<float> dq, log_energy, coherence, periodicity;
  std::vector<int8_t> period_dx, period_dy;
};

// The map of f's luma (Frame420::y, the 8-bit planes at any depth; the visible area) at strength S
// (0..2, 1 nominal): flat finer and busy coarser, edges and halftone (screentone) fields not
// coarsened, dark areas finer; the frame's mean 0 before clamping to +-35.
AqMap aq_map(const Frame420& f, double strength);
// One offset per 64x64 (the FrameSpec::sb_dq grid, ceil(width / 64) x ceil(height / 64)): the mean of
// its blocks' 25th percentile and mean, leaning to the sensitive side; within +-35.
std::vector<int> aq_sb64(const AqMap& m, int width, int height);

// For the per-block (segmentation) decision: the mean variance of the 16x16 offsets inside each coded
// superblock of coded_sb_px, and the variance of the superblocks' means.
struct AqSpread {
  double within = 0, between = 0;
};
AqSpread aq_spread(const AqMap& m, int coded_sb_px);

}  // namespace celv2
