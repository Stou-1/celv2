// A temporal dependency model (TPL) of a mini-GOP from its source frames, as libaom / AVM's
// tpl_model: for each 32x32 block, its cost coded alone (intra) and how much of later frames'
// content predicts from it, propagated back along their motion. Motion and costs are SADs on the
// 8-bit luma at half scale (the search starts at quarter scale), so this is a model, not the coder.
#pragma once

#include <vector>

#include "io/y4m.h"

namespace celv2 {

constexpr int kTplBlock = 32;  // luma samples a side

struct TplFrame {
  const Frame420* src = nullptr;
  std::vector<int> refs;  // the frames it predicts from: indices of frames before it (coding order)
};

struct TplStats {
  int cols = 0, rows = 0;            // kTplBlock blocks
  std::vector<double> intra, total;  // per block: its intra cost, and that plus all that depends on it
};

// frames in coding order; one TplStats per frame. Runs on up to `threads` threads.
std::vector<TplStats> tpl_model(const std::vector<TplFrame>& frames, int threads);

// The qindex offset of each 64x64 superblock (raster) from a frame's stats, as AVM's objective
// delta q: beta = r0 / r, r = intra / total over the superblock and r0 over the frame; the
// quantizer step becomes step / sqrt(beta). base: the frame's coded qindex; at most +-max_offset.
std::vector<int> tpl_sb_offsets(const TplStats& s, int width, int height, int base, int max_q, int max_offset);

// The share of a frame's weight that is its own: intra / total over the frame (1: nothing predicts
// from it; libaom's 1 / frame importance).
double tpl_r0(const TplStats& s);

// The coded qindex (1..max_q) whose quantizer step is nearest step(base) * ratio.
int qindex_for_step_ratio(int base, double ratio, int max_q);

}  // namespace celv2
