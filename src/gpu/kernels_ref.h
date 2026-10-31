// CPU reference implementations of the GPU decision rules (used by tests to check the kernels).
#pragma once
#include <cstdint>
#include <vector>

#include "gpu/decision_map.h"
#include "gpu/intra_common.h"
#include "io/y4m.h"

// Sum over 4x4 tiles of |4x4 Hadamard(a - b)|; w4/h4 in 4-pixel units.
int satd_ref(const uint8_t* a, int as, const uint8_t* b, int bs, int w4, int h4);
// lambda * 256 (rounded): 0.002 * qstep^2, qstep = AVM AC quantizer step in pixel units
// (defined in core/quant.cpp, shared with the encoder's intra mode search).
int64_t lambda_q8(int qindex);
DecisionMap decide_intra_ref(const Frame420& src, int sb_size, int qindex);
DecisionMap decide_inter_ref(const Frame420& src, const Frame420& prev_src, int sb_size, int qindex);
// References nearest first (refs[r] is AV2 reference r); each gets its own motion search.
DecisionMap decide_inter_ref(const Frame420& src, const std::vector<const Frame420*>& refs, int sb_size,
                             int qindex);

// Luma pyramid of celv2::kMeLevels levels (level 0 = full resolution; a reference's also holds the
// quarter-pel phases, see celv2::Plane::phase).
struct RefPyramid {
  std::vector<std::vector<uint8_t>> px;  // level l pixels
  std::vector<celv2::Plane> plane;        // level l view (p points into px[l])
  RefPyramid() = default;
  RefPyramid(const RefPyramid&) = delete;  // a copy would keep pointing at the original's pixels
  RefPyramid& operator=(const RefPyramid&) = delete;
  RefPyramid(RefPyramid&&) = default;  // moving keeps the vectors' buffers, so the views stay valid
  RefPyramid& operator=(RefPyramid&&) = default;
};
RefPyramid build_pyramid_ref(const Frame420& f, bool phases = false);
struct RefMvLevel {  // per-8x8 MVs of one level (integer pels; level 0 quarter pels)
  std::vector<int> mvx, mvy;
  int cols = 0, rows = 0;
};
// Searches 8x8 block rows [row0, row1) of level l; coarser == nullptr at the top level.
// out must be sized (cols x rows of level l) by the caller.
void me_level_rows_ref(const RefPyramid& cur, const RefPyramid& ref, int l, const RefMvLevel* coarser, int row0,
                       int row1, RefMvLevel& out);
// One superblock's decision; level0 == nullptr: intra only (ref is then unused).
SbDecision decide_sb_ref(const Frame420& src, const RefPyramid* ref, const RefMvLevel* level0, int sb_col,
                         int sb_row, int sb_size, int qindex);
// With refs[r] and its level-0 field level0[r] per reference, nearest first (both empty: intra only).
SbDecision decide_sb_ref(const Frame420& src, const std::vector<const RefPyramid*>& refs,
                         const std::vector<const RefMvLevel*>& level0, int sb_col, int sb_row, int sb_size,
                         int qindex);
// Level-0 planes and motion fields of the references, nearest first.
struct RefSet {
  std::vector<const celv2::Plane*> plane;
  std::vector<const RefMvLevel*> field;
  int size() const { return static_cast<int>(plane.size()); }
};
