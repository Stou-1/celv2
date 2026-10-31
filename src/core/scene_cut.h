// Scene-cut detection for keyframe placement, on 1/8-size luma (8x8 cell averages).
#pragma once
#include <vector>

#include "io/y4m.h"

// Frame t is a cut when its mean absolute difference from frame t-1 is over 20 (of 255), at least
// 4x the median of the previous 8 differences plus 1, and frame t+1 differs from t by less than
// half as much, so a one-frame flash (or a black frame between shots) is not a cut but the first
// frame of the new shot is. It must also differ from frame t-2 by over 20, so the frame after a
// flash, back in the same scene, is not a cut either, and leave at most half of its cells within 2
// levels of frame t-1's: held animation whose next drawing changes part of the picture (after
// identical frames, any change is far over the median) is the same shot. Tuned on an anime clip: 49 cuts in 5087
// frames, none on fast action or flashes.
// The share of 8x8 cells of b within 2 levels of a's (how much of the picture stays still).
double still_share(const Frame420& a, const Frame420& b);

class SceneCutDetector {
 public:
  // Call once per frame, in order; next is the following frame (nullptr at the end).
  bool push(const Frame420& cur, const Frame420* next);

 private:
  std::vector<int> prev2_, prev_, cur_;  // cell averages of frames t-2, t-1 and t
  std::vector<double> diffs_;            // the latest differences between consecutive frames
};
