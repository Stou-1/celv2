// --screen auto: the per-frame gate of the screen content tools (docs/superpowers/specs/2026-11-04-intrabc-palette-design.md).
#pragma once
#include "core/perceptual_aq.h"
#include "io/y4m.h"

namespace celv2 {

// The gate's two luma measures over a frame's whole 16x16 / 8x8 blocks (the 8-bit samples):
// palette, the share of 16x16s with 2 to 8 distinct values spanning kPaletteSpan levels or more (a
// flat block, or a smooth ramp of a few values, is coded well without a palette); repeat, the share
// of 8x8s on the 8x8 grid, flat ones left out, whose samples another 8x8 of the grid repeats.
struct ScreenShares {
  double palette = 0, repeat = 0;
};
ScreenShares screen_shares(const Frame420& f);

// The threshold, from the per-frame logs (q130): Flash's palette share 0.043-0.235, the regular sources'
// at most 0.031 (Dragon Maid), the rest 0.010. kPaletteSpan (not in the spec): without it the slow
// shading of test_frames.h's scene_frame (5 values a 16x16) passed.
constexpr double kPaletteShare = 0.04;
constexpr int kPaletteSpan = 32;

// The tools are on for a frame when the palette share passes. Not the repeat share, against the spec:
// it overlaps (Flash 0.02-0.25; Angel Beats up to 0.40, AM demo 0.17, AIR 0.15) and those sources gain
// nothing from the tools (--screen 1: +0.02, +0.09, +0.04%). It stays in the log.
inline bool screen_gate(const ScreenShares& s) { return s.palette >= kPaletteShare; }

// Stage B1: the share of a frame's 16x16 blocks with a period IntrabcIndex::near_copies() follows. Key frames
// above kPeriodicShare take the tools as well.
inline double periodic_share(const AqMap& m) {
  size_t n = 0;
  for (float p : m.periodicity) n += p >= 0.5f;
  return m.periodicity.empty() ? 0 : static_cast<double>(n) / m.periodicity.size();
}
constexpr double kPeriodicShare = 0.4;

}  // namespace celv2
