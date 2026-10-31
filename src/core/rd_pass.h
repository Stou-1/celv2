// The CPU RD pass over a decision engine's candidates (docs/design/2026-10-28-cpu-rd-pass.md):
// every candidate block of a 64x64 superblock is re-scored with the encoder's luma estimate - the
// prediction, the DCT at its largest transform (or 32x32 parts of a 64-point one) quantized as the
// coder does, its squared error in steps plus a weight per bit of its levels and of rough mode,
// block and partition bits - and the partition is picked bottom-up.
// Superblocks are independent here, so callers run them on any number of threads.
#pragma once
#include <vector>

#include "gpu/decision_map.h"
#include "io/y4m.h"

extern "C" {
#include "av2/common/enums.h"
}

struct RdPassParams {
  int qindex = 0;
  bool tcq = false;
  double scale[TX_SIZES_ALL] = {};  // fwd_dct()'s scale per luma transform (Encoder::luma_scale())
  // Weight of a bit in squared quantizer steps, and the side information every coded block
  // carries (mode, MV, flags). On 30 frames of live action / anime (q100-190): 0.2 and 10 bits
  // gave +2.3 / -5.1% BD-rate, 0.1 -0.8 / -11.3%, 0.05 -5.5 / -14.8%, 0.03 -8.3 / -15.7%, 0.02 -8.3
  // / -15.0%; at 0.02-0.03, 10 to 20 bits within 1.3%. 0 bits at 0.05: +1.4 / -8.1%.
  double bit_weight = 0.025;
  double block_bits = 12;
};

// Re-picks the tree and blocks of the 64x64 superblock whose top left is (sb_x, sb_y) from
// sb.cands, predicting inter blocks from refs (the engine's references, in its order). A superblock
// without candidates is left as it is.
void rd_pass_sb(const Frame420& src, const std::vector<const Frame420*>& refs, int sb_x, int sb_y,
                const RdPassParams& p, SbDecision& sb);
