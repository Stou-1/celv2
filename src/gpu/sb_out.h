// A superblock's decision as the GPU kernels write it (CUDA and Vulkan; vk/shaders/finalize.slang
// mirrors this layout field for field).
#pragma once
#include <cstdint>

#include "gpu/decision_map.h"
#include "gpu/sb_blocks.h"

namespace celv2 {

constexpr int kSbOutCap = 256;  // max tree entries / leaves per 64x64 superblock

struct SbOut {
  int ntree, nblocks;
  uint8_t tree[kSbOutCap];
  uint8_t w4[kSbOutCap], h4[kSbOutCap], mode[kSbOutCap];  // mode kNumModes = inter
  int16_t mvx[kSbOutCap], mvy[kSbOutCap];                 // quarter pel
  uint8_t ref[kSbOutCap];                                  // inter: reference index
  uint8_t ref2[kSbOutCap];                                 // compound: second reference, else 255
  int16_t mv2x[kSbOutCap], mv2y[kSbOutCap];
  // Every candidate block's best inter choice and intra mode, with costs (gpu/sb_blocks.h order).
  int64_t c_inter_cost[kSbBlocks], c_intra_cost[kSbBlocks];
  int16_t c_mvx[kSbBlocks], c_mvy[kSbBlocks], c_mv2x[kSbBlocks], c_mv2y[kSbBlocks];  // quarter pel
  uint8_t c_ref[kSbBlocks], c_ref2[kSbBlocks], c_intra[kSbBlocks];
  // And each reference's best single-reference MV, with its cost / 256 (BlockCandidates::ref_cost).
  // shortcut: 24 KB per superblock (16K: 780 MB a frame); keep the best few if memory binds.
  uint32_t c_rcost[7][kSbBlocks];
  int16_t c_rmvx[7][kSbBlocks], c_rmvy[7][kSbBlocks];  // quarter pel
};

// One superblock's SbOut as the encoder takes it: the tree, the leaves and every block's candidates
// (MVs from quarter pels to 1/8).
void sb_out_to_decision(const SbOut& o, SbDecision& d);

}  // namespace celv2
