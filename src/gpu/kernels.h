// Host-side launchers for the CUDA kernels (pointers are device memory).
#pragma once
#include <cuda_runtime.h>

#include <cstdint>

#include "gpu/sb_blocks.h"
#include "gpu/sb_out.h"

namespace celv2 {

// Plane view in device memory.
struct DevPlane {
  const uint8_t* p;
  int stride, w, h;
  size_t phase = 0;  // as celv2::Plane::phase
};

constexpr int kIntraCosts = 425 * 13;  // per superblock: 425 blocks x 13 intra modes
constexpr int kInterSlots = 4400;  // per superblock and reference: the inter tasks
inline int costs_per_sb(int nrefs) { return nrefs * kInterSlots; }  // inter costs only

// References of an inter decision, nearest first (n == 0: intra only): each one's level-0 plane
// and per-8x8 motion field (mv_cols x mv_rows, the same for all).
struct DevRefs {
  int n = 0;
  DevPlane plane[7];
  const int* mvx[7];
  const int* mvy[7];
  int mv_cols = 0, mv_rows = 0;
};

// Intra SATD * 256 of every block x mode of superblock rows [row0, row0 + nrows), K blocks per
// superblock: superblock s (raster order) at intra + s * kIntraCosts + b * kNumModes + m. They need
// only the source, so a frame's can be made ahead of its decision.
void launch_intra(DevPlane src, int sb_cols, int row0, int nrows, int k, int64_t* intra, cudaStream_t stream);
// Inter costs of every candidate of the same rows: superblock s owns costs + s * costs_per_sb(refs.n),
// reference r's inter tasks from r * kInterSlots. Nothing with no references.
void launch_costs(DevPlane src, const DevRefs& refs, int sb_cols, int row0, int nrows, int k,
                  int64_t lambda_q8, int64_t* costs, cudaStream_t stream);
// Mode / partition choice and SbOut emission for the same rows, from those (the intra mode bits
// are added here).
void launch_finalize(DevPlane src, const DevRefs& refs, int sb_cols, int row0, int nrows, int64_t lambda_q8,
                     const int64_t* intra, const int64_t* costs, SbOut* out, cudaStream_t stream);
int blocks_per_sb(int sb_cols);  // K = clamp(ceil(2 * SMs / sb_cols), 1, 16)
// dst (w/2 x h/2, stride w/2) = 2x2 average of src.
void launch_downsample(DevPlane src, uint8_t* dst, cudaStream_t stream);
// One pyramid level of the motion search; pred_* is the coarser level's field (nullptr at the top).
// refine (level 0, ref with its phases): quarter-pel MVs (celv2::me_refine).
void launch_me_level(DevPlane cur, DevPlane ref, const int* pred_x, const int* pred_y,
                     int pred_cols, int pred_rows, int* mvx, int* mvy, int cols, int rows,
                     bool refine, cudaStream_t stream);
// Quarter-pel phases 1..15 of src at dst + k * phase (dst + 0 holds src).
void launch_subpel(DevPlane src, uint8_t* dst, size_t phase, cudaStream_t stream);
void launch_satd_batch(const uint8_t* a, const uint8_t* b, int n, int w4, int h4, int* out);
// Throws std::runtime_error with the CUDA message if the last launch or call failed.
void check_cuda(const char* what);

}  // namespace celv2
