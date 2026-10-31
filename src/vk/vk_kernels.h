// The decision kernels on Vulkan, one per CUDA kernel in gpu/kernels.cu, ported one at a time.
// Each matches its CPU reference (gpu/kernels_ref, gpu/*_common.h) bit for bit.
#pragma once
#include "vk/vk_context.h"

namespace celv2::vk {

// An 8-bit plane in a device buffer: w x h samples (padded size) at `stride`, starting `offset`
// bytes in; `phase` as celv2::Plane::phase (the quarter-pel planes' spacing, 0: none).
struct DevPlane {
  const Buffer* buf = nullptr;
  int stride = 0, w = 0, h = 0;
  uint64_t phase = 0, offset = 0;
};

// gpu/kernels.h's DevRefs: the references nearest first (n == 0: intra only), each with its level-0
// plane (with its phases) and 8x8 motion field (mv_cols x mv_rows, the same for all).
struct Refs {
  int n = 0;
  DevPlane plane[7];
  const Buffer* mvx[7] = {};
  const Buffer* mvy[7] = {};
  int mv_cols = 0, mv_rows = 0;
};

class Kernels {
 public:
  explicit Kernels(Context& ctx);
  ~Kernels();
  Kernels(const Kernels&) = delete;
  Kernels& operator=(const Kernels&) = delete;
  // downsample_kernel: dst (w/2 x h/2, contiguous) = 2x2 averages of src (celv2::downsample_px).
  void downsample(const DevPlane& src, const Buffer& dst);
  // subpel_kernel: quarter-pel phases 1..15 of src at dst + k * phase (celv2::subpel_px).
  void subpel(const DevPlane& src, const Buffer& dst, uint64_t phase);
  // me_kernel: one pyramid level's 8x8 field (cols x rows) into mvx / mvy, from the coarser
  // level's field (pred_x == nullptr at the top); refine: quarter-pel refinement (level 0).
  void me_level(const DevPlane& cur, const DevPlane& ref, const Buffer* pred_x, const Buffer* pred_y, int pred_cols,
                int pred_rows, const Buffer& mvx, const Buffer& mvy, int cols, int rows, bool refine);
  // satd_kernel: out[i] = SATD of the 64x64 tiles a + 4096 i, b + 4096 i over w4 x h4 4x4s.
  void satd_batch(const Buffer& a, const Buffer& b, int n, int w4, int h4, const Buffer& out);
  // cost_kernel, as gpu/kernels.h's launch_intra / launch_costs: superblock rows [row0, row0 + nrows),
  // k workgroups a superblock; intra at s * kIntraCosts, costs at (s - row0 * sb_cols) * costs_per_sb(refs.n)
  // (the rows from row0 on: finalize() takes the same row0).
  void intra(const DevPlane& src, int sb_cols, int row0, int nrows, int k, const Buffer& intra);
  void costs(const DevPlane& src, const Refs& refs, int sb_cols, int row0, int nrows, int k, int64_t lambda_q8,
             const Buffer& costs);
  // finalize_kernel, as launch_finalize: one celv2::SbOut a superblock into out.
  void finalize(const DevPlane& src, const Refs& refs, int sb_cols, int row0, int nrows, int64_t lambda_q8,
                const Buffer& intra, const Buffer& costs, const Buffer& out);

 private:
  const Buffer& refs_buffer(const Refs& refs);

  Context& ctx_;
  Kernel downsample_, subpel_, me_, satd_, cost_, finalize_;
  Buffer tables_, table_data_, refs_;  // the task tables (built once); the current call's references
};

}  // namespace celv2::vk
