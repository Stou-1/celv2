#include "vk/vk_kernels.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

#include "gpu/inter_common.h"
#include "gpu/sb_blocks.h"
#include "vk_shaders/cost.h"
#include "vk_shaders/downsample.h"
#include "vk_shaders/finalize.h"
#include "vk_shaders/me.h"
#include "vk_shaders/satd.h"
#include "vk_shaders/subpel.h"

namespace celv2::vk {

namespace {

// The shaders' push constants (std430: these C++ layouts match it member for member).
struct PlanePush {
  uint64_t p;
  int stride, w, h;
  uint64_t phase;
};
static_assert(sizeof(PlanePush) == 32);
struct DownsamplePush {
  PlanePush src;
  uint64_t dst;
};
struct SubpelPush {
  PlanePush src;
  uint64_t dst, phase;
};
struct MePush {
  PlanePush cur, ref;
  uint64_t pred_x, pred_y, mvx, mvy;
  int pred_cols, pred_rows, cols, rows, refine;
};
static_assert(sizeof(MePush) <= 128);  // the push-constant size every Vulkan device has
struct SatdPush {
  uint64_t a, b, out;
  int n, w4, h4;
};
constexpr int kInterSlots = 4400;  // gpu/kernels.h
constexpr int kNumIntra = kSbBlocks * kNumModes;
struct KRefs {  // common.slang's KRefs
  int n;
  PlanePush p[7];
  uint64_t mvx[7], mvy[7];
  int mv_cols, mv_rows;
};
static_assert(sizeof(KRefs) == 352);
struct Tables {  // common.slang's Tables
  int n_intra_chunks, n_inter_chunks;
  uint64_t inter_first, inter_b, inter_k, intra_chunks, inter_chunks;
};
// The tables' contents, one buffer.
struct TableData {
  int inter_first[kSbBlocks + 1];
  int inter_b[kInterSlots];
  int inter_k[kInterSlots];
  int intra_chunks[kNumIntra * 4];
  int inter_chunks[kNumIntra * 4];
};
struct CostPush {
  PlanePush s;
  uint64_t tab, refs, intra, costs;
  int64_t lam;
  int sb_cols, row0, k, intra_part;
};
struct FinalizePush {
  PlanePush s;
  uint64_t tab, refs, intra, costs, out;
  int64_t lam;
  int sb_cols, row0;
};
static_assert(sizeof(FinalizePush) <= 128);

// gpu/kernels.cu's inter_tasks_kernel: the inter tasks of a superblock (block x candidate MV) and
// the chunks the warps take, the same for every superblock.
std::unique_ptr<TableData> make_tables(int* n_intra_chunks, int* n_inter_chunks) {
  auto t = std::make_unique<TableData>();
  int n = 0;
  for (int b = 0; b < kSbBlocks; b++) {
    t->inter_first[b] = n;
    int x, y, w, h;
    sb_block_geom(b, &x, &y, &w, &h);
    const int cells = mv_candidates(x, y, w, h);
    for (int k = 0; k < cells && n < kInterSlots; k++, n++) t->inter_b[n] = b, t->inter_k[n] = k;
  }
  t->inter_first[kSbBlocks] = n;
  int ni = 0, nx = 0;
  for (int b = 0; b < kSbBlocks; b++) {  // the 64x64 level: inter tasks only (kMaxInterBlockSize)
    int x, y, w, h;
    sb_block_geom(b, &x, &y, &w, &h);
    const int nsub = (w / 4) * (h / 4), lanes = nsub < 32 ? nsub : 32, per = 32 / lanes;
    for (int m = 0; m < kNumModes && b >= 5 * sb_level_offset(1); m += per, ni++) {
      const int c[4] = {b, m, kNumModes - m < per ? kNumModes - m : per, lanes};
      std::copy(c, c + 4, t->intra_chunks + ni * 4);
    }
    const int end = t->inter_first[b + 1];
    for (int i = t->inter_first[b]; i < end; i += per, nx++) {
      const int c[4] = {b, i, end - i < per ? end - i : per, lanes};
      std::copy(c, c + 4, t->inter_chunks + nx * 4);
    }
  }
  *n_intra_chunks = ni, *n_inter_chunks = nx;
  return t;
}

PlanePush plane(const DevPlane& d) { return {d.buf->addr + d.offset, d.stride, d.w, d.h, d.phase}; }
uint32_t groups(int n, int size) { return static_cast<uint32_t>((n + size - 1) / size); }

}  // namespace

Kernels::Kernels(Context& ctx)
    : ctx_(ctx),
      downsample_(ctx.kernel(kSpv_downsample, sizeof(DownsamplePush))),
      subpel_(ctx.kernel(kSpv_subpel, sizeof(SubpelPush))),
      me_(ctx.kernel(kSpv_me, sizeof(MePush))),
      satd_(ctx.kernel(kSpv_satd, sizeof(SatdPush))),
      cost_(ctx.kernel(kSpv_cost, sizeof(CostPush), 32)),
      finalize_(ctx.kernel(kSpv_finalize, sizeof(FinalizePush))) {
  Tables t;
  const std::unique_ptr<TableData> d = make_tables(&t.n_intra_chunks, &t.n_inter_chunks);
  table_data_ = ctx.buffer(sizeof(TableData));
  ctx.upload(table_data_, d.get(), sizeof(TableData));
  const uint64_t a = table_data_.addr;
  t.inter_first = a + offsetof(TableData, inter_first), t.inter_b = a + offsetof(TableData, inter_b);
  t.inter_k = a + offsetof(TableData, inter_k), t.intra_chunks = a + offsetof(TableData, intra_chunks);
  t.inter_chunks = a + offsetof(TableData, inter_chunks);
  tables_ = ctx.buffer(sizeof(Tables));
  ctx.upload(tables_, &t, sizeof(Tables));
  refs_ = ctx.buffer(sizeof(KRefs));
}

Kernels::~Kernels() {
  ctx_.free(tables_);
  ctx_.free(table_data_);
  ctx_.free(refs_);
}

const Buffer& Kernels::refs_buffer(const Refs& r) {
  KRefs k{};
  k.n = r.n;
  for (int i = 0; i < r.n; i++) k.p[i] = plane(r.plane[i]), k.mvx[i] = r.mvx[i]->addr, k.mvy[i] = r.mvy[i]->addr;
  k.mv_cols = r.mv_cols, k.mv_rows = r.mv_rows;
  ctx_.update(refs_, &k, sizeof(k));
  return refs_;
}

void Kernels::intra(const DevPlane& src, int sb_cols, int row0, int nrows, int k, const Buffer& intra) {
  const CostPush p{plane(src), tables_.addr, refs_buffer(Refs{}).addr, intra.addr, 0, 0, sb_cols, row0, k, 1};
  ctx_.dispatch(cost_, &p, static_cast<uint32_t>(sb_cols), static_cast<uint32_t>(nrows * k));
}

void Kernels::costs(const DevPlane& src, const Refs& refs, int sb_cols, int row0, int nrows, int k, int64_t lambda_q8,
                    const Buffer& costs) {
  if (refs.n == 0) return;
  const CostPush p{plane(src), tables_.addr, refs_buffer(refs).addr, 0, costs.addr, lambda_q8, sb_cols, row0, k, 0};
  ctx_.dispatch(cost_, &p, static_cast<uint32_t>(sb_cols), static_cast<uint32_t>(nrows * k));
}

void Kernels::finalize(const DevPlane& src, const Refs& refs, int sb_cols, int row0, int nrows, int64_t lambda_q8,
                       const Buffer& intra, const Buffer& costs, const Buffer& out) {
  const FinalizePush p{plane(src), tables_.addr, refs_buffer(refs).addr, intra.addr, costs.addr, out.addr, lambda_q8,
                       sb_cols, row0};
  ctx_.dispatch(finalize_, &p, static_cast<uint32_t>(sb_cols), static_cast<uint32_t>(nrows));
}

void Kernels::downsample(const DevPlane& src, const Buffer& dst) {
  const DownsamplePush p{plane(src), dst.addr};
  ctx_.dispatch(downsample_, &p, groups(src.w / 2, 16), groups(src.h / 2, 16));
}

void Kernels::subpel(const DevPlane& src, const Buffer& dst, uint64_t phase) {
  const SubpelPush p{plane(src), dst.addr, phase};
  constexpr uint32_t kPhases = 15;  // 1..15; phase 0 is src
  ctx_.dispatch(subpel_, &p, groups(src.w, 16), groups(src.h, 16), kPhases);
}

void Kernels::me_level(const DevPlane& cur, const DevPlane& ref, const Buffer* pred_x, const Buffer* pred_y,
                       int pred_cols, int pred_rows, const Buffer& mvx, const Buffer& mvy, int cols, int rows,
                       bool refine) {
  const MePush p{plane(cur), plane(ref), pred_x ? pred_x->addr : 0, pred_y ? pred_y->addr : 0, mvx.addr, mvy.addr,
                 pred_cols, pred_rows, cols, rows, refine ? 1 : 0};
  ctx_.dispatch(me_, &p, groups(cols, 8), groups(rows, 8));
}

void Kernels::satd_batch(const Buffer& a, const Buffer& b, int n, int w4, int h4, const Buffer& out) {
  const SatdPush p{a.addr, b.addr, out.addr, n, w4, h4};
  ctx_.dispatch(satd_, &p, groups(n, 64));
}

}  // namespace celv2::vk
