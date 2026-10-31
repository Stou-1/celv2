#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "gpu/inter_common.h"
#include "gpu/kernels.h"
#include "gpu/sb_blocks.h"

namespace celv2 {
namespace {

constexpr int kSquares = kSbSquares;  // 1 + 4 + 16 + 64 squares (64 down to 8), gpu/sb_blocks.h
constexpr int kBlocks = kSbBlocks;    // full, top/bottom halves, left/right halves

__device__ void block_geom(int b, int* x, int* y, int* w, int* h) { sb_block_geom(b, x, y, w, h); }
__device__ int square_id(int level, int qx, int qy) { return sb_square_id(level, qx, qy); }

// One 4x4 term of block_mode_satd() / block_inter_satd(): the sub-tile at (tx, ty) of the block.
__device__ int sub_satd(const Plane& s, const Plane& rp, int x, int y, int w, int h, int mode, int dc,
                        bool inter, int mvx, int mvy, int tx, int ty) {
  int d[16];
  const int yy = y + ty, xx = x + tx;
  const bool src_inside = yy + 4 <= s.h && xx + 4 <= s.w;  // yy, xx >= 0
  if (inter && src_inside) {
    // The common case, both 4x4 areas inside their planes: plain reads, no clamping (the same samples).
    const int ry = yy + (mvy >> 2), rx = xx + (mvx >> 2);
    if (ry >= 0 && rx >= 0 && ry + 4 <= rp.h && rx + 4 <= rp.w) {
      const uint8_t* r = rp.p + static_cast<size_t>((mvy & 3) * 4 + (mvx & 3)) * rp.phase + static_cast<size_t>(ry) * rp.stride + rx;
      const uint8_t* c = s.p + static_cast<size_t>(yy) * s.stride + xx;
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) d[i * 4 + j] = c[i * s.stride + j] - r[i * rp.stride + j];
      return hadamard4x4(d);
    }
  }
  if (!inter && mode == 0 && src_inside) {
    const uint8_t* c = s.p + static_cast<size_t>(yy) * s.stride + xx;
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++) d[i * 4 + j] = c[i * s.stride + j] - dc;
    return hadamard4x4(d);
  }
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) {
      const int p = inter ? rp.at_q(y + ty + i, x + tx + j, mvy, mvx)
                          : mode == 0 ? dc : intra_pred(s, x, y, w, h, mode, ty + i, tx + j);
      d[i * 4 + j] = s.at(y + ty + i, x + tx + j) - p;
    }
  return hadamard4x4(d);
}

constexpr int kThreads = 256;  // per block of cost_kernel / finalize_kernel (8 warps)
constexpr int kNumIntra = kBlocks * kNumModes;
constexpr int kMaxInter = kInterSlots;
static_assert(kNumIntra == kIntraCosts, "kIntraCosts assumes 425 blocks x kNumModes modes");
static_assert(kMaxRefs == 7, "DevRefs holds 7 references");

// DevRefs as kernel arguments.
struct KRefs {
  int n;
  Plane p[kMaxRefs];
  const int* mvx[kMaxRefs];
  const int* mvy[kMaxRefs];
  int mv_cols, mv_rows;
};

// Inter tasks of a superblock (block x each candidate_mv_index() MV, 4296 in all), in the order
// their costs are stored; the same for every superblock, so built once by inter_tasks_kernel.
__device__ int16_t g_inter_b[kMaxInter], g_inter_k[kMaxInter];
__device__ int16_t g_inter_first[kBlocks + 1];

// A warp's share of cost_kernel: `count` consecutive tasks of block b (intra modes from `first`, or
// inter tasks from `first`), each on `lanes` lanes (its 4x4 sub-tiles, at most 32): a block of
// 2 sub-tiles fills a warp with 16 tasks instead of leaving 30 lanes idle.
struct Chunk {
  int16_t b, first;
  int8_t count, lanes;
};
constexpr int kMaxChunks = kNumIntra;  // every task alone, at most
__device__ Chunk g_intra_chunks[kMaxChunks], g_inter_chunks[kMaxChunks];
__device__ int g_n_intra_chunks, g_n_inter_chunks;

__global__ void inter_tasks_kernel() {
  int n = 0;
  for (int b = 0; b < kBlocks; b++) {
    g_inter_first[b] = static_cast<int16_t>(n);
    int x, y, w, h;
    block_geom(b, &x, &y, &w, &h);
    const int cells = mv_candidates(x, y, w, h);
    for (int k = 0; k < cells && n < kMaxInter; k++, n++)
      g_inter_b[n] = static_cast<int16_t>(b), g_inter_k[n] = static_cast<int16_t>(k);
  }
  g_inter_first[kBlocks] = static_cast<int16_t>(n);
  int ni = 0, nx = 0;
  for (int b = 0; b < kBlocks; b++) {  // the 64x64 level: inter tasks only (kMaxInterBlockSize)
    int x, y, w, h;
    block_geom(b, &x, &y, &w, &h);
    const int nsub = (w / 4) * (h / 4), lanes = nsub < 32 ? nsub : 32, per = 32 / lanes;
    for (int m = 0; m < kNumModes && b >= 5 * sb_level_offset(1); m += per)
      g_intra_chunks[ni++] = Chunk{static_cast<int16_t>(b), static_cast<int16_t>(m),
                                   static_cast<int8_t>(kNumModes - m < per ? kNumModes - m : per), static_cast<int8_t>(lanes)};
    const int end = g_inter_first[b + 1];
    for (int t = g_inter_first[b]; t < end; t += per)
      g_inter_chunks[nx++] = Chunk{static_cast<int16_t>(b), static_cast<int16_t>(t),
                                   static_cast<int8_t>(end - t < per ? end - t : per), static_cast<int8_t>(lanes)};
  }
  g_n_intra_chunks = ni, g_n_inter_chunks = nx;
}

// Costs of every candidate of one superblock, shared by k blocks: with `intra`, block x mode into
// intra (SATD * 256: the mode bits are added by finalize_kernel, so these do not depend on the
// qindex and can be made for a frame ahead of its decision); else reference x block x candidate MV
// into costs. Chunks of tasks (see Chunk) are dealt to the k * 8 warps round-robin, so the heavy
// ones spread out.
__global__ void cost_kernel(Plane s, KRefs refs, int sb_cols, int row0, int k, int64_t lam, bool intra_part,
                            int64_t* intra, int64_t* costs) {
  __shared__ int16_t dcv[kBlocks];
  const int sbx = blockIdx.x, sby = row0 + blockIdx.y / k, share = blockIdx.y % k;
  const int ox = sbx * 64, oy = sby * 64;
  if (intra_part) {
    for (int b = threadIdx.x; b < kBlocks; b += blockDim.x) {
      int x, y, w, h;
      block_geom(b, &x, &y, &w, &h);
      dcv[b] = static_cast<int16_t>(dc_value(s, x + ox, y + oy, w, h));
    }
    __syncthreads();
  }
  const size_t sb = static_cast<size_t>(sby) * sb_cols + sbx;
  int64_t* cost = intra_part ? intra + sb * kNumIntra : costs + sb * (refs.n * kMaxInter);
  const int lane = threadIdx.x & 31, nwarps = blockDim.x >> 5;
  const int n_ic = g_n_intra_chunks, n_xc = g_n_inter_chunks;
  const int n_units = intra_part ? n_ic : refs.n * n_xc;
  for (int u = share * nwarps + (threadIdx.x >> 5); u < n_units; u += k * nwarps) {
    const bool inter = !intra_part;
    const int r = inter ? u / n_xc : 0;
    const Chunk ch = inter ? g_inter_chunks[u % n_xc] : g_intra_chunks[u];
    const int b = ch.b, lanes = ch.lanes, slot = lane / lanes, q0 = lane % lanes;
    const bool active = slot < ch.count;
    const int task = ch.first + slot;  // intra: the mode; inter: the inter task
    int x, y, w, h;
    block_geom(b, &x, &y, &w, &h);
    x += ox, y += oy;
    int mx = 0, my = 0;
    bool repeated = false;
    if (inter && active) {
      repeated = repeated_candidate(refs.mvx[r], refs.mvy[r], refs.mv_cols, refs.mv_rows, x, y, w, h, g_inter_k[task]);
      const int i = candidate_mv_index(x, y, w, h, g_inter_k[task], refs.mv_cols, refs.mv_rows);
      mx = refs.mvx[r][i], my = refs.mvy[r][i];
    }
    const int sw = w / 4, nsub = sw * (h / 4);
    int64_t sum = 0;
    if (active && !repeated)
      for (int q = q0; q < nsub; q += lanes)
        sum += sub_satd(s, inter ? refs.p[r] : s, x, y, w, h, inter ? 0 : task, dcv[b], inter, mx, my, (q % sw) * 4,
                        (q / sw) * 4);
    for (int off = lanes / 2; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off, lanes);
    if (active && q0 == 0) {
      if (inter) cost[r * kMaxInter + task] = repeated ? INT64_MAX : sum * 256 + lam * inter_bits(r);
      else cost[b * kNumModes + task] = sum * 256;
    }
  }
}

// Per-superblock choice from cost_kernel's costs: best candidate per block, then bottom-up
// partition choice and emission.
__global__ void finalize_kernel(Plane s, KRefs refs, int sb_cols, int row0, int64_t lam, const int64_t* intra, const int64_t* costs,
                                SbOut* out) {
  __shared__ int64_t best_cost[kBlocks];
  __shared__ uint8_t best_mode[kBlocks], best_ref[kBlocks], best_ref2[kBlocks];
  __shared__ int16_t best_mv[kBlocks][2], best_mv2[kBlocks][2];
  __shared__ int64_t sq_cost[kSquares];
  __shared__ uint8_t sq_choice[kSquares];
  const int sbx = blockIdx.x, sby = row0 + blockIdx.y;
  const int ox = sbx * 64, oy = sby * 64;
  const size_t sb = static_cast<size_t>(sby) * sb_cols + sbx;
  const int64_t* cost = costs + sb * (refs.n * kMaxInter);
  const int64_t* icost = intra + sb * kNumIntra;
  for (int b = threadIdx.x; b < kBlocks; b += blockDim.x) {
    // Inter candidate: each reference's first strict minimum, the best of them, then the average of
    // the two best references (kernels_ref best_mode()).
    int64_t ci = INT64_MAX;
    int bx = 0, by = 0, br = 0, br2 = 255, bx2 = 0, by2 = 0;
    int x, y, w, h;
    block_geom(b, &x, &y, &w, &h);
    int64_t rc[kMaxRefs];
    int rmx[kMaxRefs], rmy[kMaxRefs];
    int a = -1, second = -1;
    for (int r = 0; r < refs.n; r++) {
      rc[r] = INT64_MAX, rmx[r] = 0, rmy[r] = 0;
      for (int it = g_inter_first[b]; it < g_inter_first[b + 1]; it++)
        if (cost[r * kMaxInter + it] < rc[r]) {
          rc[r] = cost[r * kMaxInter + it];
          const int i = candidate_mv_index(x + ox, y + oy, w, h, g_inter_k[it], refs.mv_cols, refs.mv_rows);
          rmx[r] = refs.mvx[r][i], rmy[r] = refs.mvy[r][i];
        }
      if (a < 0 || rc[r] < rc[a]) a = r;
    }
    for (int r = 0; r < refs.n; r++)
      if (r != a && (second < 0 || rc[r] < rc[second])) second = r;
    if (a >= 0) ci = rc[a], bx = rmx[a], by = rmy[a], br = a;
    if (second >= 0 && compound_pair_ok(a, second, refs.n)) {
      const int64_t cc = block_compound_satd(s, refs.p[a], rmx[a], rmy[a], refs.p[second], rmx[second], rmy[second],
                                             x + ox, y + oy, w, h) * 256 + lam * compound_bits(a, second);
      if (cc < ci) ci = cc, br2 = second, bx2 = rmx[second], by2 = rmy[second];
    }
    // Best of DC, V, H, SMOOTH, inter (first strict minimum wins).
    // Inter frames charge intra blocks kIntraInInterBits more (kernels_ref best_mode()).
    auto intra_cost = [&](int k) {
      return icost[b * kNumModes + k] + lam * (mode_bits(k) + (refs.n > 0 ? kIntraInInterBits : 0));
    };
    int64_t c = INT64_MAX;  // the 64x64 level: inter only (kMaxBlockSize)
    int m = 0;
    for (int k = 0; k < kNumModes && b >= 5; k++) {
      const int64_t ck = intra_cost(k);
      if (ck < c) c = ck, m = k;
    }
    SbOut& oc = out[static_cast<size_t>(sby) * sb_cols + sbx];
    for (int r = 0; r < kMaxRefs; r++) {
      const bool has = r < refs.n && rc[r] != INT64_MAX;
      oc.c_rcost[r][b] = has ? static_cast<uint32_t>(min(rc[r] >> 8, static_cast<int64_t>(UINT32_MAX - 1))) : UINT32_MAX;
      oc.c_rmvx[r][b] = static_cast<int16_t>(has ? rmx[r] : 0), oc.c_rmvy[r][b] = static_cast<int16_t>(has ? rmy[r] : 0);
    }
    oc.c_inter_cost[b] = ci, oc.c_intra_cost[b] = c, oc.c_intra[b] = static_cast<uint8_t>(m);
    oc.c_mvx[b] = static_cast<int16_t>(bx), oc.c_mvy[b] = static_cast<int16_t>(by);
    oc.c_ref[b] = static_cast<uint8_t>(br), oc.c_ref2[b] = static_cast<uint8_t>(br2);
    oc.c_mv2x[b] = static_cast<int16_t>(bx2), oc.c_mv2y[b] = static_cast<int16_t>(by2);
    if (ci < c) c = ci, m = kNumModes;
    best_cost[b] = c;
    best_mode[b] = static_cast<uint8_t>(m);
    best_mv[b][0] = static_cast<int16_t>(bx);
    best_mv[b][1] = static_cast<int16_t>(by);
    best_ref[b] = static_cast<uint8_t>(br);
    best_ref2[b] = static_cast<uint8_t>(br2);
    best_mv2[b][0] = static_cast<int16_t>(bx2);
    best_mv2[b][1] = static_cast<int16_t>(by2);
  }
  __syncthreads();
  if (threadIdx.x != 0) return;
  // Bottom-up choice per square: NONE, HORZ, VERT, SPLIT (first strict minimum wins).
  const int64_t part = lam * kPartBits;
  for (int level = 3; level >= 0; level--) {
    const int n = 1 << level;
    for (int i = 0; i < n * n; i++) {
      const int sq = sb_level_offset(level) + i, b = sq * 5;
      int64_t best = INT64_MAX;
      int choice = 3;
      // Up to 32x32 any block; 64x64, 64x32 and 32x64 inter blocks in inter frames.
      if ((64 >> level) <= kMaxBlockSize || (refs.n > 0 && (64 >> level) <= kMaxInterBlockSize)) {
        const int64_t c_horz = part + best_cost[b + 1] + best_cost[b + 2];
        const int64_t c_vert = part + best_cost[b + 3] + best_cost[b + 4];
        best = best_cost[b], choice = 0;
        if (c_horz < best) best = c_horz, choice = 1;
        if (c_vert < best) best = c_vert, choice = 2;
      }
      if (level < 3) {
        const int qx = i % n, qy = i / n;
        int64_t c_split = part;
        for (int q = 0; q < 4; q++)
          c_split += sq_cost[square_id(level + 1, 2 * qx + (q & 1), 2 * qy + (q >> 1))];
        if (c_split < best) best = c_split, choice = 3;
      }
      sq_cost[sq] = best;
      sq_choice[sq] = static_cast<uint8_t>(choice);
    }
  }
  // Preorder emission with an explicit stack of squares.
  SbOut& o = out[sby * sb_cols + sbx];
  int ntree = 0, nblocks = 0;
  auto leaf = [&](int b) {
    int x, y, w, h;
    block_geom(b, &x, &y, &w, &h);
    o.tree[ntree++] = 0;
    o.w4[nblocks] = static_cast<uint8_t>(w / 4);
    o.h4[nblocks] = static_cast<uint8_t>(h / 4);
    o.mode[nblocks] = best_mode[b];
    o.mvx[nblocks] = best_mode[b] == kNumModes ? best_mv[b][0] : 0;
    o.ref[nblocks] = best_mode[b] == kNumModes ? best_ref[b] : 0;
    o.ref2[nblocks] = best_mode[b] == kNumModes ? best_ref2[b] : 255;
    o.mv2x[nblocks] = best_mode[b] == kNumModes ? best_mv2[b][0] : 0;
    o.mv2y[nblocks] = best_mode[b] == kNumModes ? best_mv2[b][1] : 0;
    o.mvy[nblocks++] = best_mode[b] == kNumModes ? best_mv[b][1] : 0;
  };
  int stack_level[16], stack_qx[16], stack_qy[16], sp = 0;
  stack_level[0] = 0, stack_qx[0] = 0, stack_qy[0] = 0, sp = 1;
  while (sp > 0) {
    sp--;
    const int level = stack_level[sp], qx = stack_qx[sp], qy = stack_qy[sp];
    const int sq = square_id(level, qx, qy), b = sq * 5;
    switch (sq_choice[sq]) {
      case 0: leaf(b); break;
      case 1: o.tree[ntree++] = 1, leaf(b + 1), leaf(b + 2); break;
      case 2: o.tree[ntree++] = 2, leaf(b + 3), leaf(b + 4); break;
      default:
        o.tree[ntree++] = 3;
        for (int q = 3; q >= 0; q--) {  // push in reverse so quadrant 0 is emitted first
          stack_level[sp] = level + 1;
          stack_qx[sp] = 2 * qx + (q & 1);
          stack_qy[sp] = 2 * qy + (q >> 1);
          sp++;
        }
        break;
    }
  }
  o.ntree = ntree;
  o.nblocks = nblocks;
}

__global__ void downsample_kernel(Plane src, uint8_t* dst) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  const int w = src.w / 2, h = src.h / 2;
  if (x < w && y < h) dst[y * w + x] = static_cast<uint8_t>(downsample_px(src, y, x));
}

__global__ void me_kernel(Plane cur, Plane ref, const int* pred_x, const int* pred_y, int pred_cols,
                          int pred_rows, int* mvx, int* mvy, int cols, int rows, bool refine) {
  const int bx = blockIdx.x * blockDim.x + threadIdx.x, by = blockIdx.y * blockDim.y + threadIdx.y;
  if (bx >= cols || by >= rows) return;
  int mx, my;
  me_cell(cur, ref, bx, by, pred_x, pred_y, pred_cols, pred_rows, &mx, &my, refine);
  if (refine) me_refine(cur, ref, bx, by, &mx, &my);
  mvx[by * cols + bx] = mx, mvy[by * cols + bx] = my;
}

// Quarter-pel phases 1..15 of src (phase 0 is src itself) at dst + k * phase.
__global__ void subpel_kernel(Plane src, uint8_t* dst, size_t phase) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, k = blockIdx.z + 1;
  if (x < src.w && y < src.h)
    dst[k * phase + static_cast<size_t>(y) * src.stride + x] = static_cast<uint8_t>(subpel_px(src, y, x, k / 4, k % 4));
}

__global__ void satd_kernel(const uint8_t* a, const uint8_t* b, int n, int w4, int h4, int* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const uint8_t* pa = a + i * 4096;
  const uint8_t* pb = b + i * 4096;
  int total = 0;
  for (int ty = 0; ty < h4 * 4; ty += 4)
    for (int tx = 0; tx < w4 * 4; tx += 4) {
      int d[16];
      for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) d[r * 4 + c] = pa[(ty + r) * 64 + tx + c] - pb[(ty + r) * 64 + tx + c];
      total += hadamard4x4(d);
    }
  out[i] = total;
}

}  // namespace

void check_cuda(const char* what) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

int blocks_per_sb(int sb_cols) {
  // One-time setup (thread-safe): the SM count and the inter task table in device memory.
  static const int sms = [] {
    int dev = 0;
    cudaDeviceProp prop;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
      throw std::runtime_error("blocks_per_sb: cannot query the CUDA device");
    inter_tasks_kernel<<<1, 1>>>();
    check_cuda("inter task table kernel");
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) throw std::runtime_error(std::string("inter task table: ") + cudaGetErrorString(e));
    return prop.multiProcessorCount;
  }();
  const int k = (2 * sms + sb_cols - 1) / sb_cols;
  return k < 1 ? 1 : k > 16 ? 16 : k;
}

static KRefs krefs(const DevRefs& d) {
  KRefs k{};
  k.n = d.n;
  for (int r = 0; r < d.n; r++)
    k.p[r] = Plane{d.plane[r].p, d.plane[r].stride, d.plane[r].w, d.plane[r].h, d.plane[r].phase}, k.mvx[r] = d.mvx[r],
    k.mvy[r] = d.mvy[r];
  k.mv_cols = d.mv_cols, k.mv_rows = d.mv_rows;
  return k;
}

void launch_intra(DevPlane src, int sb_cols, int row0, int nrows, int k, int64_t* intra, cudaStream_t stream) {
  blocks_per_sb(sb_cols);  // ensures the task tables exist
  cost_kernel<<<dim3(sb_cols, nrows * k), kThreads, 0, stream>>>(Plane{src.p, src.stride, src.w, src.h}, KRefs{}, sb_cols,
                                                                 row0, k, 0, true, intra, nullptr);
  check_cuda("intra cost kernel");
}

void launch_costs(DevPlane src, const DevRefs& refs, int sb_cols, int row0, int nrows, int k,
                  int64_t lambda_q8, int64_t* costs, cudaStream_t stream) {
  if (refs.n == 0) return;
  blocks_per_sb(sb_cols);  // ensures the task tables exist
  cost_kernel<<<dim3(sb_cols, nrows * k), kThreads, 0, stream>>>(
      Plane{src.p, src.stride, src.w, src.h}, krefs(refs), sb_cols, row0, k, lambda_q8, false, nullptr, costs);
  check_cuda("cost kernel");
}

void launch_finalize(DevPlane src, const DevRefs& refs, int sb_cols, int row0, int nrows, int64_t lambda_q8,
                     const int64_t* intra, const int64_t* costs, SbOut* out, cudaStream_t stream) {
  finalize_kernel<<<dim3(sb_cols, nrows), kThreads, 0, stream>>>(Plane{src.p, src.stride, src.w, src.h}, krefs(refs),
                                                                 sb_cols, row0, lambda_q8, intra, costs, out);
  check_cuda("finalize kernel");
}

void launch_downsample(DevPlane src, uint8_t* dst, cudaStream_t stream) {
  const dim3 threads(16, 16);
  const dim3 grid((src.w / 2 + 15) / 16, (src.h / 2 + 15) / 16);
  downsample_kernel<<<grid, threads, 0, stream>>>(Plane{src.p, src.stride, src.w, src.h}, dst);
  check_cuda("downsample kernel");
}

void launch_me_level(DevPlane cur, DevPlane ref, const int* pred_x, const int* pred_y,
                     int pred_cols, int pred_rows, int* mvx, int* mvy, int cols, int rows,
                     bool refine, cudaStream_t stream) {
  const dim3 threads(8, 8);
  const dim3 grid((cols + 7) / 8, (rows + 7) / 8);
  me_kernel<<<grid, threads, 0, stream>>>(Plane{cur.p, cur.stride, cur.w, cur.h},
                                          Plane{ref.p, ref.stride, ref.w, ref.h, ref.phase}, pred_x, pred_y,
                                          pred_cols, pred_rows, mvx, mvy, cols, rows, refine);
  check_cuda("motion search kernel");
}

void launch_subpel(DevPlane src, uint8_t* dst, size_t phase, cudaStream_t stream) {
  const dim3 threads(16, 16);
  const dim3 grid((src.w + 15) / 16, (src.h + 15) / 16, kSubpelPhases - 1);
  subpel_kernel<<<grid, threads, 0, stream>>>(Plane{src.p, src.stride, src.w, src.h}, dst, phase);
  check_cuda("subpel kernel");
}

void launch_satd_batch(const uint8_t* a, const uint8_t* b, int n, int w4, int h4, int* out) {
  satd_kernel<<<(n + 63) / 64, 64>>>(a, b, n, w4, h4, out);
  check_cuda("satd kernel");
}

}  // namespace celv2
