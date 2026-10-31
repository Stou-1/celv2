// The Vulkan kernels against their CPU references, on every discrete GPU (none: nothing to check).
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "gpu/inter_common.h"
#include "gpu/kernels.h"
#include "gpu/kernels_ref.h"
#include "vk/vk_kernels.h"

namespace {

std::vector<uint8_t> noise(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint8_t> v(n);
  for (uint8_t& x : v) x = static_cast<uint8_t>(rng());
  return v;
}

// A device buffer holding v.
celv2::vk::Buffer put(celv2::vk::Context& ctx, const void* v, size_t n) {
  celv2::vk::Buffer b = ctx.buffer(n);
  ctx.upload(b, v, n);
  return b;
}
template <class T>
std::vector<T> get(celv2::vk::Context& ctx, const celv2::vk::Buffer& b) {
  std::vector<T> v(b.size / sizeof(T));
  ctx.download(v.data(), b, b.size);
  return v;
}

// Runs fn(ctx, kernels) on every discrete GPU.
template <class F>
int each_gpu(F fn) {
  const std::vector<std::string> gpus = celv2::vk::Context::devices();
  if (gpus.empty()) std::puts("no Vulkan GPU: skipped");
  for (int g = 0; g < static_cast<int>(gpus.size()); g++) {
    celv2::vk::Context ctx(g);
    celv2::vk::Kernels k(ctx);
    if (const int r = fn(ctx, k)) return r;
  }
  return 0;
}

}  // namespace

static int downsample_matches_ref() {
  return each_gpu([](celv2::vk::Context& ctx, celv2::vk::Kernels& k) {
    // 1080p padded, a stride wider than the plane, an odd height, and 16K.
    const int sizes[][3] = {{1920, 1088, 1920}, {1918, 1078, 1920}, {64, 33, 80}, {15360, 8640, 15360}};
    for (const auto& s : sizes) {
      const int w = s[0], h = s[1], stride = s[2];
      const std::vector<uint8_t> px = noise(static_cast<size_t>(stride) * h, 7);
      celv2::vk::Buffer src = put(ctx, px.data(), px.size()), dst = ctx.buffer(static_cast<size_t>(w / 2) * (h / 2));
      k.downsample({&src, stride, w, h}, dst);
      const std::vector<uint8_t> got = get<uint8_t>(ctx, dst);
      const celv2::Plane p{px.data(), stride, w, h};
      long bad = 0;
      for (int y = 0; y < h / 2; y++)
        for (int x = 0; x < w / 2; x++) bad += got[static_cast<size_t>(y) * (w / 2) + x] != celv2::downsample_px(p, y, x);
      std::printf("%s downsample %dx%d: %ld mismatches\n", ctx.name().c_str(), w, h, bad);
      CHECK_EQ(bad, 0L);
      ctx.free(src), ctx.free(dst);
    }
    return 0;
  });
}

static int subpel_matches_ref() {
  return each_gpu([](celv2::vk::Context& ctx, celv2::vk::Kernels& k) {
    const int w = 328, h = 200, stride = 336;
    const uint64_t phase = static_cast<uint64_t>(stride) * h;
    const std::vector<uint8_t> px = noise(phase, 11);
    celv2::vk::Buffer src = put(ctx, px.data(), px.size()), dst = ctx.buffer(phase * celv2::kSubpelPhases);
    k.subpel({&src, stride, w, h}, dst, phase);
    const std::vector<uint8_t> got = get<uint8_t>(ctx, dst);
    const celv2::Plane p{px.data(), stride, w, h};
    long bad = 0;
    for (int q = 1; q < celv2::kSubpelPhases; q++)
      for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
          bad += got[q * phase + static_cast<uint64_t>(y) * stride + x] != celv2::subpel_px(p, y, x, q / 4, q % 4);
    std::printf("%s subpel %dx%d: %ld mismatches\n", ctx.name().c_str(), w, h, bad);
    CHECK_EQ(bad, 0L);
    ctx.free(src), ctx.free(dst);
    return 0;
  });
}

// One level's search with and without a coarser field, and level 0's quarter-pel refinement.
static int me_matches_ref() {
  return each_gpu([](celv2::vk::Context& ctx, celv2::vk::Kernels& k) {
    const int w = 512, h = 264, stride = 512, cols = w / 8, rows = h / 8, pcols = (cols + 1) / 2, prows = (rows + 1) / 2;
    const uint64_t phase = static_cast<uint64_t>(stride) * h;
    // cur is ref shifted by (3, -2) plus a little noise; ref carries its 16 phases.
    std::vector<uint8_t> ref = noise(phase * celv2::kSubpelPhases, 3), cur(phase);
    std::mt19937 rng(5);
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++) {
        const int sy = std::min(h - 1, std::max(0, y - 2)), sx = std::min(w - 1, x + 3);
        cur[y * stride + x] = static_cast<uint8_t>(std::min(255, ref[sy * stride + sx] + static_cast<int>(rng() % 5)));
      }
    const celv2::Plane ref0{ref.data(), stride, w, h};
    for (int q = 1; q < celv2::kSubpelPhases; q++)
      for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) ref[q * phase + y * stride + x] = static_cast<uint8_t>(celv2::subpel_px(ref0, y, x, q / 4, q % 4));
    std::vector<int> pmx(pcols * prows), pmy(pcols * prows);
    for (size_t i = 0; i < pmx.size(); i++) pmx[i] = static_cast<int>(rng() % 9) - 4, pmy[i] = static_cast<int>(rng() % 9) - 4;
    celv2::vk::Buffer bcur = put(ctx, cur.data(), cur.size()), bref = put(ctx, ref.data(), ref.size());
    celv2::vk::Buffer bpx = put(ctx, pmx.data(), pmx.size() * 4), bpy = put(ctx, pmy.data(), pmy.size() * 4);
    celv2::vk::Buffer bmx = ctx.buffer(static_cast<size_t>(cols) * rows * 4), bmy = ctx.buffer(static_cast<size_t>(cols) * rows * 4);
    for (int mode = 0; mode < 3; mode++) {  // top level, a lower level, level 0 refined
      const bool pred = mode > 0, refine = mode == 2;
      k.me_level({&bcur, stride, w, h}, {&bref, stride, w, h, refine ? phase : 0}, pred ? &bpx : nullptr,
                 pred ? &bpy : nullptr, pcols, prows, bmx, bmy, cols, rows, refine);
      const std::vector<int> gx = get<int>(ctx, bmx), gy = get<int>(ctx, bmy);
      const celv2::Plane pc{cur.data(), stride, w, h}, pr{ref.data(), stride, w, h, refine ? phase : 0};
      long bad = 0;
      for (int by = 0; by < rows; by++)
        for (int bx = 0; bx < cols; bx++) {
          int mx, my;
          celv2::me_cell(pc, pr, bx, by, pred ? pmx.data() : nullptr, pred ? pmy.data() : nullptr, pcols, prows, &mx, &my,
                        refine);
          if (refine) celv2::me_refine(pc, pr, bx, by, &mx, &my);
          bad += gx[by * cols + bx] != mx || gy[by * cols + bx] != my;
        }
      std::printf("%s me (mode %d) %dx%d cells: %ld mismatches\n", ctx.name().c_str(), mode, cols, rows, bad);
      CHECK_EQ(bad, 0L);
    }
    for (celv2::vk::Buffer* b : {&bcur, &bref, &bpx, &bpy, &bmx, &bmy}) ctx.free(*b);
    return 0;
  });
}

static int satd_matches_ref() {
  return each_gpu([](celv2::vk::Context& ctx, celv2::vk::Kernels& k) {
    const int n = 300;
    const std::vector<uint8_t> a = noise(static_cast<size_t>(n) * 4096, 21), b = noise(static_cast<size_t>(n) * 4096, 22);
    celv2::vk::Buffer ba = put(ctx, a.data(), a.size()), bb = put(ctx, b.data(), b.size()), out = ctx.buffer(n * 4);
    const int dims[][2] = {{16, 16}, {2, 4}, {16, 1}};
    for (const auto& d : dims) {
      k.satd_batch(ba, bb, n, d[0], d[1], out);
      const std::vector<int> got = get<int>(ctx, out);
      long bad = 0;
      for (int i = 0; i < n; i++) bad += got[i] != satd_ref(a.data() + i * 4096, 64, b.data() + i * 4096, 64, d[0], d[1]);
      std::printf("%s satd %dx%d: %ld mismatches\n", ctx.name().c_str(), d[0] * 4, d[1] * 4, bad);
      CHECK_EQ(bad, 0L);
    }
    ctx.free(ba), ctx.free(bb), ctx.free(out);
    return 0;
  });
}

// cost and finalize on the same inputs as the CUDA kernels: a 320x200 source (5 x 4 superblocks, the
// last row partial) and 3 references with their phases and motion fields; also intra only.
static int cost_finalize_match_cuda() {
  int ncuda = 0;
  if (cudaGetDeviceCount(&ncuda) != cudaSuccess || ncuda == 0) return std::puts("no CUDA GPU: skipped"), 0;
  const int w = 320, h = 200, stride = 320, sb_cols = 5, sb_rows = 4, nrefs = 3, cols = w / 8, rows = h / 8, k = 2;
  const uint64_t phase = static_cast<uint64_t>(stride) * h;
  std::mt19937 rng(9);
  std::vector<uint8_t> src(phase);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++)  // gradients, edges and noise, so the intra modes and splits vary
      src[y * stride + x] = static_cast<uint8_t>(((x / 40 + y / 25) % 2 ? x + y : 255 - 2 * y) / 2 + (x * y % 7) * (y > 100 ? 9 : 1) + rng() % 6);
  std::vector<std::vector<uint8_t>> ref(nrefs);
  std::vector<std::vector<int>> fx(nrefs), fy(nrefs);
  for (int r = 0; r < nrefs; r++) {
    ref[r].resize(phase * celv2::kSubpelPhases);
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++) {
        const int sy = std::min(h - 1, std::max(0, y + r - 1)), sx = std::min(w - 1, std::max(0, x - 2 * r));
        ref[r][y * stride + x] = static_cast<uint8_t>(std::min(255, src[sy * stride + sx] + static_cast<int>(rng() % (4 + 4 * r))));
      }
    const celv2::Plane p0{ref[r].data(), stride, w, h};
    for (int q = 1; q < celv2::kSubpelPhases; q++)
      for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) ref[r][q * phase + y * stride + x] = static_cast<uint8_t>(celv2::subpel_px(p0, y, x, q / 4, q % 4));
    fx[r].resize(cols * rows), fy[r].resize(cols * rows);
    for (int i = 0; i < cols * rows; i++) {  // pairs of cells share an MV, so candidates repeat
      const int c = i % cols / 2 * 2 + i / cols * cols;
      fx[r][i] = c == i ? static_cast<int>(rng() % 41) - 20 : fx[r][c];
      fy[r][i] = c == i ? static_cast<int>(rng() % 41) - 20 : fy[r][c];
    }
  }
  const int64_t lam = lambda_q8(110);
  const size_t n_sb = static_cast<size_t>(sb_cols) * sb_rows;
  const size_t intra_n = n_sb * celv2::kIntraCosts, costs_n = n_sb * celv2::costs_per_sb(nrefs), out_n = n_sb * sizeof(celv2::SbOut);

  // CUDA.
  auto dev = [](const void* p, size_t n) {
    void* d = nullptr;
    if (cudaMalloc(&d, n) != cudaSuccess) throw std::runtime_error("cudaMalloc");
    if (p) cudaMemcpy(d, p, n, cudaMemcpyHostToDevice);
    else cudaMemset(d, 0, n);
    return d;
  };
  std::vector<void*> cuda_mem;
  auto cdev = [&](const void* p, size_t n) { return cuda_mem.emplace_back(dev(p, n)); };
  const celv2::DevPlane csrc{static_cast<uint8_t*>(cdev(src.data(), src.size())), stride, w, h};
  celv2::DevRefs crefs;
  crefs.n = nrefs, crefs.mv_cols = cols, crefs.mv_rows = rows;
  for (int r = 0; r < nrefs; r++) {
    crefs.plane[r] = {static_cast<uint8_t*>(cdev(ref[r].data(), ref[r].size())), stride, w, h, phase};
    crefs.mvx[r] = static_cast<int*>(cdev(fx[r].data(), fx[r].size() * 4));
    crefs.mvy[r] = static_cast<int*>(cdev(fy[r].data(), fy[r].size() * 4));
  }
  auto* c_intra = static_cast<int64_t*>(cdev(nullptr, intra_n * 8));
  auto* c_costs = static_cast<int64_t*>(cdev(nullptr, costs_n * 8));
  auto* c_out = static_cast<celv2::SbOut*>(cdev(nullptr, out_n));
  auto* c_out0 = static_cast<celv2::SbOut*>(cdev(nullptr, out_n));
  celv2::launch_intra(csrc, sb_cols, 0, sb_rows, k, c_intra, nullptr);
  celv2::launch_costs(csrc, crefs, sb_cols, 0, sb_rows, k, lam, c_costs, nullptr);
  celv2::launch_finalize(csrc, crefs, sb_cols, 0, sb_rows, lam, c_intra, c_costs, c_out, nullptr);
  celv2::launch_finalize(csrc, celv2::DevRefs{}, sb_cols, 0, sb_rows, lam, c_intra, c_costs, c_out0, nullptr);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  std::vector<int64_t> want_intra(intra_n), want_costs(costs_n);
  std::vector<uint8_t> want_out(out_n), want_out0(out_n);
  cudaMemcpy(want_intra.data(), c_intra, intra_n * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(want_costs.data(), c_costs, costs_n * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(want_out.data(), c_out, out_n, cudaMemcpyDeviceToHost);
  cudaMemcpy(want_out0.data(), c_out0, out_n, cudaMemcpyDeviceToHost);
  for (void* p : cuda_mem) cudaFree(p);

  return each_gpu([&](celv2::vk::Context& ctx, celv2::vk::Kernels& kv) {
    std::vector<celv2::vk::Buffer> mem;
    auto vdev = [&](const void* p, size_t n) -> celv2::vk::Buffer& {
      mem.push_back(ctx.buffer(n));
      if (p) ctx.upload(mem.back(), p, n);
      else ctx.upload(mem.back(), std::vector<uint8_t>(n).data(), n);
      return mem.back();
    };
    mem.reserve(32);  // the references below point into mem
    const celv2::vk::DevPlane vsrc{&vdev(src.data(), src.size()), stride, w, h};
    celv2::vk::Refs vrefs;
    vrefs.n = nrefs, vrefs.mv_cols = cols, vrefs.mv_rows = rows;
    for (int r = 0; r < nrefs; r++) {
      vrefs.plane[r] = {&vdev(ref[r].data(), ref[r].size()), stride, w, h, phase};
      vrefs.mvx[r] = &vdev(fx[r].data(), fx[r].size() * 4);
      vrefs.mvy[r] = &vdev(fy[r].data(), fy[r].size() * 4);
    }
    const celv2::vk::Buffer &v_intra = vdev(nullptr, intra_n * 8), &v_costs = vdev(nullptr, costs_n * 8);
    const celv2::vk::Buffer &v_out = vdev(nullptr, out_n), &v_out0 = vdev(nullptr, out_n);
    kv.intra(vsrc, sb_cols, 0, sb_rows, k, v_intra);
    kv.costs(vsrc, vrefs, sb_cols, 0, sb_rows, k, lam, v_costs);
    kv.finalize(vsrc, vrefs, sb_cols, 0, sb_rows, lam, v_intra, v_costs, v_out);
    kv.finalize(vsrc, celv2::vk::Refs{}, sb_cols, 0, sb_rows, lam, v_intra, v_costs, v_out0);
    const bool intra_ok = get<int64_t>(ctx, v_intra) == want_intra, costs_ok = get<int64_t>(ctx, v_costs) == want_costs;
    const bool out_ok = get<uint8_t>(ctx, v_out) == want_out, out0_ok = get<uint8_t>(ctx, v_out0) == want_out0;
    int nblocks = 0, ninter = 0, ncomp = 0;  // what the decisions cover
    for (size_t i = 0; i < n_sb; i++) {
      const celv2::SbOut& o = reinterpret_cast<const celv2::SbOut*>(want_out.data())[i];
      nblocks += o.nblocks;
      for (int b = 0; b < o.nblocks; b++) ninter += o.mode[b] == celv2::kNumModes, ncomp += o.ref2[b] != 255;
    }
    std::printf("%s cost/finalize vs CUDA: intra %s, inter %s, decisions %s (%d blocks: %d inter, %d compound), "
                "intra-only %s\n",
                ctx.name().c_str(), intra_ok ? "same" : "DIFFER", costs_ok ? "same" : "DIFFER",
                out_ok ? "same" : "DIFFER", nblocks, ninter, ncomp, out0_ok ? "same" : "DIFFER");
    for (celv2::vk::Buffer& b : mem) ctx.free(b);
    CHECK(intra_ok && costs_ok && out_ok && out0_ok);
    return 0;
  });
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);  // a crash keeps what was printed
  int fails = 0;
  fails += RUN(downsample_matches_ref);
  fails += RUN(subpel_matches_ref);
  fails += RUN(me_matches_ref);
  fails += RUN(satd_matches_ref);
  fails += RUN(cost_finalize_match_cuda);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
