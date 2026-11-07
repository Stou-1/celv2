// The Vulkan kernels against their CPU references, on every discrete GPU (none: nothing to check).
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "gpu/inter_common.h"
#include "gpu/sb_blocks.h"
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

// Every intra cost (blocks of 32x32 and below, all modes) equals the CPU rule's: a 320x200 source, 5 x 4
// superblocks, the last row partial. The costs of inter blocks and the decisions are checked whole,
// against the CPU engine, in test_gpu_me and test_gpu_intra.
static int intra_costs_match_ref() {
  const int w = 320, h = 200, sb_cols = 5, sb_rows = 4, k = 2;
  std::vector<uint8_t> src(static_cast<size_t>(w) * h);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++)  // gradients, edges and noise, so the intra modes vary
      src[y * w + x] = static_cast<uint8_t>(((x / 40 + y / 25) % 2 ? x + y : 255 - 2 * y) / 2 + (x * y % 7));
  constexpr int kIntraCosts = celv2::kSbBlocks * celv2::kNumModes;
  const size_t n = static_cast<size_t>(sb_cols) * sb_rows * kIntraCosts;
  const celv2::Plane s{src.data(), w, w, h};
  const int offset[4] = {0, 1, 5, 21};
  return each_gpu([&](celv2::vk::Context& ctx, celv2::vk::Kernels& kv) {
    celv2::vk::Buffer sb = put(ctx, src.data(), src.size()), costs = ctx.buffer(n * 8);
    kv.intra(celv2::vk::DevPlane{&sb, w, w, h}, sb_cols, 0, sb_rows, k, costs);
    const std::vector<int64_t> g = get<int64_t>(ctx, costs);
    ctx.free(sb), ctx.free(costs);
    int bad = 0;
    for (int i = 0; i < sb_cols * sb_rows; i++)
      for (int b = 5; b < celv2::kSbBlocks; b++) {  // the 64x64 level is never costed
        const int sq = b / 5, kind = b % 5;
        int level = 0;
        while (level < 3 && sq >= offset[level + 1]) level++;
        const int idx = sq - offset[level], m = 1 << level, size = 64 >> level, half = size / 2;
        int x = (idx % m) * size, y = (idx / m) * size, bw = size, bh = size;
        if (kind == 1 || kind == 2) bh = half, y += kind == 2 ? half : 0;
        if (kind == 3 || kind == 4) bw = half, x += kind == 4 ? half : 0;
        x += (i % sb_cols) * 64, y += (i / sb_cols) * 64;
        for (int mode = 0; mode < celv2::kNumModes; mode++) {
          const int64_t want = celv2::block_mode_satd(s, x, y, bw, bh, mode) * 256;  // finalize adds the bits
          if (g[static_cast<size_t>(i) * kIntraCosts + b * celv2::kNumModes + mode] != want) bad++;
        }
      }
    CHECK_EQ(bad, 0);
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
  fails += RUN(intra_costs_match_ref);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
