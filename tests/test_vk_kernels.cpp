// The Vulkan kernels against their CPU references, on every discrete GPU (none: nothing to check).
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
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
#include "vk_shaders/probe_int64.h"
#include "vk_shaders/probe_plain.h"
#include "vk_shaders/probe_scratch.h"
#include "vk_shaders/probe_shuffle.h"
#include "vk_shaders/probe_slm.h"

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
    for (int mode = 0; mode < 6; mode++) {  // top level, a lower level, level 0 refined; by me_wide, then by me
      _putenv_s("CELV2_VK_ME_WIDE_MAX", mode < 3 ? "" : "0");
      const bool pred = mode % 3 > 0, refine = mode % 3 == 2;
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
    _putenv_s("CELV2_VK_ME_WIDE_MAX", "");
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

// V1 of the Vulkan wait spec: a family with compute and no graphics, else the first with compute;
// a forced family is taken when it has compute.
static int compute_family_rule() {
  using celv2::vk::Context;
  const VkQueueFlags g = VK_QUEUE_GRAPHICS_BIT, c = VK_QUEUE_COMPUTE_BIT, t = VK_QUEUE_TRANSFER_BIT;
  CHECK_EQ(Context::compute_family({g | c | t, t, c | t}, -1), 2u);  // the RTX 5090's order
  CHECK_EQ(Context::compute_family({g | c | t, c}, -1), 1u);         // the B50's: no transfer bit
  CHECK_EQ(Context::compute_family({t, g | c}, -1), 1u);             // no compute-only family
  CHECK_EQ(Context::compute_family({g | c | t, c | t}, 0), 0u);      // forced
  CHECK_EQ(Context::compute_family({g | c | t, t}, 1), 2u);          // forced without compute: none
  CHECK_EQ(Context::compute_family({t}, -1), 1u);                    // none
  return 0;
}

static int context_takes_a_compute_queue() {
  return each_gpu([](celv2::vk::Context& ctx, celv2::vk::Kernels&) {
    std::printf("%s: queue family flags 0x%x\n", ctx.name().c_str(), static_cast<unsigned>(ctx.queue_flags()));
    CHECK(ctx.queue_flags() & VK_QUEUE_COMPUTE_BIT);
    CHECK(!(ctx.queue_flags() & VK_QUEUE_GRAPHICS_BIT));  // both GPUs have a compute-only family
    return 0;
  });
}

// V2: the Contexts on one GPU share its VkDevice and its first queue (their own queues with
// CELV2_VK_QUEUES=many, checked on device creation only: the B50 fails running on them).
static int contexts_share_a_device() {
  for (int g = 0; g < static_cast<int>(celv2::vk::Context::devices().size()); g++) {
    {
      celv2::vk::Context a(g), b(g), c(g);
      CHECK(a.device() == b.device() && b.device() == c.device());
      CHECK(a.queue() == b.queue() && b.queue() == c.queue());
    }
    _putenv_s("CELV2_VK_QUEUES", "many");
    {
      celv2::vk::Context a(g), b(g);
      CHECK(a.device() == b.device());
      CHECK(a.queue() != b.queue());  // 4 and 8 queues
    }
    _putenv_s("CELV2_VK_QUEUES", "");
    celv2::vk::Context again(g);  // every Context closed: the device opens again
    celv2::vk::Kernels k(again);
    const std::vector<uint8_t> px = noise(64 * 32, 3);
    celv2::vk::Buffer src = put(again, px.data(), px.size()), dst = again.buffer(32 * 16);
    k.downsample({&src, 64, 64, 32}, dst);
    const std::vector<uint8_t> got = get<uint8_t>(again, dst);
    const celv2::Plane p{px.data(), 64, 64, 32};
    for (int y = 0; y < 16; y++)
      for (int x = 0; x < 32; x++) CHECK_EQ(got[y * 32 + x], celv2::downsample_px(p, y, x));
    again.free(src), again.free(dst);
  }
  return 0;
}

// Three Contexts on three threads, in batches (on the one queue they share), give what the reference gives.
static int three_threads_match_one() {
  const int w = 1920, h = 1088;
  const std::vector<uint8_t> px = noise(static_cast<size_t>(w) * h, 5);
  const celv2::Plane p{px.data(), w, w, h};
  std::vector<uint8_t> want(static_cast<size_t>(w / 2) * (h / 2));
  for (int y = 0; y < h / 2; y++)
    for (int x = 0; x < w / 2; x++) want[static_cast<size_t>(y) * (w / 2) + x] = celv2::downsample_px(p, y, x);
  {
    for (int g = 0; g < static_cast<int>(celv2::vk::Context::devices().size()); g++) {
      std::atomic<int> bad{0};
      std::vector<std::thread> ts;
      for (int t = 0; t < 3; t++)
        ts.emplace_back([&] {
          celv2::vk::Context ctx(g);
          celv2::vk::Kernels k(ctx);
          celv2::vk::Buffer src = put(ctx, px.data(), px.size()), dst = ctx.buffer(want.size());
          for (int i = 0; i < 20; i++) {
            ctx.begin_batch();
            k.downsample({&src, w, w, h}, dst);
            ctx.wait(ctx.submit());
            if (get<uint8_t>(ctx, dst) != want) bad++;
          }
          ctx.free(src), ctx.free(dst);
        });
      for (std::thread& t : ts) t.join();
      std::printf("GPU %d: %d bad of 60\n", g, bad.load());
      CHECK_EQ(bad.load(), 0);
    }
  }
  return 0;
}

// One kernel (CELV2_TEST_KERNEL: me, intra, satd or downsample; default intra) on 3 threads, each with
// its own Context, 1080p, 20 rounds each matching the first: which kernel lost the B50 with Contexts on
// separate queues (CELV2_VK_QUEUES=many, 2026-11-08). A lost device counts, not a crash. CELV2_TEST_GPU=n:
// that GPU only; CELV2_TEST_THREADS: other counts.
static int kernels_on_queues() {
  auto env = [](const char* n, int d) {
    const char* v = std::getenv(n);
    return v && *v ? std::atoi(v) : d;
  };
  const char* kv = std::getenv("CELV2_TEST_KERNEL");
  const std::string kernel = kv && *kv ? kv : "intra";
  const int only = env("CELV2_TEST_GPU", -1), threads = env("CELV2_TEST_THREADS", 3);
  const int w = 1920, h = 1088, sb_cols = w / 64, sb_rows = env("CELV2_TEST_ROWS", h / 64), cols = w / 8, rows = h / 8, n_satd = 500;  // 4 KB tiles within a plane
  const std::vector<uint8_t> a = noise(static_cast<size_t>(w) * h, 7), b = noise(static_cast<size_t>(w) * h, 8);
  constexpr int kIntraCosts = celv2::kSbBlocks * celv2::kNumModes;
  for (int g = 0; g < static_cast<int>(celv2::vk::Context::devices().size()); g++) {
    if (only >= 0 && g != only) continue;
    std::atomic<int> bad{0}, errors{0};
    std::atomic<long long> gpu_us{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < threads; t++)
      ts.emplace_back([&, t] {
        try {
          celv2::vk::Context ctx(g);
          celv2::vk::Kernels k(ctx);
          celv2::vk::Buffer ba = put(ctx, a.data(), a.size()), bb = put(ctx, b.data(), b.size());
          const size_t out_size = kernel == "intra"  ? static_cast<size_t>(sb_cols) * sb_rows * kIntraCosts * 8
                                  : kernel == "me"   ? static_cast<size_t>(cols) * rows * 4
                                  : kernel == "satd" ? static_cast<size_t>(n_satd) * 4
                                                     : a.size() / 4;
          celv2::vk::Buffer out = ctx.buffer(out_size), out2 = ctx.buffer(out_size);
          std::vector<uint8_t> first;
          for (int i = 0; i < 20; i++) {
            ctx.begin_batch();
            if (kernel == "intra") k.intra({&ba, w, w, h}, sb_cols, 0, sb_rows, 4, out);
            else if (kernel == "me") k.me_level({&ba, w, w, h}, {&bb, w, w, h}, nullptr, nullptr, 0, 0, out, out2, cols, rows, false);
            else if (kernel == "satd") k.satd_batch(ba, bb, n_satd, 16, 16, out);
            else k.downsample({&ba, w, w, h}, out);
            const auto t0 = std::chrono::steady_clock::now();
            ctx.wait(ctx.submit());
            gpu_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
            std::vector<uint8_t> got = get<uint8_t>(ctx, out);
            if (first.empty()) first = std::move(got);
            else bad += got != first;
          }
          for (celv2::vk::Buffer* p : {&ba, &bb, &out, &out2}) ctx.free(*p);
        } catch (const std::exception& e) {
          std::fprintf(stderr, "GPU %d thread %d: %s\n", g, t, e.what());
          errors++;
        }
      });
    for (std::thread& t : ts) t.join();
    std::printf("GPU %d %s: %d bad of %d, %d threads threw, %.2f ms a round\n", g, kernel.c_str(), bad.load(),
                threads * 19, errors.load(), gpu_us.load() / 1000.0 / (threads * 20));
    CHECK_EQ(bad.load() + errors.load(), 0);
  }
  return 0;
}

// kernels_on_queues with a probe shader (CELV2_TEST_PROBE: plain, slm, shuffle, scratch or int64; default slm), to
// tell which feature of the cost kernel loses the B50 on several queues. CELV2_TEST_ITERS: its loop.
static int probes_on_queues() {
  auto env = [](const char* n, int d) {
    const char* v = std::getenv(n);
    return v && *v ? std::atoi(v) : d;
  };
  const char* pv = std::getenv("CELV2_TEST_PROBE");
  const std::string probe = pv && *pv ? pv : "slm";
  const int only = env("CELV2_TEST_GPU", -1), threads = env("CELV2_TEST_THREADS", 3), iters = env("CELV2_TEST_ITERS", 4096);
  const int groups = 2048;
  struct Push {
    uint64_t out;
    int iters;
  };
  for (int g = 0; g < static_cast<int>(celv2::vk::Context::devices().size()); g++) {
    if (only >= 0 && g != only) continue;
    std::atomic<int> bad{0}, errors{0};
    std::atomic<long long> gpu_us{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < threads; t++)
      ts.emplace_back([&, t] {
        try {
          celv2::vk::Context ctx(g);
          const celv2::vk::Kernel k = probe == "plain"     ? ctx.kernel(kSpv_probe_plain, sizeof(Push))
                                      : probe == "shuffle" ? ctx.kernel(kSpv_probe_shuffle, sizeof(Push))
                                      : probe == "scratch" ? ctx.kernel(kSpv_probe_scratch, sizeof(Push))
                                      : probe == "int64"   ? ctx.kernel(kSpv_probe_int64, sizeof(Push))
                                                           : ctx.kernel(kSpv_probe_slm, sizeof(Push));
          celv2::vk::Buffer out = ctx.buffer(static_cast<size_t>(groups) * 256 * 8);  // int64 fills it
          const Push p{out.addr, iters};
          std::vector<int> first;
          for (int i = 0; i < 20; i++) {
            ctx.begin_batch();
            ctx.dispatch(k, &p, groups);
            const auto t0 = std::chrono::steady_clock::now();
            ctx.wait(ctx.submit());
            gpu_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
            std::vector<int> got = get<int>(ctx, out);
            if (first.empty()) first = std::move(got);
            else bad += got != first;
          }
          ctx.free(out);
        } catch (const std::exception& e) {
          std::fprintf(stderr, "GPU %d thread %d: %s\n", g, t, e.what());
          errors++;
        }
      });
    for (std::thread& t : ts) t.join();
    std::printf("GPU %d probe %s: %d bad of %d, %d threads threw, %.2f ms a round\n", g, probe.c_str(), bad.load(),
                threads * 19, errors.load(), gpu_us.load() / 1000.0 / (threads * 20));
    CHECK_EQ(bad.load() + errors.load(), 0);
  }
  return 0;
}

// free() waits for its own Context's work, not for another's batch on the same device.
static int free_does_not_wait_for_others() {
  for (int g = 0; g < static_cast<int>(celv2::vk::Context::devices().size()); g++) {
    celv2::vk::Context a(g), b(g);
    celv2::vk::Kernels k(a);
    const int w = 15360, h = 8640;
    celv2::vk::Buffer src = a.buffer(static_cast<size_t>(w) * h), dst = a.buffer(static_cast<size_t>(w / 2) * (h / 2));
    celv2::vk::Buffer mine = b.buffer(1 << 20);
    const auto t0 = std::chrono::steady_clock::now();
    a.begin_batch();
    for (int i = 0; i < 100; i++) k.downsample({&src, w, w, h}, dst);
    const uint64_t v = a.submit();
    b.free(mine);
    const auto t1 = std::chrono::steady_clock::now();
    a.wait(v);
    const auto t2 = std::chrono::steady_clock::now();
    const double free_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double all_ms = std::chrono::duration<double, std::milli>(t2 - t0).count();
    std::printf("%s: free() after %.1f ms, the other batch done after %.1f ms\n", a.name().c_str(), free_ms, all_ms);
    CHECK(free_ms * 2 < all_ms);
    a.free(src), a.free(dst);
  }
  return 0;
}

// CELV2_VK_TIMES=1: a batch's dispatches and copies are timed per GPU and name (the speed-first spec's step 0), and
// the results are the same as untimed.
static int kernel_times_when_asked() {
  const std::vector<std::string> gpus = celv2::vk::Context::devices();
  for (int g = 0; g < static_cast<int>(gpus.size()); g++) {
    _putenv_s("CELV2_VK_TIMES", "1");
    celv2::vk::Context ctx(g);
    _putenv_s("CELV2_VK_TIMES", "");
    celv2::vk::Kernels k(ctx);
    const std::vector<uint8_t> px = noise(64 * 32, 5);
    celv2::vk::Buffer src = ctx.buffer(px.size()), dst = ctx.buffer(32 * 16);
    celv2::vk::HostBuffer up = ctx.host_buffer(px.size()), down = ctx.host_buffer(32 * 16);
    std::memcpy(up.map, px.data(), px.size());
    ctx.begin_batch();
    ctx.copy(src, 0, up, 0, px.size());
    for (int i = 0; i < 3; i++) k.downsample({&src, 64, 64, 32}, dst);
    ctx.copy(down, 0, dst, 0, 32 * 16);
    ctx.finish();
    const celv2::Plane p{px.data(), 64, 64, 32};
    const uint8_t* got = static_cast<const uint8_t*>(down.map);
    for (int y = 0; y < 16; y++)
      for (int x = 0; x < 32; x++) CHECK_EQ(got[y * 32 + x], celv2::downsample_px(p, y, x));
    uint64_t calls = 0, copies = 0;
    double ms = -1;
    for (const celv2::vk::Context::KernelTime& t : celv2::vk::Context::kernel_times())
      if (t.device == gpus[g]) {
        if (t.kernel == "downsample") calls = t.calls, ms = t.ms;
        if (t.kernel == "upload" || t.kernel == "download") copies += t.calls;
      }
    std::printf("%s: downsample %llu calls %.4f ms\n", gpus[g].c_str(), static_cast<unsigned long long>(calls), ms);
    CHECK(calls >= 3);
    CHECK(copies >= 2);
    CHECK(ms >= 0);
    ctx.free(src), ctx.free(dst), ctx.free(up), ctx.free(down);
  }
  return 0;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);  // a crash keeps what was printed
  int fails = 0;
  fails += RUN(kernel_times_when_asked);
  fails += RUN(contexts_share_a_device);
  fails += RUN(three_threads_match_one);
  fails += RUN(kernels_on_queues);
  fails += RUN(probes_on_queues);
  fails += RUN(free_does_not_wait_for_others);
  fails += RUN(compute_family_rule);
  fails += RUN(context_takes_a_compute_queue);
  fails += RUN(downsample_matches_ref);
  fails += RUN(subpel_matches_ref);
  fails += RUN(me_matches_ref);
  fails += RUN(satd_matches_ref);
  fails += RUN(intra_costs_match_ref);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
