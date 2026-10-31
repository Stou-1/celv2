#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "check.h"
#include "gpu/cpu_engine.h"
#include "gpu/engine.h"
#include "gpu/intra_common.h"
#include "gpu/kernels_ref.h"
#include "io/y4m.h"
#include "test_frames.h"

static uint32_t g_seed = 12345;
static int rnd(int n) {
  g_seed = g_seed * 1664525u + 1013904223u;
  return static_cast<int>((g_seed >> 8) % n);
}

static Frame420 test_frame(int w, int h) {
  Frame420 f = make_frame(w, h);
  for (int r = 0; r < h; r++)
    for (int c = 0; c < w; c++) {
      // Smooth regions, edges and texture so every partition/mode wins somewhere.
      int v = (c / 37 % 2) ? 60 + r / 3 : 180 - c / 5;
      if ((r / 50 + c / 70) % 3 == 0) v += rnd(41) - 20;
      f.y[r * f.stride_y() + c] = static_cast<uint8_t>(v < 0 ? 0 : v > 255 ? 255 : v);
    }
  pad_frame(f);
  return f;
}

static int gpu_satd_matches_ref() {
  const int n = 100;
  std::vector<uint8_t> a(n * 64 * 64), b(n * 64 * 64);
  for (auto& v : a) v = static_cast<uint8_t>(rnd(256));
  for (auto& v : b) v = static_cast<uint8_t>(rnd(256));
  const std::vector<int> gpu = gpu_satd_batch(a.data(), b.data(), n, 16, 16);
  CHECK_EQ(gpu.size(), static_cast<size_t>(n));
  for (int i = 0; i < n; i++)
    CHECK_EQ(gpu[i], satd_ref(&a[i * 4096], 64, &b[i * 4096], 64, 16, 16));
  return 0;
}

static bool same(const DecisionMap& x, const DecisionMap& y) {
  if (x.sb_size != y.sb_size || x.sb_cols != y.sb_cols || x.sb_rows != y.sb_rows ||
      x.sbs.size() != y.sbs.size())
    return false;
  for (size_t i = 0; i < x.sbs.size(); i++) {
    if (x.sbs[i].tree != y.sbs[i].tree || x.sbs[i].blocks.size() != y.sbs[i].blocks.size()) {
      std::fprintf(stderr, "  superblock %zu: trees differ\n", i);
      return false;
    }
    for (size_t k = 0; k < x.sbs[i].blocks.size(); k++) {
      const auto &p = x.sbs[i].blocks[k], &q = y.sbs[i].blocks[k];
      if (p.w4 != q.w4 || p.h4 != q.h4 || p.intra != q.intra || p.inter != q.inter) {
        std::fprintf(stderr, "  superblock %zu block %zu: %dx%d mode %d inter %d vs %dx%d mode %d inter %d\n", i, k,
                     p.w4 * 4, p.h4 * 4, static_cast<int>(p.intra), p.inter, q.w4 * 4, q.h4 * 4,
                     static_cast<int>(q.intra), q.inter);
        return false;
      }
    }
  }
  return true;
}

// Every intra cost the GPU computes (blocks of 32x32 and below, all modes) equals the CPU rule's.
static int gpu_intra_costs_match_ref() {
 for (const auto& c : {std::pair<int, int>{640, 120}, std::pair<int, int>{202, 40}, std::pair<int, int>{202, 120}, std::pair<int, int>{202, 220}}) {
  const int w = c.first, h = w == 640 ? 360 : 122, q = c.second;
  const Frame420 src = w == 640 ? test_frame(w, h) : scene_frame(w, h, 3, -2);
  GpuEngine gpu(w, h, 64);
  gpu.decide_intra(src, q);
  const celv2::Plane s{src.y.data(), src.stride_y(), src.pw, src.ph};
  const int offset[4] = {0, 1, 5, 21};
  const int sb_cols = (w + 63) / 64, sb_rows = (h + 63) / 64;
  int bad = 0;
  for (int sb = 0; sb < sb_cols * sb_rows; sb++) {
    const std::vector<int64_t> g = gpu.intra_costs(sb);
    for (int b = 5; b < 425; b++) {  // the 64x64 level is never costed
      const int sq = b / 5, kind = b % 5;
      int level = 0;
      while (level < 3 && sq >= offset[level + 1]) level++;
      const int idx = sq - offset[level], n = 1 << level, size = 64 >> level, half = size / 2;
      int x = (idx % n) * size, y = (idx / n) * size, bw = size, bh = size;
      if (kind == 1 || kind == 2) bh = half, y += kind == 2 ? half : 0;
      if (kind == 3 || kind == 4) bw = half, x += kind == 4 ? half : 0;
      x += (sb % sb_cols) * 64, y += (sb / sb_cols) * 64;
      for (int m = 0; m < celv2::kNumModes; m++) {
        const int64_t want = celv2::block_mode_satd(s, x, y, bw, bh, m) * 256 + lambda_q8(q) * celv2::mode_bits(m);
        if (g[b * celv2::kNumModes + m] != want && bad++ < 5)
          std::fprintf(stderr, "  sb %d block %dx%d at (%d,%d) mode %d: gpu %lld cpu %lld\n", sb, bw, bh, x, y, m,
                       static_cast<long long>(g[b * celv2::kNumModes + m]), static_cast<long long>(want));
      }
    }
  }
  CHECK_EQ(bad, 0);
 }
  return 0;
}

static int gpu_intra_matches_ref() {
  const Frame420 src = test_frame(640, 360);
  GpuEngine gpu(640, 360, 64);
  for (int q : {40, 120, 220}) {
    const DecisionMap g = gpu.decide_intra(src, q);
    const DecisionMap r = decide_intra_ref(src, 64, q);
    CHECK(same(g, r));
    int parts[4] = {}, modes[kIntraModes] = {};
    for (const auto& sb : r.sbs) {
      for (Part p : sb.tree) parts[static_cast<int>(p)]++;
      for (const auto& b : sb.blocks) modes[static_cast<int>(b.intra)]++;
    }
    std::fprintf(stderr, "q%d parts N/H/V/S %d/%d/%d/%d modes", q, parts[0], parts[1], parts[2], parts[3]);
    for (int m = 0; m < kIntraModes; m++) std::fprintf(stderr, " %d", modes[m]);
    std::fprintf(stderr, "\n");
  }
  return 0;
}

// same() plus the motion vectors.
static bool same_mv(const DecisionMap& x, const DecisionMap& y) {
  if (!same(x, y)) return false;
  for (size_t i = 0; i < x.sbs.size(); i++)
    for (size_t k = 0; k < x.sbs[i].blocks.size(); k++) {
      const auto &p = x.sbs[i].blocks[k], &q = y.sbs[i].blocks[k];
      if (p.mv_row != q.mv_row || p.mv_col != q.mv_col) return false;
    }
  return true;
}

static int launch_collect_matches_decide() {
  for (const auto& wh : {std::pair<int, int>{202, 122}, std::pair<int, int>{640, 360}}) {
    const Frame420 src = test_frame(wh.first, wh.second), prev = test_frame(wh.first, wh.second);
    GpuEngine a(wh.first, wh.second, 64), b(wh.first, wh.second, 64);
    for (bool key : {false, true}) {
      const DecisionMap want = key ? a.decide_intra(src, 120) : a.decide_inter(src, prev, 120);
      // launch must not read the caller's frames after it returns.
      Frame420 s = src, p = prev;
      DecisionMap dm;
      b.launch(s, key ? nullptr : &p, 120, dm);
      std::fill(s.y.begin(), s.y.end(), uint8_t{0});
      std::fill(p.y.begin(), p.y.end(), uint8_t{255});
      RowGate gate;
      b.collect(dm, gate);
      CHECK(same_mv(dm, want));
      CHECK_EQ(gate.ready_rows(), dm.sb_rows);
    }
  }
  return 0;
}

// The CPU fallback must make exactly the GPU's decisions (KEY and P).
static int cpu_engine_matches_gpu() {
  for (const auto& wh :
       {std::pair<int, int>{202, 122}, std::pair<int, int>{640, 360}, std::pair<int, int>{32, 32}}) {
    const int w = wh.first, h = wh.second;
    const Frame420 prev = scene_frame(w, h), src = scene_frame(w, h, 3, -2);
    CpuEngine cpu(w, h, 64, 8);
    GpuEngine gpu(w, h, 64);
    for (bool key : {true, false}) {
      const Frame420* ref = key ? nullptr : &prev;
      DecisionMap c, g;
      RowGate cg, gg;
      // launch must not read the caller's frames after it returns.
      Frame420 s = src, p = prev;
      cpu.launch(s, key ? nullptr : &p, 120, c);
      std::fill(s.y.begin(), s.y.end(), uint8_t{0});
      std::fill(p.y.begin(), p.y.end(), uint8_t{255});
      cpu.collect(c, cg);
      gpu.launch(src, ref, 120, g);
      gpu.collect(g, gg);
      CHECK(same_mv(c, g));
      CHECK_EQ(cg.ready_rows(), c.sb_rows);
    }
  }
  return 0;
}

// same_mv() plus the reference of every inter block, and every candidate block's choices.
static bool same_ref(const DecisionMap& x, const DecisionMap& y) {
  if (!same_mv(x, y)) return false;
  for (size_t i = 0; i < x.sbs.size(); i++) {
    for (size_t k = 0; k < x.sbs[i].blocks.size(); k++)
      if (x.sbs[i].blocks[k].ref != y.sbs[i].blocks[k].ref) return false;
    const auto &p = x.sbs[i].cands, &q = y.sbs[i].cands;
    if (p.size() != q.size()) return false;
    for (size_t k = 0; k < p.size(); k++) {
      const BlockDecision &a = p[k].inter, &b = q[k].inter;
      if (p[k].inter_cost != q[k].inter_cost || p[k].intra_cost != q[k].intra_cost || p[k].intra != q[k].intra ||
          a.w4 != b.w4 || a.h4 != b.h4 || a.mv_row != b.mv_row || a.mv_col != b.mv_col || a.ref != b.ref ||
          a.ref2 != b.ref2 || a.mv2_row != b.mv2_row || a.mv2_col != b.mv2_col ||
          !std::equal(p[k].ref_cost, p[k].ref_cost + BlockCandidates::kRefs, q[k].ref_cost) ||
          !std::equal(p[k].ref_mv_col, p[k].ref_mv_col + BlockCandidates::kRefs, q[k].ref_mv_col) ||
          !std::equal(p[k].ref_mv_row, p[k].ref_mv_row + BlockCandidates::kRefs, q[k].ref_mv_row)) {
        std::fprintf(stderr, "  superblock %zu candidate %zu differs\n", i, k);
        return false;
      }
    }
  }
  return true;
}

// Three references: the GPU, the CPU reference rules and the CPU engine make the same decisions,
// and the reference the source matches best (the same scene, two frames back) gets used.
static int multi_ref_engines_match() {
  for (const auto& wh : {std::pair<int, int>{202, 122}, std::pair<int, int>{640, 360}}) {
    const int w = wh.first, h = wh.second;
    const Frame420 src = scene_frame(w, h, 6, -3), r0 = scene_frame(w, h, 40, 30, 5), r1 = scene_frame(w, h, 6, -3),
                   r2 = scene_frame(w, h, 3, -1);
    const std::vector<const Frame420*> refs{&r0, &r1, &r2};
    GpuEngine gpu(w, h, 64);
    CpuEngine cpu(w, h, 64, 8);
    const DecisionMap g = gpu.decide_inter(src, refs, 120), r = decide_inter_ref(src, refs, 64, 120);
    CHECK(same_ref(g, r));
    DecisionMap c;
    RowGate gate;
    cpu.launch(src, refs, 120, c);
    cpu.collect(c, gate);
    CHECK(same_ref(c, g));
    int used[3] = {};
    for (const auto& sb : g.sbs)
      for (const auto& b : sb.blocks)
        if (b.inter) used[b.ref]++;
    std::fprintf(stderr, "  %dx%d inter blocks per reference: %d %d %d\n", w, h, used[0], used[1], used[2]);
    CHECK(used[1] > used[0] && used[1] > used[2]);
  }
  return 0;
}

static int rows_publish_before_collect_ends() {
  const Frame420 src = test_frame(640, 360), prev = test_frame(640, 360);
  GpuEngine gpu(640, 360, 64);
  DecisionMap dm;
  gpu.launch(src, &prev, 120, dm);
  RowGate gate;
  std::atomic<bool> finished{false};
  // A hang cannot be released reliably and the threads use stack objects, so it aborts.
  std::thread watchdog([&] {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (!finished && std::chrono::steady_clock::now() < end) std::this_thread::yield();
    if (!finished) {
      std::fprintf(stderr, "rows_publish_before_collect_ends: not done within 20 s\n");
      std::abort();
    }
  });
  std::exception_ptr err;
  std::thread collector([&] {
    try {
      gpu.collect(dm, gate);
    } catch (...) {
      err = std::current_exception();
    }
  });
  int first = -1;
  try {
    gate.wait_row(0);
    first = gate.ready_rows();
  } catch (...) {
  }
  collector.join();
  finished = true;
  watchdog.join();
  if (err) std::rethrow_exception(err);
  CHECK(first >= 1);
  CHECK_EQ(gate.ready_rows(), dm.sb_rows);
  return 0;
}

static int gpu_engine_rejects_zero_size() {
  try {
    GpuEngine g(0, 0, 64);
  } catch (const std::runtime_error& e) {
    CHECK(std::string(e.what()).size() > 0);
    return 0;
  }
  CHECK(false);
  return 1;
}

// This PC has an RTX 5090 (compute 12.0); the probe must see it.
static int probe_gpu_reports_this_device() {
  const ProbeResult p = probe_gpu();
  CHECK(p.ok);
  CHECK(p.cc_major * 10 + p.cc_minor >= 75);
  CHECK(!p.name.empty());
  return 0;
}

// AV2 codes only the low 32x32 coefficients of a 64-point transform, so an intra block (one
// transform) 64 wide or high loses detail that no qindex brings back: every intra superblock
// splits, even on flat content where one 64x64 block would be cheapest. Inter blocks may be 64x64
// (the encoder splits their transform).
static int no_intra_block_wider_than_32() {
  for (const auto& wh : {std::pair<int, int>{640, 360}, std::pair<int, int>{202, 122}}) {
    const int w = wh.first, h = wh.second;
    Frame420 flat = make_frame(w, h);
    std::fill(flat.y.begin(), flat.y.end(), uint8_t{128});
    GpuEngine gpu(w, h, 64);
    const DecisionMap maps[] = {decide_intra_ref(flat, 64, 120), decide_inter_ref(flat, flat, 64, 120),
                                gpu.decide_intra(flat, 120), gpu.decide_inter(flat, flat, 120),
                                decide_intra_ref(scene_frame(w, h), 64, 40)};
    for (const DecisionMap& dm : maps)
      for (const auto& sb : dm.sbs)
        for (const auto& b : sb.blocks) CHECK(b.w4 <= (b.inter ? 16 : 8) && b.h4 <= (b.inter ? 16 : 8));
  }
  return 0;
}

int main() {
  int fails = 0;
  fails += RUN(no_intra_block_wider_than_32);
  fails += RUN(multi_ref_engines_match);
  fails += RUN(gpu_satd_matches_ref);
  fails += RUN(gpu_intra_costs_match_ref);
  fails += RUN(gpu_intra_matches_ref);
  fails += RUN(gpu_engine_rejects_zero_size);
  fails += RUN(launch_collect_matches_decide);
  fails += RUN(rows_publish_before_collect_ends);
  fails += RUN(cpu_engine_matches_gpu);
  fails += RUN(probe_gpu_reports_this_device);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
