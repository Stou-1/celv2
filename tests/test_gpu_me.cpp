#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <utility>
#include <vector>

#include "check.h"
#include "gpu/engine.h"
#include "gpu/inter_common.h"
#include "gpu/kernels_ref.h"
#include "io/y4m.h"
#include "test_frames.h"

// Fully textured picture (no flat areas, so every block has a unique best match);
// cur(x, y) = prev(x + sx, y + sy).
static Frame420 textured(int w, int h, int sx, int sy) {
  Frame420 f = make_frame(w, h);
  auto px = [](int x, int y) {
    uint32_t v = static_cast<uint32_t>(x + 4096) * 2654435761u ^ static_cast<uint32_t>(y + 4096) * 40503u;
    v ^= v >> 15;
    v *= 0x2c1b3c6du;
    v ^= v >> 12;
    return static_cast<int>(64 + (x + y) / 6 % 64 + (v & 63));
  };
  for (int r = 0; r < h; r++)
    for (int c = 0; c < w; c++) f.y[r * f.stride_y() + c] = static_cast<uint8_t>(px(c + sx, r + sy));
  for (auto& v : f.u) v = 128;
  for (auto& v : f.v) v = 128;
  pad_frame(f);
  return f;
}

static bool same(const DecisionMap& x, const DecisionMap& y) {
  if (x.sbs.size() != y.sbs.size()) return false;
  for (size_t i = 0; i < x.sbs.size(); i++) {
    if (x.sbs[i].tree != y.sbs[i].tree || x.sbs[i].blocks.size() != y.sbs[i].blocks.size())
      return false;
    for (size_t k = 0; k < x.sbs[i].blocks.size(); k++) {
      const auto &p = x.sbs[i].blocks[k], &q = y.sbs[i].blocks[k];
      if (p.w4 != q.w4 || p.h4 != q.h4 || p.inter != q.inter || p.intra != q.intra ||
          p.mv_row != q.mv_row || p.mv_col != q.mv_col)
        return false;
    }
  }
  return true;
}

static int gpu_me_matches_ref() {
  const int w = 640, h = 360;
  const Frame420 prev = textured(w, h, 0, 0);
  const Frame420 cur = textured(w, h, 5, -3);  // content moved left 5, down 3
  GpuEngine gpu(w, h, 64);
  const DecisionMap g = gpu.decide_inter(cur, prev, 120);
  const DecisionMap r = decide_inter_ref(cur, prev, 64, 120);
  CHECK(same(g, r));
  // Interior leaves (not touching the frame border): their MV must be (+5, -3) pixels, counted by
  // area (a few small blocks where the 8x8 MV field is wrong may pick intra or another MV).
  int interior = 0, hit = 0;
  for (int sr = 0; sr < r.sb_rows; sr++)
    for (int sc = 0; sc < r.sb_cols; sc++) {
      const SbDecision& sb = r.sbs[sr * r.sb_cols + sc];
      // Leaf positions follow from the preorder tree.
      struct Pos { int x, y, w, h; };
      std::vector<Pos> leaves;
      size_t t = 0;
      auto walk = [&](auto&& self, int x, int y, int bw, int bh) -> void {
        const Part p = sb.tree[t++];
        if (p == Part::None) leaves.push_back({x, y, bw, bh});
        else if (p == Part::Horz) self(self, x, y, bw, bh / 2), self(self, x, y + bh / 2, bw, bh / 2);
        else if (p == Part::Vert) self(self, x, y, bw / 2, bh), self(self, x + bw / 2, y, bw / 2, bh);
        else
          for (int q = 0; q < 4; q++)
            self(self, x + (q & 1) * bw / 2, y + (q >> 1) * bh / 2, bw / 2, bh / 2);
      };
      walk(walk, sc * 64, sr * 64, 64, 64);
      CHECK_EQ(leaves.size(), sb.blocks.size());
      for (size_t k = 0; k < leaves.size(); k++) {
        const Pos& p = leaves[k];
        if (p.x < 16 || p.y < 16 || p.x + p.w > w - 16 || p.y + p.h > h - 16) continue;
        interior += p.w * p.h;
        const BlockDecision& b = sb.blocks[k];
        if (b.inter && b.mv_col == 5 * 8 && b.mv_row == -3 * 8) hit += p.w * p.h;
      }
    }
  std::fprintf(stderr, "interior pixels %d, with the correct MV %d\n", interior, hit);
  CHECK(interior > 0);
  CHECK(hit * 10 >= interior * 9);
  return 0;
}

// Smooth, non-repeating picture (value noise: random levels on a 6-pixel grid, smoothstep-blended)
// sampled at (x + sx, y + sy): a fractional shift is predictable from quarter-pel positions.
static Frame420 smooth(int w, int h, double sx, double sy) {
  Frame420 f = make_frame(w, h);
  auto level = [](int gx, int gy) {
    uint32_t v = static_cast<uint32_t>(gx + 1000) * 2654435761u ^ static_cast<uint32_t>(gy + 1000) * 40503u;
    v ^= v >> 15, v *= 0x2c1b3c6du, v ^= v >> 12;
    return 40.0 + (v % 176);
  };
  auto ease = [](double t) { return t * t * (3 - 2 * t); };
  for (int r = 0; r < h; r++)
    for (int c = 0; c < w; c++) {
      const double x = (c + sx) / 6, y = (r + sy) / 6;
      const int gx = static_cast<int>(std::floor(x)), gy = static_cast<int>(std::floor(y));
      const double tx = ease(x - gx), ty = ease(y - gy);
      const double v = (level(gx, gy) * (1 - tx) + level(gx + 1, gy) * tx) * (1 - ty) +
                       (level(gx, gy + 1) * (1 - tx) + level(gx + 1, gy + 1) * tx) * ty;
      f.y[r * f.stride_y() + c] = static_cast<uint8_t>(std::lround(v));
    }
  for (auto& v : f.u) v = 128;
  for (auto& v : f.v) v = 128;
  pad_frame(f);
  return f;
}

// Content moved by (2.5, -1.25) pixels: most interior inter blocks find that MV (20, -10 in 1/8),
// and the GPU decides as the CPU reference.
static int gpu_me_finds_quarter_pel_motion() {
  const int w = 320, h = 192;
  const Frame420 prev = smooth(w, h, 0, 0), cur = smooth(w, h, 2.5, -1.25);
  GpuEngine gpu(w, h, 64);
  const DecisionMap g = gpu.decide_inter(cur, prev, 60);
  const DecisionMap r = decide_inter_ref(cur, prev, 64, 60);
  CHECK(same(g, r));
  int inter = 0, hit = 0;
  for (const SbDecision& sb : r.sbs)
    for (const BlockDecision& b : sb.blocks)
      if (b.inter) inter += b.w4 * b.h4, hit += (b.mv_col == 20 && b.mv_row == -10) * b.w4 * b.h4;
  std::fprintf(stderr, "inter area %d (4x4 units), at (2.5, -1.25) px %d\n", inter, hit);
  if (hit * 4 < inter * 3) {
    std::map<std::pair<int, int>, int> mvs;
    for (const SbDecision& sb : r.sbs)
      for (const BlockDecision& b : sb.blocks)
        if (b.inter) mvs[{b.mv_col, b.mv_row}] += b.w4 * b.h4;
    for (const auto& kv : mvs) std::fprintf(stderr, "  mv (%d, %d)/8: %d\n", kv.first.first, kv.first.second, kv.second);
  }
  CHECK(inter > 0);
  CHECK(hit * 4 >= inter * 3);
  return 0;
}

// Full MV pyramid, coarsest level first, searched in the given row splits (cuts = row boundaries).
static std::vector<RefMvLevel> me_pyramid(const RefPyramid& cur, const RefPyramid& ref, const std::vector<int>& cuts) {
  std::vector<RefMvLevel> lv(celv2::kMeLevels);
  for (int l = celv2::kMeLevels - 1; l >= 0; l--) {
    RefMvLevel& o = lv[l];
    o.cols = (cur.plane[l].w + 7) / 8, o.rows = (cur.plane[l].h + 7) / 8;
    o.mvx.assign(static_cast<size_t>(o.cols) * o.rows, 0), o.mvy = o.mvx;
    int a = 0;
    for (int c : cuts) {
      const int b = c < 0 ? o.rows : (c < o.rows ? c : o.rows);  // negative = end; clamp for small levels
      if (b > a) me_level_rows_ref(cur, ref, l, l == celv2::kMeLevels - 1 ? nullptr : &lv[l + 1], a, b, o);
      a = b > a ? b : a;
    }
  }
  return lv;
}

static int me_rows_split_matches_whole() {
  const int sizes[2][2] = {{202, 122}, {640, 360}};
  for (const auto& sz : sizes) {
    const Frame420 prev = scene_frame(sz[0], sz[1], 0, 0), cur = scene_frame(sz[0], sz[1], 5, 3);
    const RefPyramid pc = build_pyramid_ref(cur), pr = build_pyramid_ref(prev, true);
    const auto whole = me_pyramid(pc, pr, {-1});
    const auto split = me_pyramid(pc, pr, {1, 4, -1});
    for (int l = 0; l < celv2::kMeLevels; l++) {
      CHECK(whole[l].mvx == split[l].mvx);
      CHECK(whole[l].mvy == split[l].mvy);
    }
  }
  return 0;
}

static int decide_sb_ref_matches_decide() {
  const int sizes[2][2] = {{202, 122}, {640, 360}};
  for (const auto& sz : sizes) {
    const Frame420 prev = scene_frame(sz[0], sz[1], 0, 0), cur = scene_frame(sz[0], sz[1], 5, 3);
    const RefPyramid pc = build_pyramid_ref(cur), pr = build_pyramid_ref(prev, true);
    const auto mv = me_pyramid(pc, pr, {-1});
    const DecisionMap inter = decide_inter_ref(cur, prev, 64, 120), intra = decide_intra_ref(cur, 64, 120);
    for (int r = 0; r < inter.sb_rows; r++)
      for (int c = 0; c < inter.sb_cols; c++) {
        const SbDecision a = decide_sb_ref(cur, &pr, &mv[0], c, r, 64, 120);
        const SbDecision b = decide_sb_ref(cur, nullptr, nullptr, c, r, 64, 120);
        DecisionMap x = inter, y = intra;
        x.sbs = {a}, y.sbs = {b};
        DecisionMap xi = inter, yi = intra;
        xi.sbs = {inter.sbs[r * inter.sb_cols + c]}, yi.sbs = {intra.sbs[r * inter.sb_cols + c]};
        CHECK(same(x, xi));
        CHECK(same(y, yi));
      }
  }
  return 0;
}

int main() {
  int fails = 0;
  fails += RUN(gpu_me_matches_ref);
  fails += RUN(gpu_me_finds_quarter_pel_motion);
  fails += RUN(me_rows_split_matches_whole);
  fails += RUN(decide_sb_ref_matches_decide);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
