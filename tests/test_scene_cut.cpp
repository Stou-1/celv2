#include <cstdint>
#include <cstdio>
#include <vector>

#include "check.h"
#include "core/scene_cut.h"
#include "test_frames.h"

// Runs the detector over frames and returns the indices it calls cuts.
static std::vector<int> cuts(const std::vector<Frame420>& frames) {
  SceneCutDetector det;
  std::vector<int> out;
  for (size_t i = 0; i < frames.size(); i++)
    if (det.push(frames[i], i + 1 < frames.size() ? &frames[i + 1] : nullptr)) out.push_back(static_cast<int>(i));
  return out;
}

static Frame420 flat(int w, int h, int v) {
  Frame420 f = make_frame(w, h);
  for (auto& p : f.y) p = static_cast<uint8_t>(v);
  for (auto& p : f.u) p = 128;
  for (auto& p : f.v) p = 128;
  return f;
}

// A slow pan never cuts; switching to another scene cuts exactly on its first frame.
static int cut_on_new_scene_only() {
  const int w = 320, h = 240;
  std::vector<Frame420> f;
  for (int i = 0; i < 12; i++) f.push_back(scene_frame(w, h, 2 * i, i));
  for (int i = 0; i < 8; i++) f.push_back(scene_frame(w, h, 160 + 2 * i, 97 + i, 9));
  const std::vector<int> c = cuts(f);
  CHECK_EQ(c.size(), size_t{1});
  CHECK_EQ(c[0], 12);
  return 0;
}

// A one-frame flash differs from both neighbours: no cut on it or after it.
static int flash_is_not_a_cut() {
  const int w = 320, h = 240;
  std::vector<Frame420> f;
  for (int i = 0; i < 20; i++) f.push_back(i == 10 ? flat(w, h, 250) : scene_frame(w, h, i, 0));
  CHECK(cuts(f).empty());
  return 0;
}

// Shots separated by a black frame: the cut is the first frame of the new shot, not the black one.
static int black_frame_between_shots() {
  const int w = 320, h = 240;
  std::vector<Frame420> f;
  for (int i = 0; i < 10; i++) f.push_back(scene_frame(w, h, i, 0));
  f.push_back(flat(w, h, 0));
  for (int i = 0; i < 10; i++) f.push_back(scene_frame(w, h, 200 + i, 50, 4));
  const std::vector<int> c = cuts(f);
  CHECK_EQ(c.size(), size_t{1});
  CHECK_EQ(c[0], 11);
  return 0;
}

// Animation held for several frames whose next drawing changes a third of the picture (a
// character moving over a still background): the same shot, no cut.
static int held_drawing_change_is_not_a_cut() {
  const int w = 320, h = 240;
  const Frame420 a = scene_frame(w, h, 0, 0), other = scene_frame(w, h, 160, 97, 9);
  Frame420 b = a;
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w / 3; x++) b.y[static_cast<size_t>(y) * b.stride_y() + x] = other.y[static_cast<size_t>(y) * other.stride_y() + x];
  std::vector<Frame420> f;
  for (int i = 0; i < 16; i++) f.push_back(i < 8 ? a : b);
  CHECK(cuts(f).empty());
  return 0;
}

// A partial change leaving 40% of the picture still (C3): the same shot, coded as P frames, no cut.
static int partial_change_is_not_a_cut() {
  const int w = 320, h = 240;
  const Frame420 a = scene_frame(w, h, 0, 0), other = scene_frame(w, h, 160, 97, 9);
  Frame420 b = a;
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w * 3 / 5; x++) b.y[static_cast<size_t>(y) * b.stride_y() + x] = other.y[static_cast<size_t>(y) * other.stride_y() + x];
  std::vector<Frame420> f;
  for (int i = 0; i < 16; i++) f.push_back(i < 8 ? a : b);
  CHECK(cuts(f).empty());
  return 0;
}

// A scene inside black bars (luma 16) of `top` rows above and below and `side` columns left and right.
static Frame420 boxed(const Frame420& s, int top, int side) {
  Frame420 f = s;
  for (int y = 0; y < f.height; y++)
    for (int x = 0; x < f.width; x++)
      if (y < top || y >= f.height - top || x < side || x >= f.width - side) f.y[static_cast<size_t>(y) * f.stride_y() + x] = 16;
  return f;
}

// 2.39:1 in 16:9 (bars 27% of the picture) and 4:3 in 16:9 (pillars 25%): the bars always stay still,
// but a hard cut between two shots inside them is still a cut; a partial change inside them is not.
static int cut_inside_bars() {
  const int w = 320, h = 180;
  for (const auto [top, side] : {std::pair{24, 0}, std::pair{0, 40}}) {
    std::vector<Frame420> f;
    for (int i = 0; i < 12; i++) f.push_back(boxed(scene_frame(w, h, 2 * i, i), top, side));
    for (int i = 0; i < 8; i++) f.push_back(boxed(scene_frame(w, h, 160 + 2 * i, 97 + i, 9), top, side));
    const std::vector<int> c = cuts(f);
    CHECK_EQ(c.size(), size_t{1});
    CHECK_EQ(c[0], 12);
    const Frame420 a = boxed(scene_frame(w, h, 0, 0), top, side), other = boxed(scene_frame(w, h, 160, 97, 9), top, side);
    Frame420 b = a;  // 40% of the picture inside the bars still
    for (int y = 0; y < h; y++)
      for (int x = 0; x < side + (w - 2 * side) * 3 / 5; x++) b.y[static_cast<size_t>(y) * b.stride_y() + x] = other.y[static_cast<size_t>(y) * other.stride_y() + x];
    std::vector<Frame420> g;
    for (int i = 0; i < 16; i++) g.push_back(i < 8 ? a : b);
    CHECK(cuts(g).empty());
  }
  return 0;
}

// A dark flat sky along the top edge only (48 of 128 rows, luma 20) over a held drawing whose lower part is
// drawn again 45 levels brighter: one edge is no letterbox, so the still sky counts and it is the same shot.
static int one_dark_edge_is_not_a_bar() {
  const int w = 256, h = 128;
  std::vector<Frame420> f;
  for (int i = 0; i < 20; i++) {
    Frame420 g = flat(w, h, 20);
    for (int y = 48; y < h; y++)
      for (int x = 0; x < w; x++)
        g.y[static_cast<size_t>(y) * g.stride_y() + x] =
            static_cast<uint8_t>(60 + (x * 7 + y * 13 + (x / 8) * 31 + (y / 8) * 17) % 120 + (i >= 10 ? 45 : 0));
    f.push_back(g);
  }
  CHECK(cuts(f).empty());
  return 0;
}

// Frames smaller than the 8x8 grid still work (one cell).
static int tiny_frames() {
  std::vector<Frame420> f;
  for (int i = 0; i < 6; i++) f.push_back(flat(4, 4, i < 3 ? 10 : 200));
  const std::vector<int> c = cuts(f);
  CHECK_EQ(c.size(), size_t{1});
  CHECK_EQ(c[0], 3);
  return 0;
}

// A halftone field: dots of radius 2 (luma 40) on 200 at period p, shifted by `phase` pixels, plus
// noise of +-amp.
static Frame420 halftone(int w, int h, int p, int phase, int amp, uint32_t seed) {
  Frame420 f = make_frame(w, h);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      const int dx = (x + phase) % p - p / 2, dy = y % p - p / 2;
      seed = seed * 1664525u + 1013904223u;
      const int n = amp ? static_cast<int>(seed >> 24) % (2 * amp + 1) - amp : 0;
      const int v = (dx * dx + dy * dy <= 4 ? 40 : 200) + n;
      f.y[y * f.stride_y() + x] = static_cast<uint8_t>(v < 0 ? 0 : v > 255 ? 255 : v);
    }
  for (auto& c : f.u) c = 128;
  for (auto& c : f.v) c = 128;
  pad_frame(f);
  return f;
}

// ChangeMeter: a halftone field replaced by another of about the same mean colour (a coarser period)
// is changed nearly everywhere; the same field with a little noise is not.
static int change_meter_sees_halftone() {
  ChangeMeter m;
  CHECK(m.push(halftone(256, 128, 6, 0, 0, 1)) == 0);  // the first frame
  CHECK(m.push(halftone(256, 128, 6, 0, 1, 2)) < 0.1);
  CHECK(m.push(halftone(256, 128, 8, 3, 0, 3)) > 0.5);
  return 0;
}

int main() {
  int fails = 0;
  fails += RUN(cut_on_new_scene_only);
  fails += RUN(flash_is_not_a_cut);
  fails += RUN(black_frame_between_shots);
  fails += RUN(held_drawing_change_is_not_a_cut);
  fails += RUN(tiny_frames);
  fails += RUN(change_meter_sees_halftone);
  fails += RUN(partial_change_is_not_a_cut);
  fails += RUN(cut_inside_bars);
  fails += RUN(one_dark_edge_is_not_a_bar);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
