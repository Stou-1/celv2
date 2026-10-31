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

// Frames smaller than the 8x8 grid still work (one cell).
static int tiny_frames() {
  std::vector<Frame420> f;
  for (int i = 0; i < 6; i++) f.push_back(flat(4, 4, i < 3 ? 10 : 200));
  const std::vector<int> c = cuts(f);
  CHECK_EQ(c.size(), size_t{1});
  CHECK_EQ(c[0], 3);
  return 0;
}

int main() {
  int fails = 0;
  fails += RUN(cut_on_new_scene_only);
  fails += RUN(flash_is_not_a_cut);
  fails += RUN(black_frame_between_shots);
  fails += RUN(held_drawing_change_is_not_a_cut);
  fails += RUN(tiny_frames);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
