#include "check.h"
#include "core/intrabc_index.h"
#include "core/screen_gate.h"
#include "test_frames.h"

// The gate: off on a natural synthetic frame, on for glyphs (whose glyphs repeat). natural: scene_frame
// with a capture's noise; bare, its two-level stripes count as palette blocks (0.117, over the threshold
// real sources stay under).
static int gate_by_content() {
  Frame420 captured = scene_frame(256, 128);
  for (size_t i = 0; i < captured.y.size(); i++) {
    uint32_t n = static_cast<uint32_t>(i + 1) * 2654435761u;
    n ^= n >> 15, n *= 2246822519u, n ^= n >> 13;
    captured.y[i] = static_cast<uint8_t>(std::clamp(captured.y[i] + static_cast<int>(n >> 29) % 5 - 2, 0, 255));
  }
  const celv2::ScreenShares natural = celv2::screen_shares(captured);
  const celv2::ScreenShares glyphs = celv2::screen_shares(glyph_frame(256, 128));
  CHECK(!celv2::screen_gate(natural));
  CHECK(celv2::screen_gate(glyphs));
  CHECK(glyphs.palette > 0.5);
  CHECK(glyphs.repeat > 0.1);
  return 0;
}

// Flat 16x16s (one value) do not count as palette blocks.
static int flat_is_not_palette() {
  Frame420 f = make_frame(64, 64);
  std::fill(f.y.begin(), f.y.end(), 90);
  pad_frame(f);
  const celv2::ScreenShares s = celv2::screen_shares(f);
  CHECK(s.palette == 0);
  CHECK(s.repeat == 0);
  return 0;
}

// A 16x16 of noise at (32, 8) again at (96, 8) and (96, 72), the rest flat: from (96, 8) the index
// finds (32, 8) only, in its window, and not across its band; a flat block finds nothing.
static int index_finds_repeats() {
  Frame420 f = make_frame(160, 128);
  std::fill(f.y.begin(), f.y.end(), static_cast<uint8_t>(100));
  for (int r = 0; r < 16; r++)
    for (int c = 0; c < 16; c++) {
      uint32_t n = static_cast<uint32_t>(r * 16 + c + 1) * 2654435761u;
      n ^= n >> 15, n *= 2246822519u, n ^= n >> 13;
      const uint8_t v = static_cast<uint8_t>(n >> 24);
      for (const auto [x, y] : {std::pair{32, 8}, std::pair{96, 8}, std::pair{96, 72}}) f.y[(y + r) * f.stride_y() + x + c] = v;
    }
  const celv2::IntrabcIndex index(f, 64);
  std::vector<std::pair<int, int>> found;
  auto take = [&](int px, int py) { return found.push_back({px, py}), true; };
  index.repeats(f, 96, 8, 16, 16, 0, 160, take);
  CHECK(found == (std::vector<std::pair<int, int>>{{32, 8}}));
  found.clear();
  index.repeats(f, 96, 8, 16, 16, 64, 160, take);  // outside the columns
  CHECK(found.empty());
  index.repeats(f, 96, 72, 16, 16, 0, 160, take);  // (96, 8) is in another band
  CHECK(found.empty());
  index.repeats(f, 96, 72, 16, 16, 0, 160, take, true);  // every band above: nearest first
  CHECK(found == (std::vector<std::pair<int, int>>{{96, 8}, {32, 8}}));
  found.clear();
  index.repeats(f, 0, 0, 16, 16, 0, 160, take);  // flat
  CHECK(found.empty());
  return 0;
}

// Stage B1: copies at multiples of the map's period vector, none without a period, the best by SATD first.
static int near_copies_follow_the_period() {
  Frame420 f = scene_frame(256, 128);
  for (int y = 0; y < f.ph; y++)
    for (int x = 0; x < f.pw; x++) {
      uint32_t n = static_cast<uint32_t>(y * f.pw + x + 1) * 2654435761u;
      n ^= n >> 15;
      const int noise = static_cast<int>(n % 5) * (x < 32 ? 16 : 1);  // the far left is the worst copy
      f.y[static_cast<size_t>(y) * f.stride_y() + x] = static_cast<uint8_t>((x % 8 < 4 ? 60 : 180) + noise);
    }
  celv2::IntrabcIndex index(f, 128);
  auto left = [](int px, int) { return px < 96; };
  std::pair<int, int> got[3];
  CHECK_EQ(index.near_copies(f, 96, 32, 16, 16, left, got, 3), 0);  // no map
  auto m = std::make_shared<celv2::AqMap>();
  m->cols = 16, m->rows = 8;
  m->periodicity.assign(128, 0.9f);
  m->period_dx.assign(128, 8), m->period_dy.assign(128, 0);
  index.set_periods(m);
  const int n = index.near_copies(f, 96, 32, 16, 16, left, got, 3);
  CHECK_EQ(n, 3);
  for (int i = 0; i < n; i++) {
    CHECK_EQ(got[i].second, 32);
    CHECK(got[i].first < 96 && got[i].first >= 32);
    CHECK_EQ((96 - got[i].first) % 8, 0);
  }
  CHECK_EQ(index.near_copies(f, 96, 32, 16, 16, [](int, int) { return false; }, got, 3), 0);
  m->periodicity.assign(128, 0.4f);
  CHECK_EQ(index.near_copies(f, 96, 32, 16, 16, left, got, 3), 0);
  CHECK(celv2::periodic_share(*m) == 0);
  m->periodicity.assign(128, 0.5f);
  CHECK(celv2::periodic_share(*m) == 1);
  return 0;
}

int main() {
  int fails = 0;
  fails += RUN(gate_by_content);
  fails += RUN(flat_is_not_palette);
  fails += RUN(index_finds_repeats);
  fails += RUN(near_copies_follow_the_period);
  return fails ? 1 : 0;
}
