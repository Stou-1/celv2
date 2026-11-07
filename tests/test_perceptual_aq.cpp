#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "check.h"
#include "core/perceptual_aq.h"
#include "io/y4m.h"
#include "state/codec_state.h"

namespace {

// Deterministic uniform values in [-a, a].
struct Lcg {
  uint32_t s = 12345;
  int next(int a) {
    s = s * 1664525u + 1013904223u;
    return static_cast<int>((s >> 8) % static_cast<uint32_t>(2 * a + 1)) - a;
  }
};

// A well-mixed hash of a position (murmur3's finalizer): no repeating pattern a lag could find.
uint32_t mix(int x, int y) {
  uint32_t h = static_cast<uint32_t>(x) * 0x9e3779b1u + static_cast<uint32_t>(y) * 0x85ebca77u;
  h ^= h >> 16, h *= 0x85ebca6bu, h ^= h >> 13, h *= 0xc2b2ae35u, h ^= h >> 16;
  return h;
}

// A luma plane from v(x, y) (clamped to 0..255), chroma 128.
template <class F>
Frame420 frame(int w, int h, F v) {
  Frame420 f = make_frame(w, h);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) f.y[y * f.stride_y() + x] = static_cast<uint8_t>(std::clamp(v(x, y), 0, 255));
  std::fill(f.u.begin(), f.u.end(), 128), std::fill(f.v.begin(), f.v.end(), 128);
  pad_frame(f);
  return f;
}

// Mean dq of the 16x16 blocks whose column is in [c0, c1).
double cols_mean(const celv2::AqMap& m, int c0, int c1) {
  double s = 0;
  int n = 0;
  for (int r = 0; r < m.rows; r++)
    for (int c = c0; c < c1; c++) s += m.dq[r * m.cols + c], n++;
  return s / n;
}

double all_mean(const celv2::AqMap& m) { return cols_mean(m, 0, m.cols); }

}  // namespace

static int flat_frame_is_zero() {
  const auto m = celv2::aq_map(frame(256, 128, [](int, int) { return 128; }), 1);
  CHECK_EQ(m.cols, 16);
  CHECK_EQ(m.rows, 8);
  for (const float d : m.dq) CHECK(std::isfinite(d) && d == 0);
  return 0;
}

static int flat_finer_noise_coarser() {
  Lcg g;
  const auto m = celv2::aq_map(frame(256, 128, [&](int x, int) { return x < 128 ? 128 : 128 + g.next(20); }), 1);
  CHECK(cols_mean(m, 0, 8) < -1);
  CHECK(cols_mean(m, 8, 16) > 1);
  CHECK(std::abs(all_mean(m)) < 0.01);
  return 0;
}

static int line_art_kept() {
  // Right half: 4-pixel black vertical lines every 16 pixels on 200, or noise of the same variance per
  // 16x16 block (each block's samples 200 or 0 in the same share, scattered).
  const auto lines = celv2::aq_map(frame(256, 128, [](int x, int) { return x >= 128 && x % 16 < 4 ? 0 : 200; }), 1);
  const auto scattered = celv2::aq_map(frame(256, 128, [](int x, int y) {
    if (x < 128) return 200;
    return mix(x, y) % 4 == 0 ? 0 : 200;  // a quarter black, as the lines' 4 of 16 columns
  }), 1);
  CHECK(cols_mean(lines, 8, 16) < cols_mean(scattered, 8, 16) - 2);
  return 0;
}

// Rows 0-127, left half: round dots of radius 2 (luma 40) on 200, on a 45-degree grid of period 6
// (centres (6i + 3j, 3j)); right half: the two levels scattered in the same share (13 of 18 samples
// dark). Rows 128-191 flat, so both textured halves are on the coarsened side, where P is computed.
static int halftone_kept() {
  const auto m = celv2::aq_map(frame(256, 192, [](int x, int y) {
    if (y >= 128) return 200;
    if (x < 128) {
      for (int j = y / 3 - 1; j <= y / 3 + 1; j++)
        for (int i = (x - 3 * j) / 6 - 2; i <= (x - 3 * j) / 6 + 2; i++) {
          const int dx = x - (6 * i + 3 * j), dy = y - 3 * j;
          if (dx * dx + dy * dy <= 4) return 40;
        }
      return 200;
    }
    return mix(x, y) % 18 < 13 ? 40 : 200;
  }), 1);
  for (int r = 0; r < 8; r++)
    for (int c = 0; c < m.cols; c++) {
      const float p = m.periodicity[r * m.cols + c];
      if (c < 8) CHECK(p > 0.7f);
      else CHECK(p < 0.3f);
    }
  double dots = 0, noise = 0;
  for (int r = 0; r < 8; r++)
    for (int c = 0; c < 8; c++) dots += m.dq[r * m.cols + c] / 64, noise += m.dq[r * m.cols + c + 8] / 64;
  CHECK(dots < noise - 2);
  // The period vector where P peaked: a vector of the dot lattice ((6, 0) and (3, 3) generate it) on the
  // dots; (0, 0) where P is not computed.
  CHECK_EQ(m.period_dx.size(), m.dq.size());
  CHECK_EQ(m.period_dy.size(), m.dq.size());
  for (int r = 0; r < 8; r++)
    for (int c = 0; c < 8; c++) {
      const int dx = m.period_dx[r * m.cols + c], dy = m.period_dy[r * m.cols + c];
      CHECK(dy >= 0 && dy % 3 == 0 && ((dx - dy) % 6 + 6) % 6 == 0 && (dx != 0 || dy != 0));
    }
  for (int c = 0; c < m.cols; c++) CHECK(m.period_dx[(m.rows - 1) * m.cols + c] == 0 && m.period_dy[(m.rows - 1) * m.cols + c] == 0);
  // P is 0 where it is not computed: flat blocks, and edges (line art).
  for (const float p : celv2::aq_map(frame(256, 128, [](int, int) { return 128; }), 1).periodicity) CHECK(p == 0);
  const auto lines = celv2::aq_map(frame(256, 128, [](int x, int) { return x >= 128 && x % 16 < 4 ? 0 : 200; }), 1);
  for (int r = 0; r < lines.rows; r++)
    for (int c = 8; c < 16; c++) CHECK(lines.periodicity[r * lines.cols + c] == 0);
  CHECK_EQ(m.log_energy.size(), m.dq.size());
  CHECK_EQ(m.coherence.size(), m.dq.size());
  return 0;
}

static int dark_finer() {
  Lcg g;
  const auto m = celv2::aq_map(frame(256, 128, [&](int x, int) { return (x < 128 ? 30 : 200) + g.next(4); }), 1);
  CHECK(cols_mean(m, 0, 8) < cols_mean(m, 8, 16));
  return 0;
}

static int offsets_clamped() {
  Lcg g;
  const Frame420 f = frame(256, 128, [&](int x, int) { return x < 128 ? 128 : 128 + g.next(120); });
  const auto m = celv2::aq_map(f, 2);
  for (const float d : m.dq) CHECK(d >= -35 && d <= 35);
  for (const int d : celv2::aq_sb64(m, 256, 128)) CHECK(d >= -35 && d <= 35);
  return 0;
}

// A flat 64x64 that a busy column of 16x16 blocks enters (the spec's sky with a person's edge): its
// offset below the plain mean of its 16 blocks. (One flat block among busy ones does not move the 25th
// percentile, and the spec's rule does not aim at it.)
static int sb64_leans_sensitive() {
  Lcg g;
  const auto m = celv2::aq_map(frame(256, 128, [&](int x, int) { return x >= 48 && x < 64 ? 128 + g.next(30) : 128; }), 1);
  double plain = 0;
  for (int r = 0; r < 4; r++)
    for (int c = 0; c < 4; c++) plain += m.dq[r * m.cols + c];
  plain /= 16;
  const auto sb = celv2::aq_sb64(m, 256, 128);
  CHECK_EQ(sb.size(), 8u);
  CHECK(sb[0] < plain);
  return 0;
}

static int edge_sizes_cover_the_grid() {
  for (const auto [w, h] : {std::pair{202, 122}, std::pair{32, 32}}) {
    Lcg g;
    const auto m = celv2::aq_map(frame(w, h, [&](int x, int) { return x < w / 2 ? 100 : 100 + g.next(25); }), 1);
    CHECK_EQ(m.cols, (w + 15) / 16);
    CHECK_EQ(m.rows, (h + 15) / 16);
    CHECK_EQ(m.dq.size(), static_cast<size_t>(m.cols) * m.rows);
    for (const float d : m.dq) CHECK(std::isfinite(d));
    const auto sb = celv2::aq_sb64(m, w, h);
    CHECK_EQ(sb.size(), static_cast<size_t>((w + 63) / 64) * ((h + 63) / 64));
  }
  return 0;
}

static int ten_bit_reads_the_8bit_planes() {
  Lcg g;
  const Frame420 f8 = frame(256, 128, [&](int x, int) { return x < 128 ? 60 : 140 + g.next(30); });
  Frame420 f10 = make_frame(256, 128, 10);
  for (size_t i = 0; i < f8.y.size(); i++) f10.hy[i] = static_cast<uint16_t>(f8.y[i] * 4);
  for (size_t i = 0; i < f8.u.size(); i++) f10.hu[i] = f10.hv[i] = 512;
  round_to_8bit(f10);
  const auto a = celv2::aq_map(f8, 1), b = celv2::aq_map(f10, 1);
  CHECK(a.dq == b.dq);
  return 0;
}

static int spread_separates() {
  Lcg g;
  // Flat and noisy 16x16 blocks alternating: the map varies inside every 128x128.
  const auto checker = celv2::aq_map(frame(256, 256, [&](int x, int y) { return ((x / 16 + y / 16) % 2) ? 128 + g.next(30) : 128; }), 1);
  const auto s1 = celv2::aq_spread(checker, 128);
  CHECK(s1.within > s1.between);
  // Flat in the left 128 columns, noisy in the rest: the map varies between superblocks.
  const auto halves = celv2::aq_map(frame(256, 256, [&](int x, int) { return x < 128 ? 128 : 128 + g.next(30); }), 1);
  const auto s2 = celv2::aq_spread(halves, 128);
  CHECK(s2.between > s2.within);
  return 0;
}

// 300x200: a 5x4 grid of 64x64 offsets 0..19 in 3x2 coded 128x128 superblocks; each takes its group's
// mean rounded to a multiple of kDeltaQRes (4), the right and bottom groups averaging what they have.
static int snap_matches_coded_superblocks() {
  std::vector<int> v(20);
  for (int i = 0; i < 20; i++) v[i] = i;
  snap_sb_dq(v, 300, 200, 128);
  const std::vector<int> expect = {4, 4, 4, 4, 8, 4, 4, 4, 4, 8, 12, 12, 16, 16, 16, 12, 12, 16, 16, 16};
  CHECK(v == expect);
  std::vector<int> neg = {-2, -3, -2, -3};  // mean -2.5: -0.625 steps, rounded to -1 step
  snap_sb_dq(neg, 128, 128, 128);
  CHECK(neg == std::vector<int>(4, -4));
  std::vector<int> none;
  snap_sb_dq(none, 300, 200, 128);
  CHECK(none.empty());
  return 0;
}

int main() {
  int fails = 0;
  fails += RUN(flat_frame_is_zero);
  fails += RUN(flat_finer_noise_coarser);
  fails += RUN(line_art_kept);
  fails += RUN(halftone_kept);
  fails += RUN(dark_finer);
  fails += RUN(offsets_clamped);
  fails += RUN(sb64_leans_sensitive);
  fails += RUN(edge_sizes_cover_the_grid);
  fails += RUN(ten_bit_reads_the_8bit_planes);
  fails += RUN(spread_separates);
  fails += RUN(snap_matches_coded_superblocks);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
