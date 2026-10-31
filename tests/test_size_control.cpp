// --pass statistics and SizeControl: the qindex that fits a file size (24 qindex = half the bits).
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "check.h"
#include "core/size_control.h"
#include "io/stats.h"

namespace fs = std::filesystem;
using celv2::PassStats;
using celv2::SizeControl;

namespace {

std::string tmp(const char* name) { return (fs::temp_directory_path() / name).string(); }

bool throws_reading(const std::string& text) {
  { std::ofstream(tmp("celv2_stats_bad.txt"), std::ios::binary) << text; }
  try {
    celv2::read_stats(tmp("celv2_stats_bad.txt"));
  } catch (const std::runtime_error&) {
    return true;
  }
  return false;
}

int stats_round_trip() {
  const PassStats s{64, 48, 120, {100, 0, 5000}};
  celv2::write_stats(tmp("celv2_stats.txt"), s);
  const PassStats r = celv2::read_stats(tmp("celv2_stats.txt"));
  CHECK_EQ(r.width, 64);
  CHECK_EQ(r.height, 48);
  CHECK_EQ(r.qindex, 120);
  CHECK(r.bytes == s.bytes);
  CHECK(throws_reading("celv2-stats 2 64 48 120 1\n100\n"));  // another version
  CHECK(throws_reading("celv2-stats 1 64 48 120 3\n100\n0\n"));  // a frame line missing
  CHECK(throws_reading("hello\n"));
  CHECK(throws_reading(""));
  bool missing = false;
  try {
    celv2::read_stats(tmp("celv2_no_such_stats.txt"));
  } catch (const std::runtime_error&) {
    missing = true;
  }
  CHECK(missing);
  return 0;
}

PassStats flat(size_t frames, uint64_t bytes, int q) { return PassStats{64, 48, q, std::vector<uint64_t>(frames, bytes)}; }

int start_qindex_halves_per_24() {
  // 0.98 x 501,232 - 1,232 = 489,975: q 144 (500,000) does not fit, q 145 (485,670) does.
  CHECK_EQ(SizeControl::start_qindex(flat(100, 10000, 120), 501232), 145);
  CHECK_EQ(SizeControl::start_qindex(flat(100, 10000, 120), 1000000000), 1);
  CHECK_EQ(SizeControl::start_qindex(flat(100, 10000, 120), 2000), 255);  // nothing fits
  return 0;
}

// 32 + every frame's payload at the qindex it was coded + 12 a frame.
int update_moves_at_most_2() {
  const PassStats s = flat(96, 10000, 120);
  const uint64_t budget = 501232;
  const int start = SizeControl::start_qindex(s, budget);
  {  // twice the bytes the statistics predict: up by 2 a mini-GOP
    SizeControl sc(s, budget);
    int q = sc.qindex();
    CHECK_EQ(q, start);
    for (int g = 0; g < 4; g++) {
      for (int i = 0; i < 8; i++) sc.coded(static_cast<uint64_t>(2 * 10000 * std::pow(2.0, -(sc.qindex() - 120) / 24.0)));
      sc.update();
      CHECK_EQ(sc.qindex(), q + 2);
      q = sc.qindex();
    }
  }
  {  // as the statistics predict: never above the start, at most 2 below, and the file fits
    SizeControl sc(s, budget);
    double file = 32;
    while (sc.frames() < s.bytes.size()) {
      CHECK(sc.qindex() <= start && sc.qindex() >= start - 2);  // the qindex these frames are coded at
      for (int i = 0; i < 8; i++) {
        const uint64_t b = static_cast<uint64_t>(10000 * std::pow(2.0, -(sc.qindex() - 120) / 24.0));
        sc.coded(b);
        file += b + 12;
      }
      sc.update();
    }
    CHECK(file <= 0.98 * budget);
  }
  return 0;
}

// "even at qindex 255" only when every frame was coded at 255, not when the qindex got there late.
int coarsest_only_when_every_frame_was() {
  {
    SizeControl sc(flat(16, 10000, 120), 2000);  // nothing fits
    for (int i = 0; i < 16; i++) sc.coded(10000);
    sc.update();
    CHECK(sc.coarsest());
  }
  {
    const PassStats s = flat(96, 10000, 120);
    SizeControl sc(s, 501232);
    for (int g = 0; g < 12; g++) {  // far over the estimate: up by 2 a mini-GOP, never all at 255
      for (int i = 0; i < 8; i++) sc.coded(100000);
      sc.update();
    }
    CHECK(!sc.coarsest());
  }
  return 0;
}

// Chunks (--workers): 16-frame chunks of 10,000-byte pass-1 frames at q 120.
uint64_t at(double q, double s) { return static_cast<uint64_t>(10000 * std::pow(2.0, -(q - 120) / s)); }

// The halving step s from frames coded at two qindexes; one qindex: 24; out of range: 12 / 48.
int slope_fit() {
  auto fitted = [](double s, bool two) {
    SizeControl sc(flat(64, 10000, 120), 400000, 2);
    for (int i = 0; i < 16; i++) sc.coded(0, i, at(140, s), 140);
    for (int i = 16; i < 32; i++) sc.coded(1, i, at(two ? 150 : 140, s), two ? 150 : 140);
    return sc.slope();
  };
  CHECK(std::abs(fitted(30, true) - 30) < 0.5);
  CHECK_EQ(fitted(30, false), 24.0);
  CHECK_EQ(fitted(5, true), 12.0);
  CHECK_EQ(fitted(200, true), 48.0);
  return 0;
}

// Chunk k sees the chunks up to k - workers and itself, so what chunk 1 codes changes nothing for
// chunk 0 or for chunk 2's start (workers 2).
int chunks_see_only_finished() {
  auto run = [](double chunk1_scale, int& update0, int& start2) {
    SizeControl sc(flat(64, 10000, 120), 400000, 2);
    const int q0 = sc.chunk_start_qindex(0, 0, 16), q1 = sc.chunk_start_qindex(1, 16, 16);
    for (int i = 0; i < 8; i++) sc.coded(0, i, at(q0, 24), q0);
    for (int i = 16; i < 32; i++) sc.coded(1, i, static_cast<uint64_t>(chunk1_scale * at(q1, 24)), q1);
    update0 = sc.chunk_update(0);
    for (int i = 8; i < 16; i++) sc.coded(0, i, at(update0, 24), update0);
    start2 = sc.chunk_start_qindex(2, 32, 16);
  };
  int u_a, s_a, u_b, s_b;
  run(1, u_a, s_a);
  run(2, u_b, s_b);
  CHECK_EQ(u_a, u_b);
  CHECK_EQ(s_a, s_b);
  return 0;
}

// The first `workers` chunks start at the start qindex + 0, +1, -1 (to measure the slope early).
int dither_first_chunks() {
  const PassStats s = flat(64, 10000, 120);
  SizeControl sc(s, 400000, 3);
  const int base = SizeControl::start_qindex(s, 400000);
  CHECK_EQ(sc.chunk_start_qindex(0, 0, 16), base);
  CHECK_EQ(sc.chunk_start_qindex(1, 16, 16), base + 1);
  CHECK_EQ(sc.chunk_start_qindex(2, 32, 16), base - 1);
  return 0;
}

int coded_beyond_stats_throws() {
  SizeControl sc(flat(3, 1000, 120), 100000);
  sc.coded(1000), sc.coded(1000), sc.coded(1000);
  CHECK_EQ(sc.frames(), 3u);
  bool threw = false;
  try {
    sc.coded(1000);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw);
  return 0;
}

}  // namespace

int main() {
  int fails = 0;
  fails += RUN(stats_round_trip);
  fails += RUN(start_qindex_halves_per_24);
  fails += RUN(update_moves_at_most_2);
  fails += RUN(coarsest_only_when_every_frame_was);
  fails += RUN(slope_fit);
  fails += RUN(chunks_see_only_finished);
  fails += RUN(dither_first_chunks);
  fails += RUN(coded_beyond_stats_throws);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
