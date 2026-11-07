#include <cmath>
#include <cstdio>
#include <set>

#include "check.h"
#include "core/vmaf_probe.h"

// A scene whose VMAF falls smoothly with qindex: 100 - 3 x 2^(q / 40) (about 94 at q 40, 88 at q 80).
static double smooth(int q) { return 100 - 3 * std::pow(2.0, q / 40.0); }

// The pick meets the target and lies within a few qindex of the true crossing, in at most 5 probes.
static int finds_the_crossing() {
  for (const double target : {80.0, 88.0, 93.0, 96.0}) {
    for (const int start : {1, 60, 130, 200}) {
      int probes = 0;
      const int q = celv2::vmaf_search(target, start, smooth, 4, &probes);
      const int truth = static_cast<int>(std::floor(40 * std::log2((100 - target) / 3)));
      CHECK(probes <= 5);
      CHECK(q >= 1 && q <= 255);
      if (truth >= 1) CHECK(smooth(q) >= target - 0.5);
      if (truth >= 1 && truth <= 255) CHECK(std::abs(q - truth) <= 4);
    }
  }
  return 0;
}

// Out of reach: above the target even at 255 takes 255; below it even at 1 takes 1.
static int out_of_reach_clamps() {
  CHECK_EQ(celv2::vmaf_search(50, 130, [](int) { return 99.0; }), 255);
  CHECK_EQ(celv2::vmaf_search(99, 130, [](int) { return 40.0; }), 1);
  return 0;
}

// Each probe asks a new qindex (no qindex probed twice).
static int no_repeated_probe() {
  std::set<int> seen;
  bool repeated = false;
  celv2::vmaf_search(93, 130, [&](int q) {
    repeated = repeated || !seen.insert(q).second;
    return smooth(q);
  });
  CHECK(!repeated);
  return 0;
}

int main() {
  int fails = 0;
  fails += RUN(finds_the_crossing);
  fails += RUN(out_of_reach_clamps);
  fails += RUN(no_repeated_probe);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
