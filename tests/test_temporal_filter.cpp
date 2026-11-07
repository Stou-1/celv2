#include <cstdio>
#include <vector>

#include "check.h"
#include "core/temporal_filter.h"
#include "test_frames.h"

// The same picture with its own noise (+-6 luma) per seed.
static Frame420 noisy(const Frame420& clean, uint32_t seed) {
  Frame420 f = clean;
  uint32_t n = seed;
  for (auto& p : f.y) {
    n = n * 1103515245u + 12345u;
    const int v = p + static_cast<int>(n >> 16) % 13 - 6;
    p = static_cast<uint8_t>(v < 0 ? 0 : v > 255 ? 255 : v);
  }
  return f;
}

static double luma_mse(const Frame420& a, const Frame420& b) {
  double s = 0;
  for (size_t i = 0; i < a.y.size(); i++) s += (a.y[i] - b.y[i]) * (a.y[i] - b.y[i]);
  return s / a.y.size();
}

// A still picture: averaging in two noisy copies brings the frame closer to the clean one.
static int still_noise_is_averaged_away() {
  const Frame420 clean = scene_frame(128, 96);
  const Frame420 c = noisy(clean, 1), a = noisy(clean, 2), b = noisy(clean, 3);
  const Frame420 out = celv2::temporal_filter(c, {&a, &b});
  CHECK(luma_mse(out, clean) < 0.7 * luma_mse(c, clean));
  return 0;
}

// No neighbours, or neighbours equal to the frame: the frame comes back unchanged.
static int nothing_to_average_keeps_the_frame() {
  const Frame420 c = noisy(scene_frame(128, 96), 1);
  CHECK(celv2::temporal_filter(c, {}).y == c.y);
  CHECK(celv2::temporal_filter(c, {&c, &c}).y == c.y);
  return 0;
}

int main() {
  int fails = 0;
  fails += RUN(still_noise_is_averaged_away);
  fails += RUN(nothing_to_average_keeps_the_frame);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
