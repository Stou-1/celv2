#include <array>
#include <cmath>
#include <vector>

#include "check.h"
#include "core/warp_fit.h"

namespace {

// The 8x8 cells' MVs (1/8 pel, rounded as the engine's) of a zoom s and rotation t about (0, 0), moved
// by (u0, v0), over a square of the given size.
std::vector<std::array<double, 4>> field(int size, double s, double t, double u0, double v0) {
  const double a = s * std::cos(t) - 1, b = s * std::sin(t);
  std::vector<std::array<double, 4>> cells;
  for (int y = 4 - size / 2; y < size / 2; y += 8)
    for (int x = 4 - size / 2; x < size / 2; x += 8)
      cells.push_back({double(x), double(y), std::round(u0 + 8 * (a * x + b * y)), std::round(v0 + 8 * (-b * x + a * y))});
  return cells;
}

// A known rotation and zoom comes back within one coded step (1/64), an outlier cell left out.
int fit_finds_the_model() {
  const double s = 1.03, t = 2 * 3.14159265358979 / 180, step = 1.0 / 64;
  auto cells = field(64, s, t, 37, -21);
  celv2::RotZoom m;
  CHECK(celv2::fit_rotzoom(cells, &m));
  CHECK(std::abs(m.a - (s * std::cos(t) - 1)) < step);
  CHECK(std::abs(m.b - s * std::sin(t)) < step);
  CHECK(std::abs(m.t0 - 37) < 4 && std::abs(m.t1 + 21) < 4);
  cells[5][2] += 400;
  CHECK(celv2::fit_rotzoom(cells, &m));
  CHECK(std::abs(m.a - (s * std::cos(t) - 1)) < step);
  CHECK(std::abs(m.b - s * std::sin(t)) < step);
  return 0;
}

int fit_needs_four_cells() {
  celv2::RotZoom m;
  CHECK(celv2::fit_rotzoom(field(16, 1, 0, 0, 0), &m));  // 2x2 cells
  auto three = field(16, 1, 0, 0, 0);
  three.pop_back();
  CHECK(!celv2::fit_rotzoom(three, &m));
  return 0;
}

}  // namespace

int main() {
  int fails = 0;
  fails += RUN(fit_finds_the_model);
  fails += RUN(fit_needs_four_cells);
  return fails;
}
