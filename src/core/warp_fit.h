// A rotation / zoom model fitted to a block's 8x8 motion field, for the coder's --warp search
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace celv2 {

// A rotation / zoom fitted to the decision MVs of the block's 8x8 cells (1/8 pel): u = t0 + a x +
// b y, v = t1 - b x + a y about the block centre, by least squares, then again without the cells
// more than a pel and twice the median off. False with fewer than 4 cells.
struct RotZoom {
  double a, b, t0, t1;  // a, b per pel of position; t0 / t1: the centre's MV (1/8 pel)
};
inline bool fit_rotzoom(const std::vector<std::array<double, 4>>& cells, RotZoom* out) {  // x, y (pel), u, v (1/8 pel)
  std::vector<char> use(cells.size(), 1);
  RotZoom m{};
  for (int pass = 0; pass < 2; pass++) {
    double n = 0, sx = 0, sy = 0, su = 0, sv = 0;
    for (size_t i = 0; i < cells.size(); i++)
      if (use[i]) n++, sx += cells[i][0], sy += cells[i][1], su += cells[i][2], sv += cells[i][3];
    if (n < 4) return false;
    sx /= n, sy /= n, su /= n, sv /= n;
    double num_a = 0, num_b = 0, den = 0;
    for (size_t i = 0; i < cells.size(); i++) {
      if (!use[i]) continue;
      const double x = cells[i][0] - sx, y = cells[i][1] - sy, u = cells[i][2] - su, v = cells[i][3] - sv;
      num_a += x * u + y * v, num_b += y * u - x * v, den += x * x + y * y;
    }
    if (den <= 0) return false;
    m.a = num_a / den / 8, m.b = num_b / den / 8;
    m.t0 = su - 8 * (m.a * sx + m.b * sy), m.t1 = sv - 8 * (-m.b * sx + m.a * sy);
    std::vector<double> r(cells.size());
    for (size_t i = 0; i < cells.size(); i++) {
      const double du = cells[i][2] - (m.t0 + 8 * (m.a * cells[i][0] + m.b * cells[i][1]));
      const double dv = cells[i][3] - (m.t1 + 8 * (-m.b * cells[i][0] + m.a * cells[i][1]));
      r[i] = std::sqrt(du * du + dv * dv);
    }
    std::vector<double> sorted = r;
    std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
    const double cut = std::max(8.0, 2 * sorted[sorted.size() / 2]);
    for (size_t i = 0; i < cells.size(); i++) use[i] = r[i] <= cut;
  }
  *out = m;
  return true;
}

}  // namespace celv2
