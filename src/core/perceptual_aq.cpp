#include "core/perceptual_aq.h"

#include <algorithm>
#include <cmath>

#include "core/threads.h"

namespace celv2 {

namespace {

// Starting values (spec); each change carries its measured evidence.
constexpr double kEnergy = 4;      // qindex per doubling of the variance (x264 aq-mode 1: ~1.04 QP, 1 QP ~ 4 qindex)
constexpr double kCoherence = 1;   // a pure edge (coherence 1) is not coarsened at all
constexpr double kDark = 4;        // qindex finer at black, falling to 0 at kDarkLevel
constexpr double kDarkLevel = 64;  // 8-bit luma
constexpr double kPeriodic = 1;    // a halftone field (P >= kPeriodicHigh) is not coarsened at all
constexpr double kPeriodicLow = 0.3, kPeriodicHigh = 0.7;
constexpr float kMaxOffset = 35;   // 9 * kDeltaQRes - 1, the sb_dq range
// Masking needs texture in absolute terms: a positive d is scaled from 0 at log2(E + 1) = kMaskLow
// (variance 15) to 1 at kMaskHigh (255), psy-rd's flat and busy levels. Relative to the frame alone, a
// soft cloud edge in a frame of sky (AIR OP: frame mean log2(E + 1) 2.0) was coarsened by up to 35
// qindex. 2026-11-04, --paq 0.5, BD-rate XPSNR / SSIMULACRA2 against without: AIR +2.52 -> +2.00 /
// +1.57 -> +1.57, AM -2.68 -> -2.90 / -1.74 -> -1.88, park_joy -8.79 -> -8.94 / -8.09 -> -8.35,
// 吉村 and crowd_run within 0.05.
constexpr double kMaskLow = 4, kMaskHigh = 8;

// The largest normalized autocorrelation of the zero-mean samples v (w x h) over the lags 2..6 pixels
// away (Chebyshev; half-plane: dy >= 0, dx > 0 when dy = 0), each over the samples where the block
// overlaps its shifted self: a dot field repeats at its period (near 1), random texture does not.
double periodicity(const std::vector<double>& v, int w, int h, int& best_dx, int& best_dy) {
  double best = 0;
  best_dx = best_dy = 0;
  for (int dy = 0; dy <= 6; dy++)
    for (int dx = -6; dx <= 6; dx++) {
      if (std::max(std::abs(dx), dy) < 2 || (dy == 0 && dx <= 0) || dy >= h || std::abs(dx) >= w) continue;
      double sab = 0, saa = 0, sbb = 0;
      for (int r = 0; r + dy < h; r++)
        for (int c = std::max(0, -dx); c < w && c + dx < w; c++) {
          const double a = v[r * w + c], b = v[(r + dy) * w + c + dx];
          sab += a * b, saa += a * a, sbb += b * b;
        }
      if (saa > 0 && sbb > 0 && sab / std::sqrt(saa * sbb) > best) best = sab / std::sqrt(saa * sbb), best_dx = dx, best_dy = dy;
    }
  return best;
}

}  // namespace

AqMap aq_map(const Frame420& f, double strength) {
  AqMap m;
  m.cols = (f.width + 15) / 16, m.rows = (f.height + 15) / 16;
  const size_t n = static_cast<size_t>(m.cols) * m.rows;
  m.dq.assign(n, 0.0f), m.log_energy.assign(n, 0.0f), m.coherence.assign(n, 0.0f), m.periodicity.assign(n, 0.0f);
  m.period_dx.assign(n, 0), m.period_dy.assign(n, 0);
  const int stride = f.stride_y();
  const uint8_t* const y = f.y.data();
  auto at = [&](int c, int r) {  // clamped to the visible area
    return static_cast<int>(y[std::clamp(r, 0, f.height - 1) * stride + std::clamp(c, 0, f.width - 1)]);
  };
  std::vector<double> energy(n), coherence(n), mean(n);
  // Block rows on the pool (one row at a time made --paq 1.1-1.3x the time at 1080p); sums in a fixed
  // order, so the map is the same at any thread count.
  parallel(m.rows, [&](int br) {
    for (int bc = 0; bc < m.cols; bc++) {
      const int x0 = bc * 16, y0 = br * 16, x1 = std::min(x0 + 16, f.width), y1 = std::min(y0 + 16, f.height);
      double s = 0, s2 = 0, jxx = 0, jyy = 0, jxy = 0;
      for (int r = y0; r < y1; r++)
        for (int c = x0; c < x1; c++) {
          const double v = at(c, r);
          s += v, s2 += v * v;
          const double gx = (at(c + 1, r) - at(c - 1, r)) / 2.0, gy = (at(c, r + 1) - at(c, r - 1)) / 2.0;
          jxx += gx * gx, jyy += gy * gy, jxy += gx * gy;
        }
      const double cnt = static_cast<double>(x1 - x0) * (y1 - y0);
      const size_t i = static_cast<size_t>(br) * m.cols + bc;
      mean[i] = s / cnt;
      energy[i] = std::max(0.0, s2 / cnt - mean[i] * mean[i]);
      // Structure tensor: (l1 - l2) / (l1 + l2) = sqrt((jxx - jyy)^2 + 4 jxy^2) / (jxx + jyy).
      const double tr = jxx + jyy;
      const double ratio = tr > 0 ? std::sqrt((jxx - jyy) * (jxx - jyy) + 4 * jxy * jxy) / tr : 0;
      coherence[i] = ratio * ratio;
    }
  });
  double log_mean = 0;
  for (size_t i = 0; i < n; i++) {
    m.log_energy[i] = static_cast<float>(std::log2(energy[i] + 1)), m.coherence[i] = static_cast<float>(coherence[i]);
    log_mean += std::log2(energy[i] + 1);
  }
  log_mean /= static_cast<double>(n);
  std::vector<double> d(n);
  auto offset = [&](size_t i) {
    double v = strength * kEnergy * (std::log2(energy[i] + 1) - log_mean);
    if (v > 0) v *= std::clamp((std::log2(energy[i] + 1) - kMaskLow) / (kMaskHigh - kMaskLow), 0.0, 1.0);
    if (v > 0 && coherence[i] < 0.5) {  // halftone: busy by variance, no dominant edge, but regular
      const int bc = static_cast<int>(i % m.cols), br = static_cast<int>(i / m.cols);
      const int x0 = bc * 16, y0 = br * 16, w = std::min(16, f.width - x0), h = std::min(16, f.height - y0);
      std::vector<double> z(static_cast<size_t>(w) * h);
      for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++) z[r * w + c] = y[(y0 + r) * stride + x0 + c] - mean[i];
      int dx, dy;
      const double p = periodicity(z, w, h, dx, dy);
      m.periodicity[i] = static_cast<float>(p);
      m.period_dx[i] = static_cast<int8_t>(dx), m.period_dy[i] = static_cast<int8_t>(dy);
      v *= 1 - kPeriodic * std::clamp((p - kPeriodicLow) / (kPeriodicHigh - kPeriodicLow), 0.0, 1.0);
    }
    if (v > 0) v *= 1 - kCoherence * coherence[i];  // line art: busy by variance, not texture
    v -= strength * kDark * std::max(0.0, (kDarkLevel - mean[i]) / kDarkLevel);
    d[i] = v;
  };
  parallel(m.rows, [&](int br) {
    for (int bc = 0; bc < m.cols; bc++) offset(static_cast<size_t>(br) * m.cols + bc);
  });
  double d_mean = 0;
  for (size_t i = 0; i < n; i++) d_mean += d[i];
  d_mean /= static_cast<double>(n);
  for (size_t i = 0; i < n; i++) m.dq[i] = std::clamp(static_cast<float>(d[i] - d_mean), -kMaxOffset, kMaxOffset);
  return m;
}

std::vector<int> aq_sb64(const AqMap& m, int width, int height) {
  const int cols = (width + 63) / 64, rows = (height + 63) / 64;
  std::vector<int> out(static_cast<size_t>(cols) * rows, 0);
  std::vector<float> v;
  for (int r = 0; r < rows; r++)
    for (int c = 0; c < cols; c++) {
      v.clear();
      for (int br = r * 4; br < std::min(r * 4 + 4, m.rows); br++)
        for (int bc = c * 4; bc < std::min(c * 4 + 4, m.cols); bc++) v.push_back(m.dq[static_cast<size_t>(br) * m.cols + bc]);
      if (v.empty()) continue;
      double mean = 0;
      for (const float x : v) mean += x;
      mean /= static_cast<double>(v.size());
      std::sort(v.begin(), v.end());
      const double p25 = v[(v.size() - 1) / 4];
      out[static_cast<size_t>(r) * cols + c] = static_cast<int>(std::clamp(std::lround((p25 + mean) / 2), -35L, 35L));
    }
  return out;
}

AqSpread aq_spread(const AqMap& m, int coded_sb_px) {
  const int k = std::max(1, coded_sb_px / 16);  // 16x16 blocks a superblock side
  const int cols = (m.cols + k - 1) / k, rows = (m.rows + k - 1) / k;
  std::vector<double> means;
  double within = 0;
  for (int r = 0; r < rows; r++)
    for (int c = 0; c < cols; c++) {
      double s = 0, s2 = 0;
      int cnt = 0;
      for (int br = r * k; br < std::min(r * k + k, m.rows); br++)
        for (int bc = c * k; bc < std::min(c * k + k, m.cols); bc++) {
          const double x = m.dq[static_cast<size_t>(br) * m.cols + bc];
          s += x, s2 += x * x, cnt++;
        }
      const double mu = s / cnt;
      means.push_back(mu);
      within += s2 / cnt - mu * mu;
    }
  AqSpread out;
  if (means.empty()) return out;
  out.within = within / static_cast<double>(means.size());
  double mu = 0, s2 = 0;
  for (const double x : means) mu += x, s2 += x * x;
  mu /= static_cast<double>(means.size());
  out.between = s2 / static_cast<double>(means.size()) - mu * mu;
  return out;
}

}  // namespace celv2
