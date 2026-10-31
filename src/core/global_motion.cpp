#include "core/global_motion.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <vector>

#include "core/threads.h"

extern "C" {
#include "av2/common/warped_motion.h"
}

namespace {

constexpr int kScale = 4;       // matching on 1/4-size luma
constexpr int kBlock = 8;       // blocks of 8x8 there (32x32 pixels)
constexpr int kRange = 8;       // +-8 there (+-32 pixels)
constexpr int kStep = 12;       // a block every 12 there
constexpr int kMinVar = 64;     // flat blocks match anywhere: left out (mean squared deviation)
constexpr double kInlier = 6;   // pixels
constexpr int kIters = 300;
// Under this zoom / rotation the model is a translation, which NEARMV / NEWMV code as well.
constexpr double kMinWarp = 0.0015;

struct Match {
  double x, y, x2, y2;  // a block centre and where it lies in the reference (pixels)
};

// fn(i) for i < n on nt threads, i in turn per thread.
template <class F>
void parallel(size_t n, size_t nt, const F& fn) {
  nt = std::max<size_t>(1, std::min(n, nt));
  celv2::parallel(static_cast<int>(nt), [&](int t) {
    for (size_t i = t; i < n; i += nt) fn(i);
  });
}

template <class T>
std::vector<uint8_t> downscale(const T* p, int stride, int w, int h, size_t nt, int& dw, int& dh, int shift = 0) {
  dw = w / kScale, dh = h / kScale;
  std::vector<uint8_t> out(static_cast<size_t>(dw) * dh);
  parallel(dh, nt, [&](size_t y) {
    const T* row = p + y * kScale * stride;
    for (int x = 0; x < dw; x++) {
      int s = 0;
      for (int i = 0; i < kScale; i++)
        for (int j = 0; j < kScale; j++) s += row[static_cast<size_t>(i) * stride + x * kScale + j];
      const int n = (kScale * kScale) << shift;  // the mean, brought down to 8 bits
      out[y * dw + x] = static_cast<uint8_t>((s + n / 2) / n);
    }
  });
  return out;
}

// x2 = a x - b y + tx, y2 = b x + a y + ty from two matches.
bool solve2(const Match& p, const Match& q, double m[4]) {
  const double u = q.x - p.x, v = q.y - p.y, u2 = q.x2 - p.x2, v2 = q.y2 - p.y2;
  const double d = u * u + v * v;
  if (d < 1) return false;
  m[0] = (u * u2 + v * v2) / d, m[1] = (u * v2 - v * u2) / d;
  m[2] = p.x2 - m[0] * p.x + m[1] * p.y, m[3] = p.y2 - m[1] * p.x - m[0] * p.y;
  return true;
}

double error(const Match& p, const double m[4]) {
  const double ex = m[0] * p.x - m[1] * p.y + m[2] - p.x2, ey = m[1] * p.x + m[0] * p.y + m[3] - p.y2;
  return std::sqrt(ex * ex + ey * ey);
}

// Least squares over the inliers (centred coordinates).
void refit(const std::vector<Match>& in, double m[4]) {
  double cx = 0, cy = 0, cx2 = 0, cy2 = 0;
  for (const Match& p : in) cx += p.x, cy += p.y, cx2 += p.x2, cy2 += p.y2;
  const double n = static_cast<double>(in.size());
  cx /= n, cy /= n, cx2 /= n, cy2 /= n;
  double sa = 0, sb = 0, sd = 0;
  for (const Match& p : in) {
    const double x = p.x - cx, y = p.y - cy, x2 = p.x2 - cx2, y2 = p.y2 - cy2;
    sa += x * x2 + y * y2, sb += x * y2 - y * x2, sd += x * x + y * y;
  }
  if (sd <= 0) return;
  m[0] = sa / sd, m[1] = sb / sd;
  m[2] = cx2 - m[0] * cx + m[1] * cy, m[3] = cy2 - m[1] * cx - m[0] * cy;
}

}  // namespace

bool estimate_rotzoom(const uint8_t* cur, int cur_stride, const uint16_t* ref, int ref_stride, int ref_shift, int w, int h,
                      const struct scale_factors* sf, WarpedMotionParams* wm) {
  // Threads only past 2 FHD frames of pixels: under that, starting them costs more than they save
  // (a 1080p estimate takes ~4 ms on one).
  const size_t nt = std::min<size_t>(std::max(1u, std::thread::hardware_concurrency()),
                                     static_cast<size_t>(w) * h / (2 * 1920 * 1080));
  int dw, dh;
  const std::vector<uint8_t> c = downscale(cur, cur_stride, w, h, nt, dw, dh);
  const std::vector<uint8_t> r = downscale(ref, ref_stride, w, h, nt, dw, dh, ref_shift);
  // Block rows in turn per thread (each row's matches in its own list, joined in order).
  std::vector<int> rows;
  for (int by = kRange; by + kBlock + kRange <= dh; by += kStep) rows.push_back(by);
  std::vector<std::vector<Match>> per_row(rows.size());
  auto match_row = [&](size_t ri) {
    const int by = rows[ri];
    for (int bx = kRange; bx + kBlock + kRange <= dw; bx += kStep) {
      int sum = 0, sq = 0;
      for (int i = 0; i < kBlock; i++)
        for (int j = 0; j < kBlock; j++) {
          const int v = c[static_cast<size_t>(by + i) * dw + bx + j];
          sum += v, sq += v * v;
        }
      if (sq * kBlock * kBlock - sum * sum < kMinVar * kBlock * kBlock * kBlock * kBlock) continue;
      int best = INT32_MAX, bdx = 0, bdy = 0;
      for (int dy = -kRange; dy <= kRange; dy++)
        for (int dx = -kRange; dx <= kRange; dx++) {
          int sad = 0;
          for (int i = 0; i < kBlock && sad < best; i++)
            for (int j = 0; j < kBlock; j++)
              sad += std::abs(c[static_cast<size_t>(by + i) * dw + bx + j] -
                              r[static_cast<size_t>(by + dy + i) * dw + bx + dx + j]);
          if (sad < best) best = sad, bdx = dx, bdy = dy;
        }
      const double x = (bx + kBlock / 2.0) * kScale, y = (by + kBlock / 2.0) * kScale;
      per_row[ri].push_back({x, y, x + bdx * kScale, y + bdy * kScale});
    }
  };
  parallel(rows.size(), nt, match_row);
  std::vector<Match> matches;
  for (const auto& v : per_row) matches.insert(matches.end(), v.begin(), v.end());
  if (matches.size() < 30) return false;
  // RANSAC on pairs, a fixed sequence (the same frames give the same model).
  uint32_t seed = 12345;
  auto next = [&](size_t n) {
    seed = seed * 1103515245u + 12345u;
    return static_cast<size_t>((seed >> 8) % n);
  };
  double best_m[4] = {1, 0, 0, 0};
  size_t best_n = 0;
  for (int it = 0; it < kIters; it++) {
    const size_t i = next(matches.size()), j = next(matches.size());
    double m[4];
    if (i == j || !solve2(matches[i], matches[j], m)) continue;
    size_t n = 0;
    for (const Match& p : matches) n += error(p, m) < kInlier;
    if (n > best_n) best_n = n, std::copy(m, m + 4, best_m);
  }
  if (best_n * 10 < matches.size() * 4) return false;  // under 40% inliers
  std::vector<Match> in;
  for (const Match& p : matches)
    if (error(p, best_m) < kInlier) in.push_back(p);
  refit(in, best_m);
  if (std::fabs(best_m[0] - 1) < kMinWarp && std::fabs(best_m[1]) < kMinWarp) return false;
  // On AV2's grid (read_global_motion_params()): the alpha terms in steps of GM_ALPHA_DECODE_FACTOR,
  // the translation in steps of GM_TRANS_DECODE_FACTOR, all in 1/65536.
  auto q = [](double v, int step, int lim) {
    const long k = std::lround(v / step);
    return static_cast<int32_t>(std::max<long>(-lim, std::min<long>(lim, k)) * step);
  };
  const double one = 1 << WARPEDMODEL_PREC_BITS;
  *wm = default_warp_params;
  wm->wmtype = ROTZOOM;
  wm->wmmat[2] = q((best_m[0] - 1) * one, GM_ALPHA_DECODE_FACTOR, GM_ALPHA_MAX) + (1 << WARPEDMODEL_PREC_BITS);
  wm->wmmat[3] = q(-best_m[1] * one, GM_ALPHA_DECODE_FACTOR, GM_ALPHA_MAX);
  wm->wmmat[4] = -wm->wmmat[3];
  wm->wmmat[5] = wm->wmmat[2];
  wm->wmmat[0] = q(best_m[2] * one, GM_TRANS_DECODE_FACTOR, GM_TRANS_MAX);
  wm->wmmat[1] = q(best_m[3] * one, GM_TRANS_DECODE_FACTOR, GM_TRANS_MAX);
  av2_reduce_warp_model(wm);
  av2_get_shear_params(wm, sf);
  return !wm->invalid;
}
