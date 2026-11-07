#include "core/temporal_filter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "core/threads.h"

namespace celv2 {

namespace {

constexpr int kBlock = 16;       // luma block of one motion vector
constexpr int kDown = 4;         // the coarse search's scale
constexpr int kCoarseRange = 8;  // at 1/4 scale: +-32 pels
constexpr int kFineRange = 3;
constexpr int kWindow = 2;       // the per-sample error's window: 5x5

struct View {
  const uint8_t* p;
  int stride, w, h;
  int at(int y, int x) const { return p[std::clamp(y, 0, h - 1) * stride + std::clamp(x, 0, w - 1)]; }
};

// 4x4 averages of the luma.
std::vector<uint8_t> shrink(const Frame420& f, int* w, int* h) {
  *w = f.pw / kDown, *h = f.ph / kDown;
  std::vector<uint8_t> s(static_cast<size_t>(*w) * *h);
  for (int y = 0; y < *h; y++)
    for (int x = 0; x < *w; x++) {
      int sum = 0;
      for (int i = 0; i < kDown; i++)
        for (int j = 0; j < kDown; j++) sum += f.y[static_cast<size_t>(y * kDown + i) * f.stride_y() + x * kDown + j];
      s[static_cast<size_t>(y) * *w + x] = static_cast<uint8_t>((sum + 8) / 16);
    }
  return s;
}

int64_t sad(const View& a, const View& b, int x, int y, int n, int mx, int my) {
  int64_t s = 0;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) s += std::abs(a.at(y + i, x + j) - b.at(y + i + my, x + j + mx));
  return s;
}

// The mean of d (stride n, an h x w block) over the 5x5 window around (i, j), clipped to the block.
double window_mean(const double* d, int n, int w, int h, int i, int j) {
  double s = 0;
  int c = 0;
  for (int y = std::max(0, i - kWindow); y <= std::min(h - 1, i + kWindow); y++)
    for (int x = std::max(0, j - kWindow); x <= std::min(w - 1, j + kWindow); x++) s += d[y * n + x], c++;
  return s / c;
}

}  // namespace

double estimate_noise(const uint8_t* p, int stride, int w, int h) {
  int64_t acc = 0, count = 0;
  for (int i = 1; i < h - 1; i++)
    for (int j = 1; j < w - 1; j++) {
      const uint8_t* c = p + static_cast<size_t>(i) * stride + j;
      auto m = [&](int di, int dj) { return static_cast<int>(c[di * stride + dj]); };
      const int gx = (m(-1, -1) - m(-1, 1)) + (m(1, -1) - m(1, 1)) + 2 * (m(0, -1) - m(0, 1));
      const int gy = (m(-1, -1) - m(1, -1)) + (m(-1, 1) - m(1, 1)) + 2 * (m(-1, 0) - m(1, 0));
      if (std::abs(gx) + std::abs(gy) >= 50) continue;  // edges: not noise
      acc += std::abs(4 * m(0, 0) - 2 * (m(-1, 0) + m(1, 0) + m(0, -1) + m(0, 1)) + (m(-1, -1) + m(-1, 1) + m(1, -1) + m(1, 1)));
      count++;
    }
  return count < 16 ? -1.0 : static_cast<double>(acc) / (6 * count) * 1.25331413732;  // sqrt(pi / 2)
}

Frame420 temporal_filter(const Frame420& center, const std::vector<const Frame420*>& others, double strength) {
  Frame420 out = center;
  if (strength <= 0 || others.empty()) return out;
  // Weights from the 8-bit planes; the average of the coded samples (hy..hv above 8 bits).
  const bool high = center.bit_depth > 8;
  auto sample = [high](const Frame420& f, int plane, int y, int x) {
    const int w = plane ? f.pw / 2 : f.pw, h = plane ? f.ph / 2 : f.ph;
    const size_t i = static_cast<size_t>(std::clamp(y, 0, h - 1)) * w + std::clamp(x, 0, w - 1);
    if (high) return static_cast<int>((plane == 0 ? f.hy : plane == 1 ? f.hu : f.hv)[i]);
    return static_cast<int>((plane == 0 ? f.y : plane == 1 ? f.u : f.v)[i]);
  };
  const int cw = center.pw / 2, ch = center.ph / 2;
  // Larger noise, larger weights.
  const double n_decay[3] = {0.5 + std::log(2 * std::max(0.0, estimate_noise(center.y.data(), center.pw, center.width, center.height)) + 5),
                             0.5 + std::log(2 * std::max(0.0, estimate_noise(center.u.data(), cw, center.width / 2, center.height / 2)) + 5),
                             0.5 + std::log(2 * std::max(0.0, estimate_noise(center.v.data(), cw, center.width / 2, center.height / 2)) + 5)};
  // Motion beyond a tenth of the shorter side (in 1/8 pels, as AVM's) weighs less.
  const double dist_thresh = std::max(1.0, std::min(center.width, center.height) * 0.1);
  int sw = 0, sh = 0;
  const std::vector<uint8_t> small_c = shrink(center, &sw, &sh);
  std::vector<std::vector<uint8_t>> small_o;
  for (const Frame420* f : others) small_o.push_back(shrink(*f, &sw, &sh));
  const View cy{center.y.data(), center.pw, center.pw, center.ph};
  const View cs{small_c.data(), sw, sw, sh};
  const int bcols = (center.pw + kBlock - 1) / kBlock, brows = (center.ph + kBlock - 1) / kBlock;
  parallel(brows, [&](int br) {
    for (int bc = 0; bc < bcols; bc++) {
      const int x0 = bc * kBlock, y0 = br * kBlock;
      const int bw = std::min(kBlock, center.pw - x0), bh = std::min(kBlock, center.ph - y0);
      const int cx0 = x0 / 2, cy0 = y0 / 2, cbw = bw / 2, cbh = bh / 2;
      constexpr int kC = kBlock / 2;
      // The center's own samples at weight 1.
      double acc_y[kBlock * kBlock], cnt_y[kBlock * kBlock], acc_u[kC * kC], acc_v[kC * kC], cnt_c[2][kC * kC];
      for (int i = 0; i < bh; i++)
        for (int j = 0; j < bw; j++) acc_y[i * kBlock + j] = sample(center, 0, y0 + i, x0 + j), cnt_y[i * kBlock + j] = 1;
      for (int i = 0; i < cbh; i++)
        for (int j = 0; j < cbw; j++) {
          acc_u[i * kC + j] = sample(center, 1, cy0 + i, cx0 + j), acc_v[i * kC + j] = sample(center, 2, cy0 + i, cx0 + j);
          cnt_c[0][i * kC + j] = cnt_c[1][i * kC + j] = 1;
        }
      for (size_t k = 0; k < others.size(); k++) {
        const Frame420& o = *others[k];
        const View os{small_o[k].data(), sw, sw, sh};
        const View oy{o.y.data(), o.pw, o.pw, o.ph};
        // Coarse: the 4x4 block of the 1/4 scale, the least SAD plus 1 per step from zero; then full
        // pels around it.
        int bx = 0, by = 0;
        int64_t best = INT64_MAX;
        for (int my = -kCoarseRange; my <= kCoarseRange; my++)
          for (int mx = -kCoarseRange; mx <= kCoarseRange; mx++) {
            const int64_t s = sad(cs, os, x0 / kDown, y0 / kDown, kBlock / kDown, mx, my) + std::abs(mx) + std::abs(my);
            if (s < best) best = s, bx = mx, by = my;
          }
        int fx = bx * kDown, fy = by * kDown;
        best = INT64_MAX;
        const int sx = fx, sy = fy;
        for (int my = sy - kFineRange; my <= sy + kFineRange; my++)
          for (int mx = sx - kFineRange; mx <= sx + kFineRange; mx++) {
            const int64_t s = sad(cy, oy, x0, y0, kBlock, mx, my);
            if (s < best) best = s, fx = mx, fy = my;
          }
        // Then quarter pels (bilinear) around it by squared error: AVM searches to 1/8 pel, and a
        // whole-pel prediction of fractional motion blurs what it averages in.
        const double norm = high ? 1.0 / (1 << (center.bit_depth - 8)) : 1.0;  // to 8-bit units
        // The plane's samples of o around (x, y) moved by (mx, my) 1/16 pels of that plane, bilinear.
        auto predict = [&](int plane, int x, int y, int w, int h, int mx, int my, double* out_p) {
          const int ix = mx >> 4, iy = my >> 4, ax = mx & 15, ay = my & 15;
          for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++) {
              const int yy = y + i + iy, xx = x + j + ix;
              const double a = sample(o, plane, yy, xx), b = sample(o, plane, yy, xx + 1);
              const double c = sample(o, plane, yy + 1, xx), d = sample(o, plane, yy + 1, xx + 1);
              out_p[i * w + j] = ((a * (16 - ax) + b * ax) * (16 - ay) + (c * (16 - ax) + d * ax) * ay) / 256;
            }
        };
        double cyv[kBlock * kBlock], py[kBlock * kBlock];
        for (int i = 0; i < bh; i++)
          for (int j = 0; j < bw; j++) cyv[i * bw + j] = sample(center, 0, y0 + i, x0 + j);
        auto block_sse = [&](int mx8, int my8) {
          predict(0, x0, y0, bw, bh, 2 * mx8, 2 * my8, py);
          double e = 0;
          for (int i = 0; i < bw * bh; i++) e += (cyv[i] - py[i]) * (cyv[i] - py[i]);
          return e;
        };
        int mvx = fx * 8, mvy = fy * 8;  // 1/8 luma pels
        double best_e = block_sse(mvx, mvy);
        for (int step : {4, 2}) {
          const int cx = mvx, cyy = mvy;
          for (int k = 0; k < 9; k++) {
            if (k == 4) continue;
            const int tx = cx + (k % 3 - 1) * step, ty = cyy + (k / 3 - 1) * step;
            const double e = block_sse(tx, ty);
            if (e < best_e) best_e = e, mvx = tx, mvy = ty;
          }
        }
        // The predictions, squared differences (in 8-bit units), the block's mean and the motion's weight.
        predict(0, x0, y0, bw, bh, 2 * mvx, 2 * mvy, py);
        double pu[kC * kC], pv[kC * kC];
        predict(1, cx0, cy0, cbw, cbh, mvx, mvy, pu);
        predict(2, cx0, cy0, cbw, cbh, mvx, mvy, pv);
        double dy[kBlock * kBlock], dc[2][kC * kC];
        double sse = 0;
        for (int i = 0; i < bh; i++)
          for (int j = 0; j < bw; j++) {
            const double d = (cyv[i * bw + j] - py[i * bw + j]) * norm;
            dy[i * kBlock + j] = d * d, sse += d * d;
          }
        const double block_mse = sse / (bw * bh);
        for (int i = 0; i < cbh; i++)
          for (int j = 0; j < cbw; j++) {
            const double du = (sample(center, 1, cy0 + i, cx0 + j) - pu[i * cbw + j]) * norm;
            const double dv = (sample(center, 2, cy0 + i, cx0 + j) - pv[i * cbw + j]) * norm;
            dc[0][i * kC + j] = du * du, dc[1][i * kC + j] = dv * dv;
          }
        const double d_factor = std::max(1.0, std::hypot(mvx, mvy) / dist_thresh);
        auto weight = [&](double window, int plane) {
          const double e = (5 * window + block_mse) / 6 / 20;
          return std::exp(-std::min(e * d_factor / n_decay[plane] / strength, 7.0));
        };
        for (int i = 0; i < bh; i++)
          for (int j = 0; j < bw; j++) {
            const double w = weight(window_mean(dy, kBlock, bw, bh, i, j), 0);
            acc_y[i * kBlock + j] += w * py[i * bw + j], cnt_y[i * kBlock + j] += w;
          }
        for (int p = 0; p < 2; p++)
          for (int i = 0; i < cbh; i++)
            for (int j = 0; j < cbw; j++) {
              // The chroma window plus the 4 luma samples under the sample, as AVM's.
              double s = window_mean(dc[p], kC, cbw, cbh, i, j);
              int n = 0;
              for (int y = std::max(0, i - kWindow); y <= std::min(cbh - 1, i + kWindow); y++)
                n += std::min(cbw - 1, j + kWindow) - std::max(0, j - kWindow) + 1;
              s *= n;
              for (int a = 0; a < 2; a++)
                for (int b = 0; b < 2; b++) s += dy[(2 * i + a) * kBlock + 2 * j + b];
              const double w = weight(s / (n + 4), 1 + p);
              (p ? acc_v : acc_u)[i * kC + j] += w * (p ? pv : pu)[i * cbw + j];
              cnt_c[p][i * kC + j] += w;
            }
      }
      for (int i = 0; i < bh; i++)
        for (int j = 0; j < bw; j++) {
          const size_t at = static_cast<size_t>(y0 + i) * out.stride_y() + x0 + j;
          const long v = std::lround(acc_y[i * kBlock + j] / cnt_y[i * kBlock + j]);
          if (high) out.hy[at] = static_cast<uint16_t>(v); else out.y[at] = static_cast<uint8_t>(v);
        }
      for (int i = 0; i < cbh; i++)
        for (int j = 0; j < cbw; j++) {
          const size_t at = static_cast<size_t>(cy0 + i) * out.stride_uv() + cx0 + j;
          const long u = std::lround(acc_u[i * kC + j] / cnt_c[0][i * kC + j]);
          const long v = std::lround(acc_v[i * kC + j] / cnt_c[1][i * kC + j]);
          if (high) out.hu[at] = static_cast<uint16_t>(u), out.hv[at] = static_cast<uint16_t>(v);
          else out.u[at] = static_cast<uint8_t>(u), out.v[at] = static_cast<uint8_t>(v);
        }
    }
  });
  pad_frame(out);
  round_to_8bit(out);
  return out;
}

}  // namespace celv2
