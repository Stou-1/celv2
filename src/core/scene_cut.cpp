#include "core/scene_cut.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "core/threads.h"

namespace {

// 8x8 cell averages of the visible luma (at least one cell), in bands of cell rows on several
// threads (one thread took 30 ms a 16K frame).
std::vector<int> cells(const Frame420& f) {
  const int cw = std::max(1, f.width / 8), ch = std::max(1, f.height / 8);
  const int bw = std::max(1, f.width / cw), bh = std::max(1, f.height / ch);
  std::vector<int> out(static_cast<size_t>(cw) * ch);
  auto band = [&](int cy0, int cy1) {
    for (int cy = cy0; cy < cy1; cy++)
      for (int cx = 0; cx < cw; cx++) {
        int sum = 0;
        for (int y = 0; y < bh; y++)
          for (int x = 0; x < bw; x++) sum += f.y[static_cast<size_t>(cy * bh + y) * f.stride_y() + cx * bw + x];
        out[static_cast<size_t>(cy) * cw + cx] = sum / (bw * bh);
      }
  };
  const int n = std::max(1, std::min(static_cast<int>(std::thread::hardware_concurrency()), ch / 64));
  std::vector<std::thread> pool;
  for (int k = 1; k < n; k++) pool.emplace_back(band, ch * k / n, ch * (k + 1) / n);
  band(0, ch / n);
  for (std::thread& t : pool) t.join();
  return out;
}

double mad(const std::vector<int>& a, const std::vector<int>& b) {
  double s = 0;
  for (size_t i = 0; i < a.size(); i++) s += std::abs(a[i] - b[i]);
  return s / static_cast<double>(a.size());
}

// The share of cells within 2 levels of the other frame's.
double unchanged(const std::vector<int>& a, const std::vector<int>& b) {
  size_t n = 0;
  for (size_t i = 0; i < a.size(); i++) n += std::abs(a[i] - b[i]) <= 2;
  return static_cast<double>(n) / static_cast<double>(a.size());
}

// unchanged() without letterbox or pillarbox bars: whole cell rows (then columns) from each edge that are
// dark, flat and the same in both frames, taken only in pairs (the opposite edge's within a cell as wide): a
// dark sky along one edge is picture. Bars always stay still, and 2.39:1 bars alone are over a quarter
// of a 16:9 picture. cw: cells a row.
double unchanged_inside(const std::vector<int>& a, const std::vector<int>& b, int cw) {
  const int ch = static_cast<int>(a.size()) / cw;
  auto bar = [&](int i, int first) { return a[i] <= 40 && std::abs(a[i] - b[i]) <= 2 && std::abs(a[i] - a[first]) <= 2; };
  auto bar_row = [&](int r) {
    for (int c = 0; c < cw; c++)
      if (!bar(r * cw + c, r * cw)) return false;
    return true;
  };
  int r0 = 0, r1 = ch;
  while (r0 < r1 && bar_row(r0)) r0++;
  while (r1 > r0 && bar_row(r1 - 1)) r1--;
  if (r0 == 0 || r1 == ch || std::abs(r0 - (ch - r1)) > 1) r0 = 0, r1 = ch;
  auto bar_col = [&](int c) {
    for (int r = r0; r < r1; r++)
      if (!bar(r * cw + c, r0 * cw + c)) return false;
    return true;
  };
  int c0 = 0, c1 = cw;
  while (c0 < c1 && bar_col(c0)) c0++;
  while (c1 > c0 && bar_col(c1 - 1)) c1--;
  if (c0 == 0 || c1 == cw || std::abs(c0 - (cw - c1)) > 1) c0 = 0, c1 = cw;
  size_t n = 0;
  for (int r = r0; r < r1; r++)
    for (int c = c0; c < c1; c++) n += std::abs(a[r * cw + c] - b[r * cw + c]) <= 2;
  return static_cast<double>(n) / (static_cast<double>(r1 - r0) * (c1 - c0));
}


// Each 8x8 cell's high-frequency energy: the mean absolute deviation of its samples from its own
// average (a halftone dot field averages to flat colour in cells() but not here).
std::vector<double> cells_hf(const Frame420& f) {
  const int cw = std::max(1, f.width / 8), ch = std::max(1, f.height / 8);
  const int bw = std::max(1, f.width / cw), bh = std::max(1, f.height / ch);
  std::vector<double> out(static_cast<size_t>(cw) * ch);
  for (int cy = 0; cy < ch; cy++)
    for (int cx = 0; cx < cw; cx++) {
      int sum = 0;
      for (int y = 0; y < bh; y++)
        for (int x = 0; x < bw; x++) sum += f.y[static_cast<size_t>(cy * bh + y) * f.stride_y() + cx * bw + x];
      const double avg = static_cast<double>(sum) / (bw * bh);
      double dev = 0;
      for (int y = 0; y < bh; y++)
        for (int x = 0; x < bw; x++) dev += std::abs(f.y[static_cast<size_t>(cy * bh + y) * f.stride_y() + cx * bw + x] - avg);
      out[static_cast<size_t>(cy) * cw + cx] = dev / (bw * bh);
    }
  return out;
}

bool gop_log() {
  static const bool on = [] {
    const char* e = std::getenv("CELV2_GOP_LOG");
    return e && *e && *e != '0';
  }();
  return on;
}

}  // namespace

double still_share(const Frame420& a, const Frame420& b) { return unchanged(cells(a), cells(b)); }

double ChangeMeter::push(const Frame420& f) {
  const int cols = (f.width + 15) / 16, rows = (f.height + 15) / 16;
  std::vector<float> avg(static_cast<size_t>(cols) * rows), hf(avg.size());
  celv2::parallel(rows, [&](int r) {
    for (int c = 0; c < cols; c++) {
      const int x0 = c * 16, y0 = r * 16, x1 = std::min(x0 + 16, f.width), y1 = std::min(y0 + 16, f.height);
      int sum = 0;
      for (int y = y0; y < y1; y++)
        for (int x = x0; x < x1; x++) sum += f.y[static_cast<size_t>(y) * f.stride_y() + x];
      const double n = static_cast<double>(x1 - x0) * (y1 - y0), a = sum / n;
      double dev = 0;
      for (int y = y0; y < y1; y++)
        for (int x = x0; x < x1; x++) dev += std::abs(f.y[static_cast<size_t>(y) * f.stride_y() + x] - a);
      avg[static_cast<size_t>(r) * cols + c] = static_cast<float>(a), hf[static_cast<size_t>(r) * cols + c] = static_cast<float>(dev / n);
    }
  });
  double share = 0;
  if (avg_.size() == avg.size()) {
    size_t changed = 0;
    for (size_t i = 0; i < avg.size(); i++)
      changed += std::abs(avg[i] - avg_[i]) > 2 || std::abs(hf[i] - hf_[i]) > std::max(2.0f, 0.1f * hf_[i]);
    share = static_cast<double>(changed) / static_cast<double>(avg.size());
  }
  avg_ = std::move(avg), hf_ = std::move(hf);
  return share;
}

void SceneCutDetector::log(bool cut, const Frame420& cur) {
  std::vector<double> hf = cells_hf(cur);
  const bool have = prev_.size() == cur_.size() && prev_hf_.size() == hf.size();
  double h = 0;
  if (have) {
    for (size_t i = 0; i < hf.size(); i++) h += std::abs(hf[i] - prev_hf_[i]);
    h /= static_cast<double>(hf.size());
  }
  std::fprintf(stderr, "celv2enc: gop frame %d: mad %.2f unchanged %.3f hf %.2f cut %d\n", frame_, have ? mad(prev_, cur_) : 0.0,
               have ? unchanged_inside(prev_, cur_, cw_) : 0.0, h, cut ? 1 : 0);
  prev_hf_ = std::move(hf);
}

bool SceneCutDetector::push(const Frame420& cur, const Frame420* next) {
  prev2_ = std::move(prev_);
  prev_ = std::move(cur_);
  cur_ = cells(cur);
  cw_ = std::max(1, cur.width / 8);
  frame_++;
  const bool cut = test(next);
  if (gop_log()) log(cut, cur);
  return cut;
}

bool SceneCutDetector::test(const Frame420* next) {
  if (prev_.size() != cur_.size()) return false;  // first frame
  const double d = mad(prev_, cur_);
  std::vector<double> recent = diffs_;
  diffs_.push_back(d);
  if (diffs_.size() > 8) diffs_.erase(diffs_.begin());
  double base = 0;
  if (!recent.empty()) {
    std::nth_element(recent.begin(), recent.begin() + recent.size() / 2, recent.end());
    base = recent[recent.size() / 2];
  }
  if (d <= 20 || d <= 4 * (base + 1)) return false;
  if (prev2_.size() == cur_.size() && mad(prev2_, cur_) <= 20) return false;
  // A new drawing over a still part of the shot: a P frame. 0.25 (C3), not 0.5: S1 -1.18% BD-rate, S2 AIR +0.04%.
  if (unchanged_inside(prev_, cur_, cw_) > 0.25) return false;
  return !next || mad(cur_, cells(*next)) < d / 2;
}
