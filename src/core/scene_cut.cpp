#include "core/scene_cut.h"

#include <algorithm>
#include <cstdlib>
#include <thread>

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

}  // namespace

double still_share(const Frame420& a, const Frame420& b) { return unchanged(cells(a), cells(b)); }

bool SceneCutDetector::push(const Frame420& cur, const Frame420* next) {
  prev2_ = std::move(prev_);
  prev_ = std::move(cur_);
  cur_ = cells(cur);
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
  if (unchanged(prev_, cur_) > 0.5) return false;  // a new drawing over a still part of the shot
  return !next || mad(cur_, cells(*next)) < d / 2;
}
