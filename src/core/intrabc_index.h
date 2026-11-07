// --screen IntraBC: where a frame's source luma repeats itself. Per band of superblock rows (IntraBC's
// local window never leaves one), the hash of the 8x8 at every position, sorted; a block's exact
// repeats are found from its own top-left 8x8. Built once a frame; the RD pass and the coder share it.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "core/perceptual_aq.h"
#include "gpu/inter_common.h"
#include "io/y4m.h"

namespace celv2 {

class IntrabcIndex {
 public:
  IntrabcIndex(const Frame420& src, int band_rows);
  int band_rows() const { return band_rows_; }
  // Stage B1: the --paq map's period vectors, for near_copies().
  void set_periods(std::shared_ptr<const AqMap> m) { periods_ = std::move(m); }
  // Copies that need not match: the positions at whole multiples of the period vector under the
  // block's centre, where its periodicity is kPeriodCopy or more, that ok(px, py) lets through
  // (kCopyTries at most, nearest first); the n best by SATD on the source into out, returning how many.
  template <class F>
  int near_copies(const Frame420& src, int x, int y, int w, int h, F ok, std::pair<int, int>* out, int n) const {
    if (!periods_) return 0;
    const AqMap& m = *periods_;
    const int c = (x + w / 2) / 16, r = (y + h / 2) / 16;
    if (c >= m.cols || r >= m.rows) return 0;
    const size_t i = static_cast<size_t>(r) * m.cols + c;
    const int dx = m.period_dx[i], dy = m.period_dy[i];
    if (m.periodicity[i] < kPeriodCopy || (!dx && !dy)) return 0;
    const celv2::Plane s{src.y.data(), src.stride_y(), src.pw, src.ph};
    std::vector<std::pair<int64_t, std::pair<int, int>>> found;
    for (int k = 1; static_cast<int>(found.size()) < kCopyTries && std::max(std::abs(k * dx), std::abs(k * dy)) <= kCopyReach; k++)
      for (int sign : {-1, 1}) {
        const int px = x + sign * k * dx, py = y + sign * k * dy;
        if (px >= 0 && py >= 0 && px + w <= src.width && py + h <= src.height && ok(px, py))
          found.push_back({celv2::block_inter_satd(s, s, x, y, w, h, (px - x) * 4, (py - y) * 4), {px, py}});
      }
    std::sort(found.begin(), found.end());
    const int got = std::min<int>(n, static_cast<int>(found.size()));
    for (int k = 0; k < got; k++) out[k] = found[k].second;
    return got;
  }
  // take(px, py) for each exact repeat of the w x h block at (x, y) in its band (above_too: and the
  // bands above), in columns [x_lo, x_hi), backwards in raster order from the block, until it returns
  // false; at most kTries hash matches are compared. None where the top-left 8x8 is flat (intra
  // codes that as well).
  template <class F>
  void repeats(const Frame420& src, int x, int y, int w, int h, int x_lo, int x_hi, F take, bool above_too = false) const {
    if (x + w > src.width || y + h > src.height) return;
    bool flat = true;
    const uint64_t key = hash_at(src, x, y, &flat);
    if (flat) return;
    int tries = 0;
    for (int b = y / band_rows_, last = above_too ? 0 : b; b >= last; b--) {
      const std::vector<Entry>& band = bands_[b];
      auto it = std::lower_bound(band.begin(), band.end(), Entry{key, position(x, y)});
      while (it != band.begin() && tries < kTries) {
        --it;
        if (it->first != key) break;
        const int px = static_cast<int>(it->second & 0xFFFF), py = static_cast<int>(it->second >> 16);
        if (px < x_lo || px >= x_hi || px + w > src.width || py + h > src.height) continue;
        tries++;
        if (same(src, x, y, px, py, w, h) && !take(px, py)) return;
      }
    }
  }

 private:
  static constexpr float kPeriodCopy = 0.5f;
  static constexpr int kCopyTries = 20, kCopyReach = 128;
  std::shared_ptr<const AqMap> periods_;
  using Entry = std::pair<uint64_t, uint32_t>;  // (hash, y << 16 | x)
  static constexpr int kTries = 64;
  static uint32_t position(int x, int y) { return static_cast<uint32_t>(y) << 16 | static_cast<uint32_t>(x); }
  static uint64_t hash_at(const Frame420& src, int x, int y, bool* flat);
  static bool same(const Frame420& src, int x, int y, int px, int py, int w, int h);
  int band_rows_;
  std::vector<std::vector<Entry>> bands_;
};

}  // namespace celv2
