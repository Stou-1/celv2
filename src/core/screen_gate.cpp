#include "core/screen_gate.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace celv2 {

ScreenShares screen_shares(const Frame420& f) {
  ScreenShares s;
  const int stride = f.stride_y();
  int blocks = 0, few = 0;
  for (int y = 0; y + 16 <= f.height; y += 16)
    for (int x = 0; x + 16 <= f.width; x += 16) {
      bool seen[256] = {};
      int n = 0, lo = 255, hi = 0;
      for (int r = 0; r < 16 && n <= 8; r++)
        for (int c = 0; c < 16; c++) {
          const uint8_t v = f.y[static_cast<size_t>(y + r) * stride + x + c];
          if (!seen[v]) seen[v] = true, n++;
          lo = std::min<int>(lo, v), hi = std::max<int>(hi, v);
        }
      blocks++;
      few += n >= 2 && n <= 8 && hi - lo >= kPaletteSpan;
    }
  if (blocks) s.palette = static_cast<double>(few) / blocks;

  // 8x8 hashes (FNV-1a over the 64 samples), each with its count; a hash collision only counts a
  // block as repeated.
  std::unordered_map<uint64_t, int> count;
  std::vector<uint64_t> hashes;
  for (int y = 0; y + 8 <= f.height; y += 8)
    for (int x = 0; x + 8 <= f.width; x += 8) {
      uint64_t h = 1469598103934665603ull;
      bool flat = true;
      const uint8_t first = f.y[static_cast<size_t>(y) * stride + x];
      for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++) {
          const uint8_t v = f.y[static_cast<size_t>(y + r) * stride + x + c];
          flat = flat && v == first;
          h = (h ^ v) * 1099511628211ull;
        }
      if (flat) {
        hashes.push_back(0);  // left out, but counted in the share's base
        continue;
      }
      h |= 1;  // never 0
      hashes.push_back(h);
      count[h]++;
    }
  int repeated = 0;
  for (uint64_t h : hashes) repeated += h && count[h] > 1;
  if (!hashes.empty()) s.repeat = static_cast<double>(repeated) / hashes.size();
  return s;
}

}  // namespace celv2
