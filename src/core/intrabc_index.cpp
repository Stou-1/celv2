#include "core/intrabc_index.h"

#include "core/threads.h"

namespace celv2 {
namespace {

constexpr uint64_t kA = 0x9E3779B97F4A7C15ull, kB = 0xC2B2AE3D27D4EB4Full;

uint64_t sample(const Frame420& f, int y, int x) {
  const size_t at = static_cast<size_t>(y) * f.stride_y() + x;
  return f.bit_depth > 8 ? f.hy[at] : f.y[at];
}

}  // namespace

uint64_t IntrabcIndex::hash_at(const Frame420& src, int x, int y, bool* flat) {
  const uint64_t first = sample(src, y, x);
  uint64_t key = 0;
  for (int r = 0; r < 8; r++) {
    uint64_t v = 0;
    for (int c = 0; c < 8; c++) {
      const uint64_t s = sample(src, y + r, x + c);
      *flat = *flat && s == first;
      v = v * kA + s;
    }
    key = key * kB + v;
  }
  return key;
}

bool IntrabcIndex::same(const Frame420& src, int x, int y, int px, int py, int w, int h) {
  for (int r = 0; r < h; r++)
    for (int c = 0; c < w; c++)
      if (sample(src, py + r, px + c) != sample(src, y + r, x + c)) return false;
  return true;
}

// Rolling: each row's 8 samples at every column, then 8 of those rows, as hash_at() sums them.
IntrabcIndex::IntrabcIndex(const Frame420& src, int band_rows) : band_rows_(band_rows) {
  const int w = src.width, h = src.height, cols = w - 7;
  bands_.resize((h + band_rows - 1) / band_rows);
  if (cols <= 0) return;
  uint64_t a7 = 1, b7 = 1;
  for (int i = 0; i < 7; i++) a7 *= kA, b7 *= kB;
  ThreadPool::get().run(static_cast<int>(bands_.size()), [&](int b) {
    const int y0 = b * band_rows, rows = std::min(band_rows, h - y0);
    if (rows < 8) return;
    std::vector<uint64_t> rh(static_cast<size_t>(rows) * cols);
    for (int r = 0; r < rows; r++) {
      uint64_t v = 0;
      for (int c = 0; c < 8; c++) v = v * kA + sample(src, y0 + r, c);
      rh[static_cast<size_t>(r) * cols] = v;
      for (int c = 1; c < cols; c++) {
        v = (v - sample(src, y0 + r, c - 1) * a7) * kA + sample(src, y0 + r, c + 7);
        rh[static_cast<size_t>(r) * cols + c] = v;
      }
    }
    std::vector<Entry>& band = bands_[b];
    band.reserve(static_cast<size_t>(rows - 7) * cols);
    for (int c = 0; c < cols; c++) {
      uint64_t v = 0;
      for (int r = 0; r < 8; r++) v = v * kB + rh[static_cast<size_t>(r) * cols + c];
      for (int r = 0;; r++) {
        band.push_back({v, position(c, y0 + r)});
        if (r + 8 >= rows) break;
        v = (v - rh[static_cast<size_t>(r) * cols + c] * b7) * kB + rh[static_cast<size_t>(r + 8) * cols + c];
      }
    }
    std::sort(band.begin(), band.end());
  });
}

}  // namespace celv2
