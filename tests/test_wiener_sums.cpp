#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "check.h"
#include "core/wiener_sums.h"

namespace {

uint32_t g_rng = 2024;
int rnd(int lo, int hi) {  // [lo, hi]
  g_rng = g_rng * 1664525u + 1013904223u;
  return lo + static_cast<int>((g_rng >> 8) % static_cast<uint32_t>(hi - lo + 1));
}

// Taps within two rows and columns, as the chroma filter's: 10 reconstruction pairs, 8 luma taps, every feature once.
celv2::WienerTaps taps(int st, int lw) {
  celv2::WienerTaps t;
  t.nx = 10, t.nl = 8;
  for (int k = 0; k < t.nx; k++) t.dx[k] = rnd(0, 2) * st + rnd(-2, 2), t.px[k] = k;
  for (int k = 0; k < t.nl; k++) t.dl[k] = rnd(-2, 2) * lw + rnd(-2, 2), t.pl[k] = t.nx + k;
  return t;
}

// pattern 0: random; 1: the extremes alternating (the largest products); 2: flat.
int same_as_ref(int w, int h, int depth, bool wide_src, int pattern, int bands) {
  const int st = w + 7, lw = w, sst = w + 3, peak = (1 << depth) - 1;
  std::vector<uint16_t> x(static_cast<size_t>(st) * h), s16(static_cast<size_t>(sst) * h);
  std::vector<uint8_t> s8(static_cast<size_t>(sst) * h);
  std::vector<int16_t> l(static_cast<size_t>(lw) * h);
  for (int r = 0; r < h; r++)
    for (int c = 0; c < st; c++) {
      const bool odd = (r + c) & 1;
      x[static_cast<size_t>(r) * st + c] =
          static_cast<uint16_t>(pattern == 0 ? rnd(0, peak) : pattern == 1 ? (odd ? peak : 0) : peak / 2);
      if (c < lw)
        l[static_cast<size_t>(r) * lw + c] =
            static_cast<int16_t>(pattern == 0 ? rnd(0, peak) : pattern == 1 ? (odd ? 0 : peak) : peak / 3);
      if (c < sst) {
        s16[static_cast<size_t>(r) * sst + c] = static_cast<uint16_t>(pattern == 0 ? rnd(0, peak) : pattern == 1 ? (odd ? 0 : peak) : 7);
        s8[static_cast<size_t>(r) * sst + c] = static_cast<uint8_t>(pattern == 0 ? rnd(0, 255) : pattern == 1 ? (odd ? 0 : 255) : 7);
      }
    }
  const celv2::WienerTaps t = taps(st, lw);
  int64_t got[celv2::kWienerSums][celv2::kWienerSums], want[celv2::kWienerSums][celv2::kWienerSums];
  const uint16_t* const src16 = wide_src ? s16.data() : nullptr;
  celv2::wiener_sums(x.data(), st, l.data(), lw, src16, s8.data(), sst, w, h, t, got, bands);
  celv2::wiener_sums_ref(x.data(), st, l.data(), lw, src16, s8.data(), sst, w, h, t, want);
  for (int i = 0; i < celv2::kWienerSums; i++)
    for (int j = i; j < celv2::kWienerSums; j++) {
      if (got[i][j] != want[i][j])
        std::fprintf(stderr, "w %d h %d depth %d pattern %d bands %d: sums[%d][%d] %lld, want %lld\n", w, h, depth, pattern,
                     bands, i, j, static_cast<long long>(got[i][j]), static_cast<long long>(want[i][j]));
      CHECK_EQ(got[i][j], want[i][j]);
    }
  // wiener_rect_sums() over rectangles that tile the same samples (as the chroma units' sums do) add up to them.
  std::vector<int64_t> sum(celv2::kWienerPairs, 0), part(celv2::kWienerPairs);
  for (int y0 = 2; y0 < h - 2; y0 += 41)
    for (int x0 = 2; x0 < w - 2; x0 += 64) {
      celv2::wiener_rect_sums(x.data(), st, l.data(), lw, src16, s8.data(), sst, x0, std::min(x0 + 64, w - 2), y0,
                              std::min(y0 + 41, h - 2), t, part.data());
      for (int q = 0; q < celv2::kWienerPairs; q++) sum[q] += part[q];
    }
  for (int i = 0, q = 0; i < celv2::kWienerSums; i++)
    for (int j = i; j < celv2::kWienerSums; j++, q++) CHECK_EQ(sum[q], want[i][j]);
  return 0;
}

int sums_match_ref() {
  const int sizes[][2] = {{5, 5}, {8, 8}, {37, 23}, {130, 67}, {147, 9}, {963, 541}};
  for (const auto& s : sizes)
    for (int depth : {8, 10, 12})
      for (int pattern = 0; pattern < 3; pattern++)
        for (int bands : {1, 3, 16}) {
          if (same_as_ref(s[0], s[1], depth, depth > 8, pattern, bands)) return 1;
          if (depth == 10 && same_as_ref(s[0], s[1], depth, false, pattern, bands)) return 1;
        }
  return 0;
}

// wiener_class_sums() against its reference: 16 pairs within 3 rows and columns (as the luma filter's), random classes
// mapped by a random sub-classifier, rectangles not a multiple of the chunk or of 4 wide.
int class_sums_match_ref() {
  for (int depth : {8, 10, 12})
    for (int pattern = 0; pattern < 3; pattern++)
      for (const int classes : {1, 3, 16}) {
        const int w = 300, h = 70, st = w + 16, sst = w + 5, cst = w / 4 + 3, peak = (1 << depth) - 1;
        std::vector<uint16_t> x(static_cast<size_t>(st) * (h + 6)), s16(static_cast<size_t>(sst) * h);
        std::vector<uint8_t> s8(static_cast<size_t>(sst) * h), cls(static_cast<size_t>(cst) * (h / 4 + 1)), sub(256);
        for (size_t i = 0; i < x.size(); i++) x[i] = static_cast<uint16_t>(pattern == 0 ? rnd(0, peak) : pattern == 1 ? (i & 1 ? peak : 0) : peak / 2);
        for (size_t i = 0; i < s16.size(); i++) s16[i] = static_cast<uint16_t>(pattern == 1 ? (i & 1 ? 0 : peak) : rnd(0, peak));
        for (size_t i = 0; i < s8.size(); i++) s8[i] = static_cast<uint8_t>(rnd(0, 255));
        for (auto& c : cls) c = static_cast<uint8_t>(rnd(0, 255));
        for (auto& c : sub) c = static_cast<uint8_t>(rnd(0, classes - 1));
        celv2::WienerTaps t;
        t.nx = 16;
        for (int k = 0; k < t.nx; k++) t.dx[k] = rnd(0, 3) * st + rnd(-3, 3);
        const uint16_t* const o = x.data() + 3 * st + 4;  // 3 rows above and 4 columns left to read
        const int pairs = 17 * 18 / 2;
        const int ystep = 1 + pattern % 2;
        for (const auto& r : {std::array<int, 4>{0, w - 8, 0, h}, std::array<int, 4>{4, 133, 5, 9}, std::array<int, 4>{8, 9, 0, 1}}) {
          std::vector<int64_t> got(static_cast<size_t>(classes) * pairs), want(got.size());
          const uint16_t* const src16 = depth > 8 ? s16.data() : nullptr;
          celv2::wiener_class_sums(o, st, src16, s8.data(), sst, r[0], r[1], r[2], r[3], ystep, t, cls.data(), cst, sub.data(), classes, got.data());
          celv2::wiener_class_sums_ref(o, st, src16, s8.data(), sst, r[0], r[1], r[2], r[3], ystep, t, cls.data(), cst, sub.data(), classes, want.data());
          for (size_t i = 0; i < got.size(); i++) {
            if (got[i] != want[i])
              std::fprintf(stderr, "depth %d pattern %d classes %d rect %d-%d x %d-%d: sum %zu %lld, want %lld\n", depth, pattern, classes,
                           r[0], r[1], r[2], r[3], i, static_cast<long long>(got[i]), static_cast<long long>(want[i]));
            CHECK_EQ(got[i], want[i]);
          }
        }
      }
  return 0;
}

}  // namespace

#define RUN(fn) run_test(#fn, fn)

int main() {
  int fails = 0;
  fails += RUN(sums_match_ref);
  fails += RUN(class_sums_match_ref);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
