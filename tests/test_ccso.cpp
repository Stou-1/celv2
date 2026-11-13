#include <algorithm>
#include <cstdio>
#include <vector>

#include "check.h"
#include "core/ccso_apply.h"
#include "core/ccso_search.h"

// A plane w x h whose reconstruction is the source less `bias` where the luma is bright and
// unchanged where it is dark; the luma itself is padded as extend_ccso_border() pads it.
struct Case {
  int w = 64, h = 48, pad = CCSO_PADDING_SIZE;
  std::vector<uint16_t> luma, rec;
  std::vector<uint8_t> src;
  explicit Case(int bias) {
    const int stride = w + 2 * pad;
    luma.resize(static_cast<size_t>(stride) * (h + 2 * pad));
    rec.resize(static_cast<size_t>(w) * h);
    src.resize(static_cast<size_t>(w) * h);
    for (int y = -pad; y < h + pad; y++)
      for (int x = -pad; x < w + pad; x++) {
        const int cy = y < 0 ? 0 : y >= h ? h - 1 : y, cx = x < 0 ? 0 : x >= w ? w - 1 : x;
        luma[static_cast<size_t>(y + pad) * stride + x + pad] = static_cast<uint16_t>(cx < w / 2 ? 40 : 200 + (cy & 1));
      }
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++) {
        const int v = luma[static_cast<size_t>(y + pad) * stride + x + pad];
        src[static_cast<size_t>(y) * w + x] = static_cast<uint8_t>(v);
        rec[static_cast<size_t>(y) * w + x] = static_cast<uint16_t>(v >= 128 ? v - bias : v);
      }
  }
  CcsoChoice search() const {
    const int stride = w + 2 * pad;
    return ccso_search(luma.data() + pad * stride + pad, stride, rec.data(), w, src.data(), w, w, h, 0, 0, 1.0, 2);
  }
};

// The bright half's error is 3 levels: an offset of +3 there and none in the dark half.
static int band_bias_is_offset() {
  const CcsoChoice c = Case(3).search();
  CHECK(c.gain > 0);
  CHECK(c.max_band_log2 >= 1);
  const int top = (1 << c.max_band_log2) - 1;  // the band of luma 200
  int bright = 0, dark = 0;
  for (int k = 0; k < 16; k++) bright += c.offset[(top << 4) + k], dark += c.offset[k];
  CHECK(bright > 0);
  CHECK_EQ(dark, 0);
  return 0;
}

// No error: all offsets 0 and nothing gained.
static int exact_reconstruction_keeps_zero() {
  const CcsoChoice c = Case(0).search();
  CHECK_EQ(c.gain, 0.0);
  for (int k = 0; k < CCSO_BAND_NUM * 16; k++) CHECK_EQ(static_cast<int>(c.offset[k]), 0);
  return 0;
}

constexpr int kOffset[8] = {0, 1, -1, 3, -3, 7, -7, -10};  // setup_ccso()'s table

// One class's CcsoDelta of n samples summing sum = source - reconstruction, none clipped, at scale 1.
static CcsoDelta delta(int64_t sum, int64_t n) {
  CcsoDelta d;
  for (int i = 0; i < 8; i++) d.d[i] = n * kOffset[i] * kOffset[i] - 2 * kOffset[i] * sum;
  return d;
}

// The flags per unit on synthetic stats: one band, offset +3, four groups (2 x 2 units). The offset helps two (their
// samples 3 under the source) and hurts two (exact): mixed flags, the offset kept. Every group helped:
// all on. None helped: the plane is better off (J not under 0).
static int unit_flags() {
  CcsoChoice c;
  const int groups[4] = {0, 1, 2, 3};
  c.offset[0] = 3;
  std::vector<CcsoDelta> stats = {delta(3000, 1000), delta(3000, 1000), delta(0, 1000), delta(0, 1000)};
  std::vector<uint8_t> on;
  CHECK(ccso_unit_flags(stats.data(), 4, 1, 1.0, &c, &on, groups, 2, 2) < 0);
  CHECK(on == (std::vector<uint8_t>{1, 1, 0, 0}));
  CHECK_EQ(static_cast<int>(c.offset[0]), 3);
  stats.assign(4, delta(3000, 1000));
  CHECK(ccso_unit_flags(stats.data(), 4, 1, 1.0, &c, &on, groups, 2, 2) < 0);
  CHECK(on == (std::vector<uint8_t>{1, 1, 1, 1}));
  CHECK_EQ(static_cast<int>(c.offset[0]), 3);
  stats.assign(4, delta(0, 1000));
  CHECK(ccso_unit_flags(stats.data(), 4, 1, 1.0, &c, &on, groups, 2, 2) >= 0);
  return 0;
}

// The per-unit statistics predict the filter exactly: on a random 72x40 plane of 32x16 units (edge
// classes, 2 bands, chroma subsampled from a 144x80 luma, offsets at scale 2), each unit's error change
// at the choice's offsets, from its stats, equals the change ccso_apply_row() makes with that unit alone
// on. A third of the samples sit near 0 or 255, where the filter's clip cuts the offsets.
static int unit_stats_match_filter() {
  const int w = 72, h = 40, pad = CCSO_PADDING_SIZE, lw = 144, lh = 80, stride = lw + 2 * pad, scale = 2;
  std::vector<uint16_t> luma(static_cast<size_t>(stride) * (lh + 2 * pad)), rec(static_cast<size_t>(w) * h);
  std::vector<uint8_t> src(static_cast<size_t>(w) * h);
  uint32_t rng = 99;
  auto rnd = [&](int n) { return static_cast<int>((rng = rng * 1664525u + 1013904223u) >> 8) % n; };
  for (auto& v : luma) v = static_cast<uint16_t>(64 + rnd(128));
  for (size_t i = 0; i < rec.size(); i++) {
    const int k = rnd(6);
    rec[i] = static_cast<uint16_t>(k == 0 ? rnd(16) : k == 1 ? 240 + rnd(16) : 64 + rnd(128));
    src[i] = static_cast<uint8_t>(std::clamp(rec[i] + rnd(9) - 4, 0, 255));
  }
  celv2::CcsoPlane p;
  p.luma = luma.data() + pad * stride + pad, p.luma_stride = stride;
  p.dst = rec.data(), p.dst_stride = w, p.w = w, p.h = h, p.sx = 1, p.sy = 1, p.bit_depth = 8;
  p.bo_only = false, p.thr = 16, p.edge_clf = 0, p.max_band_log2 = 1;
  celv2::ccso_neighbours(3, stride, p.loc);
  int8_t offset[CCSO_BAND_NUM * 16] = {};
  int oidx[18];
  for (int k = 0; k < 18; k++) oidx[k] = rnd(8), offset[((k / 9) << 4) + ((k / 3 % 3) << 2) + k % 3] = static_cast<int8_t>(kOffset[oidx[k]] * scale);
  p.offset = offset;
  std::vector<CcsoDelta> stats;
  const int cols = 3, classes = ccso_unit_stats(p, src.data(), w, w, h, scale, 5, 4, cols, &stats);
  CHECK_EQ(classes, 18);
  auto sse = [&] {
    int64_t e = 0;
    for (size_t i = 0; i < rec.size(); i++) e += (src[i] - rec[i]) * (src[i] - rec[i]);
    return e;
  };
  const int64_t before = sse();
  for (int u = 0; u < 3 * cols; u++) {
    int64_t predicted = 0;
    for (int k = 0; k < classes; k++) predicted += stats[u * classes + k].d[oidx[k]];
    // Filter that unit alone: its 32x32 blocks (one a row of 16 samples here) in a copy.
    std::vector<uint16_t> keep = rec;
    std::vector<uint8_t> on(static_cast<size_t>(cols) * 2, 0);
    for (int bx = 0; bx < cols; bx++) on[static_cast<size_t>(u / cols >> 1) * cols + bx] = bx == u % cols;  // block row u / cols / 2
    p.on = on.data(), p.on_stride = cols;
    // The unit is 16 rows, a 32-row block holds two: filter the block row, then undo the other unit's rows.
    celv2::ccso_apply_row(p, u / cols >> 1);
    for (int y = (u / cols >> 1) << 5; y < std::min(h, ((u / cols >> 1) + 1) << 5); y++)
      if (y >> 4 != u / cols)
        for (int x = 0; x < w; x++) rec[static_cast<size_t>(y) * w + x] = keep[static_cast<size_t>(y) * w + x];
    CHECK_EQ(sse() - before, predicted);
    rec = keep;
  }
  return 0;
}

// The counting loop's bins16 against bin_of: every difference within 10 bits in all 16 lanes, then every pair of
// 10-bit samples (8-bit ones among them) through the subtraction the loop does, 16 neighbours a vector.
static int bins16_match_bin_of() {
  alignas(32) int16_t o[16];
  for (int d = -1023; d <= 1023; d++) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(o), ccso_bins16(_mm256_set1_epi16(static_cast<int16_t>(d))));
    for (int i = 0; i < 16; i++) CHECK_EQ(o[i], ccso_bin_of(d) + 16);
  }
  alignas(32) uint16_t n[16];
  for (int v = 0; v < 1024; v++)
    for (int n0 = 0; n0 < 1024; n0 += 16) {
      for (int i = 0; i < 16; i++) n[i] = static_cast<uint16_t>(n0 + i);
      const __m256i d = _mm256_sub_epi16(_mm256_load_si256(reinterpret_cast<const __m256i*>(n)), _mm256_set1_epi16(static_cast<int16_t>(v)));
      _mm256_store_si256(reinterpret_cast<__m256i*>(o), ccso_bins16(d));
      for (int i = 0; i < 16; i++) CHECK_EQ(o[i], ccso_bin_of(n[i] - v) + 16);
    }
  return 0;
}

int main() {
  int fails = 0;
  fails += RUN(band_bias_is_offset);
  fails += RUN(exact_reconstruction_keeps_zero);
  fails += RUN(unit_flags);
  fails += RUN(unit_stats_match_filter);
  fails += RUN(bins16_match_bin_of);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
