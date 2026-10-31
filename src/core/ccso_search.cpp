#include "core/ccso_search.h"

#include <algorithm>
#include <array>
#include <vector>

#include "core/threads.h"

extern "C" {
#include "av2/common/ccso.h"
}

namespace {

// Every edge threshold quant_sz holds, ascending. A luma difference d goes to bin +k (d > 0: d is
// over k of them) or -k (d < 0: -d over k), 0 for d = 0; then for threshold kThr[j], d > it is bin
// > j and d < -it bin < -j, so one count per bin serves every threshold.
constexpr int kThr[16] = {0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 256};
constexpr int kBins = 33;  // -16..16
constexpr int kSupports = 7;  // ext_filter_support 0..6
constexpr int kOffset[8] = {0, 1, -1, 3, -3, 7, -7, -10};  // setup_ccso()'s table, times scale_idx + 1

struct Acc {
  int64_t sum = 0, n = 0;  // of source - reconstruction
  Acc& operator+=(const Acc& o) { return sum += o.sum, n += o.n, *this; }
};

constexpr size_t kEdgeCells = static_cast<size_t>(kSupports) * 8 * kBins * kBins;  // [f][band8][b0][b1]

struct Stats {
  std::vector<Acc> edge = std::vector<Acc>(kEdgeCells);
  std::array<Acc, 256> level{};  // by luma value (band offsets only)
  Acc& at(int f, int band, int b0, int b1) { return edge[((static_cast<size_t>(f) * 8 + band) * kBins + b0) * kBins + b1]; }
};

// One counting task's: 32-bit, so half the memory to clear and merge (a task counts at most 1M
// samples of differences up to 1023, under 2^31).
struct Acc32 {
  int32_t sum = 0, n = 0;
};
struct Stats32 {
  std::vector<Acc32> edge;  // sized by the task that fills it (cleared on its thread)
  std::array<Acc32, 256> level{};
};

int bin_of(int d) {
  int k = 0;
  if (d > 0)
    while (k < 16 && d > kThr[k]) k++;
  else if (d < 0)
    while (k < 16 && -d > kThr[k]) k++;
  return d < 0 ? -k : k;
}

// The offset index of one class (n samples summing s = source - reconstruction) and its cost:
// the squared error it adds (n o^2 - 2 o s) plus its truncated unary bits.
double best_offset(const Acc& a, int scale, double sse_per_bit, int* idx) {
  double best = 1e300;
  for (int i = 0; i < 8; i++) {
    const double o = kOffset[i] * scale;
    const double j = static_cast<double>(a.n) * o * o - 2 * o * static_cast<double>(a.sum) + sse_per_bit * (i < 7 ? i + 1 : 7);
    if (j < best) best = j, *idx = i;
  }
  return best;
}

template <typename S>
CcsoChoice search(const uint16_t* luma, int luma_stride, const uint16_t* rec, int rec_stride, const S* src,
                  int src_stride, int w, int h, int ss_x, int ss_y, double sse_per_bit, int threads, int bit_depth) {
  // Bins of luma differences up to 10 bits (8-bit ones read the same bins as before).
  static const auto bins = [] {
    std::array<uint8_t, 2047> t{};
    for (int d = -1023; d <= 1023; d++) t[d + 1023] = static_cast<uint8_t>(bin_of(d) + 16);
    return t;
  }();
  const int shift = bit_depth - 8;  // luma levels and bands on the 8-bit scale
  int pos[kSupports][2];
  for (int f = 0; f < kSupports; f++) derive_ccso_sample_pos(pos[f], luma_stride, static_cast<uint8_t>(f));
  // Tasks of 64K to 1M samples, rows each; every task counts into its own Stats32, then the
  // tables are summed in slices on threads (merging 1 MB tables one after another capped 1080p
  // chroma at one thread).
  const int64_t samples = static_cast<int64_t>(w) * h;
  const int64_t per = std::clamp<int64_t>(samples / std::max(1, threads), 1 << 16, 1 << 20);
  const int nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(h, (samples + per - 1) / per)));
  std::vector<Stats32> st(nt);
  auto rows = [&](int k) {
    Stats32& s = st[k];
    s.edge.assign(kEdgeCells, Acc32{});
    for (int y = h * k / nt; y < h * (k + 1) / nt; y++) {
      const uint16_t* const l = luma + static_cast<size_t>(y << ss_y) * luma_stride;
      for (int x = 0; x < w; x++) {
        const uint16_t* const c = l + (x << ss_x);
        const int v = *c, level = v >> shift;
        const int diff = src[static_cast<size_t>(y) * src_stride + x] - rec[static_cast<size_t>(y) * rec_stride + x];
        s.level[level].sum += diff, s.level[level].n++;
        for (int f = 0; f < kSupports; f++) {
          Acc32& a = s.edge[((static_cast<size_t>(f) * 8 + (level >> 5)) * kBins + bins[c[pos[f][0]] - v + 1023]) * kBins +
                            bins[c[pos[f][1]] - v + 1023]];
          a.sum += diff, a.n++;
        }
      }
    }
  };
  celv2::parallel(nt, rows);
  Stats s;
  const int slices = std::max(1, std::min(threads, 64));
  celv2::parallel(slices, [&](int k) {
    for (size_t i = kEdgeCells * k / slices; i < kEdgeCells * (k + 1) / slices; i++)
      for (const Stats32& t : st) s.edge[i].sum += t.edge[i].sum, s.edge[i].n += t.edge[i].n;
  });
  for (const Stats32& t : st)
    for (int i = 0; i < 256; i++) s.level[i].sum += t.level[i].sum, s.level[i].n += t.level[i].n;

  CcsoChoice best;
  double best_j = 1e300;
  int idx[CCSO_BAND_NUM * 16];
  // Band offsets only: up to 64 bands of the luma value.
  for (int sc = 0; sc < 4; sc++)
    for (int b = 0; b <= 6; b++) {
      double j = sse_per_bit * (1 + 2 + 3);
      for (int band = 0; band < 1 << b; band++) {
        Acc a;
        for (int v = band << (8 - b); v < (band + 1) << (8 - b); v++) a += s.level[v];
        j += best_offset(a, sc + 1, sse_per_bit, &idx[band << 4]);
      }
      if (j < best_j) {
        best_j = j, best = CcsoChoice{};
        best.scale_idx = sc, best.max_band_log2 = b;
        for (int band = 0; band < 1 << b; band++) best.offset[band << 4] = static_cast<int8_t>(kOffset[idx[band << 4]] * (sc + 1));
      }
    }
  const double zero_j = sse_per_bit * (1 + 2 + 3 + 1);  // one band, offset 0: what the gain is against
  // Edge offsets: every support, threshold (scale_idx, quant_idx) and classifier, up to 8 bands.
  // A class is a rectangle of bins (each axis: below -jt, the middle, above jt; edge_clf 1 merges
  // the last two), so its sum comes from a summed-area table of each support and band: 4 lookups
  // instead of 33 x 33 adds for each of the 224 thresholds (that was 27 ms of a 1080p frame).
  std::vector<Acc> sat(static_cast<size_t>(kSupports) * 8 * (kBins + 1) * (kBins + 1));
  auto sat_at = [&](int f, int band, int i, int j) -> Acc& {
    return sat[((static_cast<size_t>(f) * 8 + band) * (kBins + 1) + i) * (kBins + 1) + j];
  };
  for (int f = 0; f < kSupports; f++)
    for (int band = 0; band < 8; band++)
      for (int i = 1; i <= kBins; i++)
        for (int j = 1; j <= kBins; j++) {
          Acc a = s.at(f, band, i - 1, j - 1);
          a += sat_at(f, band, i - 1, j), a += sat_at(f, band, i, j - 1);
          const Acc& d = sat_at(f, band, i - 1, j - 1);
          a.sum -= d.sum, a.n -= d.n;
          sat_at(f, band, i, j) = a;
        }
  Acc cls[8][3][3];
  for (int f = 0; f < kSupports; f++)
    for (int sc = 0; sc < 4; sc++)
      for (int q = 0; q < 4; q++) {
        const int thr = quant_sz[sc][q];
        const int jt = static_cast<int>(std::find(kThr, kThr + 16, thr) - kThr);
        for (int e = 0; e < (thr ? 2 : 1); e++) {
          // Bins [lo[c], hi[c]) of class c on one axis (bin b is difference b - 16).
          const int lo[3] = {0, 16 - jt, e ? kBins : 17 + jt}, hi[3] = {16 - jt, e ? kBins : 17 + jt, kBins};
          for (int band = 0; band < 8; band++)
            for (int c0 = 0; c0 < 3; c0++)
              for (int c1 = 0; c1 < 3; c1++) {
                Acc a = sat_at(f, band, hi[c0], hi[c1]);
                a += sat_at(f, band, lo[c0], lo[c1]);
                const Acc &x = sat_at(f, band, lo[c0], hi[c1]), &y = sat_at(f, band, hi[c0], lo[c1]);
                a.sum -= x.sum + y.sum, a.n -= x.n + y.n;
                cls[band][c0][c1] = a;
              }
          const int nd = e ? 2 : 3;
          for (int b = 0; b <= 3; b++) {
            double j = sse_per_bit * (1 + 2 + 2 + 3 + (thr ? 1 : 0) + 2);
            for (int band = 0; band < 1 << b; band++)
              for (int d0 = 0; d0 < nd; d0++)
                for (int d1 = 0; d1 < nd; d1++) {
                  Acc a;
                  for (int k = band << (3 - b); k < (band + 1) << (3 - b); k++) a += cls[k][d0][d1];
                  j += best_offset(a, sc + 1, sse_per_bit, &idx[(band << 4) + (d0 << 2) + d1]);
                }
            if (j < best_j) {
              best_j = j, best = CcsoChoice{};
              best.bo_only = false;
              best.scale_idx = sc, best.quant_idx = q, best.ext_filter_support = f, best.edge_clf = e, best.max_band_log2 = b;
              for (int band = 0; band < 1 << b; band++)
                for (int d0 = 0; d0 < nd; d0++)
                  for (int d1 = 0; d1 < nd; d1++) {
                    const int k = (band << 4) + (d0 << 2) + d1;
                    best.offset[k] = static_cast<int8_t>(kOffset[idx[k]] * (sc + 1));
                  }
            }
          }
        }
      }
  best.gain = zero_j - best_j;
  return best;
}

}  // namespace

CcsoChoice ccso_search(const uint16_t* luma, int luma_stride, const uint16_t* rec, int rec_stride, const uint8_t* src,
                       int src_stride, int w, int h, int ss_x, int ss_y, double sse_per_bit, int threads) {
  return search(luma, luma_stride, rec, rec_stride, src, src_stride, w, h, ss_x, ss_y, sse_per_bit, threads, 8);
}

CcsoChoice ccso_search(const uint16_t* luma, int luma_stride, const uint16_t* rec, int rec_stride, const uint16_t* src,
                       int src_stride, int w, int h, int ss_x, int ss_y, double sse_per_bit, int threads, int bit_depth) {
  return search(luma, luma_stride, rec, rec_stride, src, src_stride, w, h, ss_x, ss_y, sse_per_bit, threads, bit_depth);
}
