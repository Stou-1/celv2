#include "core/ccso_search.h"

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "core/ccso_apply.h"
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
  // n << 32 plus sum (|sum| < 2^31): one 64-bit add a sample and support, not two 32-bit ones.
  std::vector<int64_t> edge;  // sized by the task that fills it (cleared on its thread)
  std::array<Acc32, 256> level{};
};
// The tasks' tables, kept between frames and searches (a fresh 0.5 MB table a task paid its pages on every frame).
std::mutex g_tables_mu;
std::vector<std::vector<int64_t>> g_tables;
std::vector<int64_t> take_table() {
  std::vector<int64_t> t;
  {
    std::lock_guard<std::mutex> lock(g_tables_mu);
    if (!g_tables.empty()) t = std::move(g_tables.back()), g_tables.pop_back();
  }
  t.assign(kEdgeCells, 0);
  return t;
}
void give_table(std::vector<int64_t>&& t) {
  std::lock_guard<std::mutex> lock(g_tables_mu);
  g_tables.push_back(std::move(t));
}
Acc unpack(int64_t v) {
  const int64_t sum = static_cast<int32_t>(static_cast<uint32_t>(v));
  return {sum, (v - sum) >> 32};
}

int bin_of(int d) {
  int k = 0;
  if (d > 0)
    while (k < 16 && d > kThr[k]) k++;
  else if (d < 0)
    while (k < 16 && -d > kThr[k]) k++;
  return d < 0 ? -k : k;
}

// 16 samples from p, every one (ss 0) or every other (ss 1).
__m256i load16(const uint16_t* p, int ss) {
  const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
  if (!ss) return a;
  const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 16)), lo = _mm256_set1_epi32(0xffff);
  return _mm256_permute4x64_epi64(_mm256_packus_epi32(_mm256_and_si256(a, lo), _mm256_and_si256(b, lo)), 0xd8);
}

// bin_of(d) + 16 on 16 lanes, |d| < 2^15: the thresholds below |d| (kThr: 0..64 by 8, 80..128 by 16, 160, 192, 256).
__m256i bins16(__m256i d) {
  const __m256i a = _mm256_abs_epi16(d);
  const __m256i k8 = _mm256_min_epi16(_mm256_srli_epi16(_mm256_adds_epu16(a, _mm256_set1_epi16(7)), 3), _mm256_set1_epi16(9));
  const __m256i k16 = _mm256_min_epi16(
      _mm256_max_epi16(_mm256_sub_epi16(_mm256_srai_epi16(_mm256_add_epi16(a, _mm256_set1_epi16(15 - 64)), 4), _mm256_set1_epi16(1)),
                       _mm256_setzero_si256()),
      _mm256_set1_epi16(4));
  __m256i k = _mm256_add_epi16(k8, k16);
  for (const int t : {160, 192, 256}) k = _mm256_sub_epi16(k, _mm256_cmpgt_epi16(a, _mm256_set1_epi16(static_cast<int16_t>(t))));
  return _mm256_add_epi16(_mm256_sign_epi16(k, d), _mm256_set1_epi16(16));
}

// The frame header's bits of a filter's parameters (write_ccso()), the offsets apart.
int header_bits(bool bo_only, bool thr) { return bo_only ? 1 + 2 + 3 : 1 + 2 + 2 + 3 + (thr ? 1 : 0) + 2; }

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
    s.edge = take_table();
    for (int y = h * k / nt; y < h * (k + 1) / nt; y++) {
      const uint16_t* const l = luma + static_cast<size_t>(y << ss_y) * luma_stride;
      int x = 0;
      // 16 samples' bins and cells at a time, then their adds.
      for (; x + 16 <= w; x += 16) {
        alignas(32) uint16_t cell[kSupports][16];
        const uint16_t* const c0 = l + (x << ss_x);
        const __m256i v = load16(c0, ss_x);
        const __m256i band = _mm256_mullo_epi16(_mm256_srli_epi16(_mm256_srli_epi16(v, shift), 5), _mm256_set1_epi16(kBins * kBins));
        for (int f = 0; f < kSupports; f++) {
          const __m256i b0 = bins16(_mm256_sub_epi16(load16(c0 + pos[f][0], ss_x), v));
          const __m256i b1 = bins16(_mm256_sub_epi16(load16(c0 + pos[f][1], ss_x), v));
          const __m256i c = _mm256_add_epi16(_mm256_add_epi16(band, _mm256_set1_epi16(static_cast<int16_t>(f * 8 * kBins * kBins))),
                                             _mm256_add_epi16(_mm256_mullo_epi16(b0, _mm256_set1_epi16(kBins)), b1));
          _mm256_store_si256(reinterpret_cast<__m256i*>(cell[f]), c);
        }
        for (int i = 0; i < 16; i++) {
          const int level = c0[i << ss_x] >> shift;
          const int diff = src[static_cast<size_t>(y) * src_stride + x + i] - rec[static_cast<size_t>(y) * rec_stride + x + i];
          s.level[level].sum += diff, s.level[level].n++;
          const int64_t one = (int64_t{1} << 32) + diff;
          for (int f = 0; f < kSupports; f++) s.edge[cell[f][i]] += one;
        }
      }
      for (; x < w; x++) {
        const uint16_t* const c = l + (x << ss_x);
        const int v = *c, level = v >> shift;
        const int diff = src[static_cast<size_t>(y) * src_stride + x] - rec[static_cast<size_t>(y) * rec_stride + x];
        s.level[level].sum += diff, s.level[level].n++;
        const int64_t one = (int64_t{1} << 32) + diff;
        for (int f = 0; f < kSupports; f++)
          s.edge[((static_cast<size_t>(f) * 8 + (level >> 5)) * kBins + bins[c[pos[f][0]] - v + 1023]) * kBins +
                 bins[c[pos[f][1]] - v + 1023]] += one;
      }
    }
  };
  celv2::parallel(nt, rows);
  Stats s;
  const int slices = std::max(1, std::min(threads, 64));
  celv2::parallel(slices, [&](int k) {
    for (size_t i = kEdgeCells * k / slices; i < kEdgeCells * (k + 1) / slices; i++)
      for (const Stats32& t : st) s.edge[i] += unpack(t.edge[i]);
  });
  for (Stats32& t : st) {
    for (int i = 0; i < 256; i++) s.level[i].sum += t.level[i].sum, s.level[i].n += t.level[i].n;
    give_table(std::move(t.edge));
  }

  CcsoChoice best;
  double best_j = 1e300;
  int idx[CCSO_BAND_NUM * 16];
  // Band offsets only: up to 64 bands of the luma value.
  for (int sc = 0; sc < 4; sc++)
    for (int b = 0; b <= 6; b++) {
      double j = sse_per_bit * header_bits(true, false);
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
            double j = sse_per_bit * header_bits(false, thr > 0);
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

template <typename S>
int unit_stats(const celv2::CcsoPlane& p, const S* src, int src_stride, int w, int h, int scale, int unit_log2_x,
               int unit_log2_y, int cols, std::vector<CcsoDelta>* stats) {
  const int nd = p.bo_only ? 1 : p.edge_clf ? 2 : 3, classes = (1 << p.max_band_log2) * nd * nd;
  std::array<int16_t, CCSO_BAND_NUM * 16> compact{};
  for (int band = 0; band < 1 << p.max_band_log2; band++)
    for (int d0 = 0; d0 < nd; d0++)
      for (int d1 = 0; d1 < nd; d1++) compact[(band << 4) + (d0 << 2) + d1] = static_cast<int16_t>((band * nd + d0) * nd + d1);
  const int rows = (h + (1 << unit_log2_y) - 1) >> unit_log2_y;
  if (w > cols << unit_log2_x) throw std::invalid_argument("ccso_unit_stats: the plane is wider than its units");
  int o[8];
  for (int i = 0; i < 8; i++) o[i] = kOffset[i] * scale;
  const int maxv = (1 << p.bit_depth) - 1, lo = 10 * scale, hi = maxv - 7 * scale;  // no offset clips in [lo, hi]
  // In strips of up to 64 rows, each into its own table, then summed by unit row (by unit rows alone, 9 tasks a 1080p
  // plane, took 4 ms of the chain).
  const int strip_log2 = std::min(unit_log2_y, 6), strips = (h + (1 << strip_log2) - 1) >> strip_log2;
  const size_t cells = static_cast<size_t>(cols) * classes;
  std::vector<CcsoDelta> part(strips * cells);
  celv2::parallel(strips, [&](int r) {
    // n o^2 - 2 o (source - reconstruction) a sample, plus what the clip changes where it clips.
    std::vector<Acc> acc(cells);
    std::vector<uint16_t> idx(p.w);
    CcsoDelta* const row = part.data() + r * cells;
    for (int y = r << strip_log2; y < std::min(h, (r + 1) << strip_log2); y++) {
      celv2::ccso_class_row(p, y, idx.data());
      const S* const s = src + static_cast<size_t>(y) * src_stride;
      const uint16_t* const rec = p.dst + static_cast<ptrdiff_t>(y) * p.dst_stride;
      for (int x = 0; x < w; x++) {
        const size_t cell = static_cast<size_t>(x >> unit_log2_x) * classes + compact[idx[x]];
        const int v = rec[x], e = s[x] - v;
        acc[cell].sum += e, acc[cell].n++;
        if (v < lo || v > hi)
          for (int i = 1; i < 8; i++) {
            const int c = std::clamp(v + o[i], 0, maxv) - v;  // the clipped offset
            row[cell].d[i] += c * c - 2 * c * e - (o[i] * o[i] - 2 * o[i] * e);
          }
      }
    }
    for (size_t cell = 0; cell < acc.size(); cell++)
      for (int i = 1; i < 8; i++) row[cell].d[i] += acc[cell].n * o[i] * o[i] - 2 * o[i] * acc[cell].sum;
  });
  stats->assign(rows * cells, CcsoDelta{});
  const int per_row = 1 << (unit_log2_y - strip_log2);
  celv2::parallel(rows, [&](int r) {
    for (int k = r * per_row; k < std::min(strips, (r + 1) * per_row); k++)
      for (size_t c = 0; c < cells; c++) (*stats)[r * cells + c] += part[k * cells + c];
  });
  return classes;
}

int unary_bits(int i) { return i < 7 ? i + 1 : 7; }

}  // namespace

int ccso_bin_of(int d) { return bin_of(d); }
__m256i ccso_bins16(__m256i d) { return bins16(d); }

int ccso_unit_stats(const celv2::CcsoPlane& p, const uint8_t* src, int src_stride, int w, int h, int scale,
                    int unit_log2_x, int unit_log2_y, int cols, std::vector<CcsoDelta>* stats) {
  return unit_stats(p, src, src_stride, w, h, scale, unit_log2_x, unit_log2_y, cols, stats);
}

int ccso_unit_stats(const celv2::CcsoPlane& p, const uint16_t* src, int src_stride, int w, int h, int scale,
                    int unit_log2_x, int unit_log2_y, int cols, std::vector<CcsoDelta>* stats) {
  return unit_stats(p, src, src_stride, w, h, scale, unit_log2_x, unit_log2_y, cols, stats);
}

double ccso_unit_flags(const CcsoDelta* stats, int groups, int classes, double sse_per_bit, CcsoChoice* c,
                       std::vector<uint8_t>* on, const int* unit_group, int rows, int cols) {
  const int nd = c->bo_only ? 1 : c->edge_clf ? 2 : 3, scale = c->scale_idx + 1;
  if (classes != (1 << c->max_band_log2) * nd * nd) throw std::invalid_argument("ccso_unit_flags: the stats are not the choice's classes");
  auto table = [&](int k) { return ((k / (nd * nd)) << 4) + ((k / nd % nd) << 2) + k % nd; };  // CcsoInfo's index of class k
  const int header = header_bits(c->bo_only, quant_sz[c->scale_idx][c->quant_idx] > 0);
  // A group's error change at offset indices o (by class), and their truncated unary bits.
  auto change = [&](int g, const int* o) {
    int64_t e = 0;
    for (int k = 0; k < classes; k++) e += stats[static_cast<size_t>(g) * classes + k].d[o[k]];
    return static_cast<double>(e);
  };
  auto unary = [&](const int* o) {
    int bits = 0;
    for (int k = 0; k < classes; k++) bits += unary_bits(o[k]);
    return bits;
  };
  // Each class's best offset over the groups f marks.
  auto refit = [&](const std::vector<uint8_t>& f, int* o) {
    for (int k = 0; k < classes; k++) {
      CcsoDelta a;
      for (int g = 0; g < groups; g++)
        if (f[g]) a += stats[static_cast<size_t>(g) * classes + k];
      double best = 1e300;
      for (int i = 0; i < 8; i++)
        if (const double j = static_cast<double>(a.d[i]) + sse_per_bit * unary_bits(i); j < best) best = j, o[k] = i;
    }
  };
  auto j_on = [&](const int* o) {  // every unit on
    double e = 0;
    for (int g = 0; g < groups; g++) e += change(g, o);
    return e + sse_per_bit * (header + unary(o));
  };
  int o[CCSO_BAND_NUM * 16], all[CCSO_BAND_NUM * 16];
  for (int k = 0; k < classes; k++) o[k] = static_cast<int>(std::find(kOffset, kOffset + 8, c->offset[table(k)] / scale) - kOffset);
  refit(std::vector<uint8_t>(groups, 1), all);
  double j_all = j_on(o);
  if (const double j = j_on(all); j < j_all) j_all = j;
  else std::copy(o, o + classes, all);
  std::vector<uint8_t> f(groups), prev;
  for (int it = 0;; it++) {
    for (int g = 0; g < groups; g++) f[g] = change(g, o) < 0;
    if (it == 2 || f == prev) break;
    prev = f;
    refit(f, o);
  }
  // The flags' bits as an adaptive coder codes them: in raster order a group's flag at its first unit,
  // on av2_get_ccso_context()'s context there: a unit starts at a superblock's top, so it sees only the
  // left and bottom-left blocks, both in the left unit (context 2 if its flag is on, else 0), each
  // context's odds counted as it goes. All on costs a few bits too.
  // shortcut: a unit at a tile's left edge (not the frame's) counts the left flag, upgrade if tiles matter.
  auto flag_bits = [&](const std::vector<uint8_t>& f) {
    double n[2][2] = {{.5, .5}, {.5, .5}}, bits = 0;
    std::vector<uint8_t> seen(groups);
    for (int r = 0; r < rows; r++)
      for (int x = 0; x < cols; x++) {
        const int g = unit_group[r * cols + x];
        if (seen[g]) continue;
        seen[g] = 1;
        const int ctx = x && f[unit_group[r * cols + x - 1]];
        bits -= std::log2(n[ctx][f[g]] / (n[ctx][0] + n[ctx][1]));
        n[ctx][f[g]]++;
      }
    return bits;
  };
  j_all += sse_per_bit * flag_bits(std::vector<uint8_t>(groups, 1));
  double e = 0;
  for (int g = 0; g < groups; g++)
    if (f[g]) e += change(g, o);
  const double j_unit = e + sse_per_bit * (header + unary(o) + flag_bits(f));
  const bool unit = j_unit < j_all;
  const int* const best = unit ? o : all;
  for (int k = 0; k < classes; k++) c->offset[table(k)] = static_cast<int8_t>(kOffset[best[k]] * scale);
  if (unit) *on = f;
  else on->assign(groups, 1);
  return unit ? j_unit : j_all;
}

CcsoChoice ccso_search(const uint16_t* luma, int luma_stride, const uint16_t* rec, int rec_stride, const uint8_t* src,
                       int src_stride, int w, int h, int ss_x, int ss_y, double sse_per_bit, int threads) {
  return search(luma, luma_stride, rec, rec_stride, src, src_stride, w, h, ss_x, ss_y, sse_per_bit, threads, 8);
}

CcsoChoice ccso_search(const uint16_t* luma, int luma_stride, const uint16_t* rec, int rec_stride, const uint16_t* src,
                       int src_stride, int w, int h, int ss_x, int ss_y, double sse_per_bit, int threads, int bit_depth) {
  return search(luma, luma_stride, rec, rec_stride, src, src_stride, w, h, ss_x, ss_y, sse_per_bit, threads, bit_depth);
}
