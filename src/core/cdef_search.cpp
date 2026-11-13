#include "core/cdef_search.h"

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

extern "C" {
#include "av2/common/cdef_block.h"
}

namespace celv2 {

namespace {

int msb(int v) {  // get_msb(), v > 0
  unsigned long i;
  _BitScanReverse(&i, static_cast<unsigned long>(v));
  return static_cast<int>(i);
}

// cdef_block.c adjust_strength()
int adjust_strength(int strength, int var) {
  const int i = var >> 6 ? std::min(msb(var >> 6), 12) : 0;
  return var ? (strength * (4 + i) + 8) >> 4 : 0;
}

// cdef.h constrain() on 16 lanes, threshold > 0: sign(d) * min(|d|, max(0, threshold - (|d| >> shift))).
__m256i constrain(__m256i d, __m256i threshold, __m128i shift) {
  const __m256i a = _mm256_abs_epi16(d);
  const __m256i room = _mm256_max_epi16(_mm256_setzero_si256(), _mm256_sub_epi16(threshold, _mm256_srl_epi16(a, shift)));
  return _mm256_sign_epi16(_mm256_min_epi16(a, room), d);
}

}  // namespace

// The filter's sum is its primary part (of the level's adjusted strength along dir) plus its secondary part (of
// the secondary strength along dir +- 2, or along 0 +- 2 when the level is 0, as av2_cdef_filter_fb() passes the
// direction): 15 primary and 6 secondary parts a pixel instead of 63 whole filters, then each strength's sum,
// rounding and clamp (to the taps' range when both parts are on), as cdef_filter_block_internal().
void cdef_block_errors(const uint16_t* in, bool luma, int dir, int var, int damping, int coeff_shift,
                       const uint16_t* org16, const uint8_t* org8, int os, int vw, int vh, int64_t err[kCdefStrengths]) {
  constexpr int S = CDEF_BSTRIDE;
  const int bs = luma ? 8 : 4, nv = luma ? 4 : 1;
  damping += coeff_shift - (luma ? 0 : 1);
  // 16 lanes of a block: luma two rows of 8 a vector, chroma all four rows of 4.
  auto load = [&](int off, int q) {
    const uint16_t* const p = in + off;
    if (luma)
      return _mm256_inserti128_si256(
          _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 2 * q * S))),
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + (2 * q + 1) * S)), 1);
    const __m128i r01 = _mm_unpacklo_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)),
                                           _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p + S)));
    const __m128i r23 = _mm_unpacklo_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p + 2 * S)),
                                           _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p + 3 * S)));
    return _mm256_inserti128_si256(_mm256_castsi128_si256(r01), r23, 1);
  };
  // The source, 0 and masked off outside the visible part.
  alignas(32) int16_t org[64] = {}, vis[64] = {};
  for (int r = 0; r < std::min(vh, bs); r++)
    for (int c = 0; c < std::min(vw, bs); c++)
      org[r * bs + c] = static_cast<int16_t>(org16 ? org16[r * os + c] : org8[r * os + c]), vis[r * bs + c] = -1;

  // The strengths' constants.
  int pri_t[16] = {}, sec_t[4] = {};
  for (int l = 1; l < 16; l++) pri_t[l] = luma ? adjust_strength(l << coeff_shift, var) : l << coeff_shift;
  for (int si = 1; si < 4; si++) sec_t[si] = (si == 3 ? 4 : si) << coeff_shift;
  auto shift_of = [&](int t) { return _mm_cvtsi32_si128(std::max(0, damping - msb(t))); };

  __m256i acc[kCdefStrengths];
  for (__m256i& a : acc) a = _mm256_setzero_si256();
  const __m256i large = _mm256_set1_epi16(CDEF_VERY_LARGE);
  for (int q = 0; q < nv; q++) {
    const __m256i x = load(0, q);
    __m256i p[4], s[8], z[8];  // primary taps, secondary taps along dir and along 0
    for (int k = 0; k < 2; k++) {
      p[2 * k] = load(av2_cdef_directions[dir][k], q), p[2 * k + 1] = load(-av2_cdef_directions[dir][k], q);
      s[4 * k] = load(av2_cdef_directions[dir + 2][k], q), s[4 * k + 1] = load(-av2_cdef_directions[dir + 2][k], q);
      s[4 * k + 2] = load(av2_cdef_directions[dir - 2][k], q), s[4 * k + 3] = load(-av2_cdef_directions[dir - 2][k], q);
      z[4 * k] = load(av2_cdef_directions[2][k], q), z[4 * k + 1] = load(-av2_cdef_directions[2][k], q);
      z[4 * k + 2] = load(av2_cdef_directions[-2][k], q), z[4 * k + 3] = load(-av2_cdef_directions[-2][k], q);
    }
    // The clamp's range: the taps' (CDEF_VERY_LARGE left out of the maximum) and the pixel's.
    __m256i mn = x, mx = x;
    auto range = [&](__m256i v) {
      mn = _mm256_min_epi16(mn, v);
      mx = _mm256_max_epi16(mx, _mm256_blendv_epi8(v, x, _mm256_cmpeq_epi16(v, large)));
    };
    for (const __m256i& v : p) range(v);
    for (const __m256i& v : s) range(v);
    __m256i pd[4], sd[8], zd[8];
    for (int i = 0; i < 4; i++) pd[i] = _mm256_sub_epi16(p[i], x);
    for (int i = 0; i < 8; i++) sd[i] = _mm256_sub_epi16(s[i], x), zd[i] = _mm256_sub_epi16(z[i], x);

    __m256i prim[16], sec[4], sec0[4];
    prim[0] = sec[0] = sec0[0] = _mm256_setzero_si256();
    for (int l = 1; l < 16; l++) {
      const int t = pri_t[l];
      if (!t) {
        prim[l] = _mm256_setzero_si256();
        continue;
      }
      const int* const taps = av2_cdef_pri_taps[(t >> coeff_shift) & 1];
      const __m256i tv = _mm256_set1_epi16(static_cast<int16_t>(t));
      const __m128i sh = shift_of(t);
      const __m256i a = _mm256_add_epi16(constrain(pd[0], tv, sh), constrain(pd[1], tv, sh));
      const __m256i b = _mm256_add_epi16(constrain(pd[2], tv, sh), constrain(pd[3], tv, sh));
      prim[l] = _mm256_add_epi16(_mm256_mullo_epi16(a, _mm256_set1_epi16(static_cast<int16_t>(taps[0]))),
                                 _mm256_mullo_epi16(b, _mm256_set1_epi16(static_cast<int16_t>(taps[1]))));
    }
    for (int si = 1; si < 4; si++) {
      const __m256i tv = _mm256_set1_epi16(static_cast<int16_t>(sec_t[si]));
      const __m128i sh = shift_of(sec_t[si]);
      auto part = [&](const __m256i* d) {
        __m256i k0 = _mm256_add_epi16(_mm256_add_epi16(constrain(d[0], tv, sh), constrain(d[1], tv, sh)),
                                      _mm256_add_epi16(constrain(d[2], tv, sh), constrain(d[3], tv, sh)));
        __m256i k1 = _mm256_add_epi16(_mm256_add_epi16(constrain(d[4], tv, sh), constrain(d[5], tv, sh)),
                                      _mm256_add_epi16(constrain(d[6], tv, sh), constrain(d[7], tv, sh)));
        return _mm256_add_epi16(_mm256_mullo_epi16(k0, _mm256_set1_epi16(static_cast<int16_t>(av2_cdef_sec_taps[0]))),
                                _mm256_mullo_epi16(k1, _mm256_set1_epi16(static_cast<int16_t>(av2_cdef_sec_taps[1]))));
      };
      sec[si] = part(sd), sec0[si] = part(zd);
    }

    const __m256i o = _mm256_load_si256(reinterpret_cast<const __m256i*>(org + 16 * q));
    const __m256i m = _mm256_load_si256(reinterpret_cast<const __m256i*>(vis + 16 * q));
    const __m256i eight = _mm256_set1_epi16(8);
    for (int st = 0; st < kCdefStrengths; st++) {
      const int l = st >> 2, si = st & 3;
      __m256i y = x;
      if (st) {
        const __m256i sum = l ? _mm256_add_epi16(prim[l], sec[si]) : sec0[si];
        // (8 + sum - (sum < 0)) >> 4
        y = _mm256_add_epi16(x, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(sum, eight),
                                                                   _mm256_cmpgt_epi16(_mm256_setzero_si256(), sum)), 4));
        if (l && si && pri_t[l]) y = _mm256_min_epi16(_mm256_max_epi16(y, mn), mx);
      }
      const __m256i d = _mm256_and_si256(_mm256_sub_epi16(y, o), m);
      acc[st] = _mm256_add_epi32(acc[st], _mm256_madd_epi16(d, d));
    }
  }
  for (int st = 0; st < kCdefStrengths; st++) {
    const __m128i h = _mm_add_epi32(_mm256_castsi256_si128(acc[st]), _mm256_extracti128_si256(acc[st], 1));
    const __m128i h2 = _mm_add_epi32(h, _mm_shuffle_epi32(h, 0x4e));
    err[st] += _mm_cvtsi128_si32(_mm_add_epi32(h2, _mm_shuffle_epi32(h2, 0xb1)));
  }
}

namespace {

// sum over g of min(a[g], b[g]); once a partial sum reaches stop, that partial sum (the caller only needs to know
// the full one is not below stop: every term is at least 0).
int64_t min_sum(const int64_t* a, const int64_t* b, int n, int64_t stop) {
  __m256i acc = _mm256_setzero_si256();
  auto total = [&] {
    alignas(32) int64_t v[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(v), acc);
    return v[0] + v[1] + v[2] + v[3];
  };
  int g = 0;
  for (; g + 4 <= n; g += 4) {
    const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + g));
    const __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + g));
    acc = _mm256_add_epi64(acc, _mm256_blendv_epi8(x, y, _mm256_cmpgt_epi64(x, y)));
    if ((g & 63) == 60 && total() >= stop) return total();
  }
  int64_t sum = total();
  for (; g < n; g++) sum += std::min(a[g], b[g]);
  return sum;
}

}  // namespace

std::vector<CdefPair> cdef_choose_set(const int64_t* err, int groups, double sse_per_bit, std::vector<int8_t>* index) {
  constexpr int K = kCdefStrengths;
  auto e = [&](int g, int p, int s) { return err[(g * 2 + p) * K + s]; };
  // Candidates: the strengths that are some group's best, the most often first (12 luma, 6 chroma), the summed
  // best and 0.
  std::vector<int> cand[2];
  for (int p = 0; p < 2; p++) {
    std::array<int, K> uses{};
    std::array<int64_t, K> total{};
    for (int g = 0; g < groups; g++) {
      int best = 0;
      for (int s = 0; s < K; s++) {
        total[s] += e(g, p, s);
        if (e(g, p, s) < e(g, p, best)) best = s;
      }
      uses[best]++;
    }
    std::array<int, K> order;
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return uses[a] > uses[b]; });
    for (int i = 0; i < (p ? 6 : 12) && uses[order[i]]; i++) cand[p].push_back(order[i]);
    for (const int s : {static_cast<int>(std::min_element(total.begin(), total.end()) - total.begin()), 0})
      if (std::find(cand[p].begin(), cand[p].end(), s) == cand[p].end()) cand[p].push_back(s);
  }
  std::vector<CdefPair> pairs;
  for (const int l : cand[0])
    for (const int c : cand[1]) pairs.push_back({l, c});
  const int np = static_cast<int>(pairs.size());
  // Each group's error at each pair, pair by pair (the groups of one pair in a row: the slot search's inner loop).
  std::vector<int64_t> pe(static_cast<size_t>(groups) * np);
  auto col = [&](int k) { return &pe[static_cast<size_t>(k) * groups]; };
  for (int k = 0; k < np; k++)
    for (int g = 0; g < groups; g++) col(k)[g] = e(g, 0, pairs[k].luma) + e(g, 1, pairs[k].chroma);
  auto strength_bits = [](int s) { return 1 + (s < 4 ? 2 : 6); };  // setup_cdef()
  auto entropy = [](const std::vector<int>& counts, int total) {
    double b = 0;
    for (const int c : counts)
      if (c) b -= c * std::log2(c / static_cast<double>(total));
    return b;
  };
  // Each group's index into set, the groups' error and the bits (header 3, the strengths, the index symbols'
  // entropy: the index-0 flag, then among the rest).
  std::vector<int> set, pick(groups), count;
  auto evaluate = [&](double* bits) {
    const int n = static_cast<int>(set.size());
    count.assign(n, 0);
    int64_t error = 0;
    for (int g = 0; g < groups; g++) {
      int b = 0;
      for (int i = 1; i < n; i++)
        if (col(set[i])[g] < col(set[b])[g]) b = i;
      pick[g] = b, count[b]++, error += col(set[b])[g];
    }
    *bits = 3;
    for (const int k : set) *bits += strength_bits(pairs[k].luma) + strength_bits(pairs[k].chroma);
    if (n > 1) {
      const int c0 = *std::max_element(count.begin(), count.end());
      *bits += entropy({c0, groups - c0}, groups);
      if (n > 2) {
        std::vector<int> rest = count;
        rest.erase(std::max_element(rest.begin(), rest.end()));
        *bits += entropy(rest, groups - c0);
      }
    }
    return error;
  };
  // The pair that, in slot i of set (-1: a new slot), leaves the groups the least error (others: the min over
  // the other slots).
  std::vector<int64_t> others(groups);
  auto best_for_slot = [&](int i) {
    std::fill(others.begin(), others.end(), INT64_MAX);
    for (size_t j = 0; j < set.size(); j++)
      if (static_cast<int>(j) != i)
        for (int g = 0; g < groups; g++) others[g] = std::min(others[g], col(set[j])[g]);
    int best = -1;
    int64_t least = INT64_MAX;
    for (int k = 0; k < np; k++) {
      if (std::find(set.begin(), set.end(), k) != set.end() && (i < 0 || set[i] != k)) continue;
      const int64_t sum = min_sum(others.data(), col(k), groups, least);
      if (sum < least) least = sum, best = k;
    }
    return best;
  };
  int64_t off = 0;
  for (int g = 0; g < groups; g++) off += e(g, 0, 0) + e(g, 1, 0);
  double best_j = static_cast<double>(off);
  std::vector<int> best_set;
  for (int n = 1; n <= 8 && n <= np; n++) {
    set.push_back(best_for_slot(-1));  // greedy: the pair lowering the groups' error the most
    for (int pass = 0; pass < 2; pass++)
      for (int i = 0; i < n; i++) set[i] = best_for_slot(i);
    double bits;
    const double j = static_cast<double>(evaluate(&bits)) + sse_per_bit * bits;
    if (j < best_j) best_j = j, best_set = set;
  }
  if (best_set.empty()) return {};  // {(0,0)} alone is never kept: its J is off plus the header's bits
  set = best_set;
  double bits;
  evaluate(&bits);
  // In use order: the most used index 0.
  std::vector<int> order(set.size()), rank(set.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return count[a] > count[b]; });
  std::vector<CdefPair> out;
  for (size_t i = 0; i < order.size(); i++) rank[order[i]] = static_cast<int>(i), out.push_back(pairs[set[order[i]]]);
  index->resize(groups);
  for (int g = 0; g < groups; g++) (*index)[g] = static_cast<int8_t>(rank[pick[g]]);
  return out;
}

}  // namespace celv2
