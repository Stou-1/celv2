#include "core/quant.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <immintrin.h>
#include <limits>

#include "core/profile.h"

extern "C" {
#include "av2/common/idct.h"
#include "av2/common/quant_common.h"
#include "av2/common/scan.h"
#include "av2/common/txb_common.h"
}

tran_low_t dequant_coeff(int level, int dqv, TX_SIZE tx, int bd) {
  const int32_t max_value = (1 << (7 + bd)) - 1;
  const int32_t min_value = -(1 << (7 + bd));
  const int64_t dq_hp = (int64_t)std::abs(level) * dqv & 0xffffff;
  tran_low_t dq = (tran_low_t)ROUND_POWER_OF_TWO_64(dq_hp, QUANT_TABLE_BITS);
  dq >>= av2_get_tx_scale(tx);
  if (level < 0) dq = -dq;
  return clamp(dq, min_value, max_value);
}

int64_t lambda_q8(int qindex) {
  const double qstep = av2_ac_quant_QTX(qindex, 0, 0, AVM_BITS_8) / 8.0;
  // 4x the first guess (0.0005): on anime clips x2 / x4 / x8 / x16 gave -8.3 / -11.4 / -9.8 / -4.2%
  // BD-rate, x3 and x5 within 0.3% of x4. Larger blocks and fewer modes also make it faster.
  // Growing as qstep^1.2 rather than qstep^2 (the same at qindex 120): exponents 1.7 / 1.5 / 1.2 / 1.0
  // gave -2.5 / -3.4 / -4.0 / -3.1% BD-rate at qindex 45-180 and -3.8 / -5.7 / -7.0 / -6.5% at 30-75.
  const double pivot = av2_ac_quant_QTX(120, 0, 0, AVM_BITS_8) / 8.0;
  return std::llround(0.002 * qstep * qstep * std::pow(pivot / qstep, 0.8) * 256.0);
}

thread_local bool g_rd_zero = false;

namespace {

// Rough bits of a coefficient level, sign included (AV2's base / range / Golomb coding).
double level_bits_of(int l) {
  if (l == 0) return 0.6;
  if (l == 1) return 2.6;
  if (l == 2) return 3.6;
  if (l == 3) return 4.4;
  return 5.0 + 2.0 * std::log2(static_cast<double>(l - 2));
}

// The same values, small levels from a table (the log2 call was a large part of the trellis).
const std::array<double, 256> kLevelBits = [] {
  std::array<double, 256> t;
  for (int i = 0; i < 256; i++) t[i] = level_bits_of(i);
  return t;
}();
double level_bits(int l) { return l < 256 ? kLevelBits[l] : level_bits_of(l); }

// TCQ (decodetxb.c with tcq_mode): the state, starting at 0 on the last coefficient and moving on
// each level's parity (tcq_next_state), picks quantizer Q0 (reconstruction level * step) or Q1
// ((2 level - 1) * step / 2). The levels are a Viterbi search over the 8 states, in reverse scan
// from the last coefficient worth a level, by squared error in steps plus kBitWeight per bit.
// Returns eob; levels (signed) into qcoeff, dqcoeff as the decoder dequantizes (when given).

// A coefficient of e steps left at level 0, with --psy-trellis's term on AC.
double dropped(double e, int pos, double psy) { return e * e + (pos ? psy * e : 0); }

// The rough bits of a block's end of block and all-zero flag (the RD pass's), and of the flag alone.
const std::array<double, 1025> kEndBits = [] {
  std::array<double, 1025> t{};
  for (int i = 1; i < 1025; i++) t[i] = 2 + 1.5 * std::log2(static_cast<double>(i));
  return t;
}();
double end_bits(int eob) { return kEndBits[eob]; }
constexpr double kNoneBits = 1;

// Under this many steps a coefficient's level 1 loses to 0 under Q0 (the first state's quantizer).
double first_level(double weight, double psy) { return 0.5 + weight * (level_bits(1) - level_bits(0)) / 2 / (1 + psy); }

// e_of[pos]: each coefficient's magnitude in quantizer steps; returns eob, one past the last scan
// index whose level 1 beats 0 under Q0 (the first state's quantizer). Four at a time (n is a
// multiple of 16) by the scan indices in position order, the DC position redone with its own step.
int tcq_prep(const tran_low_t* coeff, int n, double step_dc, double step_ac, const SCAN_ORDER* so, double* e_of,
             double psy, double weight) {
  const double first = first_level(weight, 0), first_ac = first_level(weight, psy);  // AC with psy's term
  int eob = 0;
  const __m256d v_step = _mm256_set1_pd(step_ac), v_first = _mm256_set1_pd(first_ac), v_sign = _mm256_set1_pd(-0.0);
  for (int pos = 0; pos < n; pos += 4) {
    const __m256d c = _mm256_cvtepi32_pd(_mm_loadu_si128(reinterpret_cast<const __m128i*>(coeff + pos)));
    const __m256d e = _mm256_div_pd(_mm256_andnot_pd(v_sign, c), v_step);
    _mm256_storeu_pd(e_of + pos, e);
    for (int m = _mm256_movemask_pd(_mm256_cmp_pd(e, v_first, _CMP_GE_OQ)) & (pos ? 15 : 14); m; m &= m - 1)
      eob = std::max(eob, so->iscan[pos + _tzcnt_u32(m)] + 1);
  }
  e_of[0] = std::abs(static_cast<double>(coeff[0])) / step_dc;
  if (e_of[0] >= first) eob = std::max(eob, 1);  // iscan[0] is 0
  return eob;
}

// A coefficient's TCQ candidates per quantizer (Q1 for states 2, 3, 6, 7): the nearest levels and
// zero (the last coefficient must keep a level), by parity in the order tcq_levels() tries them,
// each with its squared error and weighted bits. A clamped repeat (same level, same cost) is left
// out: at most two even ones (one is zero) and one odd.
struct TcqCands {
  struct C {
    int l;
    double d2, wb;
  } even[2][2], odd[2];
  int ne[2], no[2];
};
void tcq_cands(double e, int max_level, bool last, double psy, double weight, TcqCands* t) {
  for (int q1 = 0; q1 < 2; q1++) {
    t->ne[q1] = t->no[q1] = 0;
    const int base = q1 ? static_cast<int>(e + 0.5) : static_cast<int>(e);
    const int cands[3] = {base, base + 1, 0};
    int prev = -1;
    for (int k = 0; k < 3; k++) {
      const int l = std::min(cands[k], max_level);
      if ((l == 0 && last) || (k == 2 && base == 0) || l == prev) continue;
      prev = l;
      const double rec = l == 0 ? 0 : q1 ? l - 0.5 : l;
      const TcqCands::C c{l, (e - rec) * (e - rec) + psy * std::abs(e - rec), weight * level_bits(l)};
      if (l & 1) t->odd[q1] = c, t->no[q1] = 1;
      else t->even[q1][t->ne[q1]++] = c;
    }
  }
}

// --rd-zero: a path starting here, with J so far (s: the dropped coefficients after it and its end of block), takes Q0's
// nonzero candidates into states 0 (even level) and 4 (odd), a continuing path winning a tie. step: this position's
// entries, previous state 8 marking the start (when given).
void tcq_fresh(const TcqCands& t, double s, __m256d* lo, __m256d* hi, int64_t* step) {
  constexpr int64_t kStart = 8ll << 32;
  if (t.even[0][0].l) {
    const double j = s + t.even[0][0].d2 + t.even[0][0].wb;
    if (j < _mm256_cvtsd_f64(*lo)) {
      *lo = _mm256_blend_pd(*lo, _mm256_set1_pd(j), 1);
      if (step) step[0] = t.even[0][0].l | kStart;
    }
  }
  if (t.no[0]) {
    const double j = s + t.odd[0].d2 + t.odd[0].wb;
    if (j < _mm256_cvtsd_f64(*hi)) {
      *hi = _mm256_blend_pd(*hi, _mm256_set1_pd(j), 1);
      if (step) step[4] = t.odd[0].l | kStart;
    }
  }
}

// force_eob >= 0: that end of block instead of tcq_prep()'s (the tests' oracle). --rd-zero, unforced: any end above the threshold or all
// zero by one Viterbi pass, *cost holding the end's term.
int tcq_levels(const tran_low_t* coeff, tran_low_t* qcoeff, tran_low_t* dqcoeff, TX_SIZE tx, int qindex,
               const SCAN_ORDER* so, double* cost, int bd, double psy, int force_eob = -1) {
  static const uint8_t next[8][2] = {{0, 4}, {4, 0}, {1, 5}, {5, 1}, {6, 2}, {2, 6}, {7, 3}, {3, 7}};
  const int dc_q = av2_dc_quant_QTX(qindex, 0, 0, kQDepth);
  const int ac_q = av2_ac_quant_QTX(qindex, 0, 0, kQDepth);
  const int n = get_txb_wide(tx) * get_txb_high(tx);
  const double scale = static_cast<double>(1 << (QUANT_TABLE_BITS + av2_get_tx_scale(tx)));
  const double step_dc = dc_q / scale, step_ac = ac_q / scale;
  const int max_dc = 0xffffff / (2 * dc_q), max_ac = 0xffffff / (2 * ac_q);  // qIdx = 2 level
  double e_of[1024];
  const double weight = kTcqBitWeight;
  const int prep = tcq_prep(coeff, n, step_dc, step_ac, so, e_of, psy, weight);
  const int eob = force_eob >= 0 ? force_eob : prep;
  const bool pass = force_eob < 0 && g_rd_zero;
  std::fill(qcoeff, qcoeff + n, 0);
  if (dqcoeff) std::fill(dqcoeff, dqcoeff + n, 0);
  if (!eob) {
    if (cost) {
      double j = 0;
      for (int pos = 0; pos < n; pos++)
        if (coeff[pos]) j += dropped(e_of[pos], pos, psy);  // a zero adds exactly nothing
      *cost = pass ? j + weight * kNoneBits : j;
    }
    return 0;
  }
  // Viterbi, as tcq_cost()'s vectors: path cost per state, and per scan index each state's level and previous
  // state, packed in 64 bits beside its cost. States pair up (a, a + 1 share a quantizer and lead to the same two
  // states), and each next state takes the first least of its candidates in the order they were always compared:
  // the lower state's, then the upper one's. A tie keeps the even level in the first of a pair's next states and the
  // odd one in the second.
  constexpr double kInf = std::numeric_limits<double>::infinity();
  alignas(32) static thread_local int64_t step[1024][8];  // level | previous state << 32
  __m256d lo = _mm256_setr_pd(pass ? kInf : 0, kInf, kInf, kInf), hi = _mm256_set1_pd(kInf);  // the paths of states 0-3, 4-7
  const __m256i from_lo = _mm256_setr_epi64x(0ll << 32, 1ll << 32, 2ll << 32, 3ll << 32);
  const __m256i from_hi = _mm256_setr_epi64x(4ll << 32, 5ll << 32, 6ll << 32, 7ll << 32);
  const __m256d second = _mm256_castsi256_pd(_mm256_setr_epi64x(0, -1, 0, -1));  // a pair's second next state
  const double first = first_level(weight, 0), first_ac = first_level(weight, psy);
  double tail = 0;  // --rd-zero: dropped() from the position after this one up to eob - 1
  for (int c = eob - 1; c >= 0; c--) {
    const int pos = so->scan[c];
    TcqCands t;
    tcq_cands(e_of[pos], pos == 0 ? max_dc : max_ac, !pass && c == eob - 1, pos ? psy : 0, weight, &t);
    // A missing candidate costs infinity.
    double ed[2][2], ew[2][2], od[2], ow[2];
    int64_t el[2][2], ol[2];
    for (int q1 = 0; q1 < 2; q1++) {
      for (int k = 0; k < 2; k++) {
        const bool have = k < t.ne[q1];
        ed[q1][k] = have ? t.even[q1][k].d2 : kInf, ew[q1][k] = have ? t.even[q1][k].wb : 0;
        el[q1][k] = have ? t.even[q1][k].l : 0;
      }
      od[q1] = t.no[q1] ? t.odd[q1].d2 : kInf, ow[q1] = t.no[q1] ? t.odd[q1].wb : 0;
      ol[q1] = t.no[q1] ? t.odd[q1].l : 0;
    }
    // States 0-3 and 4-7 as vectors (quantizers Q0 Q0 Q1 Q1 in each), the sums as ever.
    auto per_state = [](double q0, double q1) { return _mm256_setr_pd(q0, q0, q1, q1); };
    auto per_state_i = [](int64_t q0, int64_t q1) { return _mm256_setr_epi64x(q0, q0, q1, q1); };
    const __m256d ed0 = per_state(ed[0][0], ed[1][0]), ew0 = per_state(ew[0][0], ew[1][0]);
    const __m256d ed1 = per_state(ed[0][1], ed[1][1]), ew1 = per_state(ew[0][1], ew[1][1]);
    const __m256d odv = per_state(od[0], od[1]), owv = per_state(ow[0], ow[1]);
    const __m256i el0 = per_state_i(el[0][0], el[1][0]), el1 = per_state_i(el[0][1], el[1][1]), olv = per_state_i(ol[0], ol[1]);
    __m256d f[2], g[2];
    for (int h = 0; h < 2; h++) {
      const __m256d p = h ? hi : lo;
      const __m256i from = h ? from_hi : from_lo;
      const __m256d c0 = _mm256_add_pd(_mm256_add_pd(p, ed0), ew0), c1 = _mm256_add_pd(_mm256_add_pd(p, ed1), ew1);
      const __m256d odd = _mm256_add_pd(_mm256_add_pd(p, odv), owv);
      const __m256d even = _mm256_blendv_pd(c0, c1, _mm256_cmp_pd(c1, c0, _CMP_LT_OQ));
      const __m256d even_step = _mm256_blendv_pd(_mm256_castsi256_pd(_mm256_or_si256(el0, from)), _mm256_castsi256_pd(_mm256_or_si256(el1, from)),
                                                 _mm256_cmp_pd(c1, c0, _CMP_LT_OQ));
      const __m256d other = _mm256_permute_pd(odd, 0b0101);  // a pair's other state's odd level
      const __m256d other_step = _mm256_permute_pd(_mm256_castsi256_pd(_mm256_or_si256(olv, from)), 0b0101);
      const __m256d take = _mm256_blendv_pd(_mm256_cmp_pd(other, even, _CMP_LT_OQ), _mm256_cmp_pd(other, even, _CMP_LE_OQ), second);
      f[h] = _mm256_blendv_pd(even, other, take);
      g[h] = _mm256_blendv_pd(even_step, other_step, take);
    }
    // f[0] holds the next states 0 4 1 5, f[1] 6 2 7 3: as 0 1 4 5 and 2 3 6 7, then halves.
    const __m256d a = _mm256_permute4x64_pd(f[0], 0b11011000), d = _mm256_permute4x64_pd(f[1], 0b10001101);
    lo = _mm256_permute2f128_pd(a, d, 0x20);
    hi = _mm256_permute2f128_pd(a, d, 0x31);
    const __m256d ga = _mm256_permute4x64_pd(g[0], 0b11011000), gd = _mm256_permute4x64_pd(g[1], 0b10001101);
    _mm256_store_pd(reinterpret_cast<double*>(step[c]), _mm256_permute2f128_pd(ga, gd, 0x20));
    _mm256_store_pd(reinterpret_cast<double*>(step[c] + 4), _mm256_permute2f128_pd(ga, gd, 0x31));
    if (pass) {
      if (e_of[pos] >= (pos ? first_ac : first)) tcq_fresh(t, tail + weight * end_bits(c + 1), &lo, &hi, step[c]);
      tail += dropped(e_of[pos], pos, psy);
    }
  }
  alignas(32) double path[8];
  _mm256_store_pd(path, lo);
  _mm256_store_pd(path + 4, hi);
  int s = 0;
  for (int k = 1; k < 8; k++)
    if (path[k] < path[s]) s = k;
  const double none = tail + weight * kNoneBits;  // --rd-zero: all zero, the levels winning a tie
  const bool zero = pass && none < path[s];
  if (cost) {
    double j = zero ? none : path[s];
    for (int c = eob; c < n; c++) j += dropped(e_of[so->scan[c]], so->scan[c], psy);
    *cost = j;
  }
  if (zero) return 0;
  // Back from the DC end: the levels, then the decoder's dequantization along the states.
  int end = eob;
  for (int c = 0; c < eob; c++) {
    const int pos = so->scan[c];
    const int l = static_cast<int16_t>(step[c][s]);
    qcoeff[pos] = coeff[pos] < 0 ? -l : l;
    s = static_cast<int>(step[c][s] >> 32);
    if (s == 8) {
      end = c + 1;
      break;
    }
  }
  if (dqcoeff) {
    const int32_t max_value = (1 << (7 + bd)) - 1, min_value = -(1 << (7 + bd));
    int state = 0;
    for (int c = end - 1; c >= 0; c--) {
      const int pos = so->scan[c];
      const int l = std::abs(static_cast<int>(qcoeff[pos]));
      if (l) {
        const int64_t q_idx = std::max(0, 2 * l - ((state & 2) ? 1 : 0));
        const int64_t dq_hp = q_idx * (pos == 0 ? dc_q : ac_q) & 0xffffff;
        tran_low_t dq = static_cast<tran_low_t>(ROUND_POWER_OF_TWO_64(dq_hp, QUANT_TABLE_BITS));
        dq >>= av2_get_tx_scale(tx) + 1;
        dqcoeff[pos] = clamp(qcoeff[pos] < 0 ? -dq : dq, min_value, max_value);
      }
      state = next[state][l & 1];
    }
  }
  return end;
}

// --rd-zero on the plain path: whether no levels beat these, by J. e_at, l_at: a position's magnitude in steps and level.
template <class E, class L>
bool zero_wins(int n, int eob, const SCAN_ORDER* so, double psy, E e_at, L l_at) {
  const double m = kBitWeight;
  double levels = m * end_bits(eob), none = m * kNoneBits;
  for (int pos = 0; pos < n; pos++) {
    const double e = e_at(pos);
    const int l = l_at(pos);
    levels += (e - l) * (e - l) + (pos ? psy * std::abs(e - l) : 0) + (so->iscan[pos] < eob ? m * level_bits(l) : 0);
    none += dropped(e, pos, psy);
  }
  return none < levels;
}

// tcq_levels()'s cost alone, the same value (the same sums in the same order): without the
// levels, a state's path is the least of its candidates' costs whatever the order they are compared in.
double tcq_cost(const tran_low_t* coeff, TX_SIZE tx, int qindex, const SCAN_ORDER* so, int* eob_out, double psy) {
  if (g_rd_zero && eob_out) {  // its end of block comes from the levels
    tran_low_t q[1024];
    double c = 0;
    *eob_out = tcq_levels(coeff, q, nullptr, tx, qindex, so, &c, 8, psy);
    return c;
  }
  const int dc_q = av2_dc_quant_QTX(qindex, 0, 0, kQDepth);
  const int ac_q = av2_ac_quant_QTX(qindex, 0, 0, kQDepth);
  const int n = get_txb_wide(tx) * get_txb_high(tx);
  const double scale = static_cast<double>(1 << (QUANT_TABLE_BITS + av2_get_tx_scale(tx)));
  const double step_dc = dc_q / scale, step_ac = ac_q / scale;
  const int max_dc = 0xffffff / (2 * dc_q), max_ac = 0xffffff / (2 * ac_q);
  double e_of[1024];
  const double weight = kTcqBitWeight;
  const int eob = tcq_prep(coeff, n, step_dc, step_ac, so, e_of, psy, weight);
  const bool pass = g_rd_zero;
  if (eob_out) *eob_out = eob;
  if (!eob) {
    double j = 0;
    for (int pos = 0; pos < n; pos++)
      if (coeff[pos]) j += dropped(e_of[pos], pos, psy);  // a zero adds exactly nothing
    return pass ? j + weight * kNoneBits : j;
  }
  // Each state's path from the states before it: a state's quantizer (Q1 for states 2, 3, 6, 7)
  // and each level's parity give the next state (tcq_levels()'s table), so states pair up: a and
  // b = a + 1 lead to e (even level from a, odd from b) and o (odd from a, even from b).
  constexpr double kInf = std::numeric_limits<double>::infinity();
  __m256d lo = _mm256_setr_pd(pass ? kInf : 0, kInf, kInf, kInf), hi = _mm256_set1_pd(kInf);  // the paths of states 0-3, 4-7
  const double first = first_level(weight, 0), first_ac = first_level(weight, psy);
  double tail = 0;  // as tcq_levels()
  for (int c = eob - 1; c >= 0; c--) {
    const int pos = so->scan[c];
    TcqCands t;
    tcq_cands(e_of[pos], pos == 0 ? max_dc : max_ac, !pass && c == eob - 1, pos ? psy : 0, weight, &t);
    // A missing candidate costs infinity.
    double ed[2][2], ew[2][2], od[2], ow[2];
    for (int q1 = 0; q1 < 2; q1++) {
      for (int k = 0; k < 2; k++)
        ed[q1][k] = k < t.ne[q1] ? t.even[q1][k].d2 : kInf, ew[q1][k] = k < t.ne[q1] ? t.even[q1][k].wb : 0;
      od[q1] = t.no[q1] ? t.odd[q1].d2 : kInf, ow[q1] = t.no[q1] ? t.odd[q1].wb : 0;
    }
    // States 0-3 and 4-7 as vectors (quantizers Q0 Q0 Q1 Q1 in each), the sums as tcq_levels()'s.
    auto per_state = [](double q0, double q1) { return _mm256_setr_pd(q0, q0, q1, q1); };
    const __m256d ed0 = per_state(ed[0][0], ed[1][0]), ew0 = per_state(ew[0][0], ew[1][0]);
    const __m256d ed1 = per_state(ed[0][1], ed[1][1]), ew1 = per_state(ew[0][1], ew[1][1]);
    const __m256d odv = per_state(od[0], od[1]), owv = per_state(ow[0], ow[1]);
    __m256d f[2];
    for (int h = 0; h < 2; h++) {
      const __m256d p = h ? hi : lo;
      const __m256d even = _mm256_min_pd(_mm256_add_pd(_mm256_add_pd(p, ed0), ew0), _mm256_add_pd(_mm256_add_pd(p, ed1), ew1));
      const __m256d odd = _mm256_add_pd(_mm256_add_pd(p, odv), owv);
      // a = 0, 2 (4, 6): even from a or odd from a + 1, and odd from a or even from a + 1
      f[h] = _mm256_min_pd(even, _mm256_permute_pd(odd, 0b0101));
    }
    // f[0] holds the next states 0 4 1 5, f[1] 6 2 7 3: as 0 1 4 5 and 2 3 6 7, then halves.
    const __m256d a = _mm256_permute4x64_pd(f[0], 0b11011000), b = _mm256_permute4x64_pd(f[1], 0b10001101);
    lo = _mm256_permute2f128_pd(a, b, 0x20);
    hi = _mm256_permute2f128_pd(a, b, 0x31);
    if (pass) {
      if (e_of[pos] >= (pos ? first_ac : first)) tcq_fresh(t, tail + weight * end_bits(c + 1), &lo, &hi, nullptr);
      tail += dropped(e_of[pos], pos, psy);
    }
  }
  alignas(32) double path[8];
  _mm256_store_pd(path, lo);
  _mm256_store_pd(path + 4, hi);
  double j = *std::min_element(path, path + 8);
  if (pass) j = std::min(j, tail + weight * kNoneBits);
  for (int c = eob; c < n; c++)
    if (coeff[so->scan[c]]) j += dropped(e_of[so->scan[c]], so->scan[c], psy);
  return j;
}

}  // namespace

double coeff_level_bits(int l) { return level_bits(l); }

double tcq_end_cost(const tran_low_t* coeff, tran_low_t* qcoeff, tran_low_t* dqcoeff, TX_SIZE tx, int qindex, TX_TYPE tx_type, int end, double psy) {
  const SCAN_ORDER* const so = get_scan(tx, tx_type);
  const double weight = kTcqBitWeight;
  const double scale = static_cast<double>(1 << (QUANT_TABLE_BITS + av2_get_tx_scale(tx)));
  const int pos = end ? so->scan[end - 1] : 0;
  const double step = (pos ? av2_ac_quant_QTX(qindex, 0, 0, kQDepth) : av2_dc_quant_QTX(qindex, 0, 0, kQDepth)) / scale;
  if (end && std::abs(static_cast<double>(coeff[pos])) / step < first_level(weight, pos ? psy : 0)) return -1;
  double c = 0;
  tcq_levels(coeff, qcoeff, dqcoeff, tx, qindex, so, &c, 8, psy, end);
  return c + weight * (end ? end_bits(end) : kNoneBits);
}

double quantize_cost(const tran_low_t* coeff, TX_SIZE tx, int qindex, TX_TYPE tx_type, int* eob_out, bool tcq, double psy) {
  celv2::HotScope prof(celv2::Stage::Quant);
  if (tcq) return tcq_cost(coeff, tx, qindex, get_scan(tx, tx_type), eob_out, psy);
  // quantize()'s choices step for step (levels, then the trailing +-1 trimming), without the
  // dequantization; level 0 (|coeff| under half a step, most of them) takes no level_bits call.
  const int dc_q = av2_dc_quant_QTX(qindex, 0, 0, kQDepth);
  const int ac_q = av2_ac_quant_QTX(qindex, 0, 0, kQDepth);
  const int n = get_txb_wide(tx) * get_txb_high(tx);
  const double scale = static_cast<double>(1 << (QUANT_TABLE_BITS + av2_get_tx_scale(tx)));
  const SCAN_ORDER* const so = get_scan(tx, tx_type);
  const double step_dc = dc_q / scale, step_ac = ac_q / scale;
  const int max_dc = 0xffffff / dc_q, max_ac = 0xffffff / ac_q;
  const double weight = kBitWeight;
  double e_of[1024];
  int16_t level_of[1024];
  for (int pos = 0; pos < n; pos++) {
    const double e = std::abs(static_cast<double>(coeff[pos])) / (pos == 0 ? step_dc : step_ac);
    e_of[pos] = e;
    if (e < 0.5) {
      level_of[pos] = 0;
      continue;
    }
    const int hi = static_cast<int>(std::floor(e + 0.5));
    int level = 0;
    double best = 1e300;
    for (int l : {hi, hi - 1, 0}) {
      if (l < 0) continue;
      const double j = (e - l) * (e - l) + (pos ? psy * std::abs(e - l) : 0) + weight * level_bits(l);
      if (j < best) best = j, level = l;
    }
    const int max_level = pos == 0 ? max_dc : max_ac;
    level_of[pos] = static_cast<int16_t>(std::min(level, max_level));
  }
  int last = n - 1;
  while (last >= 0 && !level_of[so->scan[last]]) last--;
  while (last >= 0 && level_of[so->scan[last]] == 1) {
    int prev = last - 1;
    while (prev >= 0 && !level_of[so->scan[prev]]) prev--;
    const double e = e_of[so->scan[last]];
    const double dj = (e * e - (e - 1) * (e - 1)) + (so->scan[last] ? psy * (e - std::abs(e - 1)) : 0) -
                      weight * (level_bits(1) + 0.6 * (last - prev - 1) + 1.0);
    if (dj >= 0) break;
    level_of[so->scan[last]] = 0;
    last = prev;
  }
  if (g_rd_zero && last >= 0 && zero_wins(n, last + 1, so, psy, [&](int pos) { return e_of[pos]; }, [&](int pos) { return static_cast<int>(level_of[pos]); })) {
    std::fill(level_of, level_of + n, static_cast<int16_t>(0));
    last = -1;
  }
  const int eob = last + 1;  // in scan order
  if (eob_out) *eob_out = eob;
  double j = 0;
  for (int pos = 0; pos < n; pos++) {
    const double e = e_of[pos];
    const int l = level_of[pos];
    j += (e - l) * (e - l) + (pos ? psy * std::abs(e - l) : 0) + (so->iscan[pos] < eob ? weight * level_bits(l) : 0);
  }
  return j;
}
int quantize(const tran_low_t* coeff, tran_low_t* qcoeff, tran_low_t* dqcoeff, TX_SIZE tx,
             int qindex, int plane, TX_TYPE tx_type, double* cost, bool tcq, int bd, double psy) {
  celv2::HotScope prof(celv2::Stage::Quant);
  if (tcq) return tcq_levels(coeff, qcoeff, dqcoeff, tx, qindex, get_scan(tx, tx_type), cost, bd, psy);
  const int dc_q = av2_dc_quant_QTX(qindex, 0, 0, kQDepth);
  const int ac_q = av2_ac_quant_QTX(qindex, 0, 0, kQDepth);
  (void)plane;  // all plane delta-q values are 0 in celv2
  const int n = get_txb_wide(tx) * get_txb_high(tx);
  const double scale = static_cast<double>(1 << (QUANT_TABLE_BITS + av2_get_tx_scale(tx)));
  const SCAN_ORDER* const so = get_scan(tx, tx_type);
  const double step_dc = dc_q / scale, step_ac = ac_q / scale;
  // Keep level * dqv inside the decoder's 24-bit mask.
  const int max_dc = 0xffffff / dc_q, max_ac = 0xffffff / ac_q;
  const double weight = kBitWeight;
  // Each level: the cheapest of round-to-nearest, one less and zero, by squared error (in steps)
  // plus kBitWeight per estimated bit.
  for (int pos = 0; pos < n; pos++) {
    const double e = std::abs(static_cast<double>(coeff[pos])) / (pos == 0 ? step_dc : step_ac);
    const int hi = static_cast<int>(std::floor(e + 0.5));
    int level = 0;
    double best = 1e300;
    for (int l : {hi, hi - 1, 0}) {
      if (l < 0) continue;
      const double j = (e - l) * (e - l) + (pos ? psy * std::abs(e - l) : 0) + weight * level_bits(l);
      if (j < best) best = j, level = l;
    }
    const int max_level = pos == 0 ? max_dc : max_ac;
    if (level > max_level) level = max_level;
    qcoeff[pos] = coeff[pos] < 0 ? -level : level;
  }
  // Trailing +-1 levels: dropping one also drops the zeros coded before it (back to the previous
  // nonzero level in scan order) and moves the end of block, so it pays off when the added
  // distortion is small.
  int last = n - 1;
  while (last >= 0 && !qcoeff[so->scan[last]]) last--;
  while (last >= 0 && std::abs(qcoeff[so->scan[last]]) == 1) {
    int prev = last - 1;
    while (prev >= 0 && !qcoeff[so->scan[prev]]) prev--;
    const int pos = so->scan[last];
    const double e = std::abs(static_cast<double>(coeff[pos])) / (pos == 0 ? step_dc : step_ac);
    const double dj = (e * e - (e - 1) * (e - 1)) + (pos ? psy * (e - std::abs(e - 1)) : 0) -
                      weight * (level_bits(1) + 0.6 * (last - prev - 1) + 1.0);
    if (dj >= 0) break;
    qcoeff[pos] = 0;
    last = prev;
  }
  if (g_rd_zero && last >= 0 &&
      zero_wins(n, last + 1, so, psy, [&](int pos) { return std::abs(static_cast<double>(coeff[pos])) / (pos == 0 ? step_dc : step_ac); },
                [&](int pos) { return std::abs(static_cast<int>(qcoeff[pos])); }))
    std::fill(qcoeff, qcoeff + n, 0);
  int eob = 0;
  for (int pos = 0; pos < n; pos++) {
    dqcoeff[pos] = qcoeff[pos] ? dequant_coeff(qcoeff[pos], pos == 0 ? dc_q : ac_q, tx, bd) : 0;
    if (qcoeff[pos] && so->iscan[pos] + 1 > eob) eob = so->iscan[pos] + 1;
  }
  if (cost) {
    double j = 0;
    for (int pos = 0; pos < n; pos++) {
      const double e = std::abs(static_cast<double>(coeff[pos])) / (pos == 0 ? step_dc : step_ac);
      const int l = std::abs(static_cast<int>(qcoeff[pos]));
      j += (e - l) * (e - l) + (pos ? psy * std::abs(e - l) : 0) + (so->iscan[pos] < eob ? weight * level_bits(l) : 0);
    }
    *cost = j;
  }
  return eob;
}

// fwd_dct() is orthonormal times `scale`, so the loss is the residual's energy scaled less the kept
// coefficients'.
double dropped_cost(const int16_t* res, int stride, TX_SIZE tx, double scale, const tran_low_t* coeff,
                    int qindex) {
  const int w = tx_size_wide[tx], h = tx_size_high[tx];
  if (w <= 32 && h <= 32) return 0;
  // Integer sums: exact, as the double sums of these integers were.
  int64_t all = 0, kept = 0;
  for (int r = 0; r < h; r++)
    for (int c = 0; c < w; c++) all += res[r * stride + c] * res[r * stride + c];
  for (int i = 0; i < std::min(w, 32) * std::min(h, 32); i++) kept += static_cast<int64_t>(coeff[i]) * coeff[i];
  const double step = av2_ac_quant_QTX(qindex, 0, 0, kQDepth) /
                      static_cast<double>(1 << (QUANT_TABLE_BITS + av2_get_tx_scale(tx)));
  return std::max(0.0, static_cast<double>(all) * scale * scale - static_cast<double>(kept)) / (step * step);
}
