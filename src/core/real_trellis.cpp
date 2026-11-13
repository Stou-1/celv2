#include "core/real_trellis.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "core/quant.h"  // kQDepth

extern "C" {
#include "av2/common/hr_coding.h"
#include "av2/common/idct.h"
#include "av2/common/quant_common.h"
#include "av2/common/scan.h"
#include "av2/common/txb_common.h"
}

namespace celv2::real {
namespace {

const std::array<int32_t, CDF_PROB_TOP + 1> kBits = [] {
  std::array<int32_t, CDF_PROB_TOP + 1> t{};
  for (int p = 0; p <= CDF_PROB_TOP; p++)
    t[p] = static_cast<int32_t>(std::llround(-std::log2(std::max(p, 1) / static_cast<double>(CDF_PROB_TOP)) * 512));
  return t;
}();
inline int32_t bits(const avm_cdf_prob* cdf, int v) { return kBits[(v ? cdf[v - 1] : CDF_PROB_TOP) - cdf[v]]; }

int32_t eob_bits(const Block& b, int eob) {
  const FRAME_CONTEXT* const fc = b.fc;
  int eob_pt = 1;
  while (eob_pt + 1 < 12 && av2_eob_group_start[eob_pt + 1] <= eob) eob_pt++;
  const int sym = eob_pt - 1, ms = txsize_log2_minus4[b.tx];
  int32_t r = 0;
  switch (ms) {
    case 0: r = bits(fc->eob_flag_cdf16[b.pl_ctx], sym); break;
    case 1: r = bits(fc->eob_flag_cdf32[b.pl_ctx], sym); break;
    case 2: r = bits(fc->eob_flag_cdf64[b.pl_ctx], sym); break;
    case 3: r = bits(fc->eob_flag_cdf128[b.pl_ctx], sym); break;
    default: {
      const avm_cdf_prob* const cdf = ms == 4 ? fc->eob_flag_cdf256[b.pl_ctx] : ms == 5 ? fc->eob_flag_cdf512[b.pl_ctx] : fc->eob_flag_cdf1024[b.pl_ctx];
      const int low = AVMMIN(sym, EOB_PT_INDEX_COUNT - 1);
      r = bits(cdf, low);
      if (low == EOB_PT_INDEX_COUNT - 1) r += ms == 4 ? 512 : 1024;
    }
  }
  const int extra = eob - av2_eob_group_start[eob_pt], n = av2_eob_offset_bits[eob_pt];
  if (n > 0) r += bits(fc->eob_extra_cdf, (extra >> (n - 1)) & 1) + (n - 1) * 512;
  return r;
}

int32_t low_range(const avm_cdf_prob* cdf, int br) {
  int32_t r = 0;
  for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
    const int k = AVMMIN(br - idx, BR_CDF_SIZE - 1);
    r += bits(cdf, k);
    if (k < BR_CDF_SIZE - 1) break;
  }
  return r;
}

int32_t high_range(int level, int ctx) {  // Golomb bits, whole
  const int m = get_adaptive_param(ctx), k = m + 1, cmax = AVMMIN(m + 4, 6), q = level >> m;
  if (q < cmax) return (q + 1 + m) * 512;
  const int length = get_msb(level - (cmax << m) + (1 << k)) + 1;
  return (cmax + length - 1 - k + length) * 512;
}

// What the base and range symbols code of a (the writer's abs_level()).
int coded_level(const Block& b, int a, int row, int col, TX_CLASS tx_class) {
  if (!b.tcq) return a;
  const int max_br = get_lf_limits(row, col, tx_class, b.plane) ? LF_MAX_BASE_BR_RANGE : MAX_BASE_BR_RANGE;
  return a >= max_br - 1 ? max_br - 1 + ((a - max_br + 1) & 1) : a;
}

// levels_and_signs()'s last coefficient, a > 0: its symbols' bits, *level what levels[] then holds.
int32_t last_bits(const Block& b, int a, int c, int pos, int bwl, int height, TX_CLASS tx_class, int txs_ctx, int* level) {
  const FRAME_CONTEXT* const fc = b.fc;
  const int ctx = get_lower_levels_ctx_eob(bwl, height, c), row = pos >> bwl, col = pos - (row << bwl);
  const bool limits = get_lf_limits(row, col, tx_class, b.plane);
  int32_t r;
  if (b.plane > 0) {
    if (limits) {
      *level = AVMMIN(a, LF_BASE_SYMBOLS - 1);
      return bits(fc->coeff_base_lf_eob_uv_cdf[ctx], *level - 1);
    }
    *level = AVMMIN(a, 3);
    r = bits(fc->coeff_base_eob_uv_cdf[ctx], *level - 1);
    if (*level > NUM_BASE_LEVELS) {
      r += low_range(fc->coeff_br_uv_cdf[0], a - 1 - NUM_BASE_LEVELS);
      *level = AVMMIN(a, MAX_BASE_BR_RANGE);
    }
    return r;
  }
  if (limits) {
    *level = AVMMIN(a, LF_BASE_SYMBOLS - 1);
    r = bits(fc->coeff_base_lf_eob_cdf[txs_ctx][ctx], *level - 1);
    if (*level > LF_NUM_BASE_LEVELS) {
      r += low_range(fc->coeff_br_lf_cdf[get_br_ctx_lf_eob(pos, tx_class)], a - 1 - LF_NUM_BASE_LEVELS);
      *level = AVMMIN(a, LF_MAX_BASE_BR_RANGE);
    }
    return r;
  }
  *level = AVMMIN(a, 3);
  r = bits(fc->coeff_base_eob_cdf[txs_ctx][ctx], *level - 1);
  if (*level > NUM_BASE_LEVELS) {
    r += low_range(fc->coeff_br_cdf[0], a - 1 - NUM_BASE_LEVELS);
    *level = AVMMIN(a, MAX_BASE_BR_RANGE);
  }
  return r;
}

// levels_and_signs()'s code_one(), on a levels array that holds this path's neighbours of pos.
int32_t mid_bits(const Block& b, const uint8_t* levels, int a, int pos, int bwl, bool use_2d, TX_CLASS tx_class, int txs_ctx, int q_i,
                 int* level) {
  const FRAME_CONTEXT* const fc = b.fc;
  const int row = pos >> bwl, col = pos - (row << bwl), plane = b.plane;
  const bool limits = get_lf_limits(row, col, use_2d ? TX_CLASS_2D : tx_class, plane);
  int32_t r;
  if (plane > 0) {
    if (limits) {
      const int ctx = use_2d ? get_lower_levels_ctx_lf_2d_chroma(levels, pos, bwl, plane) : get_lower_levels_lf_ctx_chroma(levels, pos, bwl, tx_class, plane);
      *level = AVMMIN(a, LF_BASE_SYMBOLS - 1);
      return bits(fc->coeff_base_lf_uv_cdf[ctx], *level);
    }
    const int ctx = use_2d ? get_lower_levels_ctx_2d_chroma(levels, pos, bwl, plane) : get_lower_levels_ctx_chroma(levels, pos, bwl, tx_class, plane);
    *level = AVMMIN(a, 3);
    r = bits(fc->coeff_base_uv_cdf[ctx], *level);
    if (*level > NUM_BASE_LEVELS) {
      const int br = use_2d ? get_br_ctx_2d_chroma(levels, pos, bwl) : get_br_ctx_chroma(levels, pos, bwl, tx_class);
      r += low_range(fc->coeff_br_uv_cdf[br], a - 1 - NUM_BASE_LEVELS);
      *level = AVMMIN(a, MAX_BASE_BR_RANGE);
    }
    return r;
  }
  if (limits) {
    const int ctx = use_2d ? get_lower_levels_ctx_lf_2d(levels, pos, bwl) : get_lower_levels_lf_ctx(levels, pos, bwl, tx_class);
    *level = AVMMIN(a, LF_BASE_SYMBOLS - 1);
    r = bits(fc->coeff_base_lf_cdf[txs_ctx][ctx][q_i], *level);
    if (*level > LF_NUM_BASE_LEVELS) {
      const int br = use_2d ? get_br_lf_ctx_2d(levels, pos, bwl) : get_br_lf_ctx(levels, pos, bwl, tx_class);
      r += low_range(fc->coeff_br_lf_cdf[br], a - 1 - LF_NUM_BASE_LEVELS);
      *level = AVMMIN(a, LF_MAX_BASE_BR_RANGE);
    }
    return r;
  }
  const int ctx = use_2d ? get_lower_levels_ctx_2d(levels, pos, bwl, plane) : get_lower_levels_ctx(levels, pos, bwl, tx_class, plane);
  *level = AVMMIN(a, 3);
  r = bits(fc->coeff_base_cdf[txs_ctx][ctx][q_i], *level);
  if (*level > NUM_BASE_LEVELS) {
    const int br = use_2d ? get_br_ctx_2d(levels, pos, bwl) : get_br_ctx(levels, pos, bwl, tx_class);
    r += low_range(fc->coeff_br_cdf[br], a - 1 - NUM_BASE_LEVELS);
    *level = AVMMIN(a, MAX_BASE_BR_RANGE);
  }
  return r;
}

// The sign and high range of a nonzero level: a in full, level as levels[] holds it, *avg the path's hr_level_avg.
int32_t sign_bits(const Block& b, int a, int level, int sign, int c, int pos, int bwl, TX_CLASS tx_class, int* avg) {
  const int row = pos >> bwl, col = pos - (row << bwl), plane = b.plane;
  const bool dc_2dtx = c == 0, dc_hor = col == 0 && tx_class == TX_CLASS_HORIZ, dc_ver = row == 0 && tx_class == TX_CLASS_VERT;
  int32_t r = 512;
  if ((dc_2dtx || dc_hor || dc_ver) && plane == AVM_PLANE_Y) r = bits(b.fc->dc_sign_cdf[get_plane_type(plane)][0][dc_2dtx ? b.dc_sign_ctx : 0], sign);
  const int lf = get_lf_limits(row, col, tx_class, plane);
  const int max_br = lf ? (plane == 0 ? LF_MAX_BASE_BR_RANGE : LF_NUM_BASE_LEVELS + 1) : MAX_BASE_BR_RANGE;
  if (b.tcq ? level >= max_br - 1 : level >= max_br) {
    const int hr = b.tcq ? (a - level) >> 1 : a - level;
    r += high_range(hr, *avg);
    *avg = (*avg + hr) >> 1;
  }
  return r;
}

// The neighbours a level's contexts read, by class (get_nz_mag(), get_br_ctx() and their low-frequency and chroma kin).
struct Nbrs {
  int n;
  int8_t dr[5], dc[5];
};
constexpr Nbrs kNbrs[3] = {
    {5, {0, 1, 1, 0, 2}, {1, 0, 1, 2, 0}},  // TX_CLASS_2D
    {5, {0, 1, 0, 0, 0}, {1, 0, 2, 3, 4}},  // TX_CLASS_HORIZ
    {5, {0, 1, 2, 3, 4}, {1, 0, 0, 0, 0}},  // TX_CLASS_VERT
};

// The farthest scan index ahead of a coefficient that one of its neighbours has, by size and class. get_scan() reads only
// the primary type and every type of a class shares one scan, so one type per class stands for all.
int window(TX_SIZE tx, TX_CLASS cls) {
  static const auto table = [] {
    static constexpr TX_TYPE kRep[3] = {DCT_DCT, H_DCT, V_DCT};  // TX_CLASS_2D, _HORIZ, _VERT
    std::array<std::array<int, 3>, TX_SIZES_ALL> t{};
    for (int s = 0; s < TX_SIZES_ALL; s++) {
      const TX_SIZE ts = static_cast<TX_SIZE>(s);
      const int bwl = get_txb_bwl(ts), width = get_txb_wide(ts), height = get_txb_high(ts);
      for (int k = 0; k < 3; k++) {
        const SCAN_ORDER* const so = get_scan(ts, kRep[k]);
        int most = 1;
        for (int pos = 0; pos < width * height; pos++) {
          const int row = pos >> bwl, col = pos - (row << bwl);
          for (int n = 0; n < kNbrs[k].n; n++) {
            const int r = row + kNbrs[k].dr[n], c = col + kNbrs[k].dc[n];
            if (r < height && c < width) most = std::max(most, so->iscan[(r << bwl) + c] - so->iscan[pos]);
          }
        }
        t[s][k] = most;
      }
    }
    return t;
  }();
  return table[tx][cls];
}

}  // namespace

int64_t lam_of(int qindex, TX_SIZE tx, double weight) {
  const double step = av2_dc_quant_QTX(qindex, 0, 0, kQDepth) / static_cast<double>(1 << (QUANT_TABLE_BITS + av2_get_tx_scale(tx)));
  return std::llround(weight * step * step * 2048);
}

int32_t bits512(const avm_cdf_prob* cdf, int v) { return bits(cdf, v); }

int levels(const Block& b, const tran_low_t* coeff, tran_low_t* q, tran_low_t* dq, int64_t* rate) {
  static const uint8_t next[8][2] = {{0, 4}, {4, 0}, {1, 5}, {5, 1}, {6, 2}, {2, 6}, {7, 3}, {3, 7}};
  constexpr int64_t kInf = std::numeric_limits<int64_t>::max();
  constexpr int kStart = 8;
  const TX_SIZE tx = b.tx;
  const SCAN_ORDER* const so = get_scan(tx, b.type);
  const TX_CLASS tx_class = tx_type_to_class[get_primary_tx_type(b.type)];
  const TX_SIZE txs_ctx = get_txsize_entropy_ctx(tx);
  const int bwl = get_txb_bwl(tx), width = get_txb_wide(tx), height = get_txb_high(tx), n = width * height;
  const int dc_q = av2_dc_quant_QTX(b.qindex, 0, 0, kQDepth), ac_q = av2_ac_quant_QTX(b.qindex, 0, 0, kQDepth);
  const int shift = av2_get_tx_scale(tx), up = QUANT_TABLE_BITS + shift;
  const int states = b.tcq ? 8 : 1;
  // A level's reconstruction: qi half steps under TCQ, whole ones on the plain path.
  auto rebuilt = [&](int qi, int dqv) {
    const int64_t hp = static_cast<int64_t>(qi) * dqv & 0xffffff;
    return static_cast<int64_t>(static_cast<tran_low_t>(ROUND_POWER_OF_TWO_64(hp, QUANT_TABLE_BITS)) >> (shift + (b.tcq ? 1 : 0)));
  };
  std::fill(q, q + n, 0);
  std::fill(dq, dq + tx_size_2d[tx], 0);
  int top = n - 1;
  while (top >= 0) {  // the last coefficient worth half a step
    const int pos = so->scan[top];
    if ((static_cast<int64_t>(std::abs(coeff[pos])) << (up + 1)) >= (pos ? ac_q : dc_q)) break;
    top--;
  }
  if (top < 0) {
    *rate = b.flag[1];
    return 0;
  }
  // J = e^2 << kE + lam x bits512: kE 20 (lam_of()'s scale) unless a block's squares would leave int64, then lam with it.
  int64_t amax = 0;
  for (int c = 0; c <= top; c++) amax = std::max<int64_t>(amax, std::abs(coeff[so->scan[c]]));
  int abits = 0;
  while ((amax + 1) >> abits) abits++;
  const int kE = std::clamp(61 - 11 - 2 * abits, 0, 20);  // n <= 2^10 coefficients, and a margin bit
  const int64_t lam = kE == 20 ? b.lam : (b.lam + (int64_t{1} << (19 - kE))) >> (20 - kE);
  const int win = window(tx, tx_class);
  // Each state's path, by scan index: its levels as levels[] holds them, only [c, c + win] still read.
  static thread_local uint8_t path_lv[2][8][1024 + 1024 + 1];
  static thread_local uint8_t pad[TX_PAD_2D];  // the neighbours of the coefficient at hand, for AVM's context functions
  std::memset(pad, 0, static_cast<size_t>(height + TX_PAD_VER) * (width + TX_PAD_HOR) + TX_PAD_END);
  struct Step {
    int a;
    int from;
  };
  static thread_local Step step[1024][8];
  int cur = 0;
  int64_t path[8], bits_of[8];
  int avg[8];
  std::fill(path, path + 8, kInf);
  const Nbrs& nb = kNbrs[tx_class];
  for (int c = top; c >= 0; c--) {
    const int pos = so->scan[c], row = pos >> bwl, col = pos - (row << bwl);
    const int64_t abs_c = std::abs(coeff[pos]);
    const int sign = coeff[pos] < 0, dqv = pos ? ac_q : dc_q;
    const int max_level = 0xffffff / ((b.tcq ? 2 : 1) * dqv);
    const int64_t d0 = abs_c * abs_c;
    // Candidates per quantizer: its even and odd levels nearest (av2_pre_quant()), or round and one less; zero on its own.
    struct Cand {
      int a;
      int64_t d;  // e^2 - abs_c^2 << kE
    } cand[2][2];
    int nc[2] = {0, 0};
    auto add = [&](int q1, int a) {
      a = std::min(a, max_level);
      if (a <= 0) return;
      for (int k = 0; k < nc[q1]; k++)
        if (cand[q1][k].a == a) return;
      const int64_t e = abs_c - rebuilt(b.tcq ? 2 * a - q1 : a, dqv);
      cand[q1][nc[q1]++] = {a, (e * e - d0) * (int64_t{1} << kE)};
    };
    if (b.tcq) {
      const int base = std::max<int64_t>(1, ((abs_c << (up + 1)) / dqv) - 1);
      for (int qi = base; qi < base + 4; qi++) add(qi & 1, (qi + 1) >> 1);
    } else {
      const int hi = static_cast<int>(((abs_c << (up + 1)) + dqv) / (2 * dqv));
      add(0, hi), add(0, hi - 1);
    }
    int64_t nj[8], nbits[8];
    int na[8], nfrom[8], nlevel[8], navg[8];
    std::fill(nj, nj + 8, kInf);
    auto offer = [&](int ns, int64_t j, int64_t r, int a, int from, int level, int av) {
      if (j < nj[ns]) nj[ns] = j, nbits[ns] = r, na[ns] = a, nfrom[ns] = from, nlevel[ns] = level, navg[ns] = av;
    };
    const bool use_2d = c >= 1 && tx_class == TX_CLASS_2D;
    for (int s = 0; s < states; s++) {
      if (path[s] == kInf) continue;
      const uint8_t* const lv = path_lv[cur][s];
      for (int k = 0; k < nb.n; k++) {
        const int r = row + nb.dr[k], cc = col + nb.dc[k];
        if (r < height && cc < width) pad[get_padded_idx((r << bwl) + cc, bwl)] = lv[so->iscan[(r << bwl) + cc]];
      }
      const int q1 = b.tcq ? tcq_quant(s) : 0;
      int level = 0;
      const int32_t zero = mid_bits(b, pad, 0, pos, bwl, use_2d, tx_class, txs_ctx, q1, &level);
      offer(b.tcq ? next[s][0] : 0, path[s] + lam * zero, bits_of[s] + zero, 0, s, 0, avg[s]);
      for (int k = 0; k < nc[q1]; k++) {
        const int a = cand[q1][k].a;
        int32_t r = mid_bits(b, pad, coded_level(b, a, row, col, tx_class), pos, bwl, use_2d, tx_class, txs_ctx, q1, &level);
        int av = avg[s];
        r += sign_bits(b, a, level, sign, c, pos, bwl, tx_class, &av);
        offer(b.tcq ? next[s][a & 1] : 0, path[s] + cand[q1][k].d + lam * r, bits_of[s] + r, a, s, level, av);
      }
    }
    const int32_t end = eob_bits(b, c + 1);
    for (int k = 0; k < nc[0]; k++) {
      const int a = cand[0][k].a;
      int level = 0, av = 0;
      int32_t r = end + last_bits(b, coded_level(b, a, row, col, tx_class), c, pos, bwl, height, tx_class, txs_ctx, &level);
      r += sign_bits(b, a, level, sign, c, pos, bwl, tx_class, &av);
      offer(b.tcq ? next[0][a & 1] : 0, cand[0][k].d + lam * r, r, a, kStart, level, av);
    }
    const int nxt = cur ^ 1;
    for (int s = 0; s < states; s++) {
      path[s] = nj[s];
      if (nj[s] == kInf) continue;
      bits_of[s] = nbits[s], avg[s] = navg[s];
      step[c][s] = {na[s], nfrom[s]};
      uint8_t* const dst = path_lv[nxt][s];
      if (nfrom[s] == kStart) std::memset(dst + c + 1, 0, win);
      else std::memcpy(dst + c + 1, path_lv[cur][nfrom[s]] + c + 1, win);
      dst[c] = static_cast<uint8_t>(nlevel[s]);
    }
    cur = nxt;
  }
  int s = -1;
  for (int k = 0; k < states; k++)
    if (path[k] < kInf && (s < 0 || path[k] < path[s])) s = k;
  if (s < 0 || lam * b.flag[1] <= path[s] + lam * b.flag[0]) {
    *rate = b.flag[1];
    return 0;
  }
  *rate = bits_of[s] + b.flag[0];
  int end = 0;
  for (int c = 0;; c++) {
    const Step st = step[c][s];
    q[so->scan[c]] = coeff[so->scan[c]] < 0 ? -st.a : st.a;
    s = st.from;
    if (s == kStart) {
      end = c + 1;
      break;
    }
  }
  const int32_t max_value = (1 << (7 + b.bd)) - 1, min_value = -(1 << (7 + b.bd));
  int state = 0;
  for (int c = end - 1; c >= 0; c--) {
    const int pos = so->scan[c], a = std::abs(static_cast<int>(q[pos]));
    if (a) {
      const tran_low_t d = static_cast<tran_low_t>(rebuilt(b.tcq ? 2 * a - tcq_quant(state) : a, pos ? ac_q : dc_q));
      dq[pos] = clamp(q[pos] < 0 ? -d : d, min_value, max_value);
    }
    if (b.tcq) state = next[state][a & 1];
  }
  return end;
}

}  // namespace celv2::real
