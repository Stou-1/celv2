// Q0 of docs/superpowers/specs/2026-11-06-real-cost-trellis-design.md: levels by the coder's own bits (X_ORACLE_Q), a measurement
// in the replay. Each path carries its levels, so a level's rate comes by the contexts block_writer.cpp writes with; rates are
// -log2 of the fixed CDFs given, as SymbolLog::bits() adds them. Mirrors BlockWriter's eob_symbols() and levels_and_signs():
// BlockWriter::oracle_bits() and X_ORACLE_QCHECK prove it.
// shortcut: full levels copied per state and step, slow; Q1 takes avmenc's diagonals or the spec's short way.

namespace real {

const std::vector<double> kBits = [] {
  std::vector<double> t(CDF_PROB_TOP + 1);
  for (int p = 0; p <= CDF_PROB_TOP; p++) t[p] = -std::log2(std::max(p, 1) / static_cast<double>(CDF_PROB_TOP));
  return t;
}();
double bits(const avm_cdf_prob* cdf, int v) { return kBits[(v ? cdf[v - 1] : CDF_PROB_TOP) - cdf[v]]; }

struct Block {
  const FRAME_CONTEXT* fc;
  TX_SIZE tx;
  TX_TYPE type;
  int plane, pl_ctx, dc_sign_ctx, qindex, bd;
  bool tcq;
  double flag[2];  // the all-zero flag's bits: coded, all zero
  double weight;   // of a bit, in squared DC steps
};

double eob_bits(const Block& b, int eob) {
  const FRAME_CONTEXT* const fc = b.fc;
  int eob_pt = 1;
  while (eob_pt + 1 < 12 && av2_eob_group_start[eob_pt + 1] <= eob) eob_pt++;
  const int sym = eob_pt - 1, ms = txsize_log2_minus4[b.tx];
  double r = 0;
  switch (ms) {
    case 0: r = bits(fc->eob_flag_cdf16[b.pl_ctx], sym); break;
    case 1: r = bits(fc->eob_flag_cdf32[b.pl_ctx], sym); break;
    case 2: r = bits(fc->eob_flag_cdf64[b.pl_ctx], sym); break;
    case 3: r = bits(fc->eob_flag_cdf128[b.pl_ctx], sym); break;
    default: {
      const avm_cdf_prob* const cdf = ms == 4 ? fc->eob_flag_cdf256[b.pl_ctx] : ms == 5 ? fc->eob_flag_cdf512[b.pl_ctx] : fc->eob_flag_cdf1024[b.pl_ctx];
      const int low = AVMMIN(sym, EOB_PT_INDEX_COUNT - 1);
      r = bits(cdf, low);
      if (low == EOB_PT_INDEX_COUNT - 1) r += ms == 4 ? 1 : 2;
    }
  }
  const int extra = eob - av2_eob_group_start[eob_pt], n = av2_eob_offset_bits[eob_pt];
  if (n > 0) r += bits(fc->eob_extra_cdf, (extra >> (n - 1)) & 1) + (n - 1);
  return r;
}

double low_range(const avm_cdf_prob* cdf, int br) {
  double r = 0;
  for (int idx = 0; idx < COEFF_BASE_RANGE; idx += BR_CDF_SIZE - 1) {
    const int k = AVMMIN(br - idx, BR_CDF_SIZE - 1);
    r += bits(cdf, k);
    if (k < BR_CDF_SIZE - 1) break;
  }
  return r;
}

double high_range(int level, int ctx) {
  const int m = get_adaptive_param(ctx), k = m + 1, cmax = AVMMIN(m + 4, 6), q = level >> m;
  if (q < cmax) return q + 1 + m;
  const int length = get_msb(level - (cmax << m) + (1 << k)) + 1;
  return cmax + length - 1 - k + length;
}

// What the base and range symbols code of a (the writer's abs_level()).
int coded_level(const Block& b, int a, int row, int col, TX_CLASS tx_class) {
  if (!b.tcq) return a;
  const int max_br = get_lf_limits(row, col, tx_class, b.plane) ? LF_MAX_BASE_BR_RANGE : MAX_BASE_BR_RANGE;
  return a >= max_br - 1 ? max_br - 1 + ((a - max_br + 1) & 1) : a;
}

// levels_and_signs()'s last coefficient, a > 0: its symbols' bits, *level what levels[] then holds.
double last_bits(const Block& b, int a, int c, int pos, int bwl, int height, TX_CLASS tx_class, int txs_ctx, int* level) {
  const FRAME_CONTEXT* const fc = b.fc;
  const int ctx = get_lower_levels_ctx_eob(bwl, height, c), row = pos >> bwl, col = pos - (row << bwl);
  const bool limits = get_lf_limits(row, col, tx_class, b.plane);
  double r;
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

// levels_and_signs()'s code_one().
double mid_bits(const Block& b, const uint8_t* levels, int a, int pos, int bwl, bool use_2d, TX_CLASS tx_class, int txs_ctx, int q_i, int* level) {
  const FRAME_CONTEXT* const fc = b.fc;
  const int row = pos >> bwl, col = pos - (row << bwl), plane = b.plane;
  const bool limits = get_lf_limits(row, col, use_2d ? TX_CLASS_2D : tx_class, plane);
  double r;
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
double sign_bits(const Block& b, int a, int level, int sign, int c, int pos, int bwl, TX_CLASS tx_class, int* avg) {
  const int row = pos >> bwl, col = pos - (row << bwl), plane = b.plane;
  const bool dc_2dtx = c == 0, dc_hor = col == 0 && tx_class == TX_CLASS_HORIZ, dc_ver = row == 0 && tx_class == TX_CLASS_VERT;
  double r = 1;
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

// Levels of coeff by J = squared error in DC steps + weight * bits, from the last coefficient worth a level down to DC, in TCQ's
// eight states or the plain path's one: each state keeps its best path, a path may start (end of block) at any position, and
// all zero is one more end, by the flag (av2_trellis_quant(), av2_optimize_txb_new()). *rate: the bits of what it chose.
int levels(const Block& b, const tran_low_t* coeff, tran_low_t* q, tran_low_t* dq, double* rate) {
  static const uint8_t next[8][2] = {{0, 4}, {4, 0}, {1, 5}, {5, 1}, {6, 2}, {2, 6}, {7, 3}, {3, 7}};
  constexpr double kInf = std::numeric_limits<double>::infinity();
  constexpr int kStart = 8;
  const TX_SIZE tx = b.tx;
  const SCAN_ORDER* const so = get_scan(tx, b.type);
  const TX_CLASS tx_class = tx_type_to_class[get_primary_tx_type(b.type)];
  const TX_SIZE txs_ctx = get_txsize_entropy_ctx(tx);
  const int bwl = get_txb_bwl(tx), width = get_txb_wide(tx), height = get_txb_high(tx), n = width * height;
  const int dc_q = av2_dc_quant_QTX(b.qindex, 0, 0, kQDepth), ac_q = av2_ac_quant_QTX(b.qindex, 0, 0, kQDepth);
  const int shift = av2_get_tx_scale(tx);
  const double scale = static_cast<double>(1 << (QUANT_TABLE_BITS + shift));
  const double dc_step = dc_q / scale, inv = 1 / (dc_step * dc_step);
  const int states = b.tcq ? 8 : 1;
  // A level's reconstruction: qi half steps under TCQ, whole ones on the plain path.
  auto rebuilt = [&](int qi, int dqv) {
    const int64_t hp = static_cast<int64_t>(qi) * dqv & 0xffffff;
    return static_cast<tran_low_t>(ROUND_POWER_OF_TWO_64(hp, QUANT_TABLE_BITS)) >> (shift + (b.tcq ? 1 : 0));
  };
  std::fill(q, q + n, 0);
  std::fill(dq, dq + n, 0);
  int top = n - 1;
  while (top >= 0) {
    const int pos = so->scan[top];
    if (std::abs(static_cast<double>(coeff[pos])) >= 0.5 * (pos ? ac_q : dc_q) / scale) break;
    top--;
  }
  if (top < 0) {
    *rate = b.flag[1];
    return 0;
  }
  const size_t lv_size = static_cast<size_t>(height + TX_PAD_VER) * (width + TX_PAD_HOR) + TX_PAD_END;
  static thread_local uint8_t buf[2][8][TX_PAD_2D];
  struct Step {
    int a;
    int from;
  };
  static thread_local Step step[1024][8];
  int cur = 0;
  double path[8], bits_of[8];
  int avg[8];
  std::fill(path, path + 8, kInf);
  for (int c = top; c >= 0; c--) {
    const int pos = so->scan[c], row = pos >> bwl, col = pos - (row << bwl);
    const double abs_c = std::abs(static_cast<double>(coeff[pos]));
    const int sign = coeff[pos] < 0, dqv = pos ? ac_q : dc_q;
    const int max_level = 0xffffff / ((b.tcq ? 2 : 1) * dqv);
    const double d0 = abs_c * abs_c * inv;
    // Candidates per quantizer: its even and odd levels nearest (av2_pre_quant()), or round and one less; zero comes on its own.
    struct Cand {
      int a;
      double d;
    } cand[2][2];
    int nc[2] = {0, 0};
    auto add = [&](int q1, int a) {
      a = std::min(a, max_level);
      if (a <= 0) return;
      for (int k = 0; k < nc[q1]; k++)
        if (cand[q1][k].a == a) return;
      const double e = abs_c - rebuilt(b.tcq ? 2 * a - q1 : a, dqv);
      cand[q1][nc[q1]++] = {a, e * e * inv - d0};
    };
    if (b.tcq) {
      const int base = std::max(1, static_cast<int>(abs_c * 2 * scale / dqv) - 1);
      for (int qi = base; qi < base + 4; qi++) add(qi & 1, (qi + 1) >> 1);
    } else {
      const int hi = static_cast<int>(abs_c * scale / dqv + 0.5);
      add(0, hi), add(0, hi - 1);
    }
    double nj[8], nb[8];
    int na[8], nfrom[8], nlevel[8], navg[8];
    std::fill(nj, nj + 8, kInf);
    auto offer = [&](int ns, double j, double r, int a, int from, int level, int av) {
      if (j < nj[ns]) nj[ns] = j, nb[ns] = r, na[ns] = a, nfrom[ns] = from, nlevel[ns] = level, navg[ns] = av;
    };
    const bool use_2d = c >= 1 && tx_class == TX_CLASS_2D;
    for (int s = 0; s < states; s++) {
      if (path[s] == kInf) continue;
      const uint8_t* const lv = set_levels(buf[cur][s], width);
      const int q1 = b.tcq ? tcq_quant(s) : 0;
      int level = 0;
      const double zero = mid_bits(b, lv, 0, pos, bwl, use_2d, tx_class, txs_ctx, q1, &level);
      offer(b.tcq ? next[s][0] : 0, path[s] + b.weight * zero, bits_of[s] + zero, 0, s, 0, avg[s]);
      for (int k = 0; k < nc[q1]; k++) {
        const int a = cand[q1][k].a;
        double r = mid_bits(b, lv, coded_level(b, a, row, col, tx_class), pos, bwl, use_2d, tx_class, txs_ctx, q1, &level);
        int av = avg[s];
        r += sign_bits(b, a, level, sign, c, pos, bwl, tx_class, &av);
        offer(b.tcq ? next[s][a & 1] : 0, path[s] + cand[q1][k].d + b.weight * r, bits_of[s] + r, a, s, level, av);
      }
    }
    const double end = eob_bits(b, c + 1);
    for (int k = 0; k < nc[0]; k++) {
      const int a = cand[0][k].a;
      int level = 0, av = 0;
      double r = end + last_bits(b, coded_level(b, a, row, col, tx_class), c, pos, bwl, height, tx_class, txs_ctx, &level);
      r += sign_bits(b, a, level, sign, c, pos, bwl, tx_class, &av);
      offer(b.tcq ? next[0][a & 1] : 0, cand[0][k].d + b.weight * r, r, a, kStart, level, av);
    }
    const int nxt = cur ^ 1;
    for (int s = 0; s < states; s++) {
      path[s] = nj[s];
      if (nj[s] == kInf) continue;
      bits_of[s] = nb[s], avg[s] = navg[s];
      step[c][s] = {na[s], nfrom[s]};
      if (nfrom[s] == kStart) std::memset(buf[nxt][s], 0, lv_size);
      else std::memcpy(buf[nxt][s], buf[cur][nfrom[s]], lv_size);
      set_levels(buf[nxt][s], width)[get_padded_idx(pos, bwl)] = static_cast<uint8_t>(nlevel[s]);
    }
    cur = nxt;
  }
  int s = -1;
  for (int k = 0; k < states; k++)
    if (path[k] < kInf && (s < 0 || path[k] < path[s])) s = k;
  if (s < 0 || b.weight * b.flag[1] <= path[s] + b.weight * b.flag[0]) {
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
      const tran_low_t d = rebuilt(b.tcq ? 2 * a - tcq_quant(state) : a, pos ? ac_q : dc_q);
      dq[pos] = clamp(q[pos] < 0 ? -d : d, min_value, max_value);
    }
    if (b.tcq) state = next[state][a & 1];
  }
  return end;
}

}  // namespace real
