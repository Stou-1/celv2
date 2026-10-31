#include "core/rd_pass.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <immintrin.h>

#include "core/quant.h"
#include "core/transform.h"
#include "gpu/inter_common.h"
#include "gpu/sb_blocks.h"

extern "C" {
#include "av2/common/idct.h"
#include "av2/common/quant_common.h"
#include "av2/common/scan.h"
#include "av2/common/txb_common.h"
}

namespace {

constexpr double kInf = 1e300;
// The pass's charge for a compound block over its reference indices (the engine's is 20), and the
// pairs it tries: a reference's best within kPairRatio of the cheapest one's engine cost. Against no
// pairs, 30 frames live action / anime: 1.5 and 24 bits -1.48 / -1.76% BD-rate but 120-240 frames
// anime C +0.10%, anime D +0.16%, anime A +0.46%; 1.2 and 32 / 36 / 40 bits -1.64 / -1.40%,
// -1.28 / -1.07%, -1.14 / -0.83% (30 frames), anime C -0.03% each, anime D -0.18 / -0.17 / -0.21%,
// anime A -0.02 / -0.08 / -0.13%. anime A q130 7.7 -> 6.9 fps.
constexpr int kCompoundBits = 36;
constexpr double kPairRatio = 1.2;

// Squared error in quantizer steps and estimated bits.
struct Cost {
  double d = 0, bits = 0;
  double j(const RdPassParams& p) const { return d + p.bit_weight * bits; }
  Cost& operator+=(const Cost& o) { return d += o.d, bits += o.bits, *this; }
};

TX_SIZE tx_of(int w, int h) {
  for (int t = 0; t < TX_SIZES_ALL; t++)
    if (tx_size_wide[t] == w && tx_size_high[t] == h) return static_cast<TX_SIZE>(t);
  return TX_INVALID;
}

// One transform of res (stride w of the block) quantized as the coder does: squared error in steps
// (with what a 64-point one drops), and the bits of its levels and end of block - weighed like the
// side bits, unlike quantize_cost()'s RDOQ weight.
Cost tx_cost(const int16_t* res, int stride, TX_SIZE t, const RdPassParams& p) {
  tran_low_t coeff[1024], q[1024], dq[1024];
  fwd_dct(res, stride, t, p.scale[t], coeff);
  const int eob = quantize(coeff, q, dq, t, p.qindex, 0, DCT_DCT, nullptr, p.tcq);
  const double div = static_cast<double>(1 << (QUANT_TABLE_BITS + av2_get_tx_scale(t)));
  const double step_dc = av2_dc_quant_QTX(p.qindex, 0, 0, AVM_BITS_8) / div;
  const double step_ac = av2_ac_quant_QTX(p.qindex, 0, 0, AVM_BITS_8) / div;
  const int n = get_txb_wide(t) * get_txb_high(t);
  const int16_t* const iscan = get_scan(t, DCT_DCT)->iscan;
  Cost c;
  c.bits = eob ? 2 + 1.5 * std::log2(static_cast<double>(eob)) : 1;
  for (int pos = 0; pos < n; pos++) {
    const double e = (coeff[pos] - dq[pos]) / (pos == 0 ? step_dc : step_ac);
    c.d += e * e;
    if (iscan[pos] < eob) c.bits += coeff_level_bits(std::abs(static_cast<int>(q[pos])));
  }
  c.d += dropped_cost(res, stride, t, p.scale[t], coeff, p.qindex);
  return c;
}

// subpel_px() of every sample of the w x h block at (x, y) of r moved by (qx, qy) quarter pels, into
// pred (stride w): the same sums done separably, each tap row's horizontal sum once.
void predict(const celv2::Plane& r, int x, int y, int w, int h, int qx, int qy, uint8_t* pred) {
  const int ix = x + (qx >> 2), iy = y + (qy >> 2), fx = qx & 3, fy = qy & 3;
  int tx[8], ty[8], j0 = 8, j1 = -1, k0 = 8, k1 = -1;
  for (int k = 0; k < 8; k++) {
    tx[k] = celv2::subpel_tap(fx, k), ty[k] = celv2::subpel_tap(fy, k);
    if (tx[k]) j0 = std::min(j0, k), j1 = k;
    if (ty[k]) k0 = std::min(k0, k), k1 = k;
  }
  alignas(32) int hs[(64 + 7) * 64];
  const int rows = h + k1 - k0;
  const bool inside = ix + j0 - 3 >= 0 && ix + w - 1 + j1 - 3 < r.w;
  // 8 samples at a time where the block is 8-wide (integer sums: the same values in any order).
  const bool wide = w % 8 == 0;
  for (int i = 0; i < rows; i++) {
    const int yy = std::clamp(iy + k0 - 3 + i, 0, r.h - 1);
    const uint8_t* const row = r.p + static_cast<size_t>(yy) * r.stride;
    int* const out = hs + i * w;
    if (inside && wide) {
      for (int j = 0; j < w; j += 8) {
        __m256i s = _mm256_setzero_si256();
        for (int k = j0; k <= j1; k++) {
          const __m256i px = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(row + ix + j + k - 3)));
          s = _mm256_add_epi32(s, _mm256_mullo_epi32(px, _mm256_set1_epi32(tx[k])));
        }
        _mm256_store_si256(reinterpret_cast<__m256i*>(out + j), s);
      }
      continue;
    }
    for (int j = 0; j < w; j++) {
      int s = 0;
      if (inside)
        for (int k = j0; k <= j1; k++) s += tx[k] * row[ix + j + k - 3];
      else
        for (int k = j0; k <= j1; k++) s += tx[k] * row[std::clamp(ix + j + k - 3, 0, r.w - 1)];
      out[j] = s;
    }
  }
  for (int i = 0; i < h; i++) {
    if (wide) {
      for (int j = 0; j < w; j += 8) {
        __m256i v = _mm256_set1_epi32(1 << 13);
        for (int k = k0; k <= k1; k++)
          v = _mm256_add_epi32(v, _mm256_mullo_epi32(_mm256_set1_epi32(ty[k]),
                                                     _mm256_load_si256(reinterpret_cast<const __m256i*>(hs + (i + k - k0) * w + j))));
        // >> 14, then saturating packs: the clamp to 0..255
        const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(_mm256_srai_epi32(v, 14), _mm256_setzero_si256()), 0b1000);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(pred + i * w + j), _mm_packus_epi16(_mm256_castsi256_si128(p16), _mm_setzero_si128()));
      }
      continue;
    }
    for (int j = 0; j < w; j++) {
      int v = 0;
      for (int k = k0; k <= k1; k++) v += ty[k] * hs[(i + k - k0) * w + j];
      v = (v + (1 << 13)) >> 14;
      pred[i * w + j] = static_cast<uint8_t>(v < 0 ? 0 : v > 255 ? 255 : v);
    }
  }
}

// The block's luma cost: its largest transform, or for a 64-wide or -high block also 32x32 parts
// (what TileCoder::choose_tx_partition() picks from).
Cost residual_cost(const int16_t* res, int w, int h, const RdPassParams& p) {
  Cost c = tx_cost(res, w, tx_of(w, h), p);
  if (w == 64 || h == 64) {
    Cost parts;
    for (int y = 0; y < h && parts.j(p) < c.j(p); y += 32)
      for (int x = 0; x < w; x += 32) parts += tx_cost(res + y * w + x, w, TX_32X32, p);
    if (parts.j(p) < c.j(p)) c = parts;
  }
  return c;
}

struct Pass {
  const celv2::Plane& s;
  const std::vector<celv2::Plane>& refs;
  int sb_x, sb_y;
  const RdPassParams& p;
  const std::vector<BlockCandidates>& cands;
  bool inter_frame;

  // The source sample, the edge repeated past the frame.
  int src(int y, int x) const { return s.at(y, x); }

  Cost inter_cost(const BlockDecision& d, int x, int y, int w, int h) const {
    int16_t res[64 * 64];
    const celv2::Plane& r0 = refs[d.ref];
    const int mx = d.mv_col / 2, my = d.mv_row / 2;  // quarter pels
    // AV2 has no compound 8x4 / 4x8 blocks (is_comp_ref_allowed): the coder predicts them from d.ref.
    const bool comp = d.ref2 != BlockDecision::kSingleRef && std::min(w, h) >= 8;
    const int mx2 = d.mv2_col / 2, my2 = d.mv2_row / 2;
    uint8_t p0[64 * 64], p1[64 * 64];
    predict(r0, x, y, w, h, mx, my, p0);
    if (comp) predict(refs[d.ref2], x, y, w, h, mx2, my2, p1);
    for (int i = 0; i < h; i++)
      for (int j = 0; j < w; j++) {
        const int k = i * w + j;
        const int pred = comp ? (p0[k] + p1[k] + 1) >> 1 : p0[k];
        res[k] = static_cast<int16_t>(src(y + i, x + j) - pred);
      }
    Cost c = residual_cost(res, w, h, p);
    c.bits += comp ? kCompoundBits + d.ref + d.ref2 : celv2::inter_bits(d.ref);
    return c;
  }

  Cost intra_cost(int mode, int x, int y, int w, int h) const {
    int16_t res[64 * 64];
    for (int i = 0; i < h; i++)
      for (int j = 0; j < w; j++)
        res[i * w + j] = static_cast<int16_t>(src(y + i, x + j) - celv2::intra_pred(s, x, y, w, h, mode, i, j));
    Cost c = residual_cost(res, w, h, p);
    c.bits += celv2::mode_bits(mode) + (inter_frame ? celv2::kIntraInInterBits : 0);
    return c;
  }

  // Block b's best choice and its cost (kInf when it has none); its inter choice also from the
  // inter candidates of the blocks in pool.
  double block(int b, BlockDecision& out, const std::vector<int>& pool) const {
    const BlockCandidates& c = cands[b];
    int x, y, w, h;
    celv2::sb_block_geom(b, &x, &y, &w, &h);
    x += sb_x, y += sb_y;
    double best = kInf;
    if (c.intra_cost != INT64_MAX) {
      best = intra_cost(static_cast<int>(c.intra), x, y, w, h).j(p);
      out = BlockDecision{};
      out.w4 = static_cast<uint8_t>(w / 4), out.h4 = static_cast<uint8_t>(h / 4);
      out.intra = c.intra;
    }
    if (c.inter_cost != INT64_MAX) {
      const double j = inter_cost(c.inter, x, y, w, h).j(p);
      if (j < best) best = j, out = c.inter;
      BlockDecision tried[17];  // c.inter, the pool's (at most 10) and the pairs (at most 6)
      int n = 0;
      tried[n++] = c.inter;
      auto same = [](const BlockDecision& a, const BlockDecision& e) {
        return a.ref == e.ref && a.mv_col == e.mv_col && a.mv_row == e.mv_row && a.ref2 == e.ref2 &&
               (a.ref2 == BlockDecision::kSingleRef || (a.mv2_col == e.mv2_col && a.mv2_row == e.mv2_row));
      };
      for (int k : pool) {
        if (k == b || cands[k].inter_cost == INT64_MAX) continue;
        if (std::any_of(tried, tried + n, [&](const BlockDecision& t) { return same(t, cands[k].inter); })) continue;
        tried[n++] = cands[k].inter;
        BlockDecision d = cands[k].inter;
        d.w4 = static_cast<uint8_t>(w / 4), d.h4 = static_cast<uint8_t>(h / 4);
        const double jk = inter_cost(d, x, y, w, h).j(p);
        if (jk < best) best = jk, out = d;
      }
      // The cheapest reference's best MV averaged with each other reference's best within kPairRatio
      // of its engine cost (as single candidates instead they lost: +0.5 to +1.5% BD-rate).
      int first = -1;
      for (int r = 0; r < BlockCandidates::kRefs; r++)
        if (c.ref_cost[r] != UINT32_MAX && (first < 0 || c.ref_cost[r] < c.ref_cost[first])) first = r;
      for (int r = 0; r < BlockCandidates::kRefs && first >= 0 && std::min(w, h) >= 8; r++) {
        if (r == first || c.ref_cost[r] == UINT32_MAX || c.ref_cost[r] > c.ref_cost[first] * kPairRatio ||
            !celv2::compound_pair_ok(first, r, static_cast<int>(refs.size())))
          continue;
        BlockDecision d = c.inter;
        d.ref = static_cast<uint8_t>(first), d.mv_col = c.ref_mv_col[first], d.mv_row = c.ref_mv_row[first];
        d.ref2 = static_cast<uint8_t>(r), d.mv2_col = c.ref_mv_col[r], d.mv2_row = c.ref_mv_row[r];
        if (std::any_of(tried, tried + n, [&](const BlockDecision& t) { return same(t, d); })) continue;
        tried[n++] = d;
        const double jk = inter_cost(d, x, y, w, h).j(p);
        if (jk < best) best = jk, out = d;
      }
    }
    return best + p.bit_weight * p.block_bits;
  }

  // The square's subtree (appended to tree / blocks) and its cost: NONE, HORZ, VERT, then SPLIT
  // (first strict minimum), as the engines choose.
  double square(int level, int qx, int qy, std::vector<Part>& tree, std::vector<BlockDecision>& blocks) const {
    const int b = celv2::sb_square_id(level, qx, qy) * 5, size = 64 >> level;
    const double part = p.bit_weight * celv2::kPartBits;
    BlockDecision d[5];
    // Inter MVs to try on each block: those the engine found for the square's own blocks, its
    // parent and its children (120 frames 1080p: -0.6 to -2.1% BD-rate, anime A q130 9.9 -> 7.7 fps;
    // the parent alone gave about 80% of it for 16% of the time).
    std::vector<int> pool;
    if (inter_frame) {
      for (int k = 0; k < 5; k++) pool.push_back(b + k);
      if (level > 0) pool.push_back(celv2::sb_square_id(level - 1, qx / 2, qy / 2) * 5);
      if (level < 3)
        for (int q = 0; q < 4; q++) pool.push_back(celv2::sb_square_id(level + 1, 2 * qx + (q & 1), 2 * qy + (q >> 1)) * 5);
    }
    double best = kInf;
    int choice = 3;
    if (size <= celv2::kMaxBlockSize || (inter_frame && size <= celv2::kMaxInterBlockSize)) {
      const double n = block(b, d[0], pool);
      const double hz = part + block(b + 1, d[1], pool) + block(b + 2, d[2], pool);
      const double vt = part + block(b + 3, d[3], pool) + block(b + 4, d[4], pool);
      best = n, choice = 0;
      if (hz < best) best = hz, choice = 1;
      if (vt < best) best = vt, choice = 2;
    }
    std::vector<Part> st;
    std::vector<BlockDecision> sbk;
    if (size > 8) {
      double split = part;
      for (int q = 0; q < 4 && split < best; q++)
        split += square(level + 1, 2 * qx + (q & 1), 2 * qy + (q >> 1), st, sbk);
      if (split < best) best = split, choice = 3;
    }
    auto leaf = [&](int k) { blocks.push_back(d[k]); };
    switch (choice) {
      case 0: tree.push_back(Part::None), leaf(0); break;
      case 1: tree.push_back(Part::Horz), tree.push_back(Part::None), tree.push_back(Part::None), leaf(1), leaf(2); break;
      case 2: tree.push_back(Part::Vert), tree.push_back(Part::None), tree.push_back(Part::None), leaf(3), leaf(4); break;
      default:
        tree.push_back(Part::Split);
        tree.insert(tree.end(), st.begin(), st.end());
        blocks.insert(blocks.end(), sbk.begin(), sbk.end());
    }
    return best;
  }
};

}  // namespace

void rd_pass_sb(const Frame420& src, const std::vector<const Frame420*>& refs, int sb_x, int sb_y,
                const RdPassParams& p, SbDecision& sb) {
  if (sb.cands.size() != static_cast<size_t>(celv2::kSbBlocks)) return;
  const celv2::Plane s{src.y.data(), src.stride_y(), src.pw, src.ph};
  std::vector<celv2::Plane> rp;
  for (const Frame420* r : refs) rp.push_back(celv2::Plane{r->y.data(), r->stride_y(), r->pw, r->ph});
  const Pass pass{s, rp, sb_x, sb_y, p, sb.cands, !refs.empty()};
  std::vector<Part> tree;
  std::vector<BlockDecision> blocks;
  pass.square(0, 0, 0, tree, blocks);
  sb.tree = std::move(tree);
  sb.blocks = std::move(blocks);
}
