#include "core/rd_pass.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>

#include "core/intra_block.h"
#include "core/residual.h"
#include "core/profile.h"
#include "core/psy.h"
#include "core/quant.h"
#include "core/rd_int.h"
#include "core/transform.h"
#include "gpu/inter_common.h"
#include "gpu/sb_blocks.h"
#ifdef CELV2_ORACLE
#include "oracle.h"
#endif

extern "C" {
#include "av2/common/idct.h"
#include "av2/common/quant_common.h"
#include "av2/common/scan.h"
#include "av2/common/txb_common.h"
}

namespace {

// Costs are integers (C17 step 1, core/rd_int.h): distortion in J (1/(1024 dqv^2) of a squared step), bits in
// 1/512 bit, a trial's J = distortion + lam x bits, lam = llround(2 bit_weight dqv^2) for the superblock's qindex.
namespace rdi = celv2::rdi;
constexpr int64_t kInf = rdi::kInf;
// The pass's charge for a compound block over its reference indices (the engine's is 20), and the
// pairs it tries: a reference's best within kPairRatio of the cheapest one's engine cost. Against no
// pairs, 30 frames live action / anime: 1.5 and 24 bits -1.48 / -1.76% BD-rate but 120-240 frames
// anime C +0.10%, anime D +0.16%, anime A +0.46%; 1.2 and 32 / 36 / 40 bits -1.64 / -1.40%,
// -1.28 / -1.07%, -1.14 / -0.83% (30 frames), anime C -0.03% each, anime D -0.18 / -0.17 / -0.21%,
// anime A -0.02 / -0.08 / -0.13%. anime A q130 7.7 -> 6.9 fps.
constexpr int kCompoundBits = 36;
constexpr double kPairRatio = 1.2;
// The cost of a 64-wide or -high intra block, scaled: its estimate predicts all of it from the
// outer edge, where the coder's 32x32 transforms predict from the parts before them. One frame,
// q100-190, against no 64 intra blocks (AIR / Dragon Maid / AM / Flash f20 / f100): x1 -12.0 / -9.4 /
// +0.1 / +1.9 / -0.9%, x1.1 -12.3 / -9.3 / -1.2 / +1.3 / -1.3%, x1.25 -9.7 / -9.3 / -1.4 / +1.1 / -1.2%;
// +24 bits instead -10.4 / -8.9 / -0.0 / +1.5 / -1.5%. x1.1, as J + J / 10.
int64_t intra64_scaled(int64_t j) { return rdi::add(j, j / 10); }
// A palette block's index map, in bits per sample and bit of colour index (rough: runs code cheaply).
// Flash 30 frames q100-190 against --screen 0: 1 -3.24%, 0.5 -4.00%, 0.25 -4.32%, 0.15 -4.35%, 0.1
// -4.34%, 0.05 -4.27%.
constexpr double kPaletteMapBits = 0.15;
// An IntraBC block's flags and vector, and what its copy of a reconstruction (not of the source)
// costs. The same Flash, map bits 0.5: 12 -4.01%, 24 -4.00%, 48 -4.06%.
constexpr double kIntraBcBits = 24;
// The side prices: a compound block's bias over its priced symbols (kCompoundBits held 36 over the reference
// indices). Those are an approximation, not the writer's: the comp flag and NEW_NEWMV from the CDFs, but each
// reference as a single one (the single_ref table twice, not the compound reference tree), the bias swept. 33
// frames, 8 sources, at bit weight 0.025, BD-rate YUV / Y against the constants: 20 -0.70 / -0.43%, 30 -0.46 / -0.14%.
constexpr int kSideCompBias = 20;

// Distortion in J and bits in 1/512 bit.
struct Cost {
  int64_t d = 0, bits = 0;
  int64_t j(int64_t lam) const { return rdi::add(d, lam * bits); }
  Cost& operator+=(const Cost& o) { return d = rdi::add(d, o.d), bits += o.bits, *this; }
};

TX_SIZE tx_of(int w, int h) {
  for (int t = 0; t < TX_SIZES_ALL; t++)
    if (tx_size_wide[t] == w && tx_size_high[t] == h) return static_cast<TX_SIZE>(t);
  return TX_INVALID;
}

// One transform of res (stride w of the block) quantized as the coder does: squared error in steps
// (with what a 64-point one drops), and the bits of its levels and end of block - weighed like the
// side bits, unlike quantize_cost()'s RDOQ weight.
// psy (--psy-rd): the source block (stride as res) and its weight times the strength; 0 / null: none.
// trellis (--psy-trellis): quantize()'s psy.
// CELV2_RD_STATS: what this thread's superblock has costed so far (hashes), cleared per superblock.
thread_local std::unordered_set<uint64_t> t_seen_tx, t_seen_inter;
uint64_t mix(uint64_t h, uint64_t v) { return (h ^ v) * 0x100000001b3ull; }

// A tx_cost() call for the counts: a repeat when its size and residual match an earlier one's (the same cost
// unless --psy-* weighs it).
void note_tx(const int16_t* res, int stride, TX_SIZE t, bool weighted) {
  const int w = tx_size_wide[t], h = tx_size_high[t];
  celv2::count(celv2::Count::TxCalls);
  celv2::count(celv2::Count::TxSamples, static_cast<uint64_t>(w) * h);
  if (t == TX_32X32) celv2::count(celv2::Count::Tx32Calls);
  if (weighted) return celv2::count(celv2::Count::TxWeighted);
  uint64_t k = mix(0xcbf29ce484222325ull, t);
  for (int i = 0; i < h; i++)
    for (int j = 0; j < w; j++) k = mix(k, static_cast<uint16_t>(res[i * stride + j]));
  if (t_seen_tx.insert(k).second) return;
  celv2::count(celv2::Count::TxRepeats);
  celv2::count(celv2::Count::TxRepeatSamples, static_cast<uint64_t>(w) * h);
  if (t == TX_32X32) celv2::count(celv2::Count::Tx32Repeats);
}

// The rank of mode `pick` among all intra modes by SATD (0: the lowest; ties to the lower mode).
int satd_rank(const celv2::Plane& s, int x, int y, int w, int h, int pick) {
  int64_t satd[kIntraModes];
  for (int m = 0; m < kIntraModes; m++) satd[m] = celv2::block_mode_satd(s, x, y, w, h, m);
  int rank = 0;
  for (int m = 0; m < kIntraModes; m++) rank += satd[m] < satd[pick] || (satd[m] == satd[pick] && m < pick);
  return rank;
}

// C20 (the speed-first spec's step 2): a residual this superblock has costed before takes the first call's cost
// (the same size, residual and frame parameters give the same cost: the same bytes). Per thread, cleared per
// superblock; off under --psy-rd / --psy-trellis (their costs read the source) and with CELV2_TX_MEMO=0.
// Step 0: identical residuals were 38% of S1's transformed samples, 73% of Dragon Maid's.
struct TxMemo {
  static constexpr uint32_t kSlots = 8192;               // a superblock costs about 1300 transforms (S1)
  static constexpr size_t kPool = size_t{1} << 20;       // stored samples, at most (2 MB)
  struct Slot {
    uint64_t key = 0;
    uint32_t gen = 0, at = 0;
    TX_SIZE t = TX_4X4;
    int qindex = 0;  // and the parameters tx_cost() reads, so no stale entry can match
    bool tcq = false;
    Cost c;
  };
  std::vector<Slot> slots = std::vector<Slot>(kSlots);
  std::vector<int16_t> pool;
  uint32_t gen = 1, used = 0;
  void clear() { gen++, used = 0, pool.clear(); }
};
// The pass starts its threads anew each frame (Pipeline::collect), so memos are kept for the process: a thread
// borrows one and gives it back when it ends. A memo made per thread cost more than it saved (S1 -20% fps: the
// allocations stalled the threads, 32 cores busy fell to 21).
std::mutex g_memo_mu;
std::vector<std::unique_ptr<TxMemo>> g_memo_free;
struct MemoLease {
  TxMemo* m = nullptr;
  TxMemo* get() {
    if (!m) {
      {
        const std::lock_guard<std::mutex> lock(g_memo_mu);
        if (!g_memo_free.empty()) m = g_memo_free.back().release(), g_memo_free.pop_back();
      }
      if (!m) m = new TxMemo;
      m->clear();
    }
    return m;
  }
  ~MemoLease() {
    if (!m) return;
    const std::lock_guard<std::mutex> lock(g_memo_mu);
    g_memo_free.emplace_back(m);
  }
};
thread_local MemoLease t_memo;

bool tx_memo_on() {
  static const bool on = [] {
    const char* e = std::getenv("CELV2_TX_MEMO");
    return !(e && *e == '0');
  }();
  return on;
}

// CELV2_TX_DUMP=<file>: tx_cost()'s inputs, every CELV2_TX_DUMP_EVERY'th call (default 16), for the benches of
// tools/c17 (C17 step 0b, the TCQ plan's step 0). A record: TX_SIZE, tcq (one byte each), qindex (int16), scale
// (double), the w x h residual (int16, rows packed). <file>.counts gets each size's calls and uncached calls at exit.
struct TxDump {
  FILE* f = nullptr;
  std::string path;
  uint32_t every = 16;
  std::mutex mu;
  std::atomic<uint64_t> n{0};
  std::atomic<uint64_t> calls[TX_SIZES_ALL] = {}, uncached[TX_SIZES_ALL] = {};
  TxDump() {
    const char* e = std::getenv("CELV2_TX_DUMP");
    if (!e || !*e) return;
    path = e;
    f = std::fopen(e, "wb");
    if (const char* k = std::getenv("CELV2_TX_DUMP_EVERY")) every = std::max(1, std::atoi(k));
  }
  ~TxDump() {
    if (!f) return;
    std::fclose(f);
    if (FILE* c = std::fopen((path + ".counts").c_str(), "w")) {
      std::fprintf(c, "every %u\n", every);
      for (int t = 0; t < TX_SIZES_ALL; t++)
        if (calls[t]) std::fprintf(c, "%d %llu %llu\n", t, static_cast<unsigned long long>(calls[t]),
                                   static_cast<unsigned long long>(uncached[t]));
      std::fclose(c);
    }
  }
  void note(const int16_t* res, int stride, TX_SIZE t, const RdPassParams& p) {
    calls[t]++;
    if (n++ % every) return;
    const int w = tx_size_wide[t], h = tx_size_high[t];
    const uint8_t head[2] = {static_cast<uint8_t>(t), static_cast<uint8_t>(p.tcq)};
    const int16_t q = static_cast<int16_t>(p.qindex);
    std::lock_guard<std::mutex> lock(mu);
    std::fwrite(head, 1, 2, f), std::fwrite(&q, 2, 1, f), std::fwrite(&p.scale[t], 8, 1, f);
    for (int i = 0; i < h; i++) std::fwrite(res + i * stride, 2, w, f);
  }
};
TxDump& tx_dump() {
  static TxDump d;
  return d;
}

uint64_t res_hash(const int16_t* res, int stride, int w, int h, TX_SIZE t) {
  uint64_t k = 0x9e3779b97f4a7c15ull ^ static_cast<uint64_t>(t);
  for (int i = 0; i < h; i++)
    for (int j = 0; j < w; j += 4) {  // every size is a multiple of 4 wide
      uint64_t v;
      std::memcpy(&v, res + i * stride + j, 8);
      k = (k ^ v) * 0xff51afd7ed558ccdull;
      k ^= k >> 29;
    }
  return k;
}

Cost tx_cost_uncached(const int16_t* res, int stride, TX_SIZE t, const RdPassParams& p, const int16_t* src,
                      double psy, double trellis);

Cost tx_cost(const int16_t* res, int stride, TX_SIZE t, const RdPassParams& p, const int16_t* src = nullptr,
             double psy = 0, double trellis = 0) {
  celv2::RdTimer timer(celv2::RdPart::TxCost);
  if (celv2::rd_stats()) note_tx(res, stride, t, (src && psy > 0) || trellis > 0);
  if (tx_dump().f) tx_dump().note(res, stride, t, p);
  if ((src && psy > 0) || trellis > 0 || !tx_memo_on()) return tx_cost_uncached(res, stride, t, p, src, psy, trellis);
  TxMemo& m = *t_memo.get();
  const int w = tx_size_wide[t], h = tx_size_high[t];
  uint64_t key;
  uint32_t s;
  {
    celv2::RdTimer th(celv2::RdPart::TxHash);
    key = res_hash(res, stride, w, h, t);
    s = static_cast<uint32_t>(key) & (TxMemo::kSlots - 1);
    for (; m.slots[s].gen == m.gen; s = (s + 1) & (TxMemo::kSlots - 1)) {
      const TxMemo::Slot& e = m.slots[s];
      if (e.key != key || e.t != t || e.qindex != p.qindex || e.tcq != p.tcq) continue;
      bool same = true;
      for (int i = 0; i < h && same; i++) same = !std::memcmp(m.pool.data() + e.at + i * w, res + i * stride, w * 2);
      if (same) return e.c;
    }
  }
  const Cost c = tx_cost_uncached(res, stride, t, p, src, psy, trellis);
  if (m.used * 4 < TxMemo::kSlots * 3 && m.pool.size() + w * h <= TxMemo::kPool) {
    TxMemo::Slot& e = m.slots[s];
    e.key = key, e.gen = m.gen, e.t = t, e.c = c, e.at = static_cast<uint32_t>(m.pool.size());
    e.qindex = p.qindex, e.tcq = p.tcq;
    for (int i = 0; i < h; i++) m.pool.insert(m.pool.end(), res + i * stride, res + i * stride + w);
    m.used++;
  }
  return c;
}

// What a 64-point transform leaves out, in squared coefficients (0 below 64): fwd_dct() is orthonormal times scale, so
// the residual's energy scaled less the kept coefficients' (dropped_cost() in steps). The transform stays fwd_dct()
// (the coder's), in float: AVM's integer DCT in its place, at the coder's scale per frequency, lost 0.1-0.4% BD-rate
// on anime (AVM's rows are not quite orthogonal; docs/reports/2026-11-09-c17-step1.md). CPU only, as is the transform.
int64_t dropped_energy(const int16_t* res, int stride, TX_SIZE t, double scale, const int32_t* coeff) {
  const int w = tx_size_wide[t], h = tx_size_high[t];
  if (w <= 32 && h <= 32) return 0;
  int64_t all = 0, kept = 0;
  for (int r = 0; r < h; r++)
    for (int c = 0; c < w; c++) all += res[r * stride + c] * res[r * stride + c];
  for (int i = 0; i < std::min(w, 32) * std::min(h, 32); i++) kept += static_cast<int64_t>(coeff[i]) * coeff[i];
  return std::max<int64_t>(0, std::llround(static_cast<double>(all) * scale * scale - static_cast<double>(kept)));
}
Cost tx_cost_uncached(const int16_t* res, int stride, TX_SIZE t, const RdPassParams& p, const int16_t* src,
                      double psy, double trellis) {
  celv2::RdTimer timer(celv2::RdPart::TxUncached);
  if (tx_dump().f) tx_dump().uncached[t]++;
  int32_t coeff[1024], q[1024], dq[1024];
  int64_t dropped;  // a 64-point transform's left-out coefficients, squared
  {
    celv2::RdTimer td(celv2::RdPart::Fdct);
    fwd_dct(res, stride, t, p.scale[t], coeff);
    dropped = dropped_energy(res, stride, t, p.scale[t], coeff);
  }
  rdi::TxCost tc;
  {
    celv2::RdTimer tq(celv2::RdPart::Quant);
    if (trellis > 0) {  // --psy-trellis (CPU only): quantize()'s levels for its term, costed as the rest
      const int eob = quantize(coeff, q, dq, t, p.qindex, 0, DCT_DCT, nullptr, p.tcq, 8, trellis);
      tc = rdi::sums(coeff, dropped, q, eob, t, p.qindex, p.tcq, dq);
    } else {
      tc = rdi::quant_cost(coeff, dropped, t, p.qindex, p.tcq, g_rd_zero, q, dq);
    }
  }
  Cost c{tc.d, tc.bits};
  if (src && psy > 0) {  // --psy-rd (CPU only): its term in squared steps, the source by the same transform
    int32_t S[1024];
    fwd_dct(src, stride, t, p.scale[t], S);
    const double dqv = av2_ac_quant_QTX(p.qindex, 0, 0, kQDepth);
    const double step = dqv / (1 << (QUANT_TABLE_BITS + av2_get_tx_scale(t)));
    const double term = psy * celv2::kPsyScale * celv2::psy_term(S, coeff, dq, get_txb_wide(t) * get_txb_high(t), step);
    c.d = rdi::add(c.d, std::llround(term * 1024 * dqv * dqv));
  }
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
Cost residual_cost(const int16_t* res, int w, int h, const RdPassParams& p, int64_t lam, const int16_t* src = nullptr,
                   double psy = 0, double trellis = 0) {
  Cost c = tx_cost(res, w, tx_of(w, h), p, src, psy, trellis);
#ifdef CELV2_ORACLE
  if (std::min(w, h) < 32) return c;  // avmenc's 64x16 and the like: no 32x32 parts
#endif
  if (w == 64 || h == 64) {
    Cost parts;
    for (int y = 0; y < h && parts.j(lam) < c.j(lam); y += 32)
      for (int x = 0; x < w; x += 32) parts += tx_cost(res + y * w + x, w, TX_32X32, p, src ? src + y * w + x : nullptr, psy, trellis);
    if (parts.j(lam) < c.j(lam)) c = parts;
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
  const Frame420& f;
  const std::vector<int16_t>& wide_mv;  // SbDecision::wide_mv
  const celv2::Plane* chroma_src;      // U, V (W4)
  const std::vector<std::array<celv2::Plane, 2>>& chroma_refs;  // each reference's U, V
  int64_t lam;  // rdi::lam_of(p.qindex, p.bit_weight)

  int64_t of_bits(double bits) const { return lam * rdi::bits512(bits); }

  // The source sample, the edge repeated past the frame.
  int src(int y, int x) const { return s.at(y, x); }

  // --psy-trellis: the block's weight times the strength (0: off).
  double trellis_of(int x, int y, int w, int h) const {
    if (p.trellis <= 0 || !p.map) return 0;
    return p.trellis * celv2::psy_weight(*p.map, x, y, w, h, p.psy_lines ? celv2::PsyRule::Line : celv2::PsyRule::Texture);
  }

  // --psy-rd: the block's weight times the strength (0: off), and its source samples into out.
  double psy_of(int x, int y, int w, int h, int16_t* out) const {
    if (p.psy <= 0 || !p.map) return 0;
    const double wt = p.psy * celv2::psy_weight(*p.map, x, y, w, h, p.psy_lines ? celv2::PsyRule::Line : celv2::PsyRule::Texture);
    if (wt <= 0) return 0;
    for (int i = 0; i < h; i++)
      for (int j = 0; j < w; j++) out[i * w + j] = static_cast<int16_t>(src(y + i, x + j));
    return wt;
  }

  // The predictor of a reference's MV (shortcut for the coder's MV stack): the enclosing block's engine MV for it.
  void pred_mv(int pred, int r, int& row, int& col) const {
    const BlockCandidates& pc = cands[pred];
    if (r < BlockCandidates::kRefs && pc.ref_cost[r] != UINT32_MAX) {
      row = pc.ref_mv_row[r], col = pc.ref_mv_col[r];
    } else if (pc.inter_cost != INT64_MAX && pc.inter.ref == r) {
      row = pc.inter.mv_row, col = pc.inter.mv_col;
    } else {
      row = col = 0;
    }
  }
  // The side prices: an inter block's side bits (1/512 bit) from the frame's CDFs: NEARMV where its MV is block
  // pred's for its reference, else NEWMV. The MV difference is not priced: from the CDFs' shell classes, against this
  // predictor or the squares left and above too, it lost on anime (docs/design/tried-not-kept.md).
  int64_t inter_side(const BlockDecision& d, bool comp, bool sub8, int pred) const {
    const SideCosts& t = *p.side;
    if (comp)
      return t.inter_block + t.compound + t.ref[std::min<int>(d.ref, 7)] + t.ref[std::min<int>(d.ref2, 7)] + t.new_new +
             kSideCompBias * 512;
    int row, col;
    pred_mv(pred, d.ref, row, col);
    // 8x4 / 4x8: the writer codes no skip_mode, is_inter (forced, offset_started) or comp flag (is_comp_ref_allowed);
    // a compound candidate of that size comes here as single (comp is false), as the writer codes it from d.ref.
    const int flags = sub8 ? t.skip_txfm : t.inter_block + t.single;
    return flags + t.ref[std::min<int>(d.ref, 7)] + (d.mv_row == row && d.mv_col == col ? t.nearmv : t.newmv);
  }

  Cost inter_cost(const BlockDecision& d, int x, int y, int w, int h, int pred) const {
    if (celv2::rd_stats()) {
      celv2::count(celv2::Count::InterCalls);
      uint64_t k = 0xcbf29ce484222325ull;
      for (const int v : {x, y, w, h, int{d.ref}, int{d.mv_col}, int{d.mv_row}, int{d.ref2}, int{d.mv2_col}, int{d.mv2_row}})
        k = mix(k, static_cast<uint32_t>(v));
      if (!t_seen_inter.insert(k).second) celv2::count(celv2::Count::InterRepeats);
    }
    celv2::RdTimer timer(celv2::RdPart::InterCost);
    int16_t res[64 * 64];
    const celv2::Plane& r0 = refs[d.ref];
    const int mx = d.mv_col / 2, my = d.mv_row / 2;  // quarter pels
    // AV2 has no compound 8x4 / 4x8 blocks (is_comp_ref_allowed): the coder predicts them from d.ref.
    // shortcut: 8x4/4x8 only; thin 4xN (4x16, 16x4; ORACLE leaf() alone) codes skip_mode and comp, priced here as single.
    const bool comp = d.ref2 != BlockDecision::kSingleRef && std::min(w, h) >= 8;
    const int mx2 = d.mv2_col / 2, my2 = d.mv2_row / 2;
    uint8_t p0[64 * 64], p1[64 * 64];
    {
      celv2::RdTimer tp(celv2::RdPart::InterPred);
      predict(r0, x, y, w, h, mx, my, p0);
      if (comp) predict(refs[d.ref2], x, y, w, h, mx2, my2, p1);
    }
    {
      celv2::RdTimer tr(celv2::RdPart::InterRes);
      if (comp) celv2::average_into(p0, p1, w * h);
      celv2::block_residual(s, x, y, w, h, p0, res);
    }
    int16_t sblk[64 * 64];
    double psy, trellis;
    {
      celv2::RdTimer tq(celv2::RdPart::PsyOf);
      psy = psy_of(x, y, w, h, sblk), trellis = trellis_of(x, y, w, h);
    }
    Cost c = residual_cost(res, w, h, p, lam, sblk, psy, trellis);
    if (p.side) c.bits += inter_side(d, comp, std::min(w, h) < 8, pred);
    else c.bits += 512 * (comp ? kCompoundBits + d.ref + d.ref2 : celv2::inter_bits(d.ref));
    if (p.chroma && std::min(w, h) >= 8)
      for (int pl = 0; pl < 2; pl++) {
        // shortcut: chroma MVs rounded to quarter pels, where the coder has eighths; enough to measure.
        const int cw = w / 2, ch = h / 2;
        {
          celv2::RdTimer tp(celv2::RdPart::InterPred);
          predict(chroma_refs[d.ref][pl], x / 2, y / 2, cw, ch, mx / 2, my / 2, p0);
          if (comp) predict(chroma_refs[d.ref2][pl], x / 2, y / 2, cw, ch, mx2 / 2, my2 / 2, p1);
        }
        {
          celv2::RdTimer tr(celv2::RdPart::InterRes);
          if (comp) celv2::average_into(p0, p1, cw * ch);
          celv2::block_residual(chroma_src[pl], x / 2, y / 2, cw, ch, p0, res);
        }
        c += residual_cost(res, cw, ch, p, lam);
      }
    return c;
  }

  Cost intra_cost(int mode, int x, int y, int w, int h) const {
    celv2::RdTimer timer(celv2::RdPart::IntraCost);
    int16_t res[64 * 64];
    uint8_t pred[64 * 64];
    {
      celv2::RdTimer tp(celv2::RdPart::IntraPred);
      celv2::intra_pred_block(s, x, y, w, h, mode, pred);  // intra_pred() / dc_value()'s values, the edges read once
      celv2::block_residual(s, x, y, w, h, pred, res);
    }
    int16_t sblk[64 * 64];
    double psy, trellis;
    {
      celv2::RdTimer tq(celv2::RdPart::PsyOf);
      psy = psy_of(x, y, w, h, sblk), trellis = trellis_of(x, y, w, h);
    }
    Cost c = residual_cost(res, w, h, p, lam, sblk, psy, trellis);
    c.bits += 512 * (celv2::mode_bits(mode) + (inter_frame ? celv2::kIntraInInterBits : 0));
    if (p.chroma && std::min(w, h) >= 8)
      for (int pl = 0; pl < 2; pl++) {
        // shortcut: the luma mode on chroma, where the coder searches its own and CfL.
        const celv2::Plane& cs = chroma_src[pl];
        const int cw = w / 2, ch = h / 2, cx = x / 2, cy = y / 2;
        celv2::intra_pred_block(cs, cx, cy, cw, ch, mode, pred);
        celv2::block_residual(cs, cx, cy, cw, ch, pred, res);
        c += residual_cost(res, cw, ch, p, lam);
      }
    return c;
  }

  // Bits of the block as a palette, or -1 where it is not one: its colours and index map.
  double palette_bits(int x, int y, int w, int h) const {
    int v[9], n = 0;
    for (int i = 0; i < h; i++)
      for (int j = 0; j < w; j++) {
        const int s0 = src(y + i, x + j);
        if (std::find(v, v + n, s0) != v + n) continue;
        if (n == 8) return -1;
        v[n++] = s0;
      }
    if (n < 2) return -1;
    return celv2::mode_bits(0) + (inter_frame ? celv2::kIntraInInterBits : 0) + 8.0 * n + kPaletteMapBits * w * h * std::log2(n);
  }

  // Block b's best choice and its cost (kInf when it has none); its inter choice also from the
  // inter candidates of the blocks in pool.
  // pred: the block whose engine MVs predict this one's (SideCosts).
  int64_t block(int b, BlockDecision& out, const std::vector<int>& pool, int pred) const {
    const BlockCandidates& c = cands[b];
    int x, y, w, h;
    celv2::sb_block_geom(b, &x, &y, &w, &h);
    x += sb_x, y += sb_y;
    int64_t best = kInf;
#ifdef CELV2_ORACLE
    static const bool g11 = std::getenv("X_G11") != nullptr;  // G1.1's gate: inter frames score these intra blocks too
#else
    constexpr bool g11 = false;
#endif
    if (c.intra_cost == INT64_MAX && (!inter_frame || g11) && p.intra64 && std::max(w, h) == 64) {
      out = BlockDecision{};  // no engine candidate this large: every mode scored here
      out.w4 = static_cast<uint8_t>(w / 4), out.h4 = static_cast<uint8_t>(h / 4);
      for (int m = 0; m < kIntraModes; m++) {
        const int64_t jm = intra_cost(m, x, y, w, h).j(lam);
        if (jm < best) best = jm, out.intra = static_cast<IntraMode>(m);
      }
      if (celv2::rd_stats()) {
        const int rank = satd_rank(s, x, y, w, h, static_cast<int>(out.intra));
        celv2::count(celv2::Count::Intra64Blocks);
        if (rank >= 3) celv2::count(celv2::Count::Intra64Miss3);
        if (rank >= 4) celv2::count(celv2::Count::Intra64Miss4);
      }
      best = intra64_scaled(best);
    } else if (c.intra_cost != INT64_MAX && !(inter_frame && std::min(w, h) < 8)) {
      // shortcut: inter frames' 8x4 / 4x8 take no intra: 4:2:0 offset_started forces them inter (the writer's NEARMV); 4:4:4 would.
      best = intra_cost(static_cast<int>(c.intra), x, y, w, h).j(lam);
      out = BlockDecision{};
      out.w4 = static_cast<uint8_t>(w / 4), out.h4 = static_cast<uint8_t>(h / 4);
      out.intra = c.intra;
      for (int m = 0; m < kIntraModes && p.all_intra_modes; m++) {
        if (m == static_cast<int>(c.intra)) continue;
        const int64_t jm = intra_cost(m, x, y, w, h).j(lam);
        if (jm < best) best = jm, out.intra = static_cast<IntraMode>(m);
      }
      if (p.all_intra_modes && celv2::rd_stats()) {
        const int rank = satd_rank(s, x, y, w, h, static_cast<int>(out.intra));
        celv2::count(celv2::Count::RdIntraBlocks);
        if (rank >= 3) celv2::count(celv2::Count::RdIntraMiss3);
        if (rank >= 4) celv2::count(celv2::Count::RdIntraMiss4);
      }
    }
    if (c.inter_cost != INT64_MAX) {
      const int64_t j = inter_cost(c.inter, x, y, w, h, pred).j(lam);
      if (j < best) best = j, out = c.inter;
      BlockDecision tried[17 + 7 + 7 + 21];  // c.inter, the pool's (at most 10), the pairs (at most 6); W1, W2 and W6's
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
        const int64_t jk = inter_cost(d, x, y, w, h, pred).j(lam);
        if (jk < best) best = jk, out = d;
      }
      auto single = [&](int r, int col, int row) {
        BlockDecision d = c.inter;
        d.ref = static_cast<uint8_t>(r), d.mv_col = static_cast<int16_t>(col), d.mv_row = static_cast<int16_t>(row);
        d.ref2 = BlockDecision::kSingleRef, d.mv2_col = d.mv2_row = 0;
        if (std::any_of(tried, tried + n, [&](const BlockDecision& t) { return same(t, d); })) return;
        tried[n++] = d;
        const int64_t jk = inter_cost(d, x, y, w, h, pred).j(lam);
        if (jk < best) best = jk, out = d;
      };
      for (int r = 0; r < BlockCandidates::kRefs && p.per_ref; r++)
        if (c.ref_cost[r] != UINT32_MAX) single(r, c.ref_mv_col[r], c.ref_mv_row[r]);
      for (int r = 0; r < static_cast<int>(refs.size()) && !wide_mv.empty(); r++) {
        const int16_t* const m = &wide_mv[(static_cast<size_t>(b) * BlockCandidates::kRefs + r) * 2];
        if (m[0] != SbDecision::kNoMv) single(r, m[0], m[1]);
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
        const int64_t jk = inter_cost(d, x, y, w, h, pred).j(lam);
        if (jk < best) best = jk, out = d;
      }
      for (int a = 0; a < BlockCandidates::kRefs && p.all_pairs && std::min(w, h) >= 8; a++)
        for (int e = a + 1; e < BlockCandidates::kRefs; e++) {
          if (c.ref_cost[a] == UINT32_MAX || c.ref_cost[e] == UINT32_MAX || !celv2::compound_pair_ok(a, e, static_cast<int>(refs.size()))) continue;
          BlockDecision d = c.inter;
          d.ref = static_cast<uint8_t>(a), d.mv_col = c.ref_mv_col[a], d.mv_row = c.ref_mv_row[a];
          d.ref2 = static_cast<uint8_t>(e), d.mv2_col = c.ref_mv_col[e], d.mv2_row = c.ref_mv_row[e];
          if (std::any_of(tried, tried + n, [&](const BlockDecision& t) { return same(t, d); })) continue;
          tried[n++] = d;
          const int64_t jk = inter_cost(d, x, y, w, h, pred).j(lam);
          if (jk < best) best = jk, out = d;
        }
    }
    if (p.screen && std::min(w, h) >= 8 && std::max(w, h) <= 64 && x + w <= s.w && y + h <= s.h) {
      const double bits = palette_bits(x, y, w, h);
      if (bits >= 0 && of_bits(bits) < best) {
        best = of_bits(bits);
        out = BlockDecision{};
        out.w4 = static_cast<uint8_t>(w / 4), out.h4 = static_cast<uint8_t>(h / 4);
        out.intra = IntraMode::DC;
      }
    }
    if (p.ibc && std::min(w, h) >= 8 && std::max(w, h) <= 64 && w + h < 128) {
      const int sb = p.ibc->band_rows(), left = x / sb * sb, top = y / sb * sb;
      const bool global = !inter_frame;  // key frames reach every superblock row above
      int dist = -1;
      p.ibc->repeats(f, x, y, w, h, global ? 0 : std::max(0, left - 2 * sb), global ? f.width : left + sb, [&](int px, int py) {
        // Not coded yet, or past the wavefront above (TileCoder::intrabc_candidates()).
        if (py >= top ? py + h > y && px + w > x : py + h > top || (px + w - 1) / sb > x / sb + y / sb - (py + h - 1) / sb) return true;
        dist = std::abs(px - x) + std::abs(py - y);
        return false;
      }, global);
      const int64_t j = of_bits(celv2::mode_bits(0) + (inter_frame ? celv2::kIntraInInterBits : 0) + kIntraBcBits + 2 * std::log2(dist + 1.0));
      if (dist >= 0 && j < best) {
        best = j;
        out = BlockDecision{};
        out.w4 = static_cast<uint8_t>(w / 4), out.h4 = static_cast<uint8_t>(h / 4);
        out.intra = IntraMode::DC;
      }
    }
    return rdi::add(best, of_bits(p.block_bits));
  }

  // The square's subtree (appended to tree / blocks) and its cost: NONE, HORZ, VERT, then SPLIT
  // (first strict minimum), as the engines choose.
  int64_t square(int level, int qx, int qy, std::vector<Part>& tree, std::vector<BlockDecision>& blocks) const {
    const int b = celv2::sb_square_id(level, qx, qy) * 5, size = 64 >> level;
    const int64_t part = lam * 512 * celv2::kPartBits;
    // The side prices: the partition symbols' bits; without them NONE 0, any other kPartBits.
    const SideCosts* const t = p.side;
    auto sq_bits = [&](int k) { return t ? lam * t->part_sq[level][k] : k ? part : 0; };
    auto half_bits = [&](int dir, int split) { return t ? lam * t->part_half[level][dir][split] : split ? part : 0; };
    // The side prices' MV predictor: the square's own block takes its parent's (the 64x64 its own), a half the
    // square's.
    const int parent = level > 0 ? celv2::sb_square_id(level - 1, qx / 2, qy / 2) * 5 : b;
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
    // The quarters' subtrees, each searched once: SPLIT takes all four, and (AV2 partitions each half
    // again) a half of HORZ / VERT may be its two quarters instead of one block.
    std::vector<Part> qt[4];
    std::vector<BlockDecision> qb[4];
    int64_t qc[4] = {kInf, kInf, kInf, kInf};
    if (size > 8)
      for (int q = 0; q < 4; q++) qc[q] = square(level + 1, 2 * qx + (q & 1), 2 * qy + (q >> 1), qt[q], qb[q]);
    int64_t best = kInf;
    int choice = 3;
    bool halves_split[2] = {false, false};  // the chosen HORZ / VERT's halves taken as their quarters
    if (size <= celv2::kMaxBlockSize || ((inter_frame || p.intra64) && size <= celv2::kMaxInterBlockSize)) {
      best = rdi::add(block(b, d[0], pool, parent), sq_bits(0)), choice = 0;
      // HORZ: halves 1 (quarters 0, 1) and 2 (quarters 2, 3); VERT: 3 (0, 2) and 4 (1, 3).
      const int quarters[4][2] = {{0, 1}, {2, 3}, {0, 2}, {1, 3}};
      for (int dir = 0; dir < 2; dir++) {
        int64_t cost = sq_bits(1 + dir);
        bool split[2];
        for (int h = 0; h < 2; h++) {
          const int64_t whole = rdi::add(block(b + 1 + 2 * dir + h, d[1 + 2 * dir + h], pool, b), half_bits(dir, 0));
          const int* qs = quarters[2 * dir + h];
          const int64_t two = rdi::add(rdi::add(half_bits(dir, 1), qc[qs[0]]), qc[qs[1]]);
          split[h] = two < whole;
          cost = rdi::add(cost, split[h] ? two : whole);
        }
        if (cost < best) best = cost, choice = 1 + dir, halves_split[0] = split[0], halves_split[1] = split[1];
      }
    }
    if (size > 8) {
      const int64_t all = rdi::add(rdi::add(rdi::add(rdi::add(sq_bits(3), qc[0]), qc[1]), qc[2]), qc[3]);
      if (all < best) best = all, choice = 3;
    }
    auto quarter = [&](int q) {
      tree.insert(tree.end(), qt[q].begin(), qt[q].end());
      blocks.insert(blocks.end(), qb[q].begin(), qb[q].end());
    };
    switch (choice) {
      case 0: tree.push_back(Part::None), blocks.push_back(d[0]); break;
      case 1:
      case 2: {
        const int dir = choice - 1;
        const int quarters[4][2] = {{0, 1}, {2, 3}, {0, 2}, {1, 3}};
        tree.push_back(dir == 0 ? Part::Horz : Part::Vert);
        for (int h = 0; h < 2; h++)
          if (halves_split[h]) {
            tree.push_back(dir == 0 ? Part::Vert : Part::Horz);  // a 2:1 half into its two squares
            quarter(quarters[2 * dir + h][0]), quarter(quarters[2 * dir + h][1]);
          } else {
            tree.push_back(Part::None), blocks.push_back(d[1 + 2 * dir + h]);
          }
        break;
      }
      default:
        tree.push_back(Part::Split);
        for (int q = 0; q < 4; q++) quarter(q);
    }
    return best;
  }

#ifdef CELV2_ORACLE
  // L1: the leaf at (bx, by) in the superblock, w x h, as square() would pick it. A shape the engines have no
  // candidates for (4:1, 8:1, 4x4) takes the inter choices of the square around it, and its intra mode, or on
  // a key frame the best of all, each scored on the leaf.
  BlockDecision leaf(int bx, int by, int w, int h) const {
    BlockDecision out;
    const int x = sb_x + bx, y = sb_y + by;
    if (x >= s.w || y >= s.h) return out;  // past the frame: a stand-in
    const int size = std::max({w, h, 8});
    int level = 0;
    while ((64 >> level) > size) level++;
    const int qx = bx / size, qy = by / size, b = celv2::sb_square_id(level, qx, qy) * 5;
    std::vector<int> pool;
    if (inter_frame) {
      for (int k = 0; k < 5; k++) pool.push_back(b + k);
      if (level > 0) pool.push_back(celv2::sb_square_id(level - 1, qx / 2, qy / 2) * 5);
      if (level < 3)
        for (int q = 0; q < 4; q++) pool.push_back(celv2::sb_square_id(level + 1, 2 * qx + (q & 1), 2 * qy + (q >> 1)) * 5);
    }
    const int lo = std::min(w, h);
    if ((w == h && w >= 8) || (lo * 2 == std::max(w, h) && lo >= 4)) {
      block(celv2::sb_block_index(bx, by, w, h), out, pool, b);
      return out;
    }
    out.w4 = static_cast<uint8_t>(w / 4), out.h4 = static_cast<uint8_t>(h / 4);
    const BlockCandidates& c = cands[b];
    int64_t best = kInf;
    if (!inter_frame) {
      for (int m = 0; m < kIntraModes; m++) {
        const int64_t j = intra_cost(m, x, y, w, h).j(lam);
        if (j < best) best = j, out.intra = static_cast<IntraMode>(m);
      }
    } else if (c.intra_cost != INT64_MAX && (lo >= 8 || w == h)) {  // 4x16 / 16x4 as block()'s 8x4 / 4x8; 4x4 is intra
      best = intra_cost(static_cast<int>(c.intra), x, y, w, h).j(lam), out.intra = c.intra;
    }
    for (int k : pool) {
      if (cands[k].inter_cost == INT64_MAX) continue;
      BlockDecision d = cands[k].inter;
      d.w4 = out.w4, d.h4 = out.h4;
      const int64_t j = inter_cost(d, x, y, w, h, b).j(lam);
      if (j < best) best = j, out = d;
    }
    return out;
  }

  // The imported tree from pos, for the node at (x, y), w x h: its leaves into out.
  void leaves(const std::vector<Part>& tree, size_t& pos, int x, int y, int w, int h, std::vector<BlockDecision>& out) const {
    const Part part = tree[pos++];
    if (part == Part::None) {
      out.push_back(leaf(x, y, w, h));
    } else if (part == Part::Horz) {
      leaves(tree, pos, x, y, w, h / 2, out), leaves(tree, pos, x, y + h / 2, w, h / 2, out);
    } else if (part == Part::Vert) {
      leaves(tree, pos, x, y, w / 2, h, out), leaves(tree, pos, x + w / 2, y, w / 2, h, out);
    } else {
      for (int i = 0; i < 4; i++) leaves(tree, pos, x + (i & 1) * w / 2, y + (i >> 1) * h / 2, w / 2, h / 2, out);
    }
  }
#endif
};

// CELV2_WIDE W2: every block's best MV per reference: 8x8 SADs at every whole pel within +-kReach of
// the block's place, summed per block, a pel from the engine's MV charged as the engine's search does,
// then the half and quarter pels around it by SAD.
void full_search(const celv2::Plane& s, const std::vector<celv2::Plane>& refs, int sb_x, int sb_y,
                 const std::vector<BlockCandidates>& cands, std::vector<int16_t>& out) {
  constexpr int kReach = 64, kN = 2 * kReach + 1, kWin = 64 + 2 * kReach;
  static thread_local std::vector<uint8_t> win(kWin * kWin);
  static thread_local std::vector<uint16_t> cells(static_cast<size_t>(64) * kN * kN);
  static thread_local std::vector<uint32_t> sum(kN * kN);
  alignas(16) uint8_t cur[64 * 64];
  out.assign(static_cast<size_t>(celv2::kSbBlocks) * BlockCandidates::kRefs * 2, SbDecision::kNoMv);
  for (int i = 0; i < 64; i++)
    for (int j = 0; j < 64; j++) cur[i * 64 + j] = static_cast<uint8_t>(s.at(sb_y + i, sb_x + j));
  for (size_t r = 0; r < refs.size() && r < BlockCandidates::kRefs; r++) {
    for (int i = 0; i < kWin; i++)
      for (int j = 0; j < kWin; j++) win[i * kWin + j] = static_cast<uint8_t>(refs[r].at(sb_y + i - kReach, sb_x + j - kReach));
    for (int oy = 0; oy < kN; oy++)
      for (int ox = 0; ox < kN; ox++)
        for (int c = 0; c < 64; c++) {
          const int cx = (c & 7) * 8, cy = (c >> 3) * 8;
          __m128i acc = _mm_setzero_si128();
          for (int i = 0; i < 8; i++)
            acc = _mm_add_epi64(acc, _mm_sad_epu8(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(cur + (cy + i) * 64 + cx)),
                                                  _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&win[(oy + cy + i) * kWin + ox + cx]))));
          cells[static_cast<size_t>(c) * kN * kN + oy * kN + ox] = static_cast<uint16_t>(_mm_cvtsi128_si32(acc));
        }
    for (int b = 0; b < celv2::kSbBlocks; b++) {
      int x, y, w, h;
      celv2::sb_block_geom(b, &x, &y, &w, &h);
      if (std::min(w, h) < 8 || cands[b].inter_cost == INT64_MAX) continue;
      std::fill(sum.begin(), sum.end(), 0u);
      for (int cy = y / 8; cy < (y + h) / 8; cy++)
        for (int cx = x / 8; cx < (x + w) / 8; cx++) {
          const uint16_t* const src = &cells[static_cast<size_t>(cy * 8 + cx) * kN * kN];
          for (int k = 0; k < kN * kN; k++) sum[k] += src[k];
        }
      const BlockCandidates& cand = cands[b];
      const bool own = cand.ref_cost[r] != UINT32_MAX;
      const int ex = own ? cand.ref_mv_col[r] / 8 : 0, ey = own ? cand.ref_mv_row[r] / 8 : 0;
      int64_t least = INT64_MAX;
      int bx = 0, by = 0;
      for (int k = 0; k < kN * kN; k++) {
        const int dx = k % kN - kReach, dy = k / kN - kReach;
        const int64_t j = sum[k] + static_cast<int64_t>(celv2::kMePenalty) * (std::abs(dx - ex) + std::abs(dy - ey));
        if (j < least) least = j, bx = dx, by = dy;
      }
      uint8_t pred[64 * 64];
      auto sad = [&](int qx, int qy) {
        predict(refs[r], sb_x + x, sb_y + y, w, h, qx, qy, pred);
        int64_t t = 0;
        for (int i = 0; i < h; i++)
          for (int j = 0; j < w; j++) t += std::abs(cur[(y + i) * 64 + x + j] - pred[i * w + j]);
        return t;
      };
      int qx = bx * 4, qy = by * 4;
      int64_t now = sad(qx, qy);
      for (int step = 2; step >= 1; step--) {
        const int cx = qx, cy = qy;
        for (int dy = -1; dy <= 1; dy++)
          for (int dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue;
            const int64_t t = sad(cx + dx * step, cy + dy * step);
            if (t < now) now = t, qx = cx + dx * step, qy = cy + dy * step;
          }
      }
      int16_t* const m = &out[(static_cast<size_t>(b) * BlockCandidates::kRefs + r) * 2];
      m[0] = static_cast<int16_t>(qx * 2), m[1] = static_cast<int16_t>(qy * 2);
    }
  }
}

}  // namespace

void rd_pass_sb(const Frame420& src, const std::vector<const Frame420*>& refs, int sb_x, int sb_y,
                const RdPassParams& p, SbDecision& sb) {
  if (sb.cands.size() != static_cast<size_t>(celv2::kSbBlocks)) return;
  celv2::RdTimer timer(celv2::RdPart::Sb);
  if (celv2::rd_stats()) t_seen_tx.clear(), t_seen_inter.clear();
  if (t_memo.m) t_memo.m->clear();
  const celv2::Plane s{src.y.data(), src.stride_y(), src.pw, src.ph};
  std::vector<celv2::Plane> rp;
  for (const Frame420* r : refs) rp.push_back(celv2::Plane{r->y.data(), r->stride_y(), r->pw, r->ph});
  if (p.full_search && !refs.empty() && sb.wide_mv.empty()) full_search(s, rp, sb_x, sb_y, sb.cands, sb.wide_mv);
  const celv2::Plane cs[2] = {{src.u.data(), src.stride_uv(), src.pw / 2, src.ph / 2}, {src.v.data(), src.stride_uv(), src.pw / 2, src.ph / 2}};
  std::vector<std::array<celv2::Plane, 2>> crp;
  for (const Frame420* r : refs)
    crp.push_back({celv2::Plane{r->u.data(), r->stride_uv(), r->pw / 2, r->ph / 2}, celv2::Plane{r->v.data(), r->stride_uv(), r->pw / 2, r->ph / 2}});
  const Pass pass{s, rp, sb_x, sb_y, p, sb.cands, !refs.empty(), src, sb.wide_mv, cs, crp, rdi::lam_of(p.qindex, p.bit_weight)};
  std::vector<Part> tree;
  std::vector<BlockDecision> blocks;
#ifdef CELV2_ORACLE
  if (sb.forced) {
    size_t pos = 0;
    pass.leaves(sb.forced->tree, pos, 0, 0, 64, 64, blocks);
    sb.tree = sb.forced->tree;
    sb.blocks = std::move(blocks);
    return;
  }
#endif
  pass.square(0, 0, 0, tree, blocks);
  sb.tree = std::move(tree);
  sb.blocks = std::move(blocks);
}
