#include "core/tpl.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <immintrin.h>
#include <stdexcept>

#include "core/profile.h"
#include "core/threads.h"

extern "C" {
#include "av2/common/quant_common.h"
}

namespace celv2 {

namespace {

// Half-scale block (kTplBlock / 2 = 16) and the search: quarter scale +-16 (+-64 luma samples),
// then half scale +-3 around it.
constexpr int kHalf = kTplBlock / 2, kQuarter = kTplBlock / 4, kRangeQ = 16, kRefine = 3;
constexpr int kPadHalf = 2 * kRangeQ + kRefine + kHalf;  // every searched block stays inside the padding

// An 8-bit plane with `pad` replicated samples on every side.
struct Plane {
  int w = 0, h = 0, pad = 0, stride = 0;
  std::vector<uint8_t> px;
  const uint8_t* at(int x, int y) const { return &px[static_cast<size_t>(y + pad) * stride + x + pad]; }
};

void replicate(Plane& p) {
  for (int y = 0; y < p.h; y++) {
    uint8_t* r = &p.px[static_cast<size_t>(y + p.pad) * p.stride];
    std::fill(r, r + p.pad, r[p.pad]);
    std::fill(r + p.pad + p.w, r + p.stride, r[p.pad + p.w - 1]);
  }
  for (int y = 0; y < p.pad; y++) {
    std::copy_n(&p.px[static_cast<size_t>(p.pad) * p.stride], p.stride, &p.px[static_cast<size_t>(y) * p.stride]);
    std::copy_n(&p.px[static_cast<size_t>(p.pad + p.h - 1) * p.stride], p.stride, &p.px[static_cast<size_t>(p.pad + p.h + y) * p.stride]);
  }
}

// 2x2 averages of a w x h 8-bit plane (rounded), with `pad`.
Plane half_of(const uint8_t* s, int stride, int w, int h, int pad) {
  Plane p;
  p.w = std::max(1, w / 2), p.h = std::max(1, h / 2), p.pad = pad, p.stride = p.w + 2 * pad;
  p.px.resize(static_cast<size_t>(p.stride) * (p.h + 2 * pad));
  for (int y = 0; y < p.h; y++) {
    const uint8_t* a = s + static_cast<size_t>(std::min(2 * y, h - 1)) * stride;
    const uint8_t* b = s + static_cast<size_t>(std::min(2 * y + 1, h - 1)) * stride;
    uint8_t* d = &p.px[static_cast<size_t>(y + pad) * p.stride + pad];
    for (int x = 0; x < p.w; x++) {
      const int x0 = std::min(2 * x, w - 1), x1 = std::min(2 * x + 1, w - 1);
      d[x] = static_cast<uint8_t>((a[x0] + a[x1] + b[x0] + b[x1] + 2) >> 2);
    }
  }
  replicate(p);
  return p;
}

int sad16(const uint8_t* a, int as, const uint8_t* b, int bs) {
  __m128i acc = _mm_setzero_si128();
  for (int y = 0; y < 16; y++)
    acc = _mm_add_epi64(acc, _mm_sad_epu8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + y * as)),
                                          _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + y * bs))));
  return _mm_cvtsi128_si32(acc) + _mm_extract_epi32(acc, 2);
}

int sad8(const uint8_t* a, int as, const uint8_t* b, int bs) {
  __m128i acc = _mm_setzero_si128();
  for (int y = 0; y < 8; y++)
    acc = _mm_add_epi64(acc, _mm_sad_epu8(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(a + y * as)),
                                          _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + y * bs))));
  return _mm_cvtsi128_si32(acc);
}

// The least SAD of DC, vertical and horizontal prediction from the row above and the column left.
int intra16(const Plane& p, int x, int y) {
  const uint8_t* top = p.at(x, y - 1);
  int left[16], sum = 0;
  for (int i = 0; i < 16; i++) left[i] = *p.at(x - 1, y + i), sum += top[i] + left[i];
  const int dc = (sum + 16) >> 5;
  int c_dc = 0, c_v = 0, c_h = 0;
  for (int i = 0; i < 16; i++) {
    const uint8_t* r = p.at(x, y + i);
    for (int j = 0; j < 16; j++) c_dc += std::abs(r[j] - dc), c_v += std::abs(r[j] - top[j]), c_h += std::abs(r[j] - left[i]);
  }
  return std::min({c_dc, c_v, c_h});
}

struct Motion {
  int cost = INT32_MAX;
  int ref = -1, dx = 0, dy = 0;  // in half-scale samples
};

}  // namespace

std::vector<TplStats> tpl_model(const std::vector<TplFrame>& frames, int threads) {
  celv2::StageScope prof(celv2::Stage::Tpl);
  const int n = static_cast<int>(frames.size());
  if (n == 0) return {};
  const int w = frames[0].src->width, h = frames[0].src->height;
  for (int f = 0; f < n; f++) {
    if (frames[f].src->width != w || frames[f].src->height != h) throw std::invalid_argument("tpl_model: frame sizes differ");
    for (int r : frames[f].refs)
      if (r < 0 || r >= f) throw std::invalid_argument("tpl_model: a reference must come earlier in coding order");
  }
  const int cols = (w + kTplBlock - 1) / kTplBlock, rows = (h + kTplBlock - 1) / kTplBlock;
  threads = std::max(1, threads);

  // Half and quarter scale planes, each frame on its own thread.
  std::vector<Plane> half(n), quarter(n);
  std::atomic<int> next{0};
  parallel(std::min(threads, n), [&](int) {
    for (int f; (f = next++) < n;) {
      const Frame420& s = *frames[f].src;
      half[f] = half_of(s.y.data(), s.stride_y(), w, h, kPadHalf);
      const Plane& hp = half[f];
      quarter[f] = half_of(hp.at(0, 0), hp.stride, hp.w, hp.h, kPadHalf / 2);
    }
  });

  // Intra costs and the best motion per block, rows of blocks on threads.
  std::vector<TplStats> st(n);
  std::vector<std::vector<Motion>> mo(n);
  for (int f = 0; f < n; f++) {
    st[f].cols = cols, st[f].rows = rows;
    st[f].intra.assign(static_cast<size_t>(cols) * rows, 0);
    st[f].total.assign(static_cast<size_t>(cols) * rows, 0);
    mo[f].assign(static_cast<size_t>(cols) * rows, Motion{});
  }
  next = 0;
  const int tasks = n * rows;
  parallel(std::min(threads, tasks), [&](int) {
    for (int t; (t = next++) < tasks;) {
      const int f = t / rows, by = t % rows;
      const Plane& cur = half[f];
      const Plane& curq = quarter[f];
      for (int bx = 0; bx < cols; bx++) {
        const int x = bx * kHalf, y = by * kHalf;
        const size_t i = static_cast<size_t>(by) * cols + bx;
        st[f].intra[i] = std::max(1, intra16(cur, x, y));
        Motion& best = mo[f][i];
        for (int r : frames[f].refs) {
          const Plane& ref = half[r];
          const Plane& refq = quarter[r];
          int qx = 0, qy = 0, qc = INT32_MAX;
          for (int dy = -kRangeQ; dy <= kRangeQ; dy++)
            for (int dx = -kRangeQ; dx <= kRangeQ; dx++) {
              const int c = sad8(curq.at(x / 2, y / 2), curq.stride, refq.at(x / 2 + dx, y / 2 + dy), refq.stride);
              if (c < qc) qc = c, qx = dx, qy = dy;
            }
          for (int dy = 2 * qy - kRefine; dy <= 2 * qy + kRefine; dy++)
            for (int dx = 2 * qx - kRefine; dx <= 2 * qx + kRefine; dx++) {
              const int c = sad16(cur.at(x, y), cur.stride, ref.at(x + dx, y + dy), ref.stride);
              if (c < best.cost) best = Motion{c, r, dx, dy};
            }
        }
      }
    }
  });

  // Back along coding order: a block's total (its intra cost plus what flowed into it) passes the
  // share prediction saves, 1 - inter / intra, to the blocks its motion covers in the reference.
  for (int f = n - 1; f >= 0; f--) {
    for (int by = 0; by < rows; by++)
      for (int bx = 0; bx < cols; bx++) {
        const size_t i = static_cast<size_t>(by) * cols + bx;
        st[f].total[i] += st[f].intra[i];
        const Motion& m = mo[f][i];
        if (m.ref < 0 || m.cost >= st[f].intra[i]) continue;
        const double flow = st[f].total[i] * (1.0 - m.cost / st[f].intra[i]);
        const int px = bx * kTplBlock + 2 * m.dx, py = by * kTplBlock + 2 * m.dy;  // luma samples
        const int cx = static_cast<int>(std::floor(px / static_cast<double>(kTplBlock)));
        const int cy = static_cast<int>(std::floor(py / static_cast<double>(kTplBlock)));
        for (int k = 0; k < 4; k++) {
          const int gx = cx + (k & 1), gy = cy + (k >> 1);
          if (gx < 0 || gy < 0 || gx >= cols || gy >= rows) continue;
          const int ox = std::min(px + kTplBlock, (gx + 1) * kTplBlock) - std::max(px, gx * kTplBlock);
          const int oy = std::min(py + kTplBlock, (gy + 1) * kTplBlock) - std::max(py, gy * kTplBlock);
          if (ox > 0 && oy > 0)
            st[m.ref].total[static_cast<size_t>(gy) * cols + gx] += flow * ox * oy / (kTplBlock * kTplBlock);
        }
      }
  }
  return st;
}

double tpl_r0(const TplStats& s) {
  double intra = 0, total = 0;
  for (size_t i = 0; i < s.intra.size(); i++) intra += s.intra[i], total += s.total[i];
  return intra > 0 && total > 0 ? intra / total : 1.0;
}

int qindex_for_step_ratio(int base, double ratio, int max_q) {
  auto step = [](int q) { return av2_dc_quant_QTX(q, 0, 0, AVM_BITS_12); };
  const double want = step(base) * ratio;
  int q = base;
  while (q > 1 && std::abs(step(q - 1) - want) < std::abs(step(q) - want)) q--;
  while (q < max_q && std::abs(step(q + 1) - want) < std::abs(step(q) - want)) q++;
  return q;
}

std::vector<int> tpl_sb_offsets(const TplStats& s, int width, int height, int base, int max_q, int max_offset) {
  const int sb_cols = (width + 63) / 64, sb_rows = (height + 63) / 64, per = 64 / kTplBlock;
  double all_intra = 0, all_total = 0;
  for (size_t i = 0; i < s.intra.size(); i++) all_intra += s.intra[i], all_total += s.total[i];
  std::vector<int> out(static_cast<size_t>(sb_cols) * sb_rows, 0);
  if (all_intra <= 0 || all_total <= 0) return out;
  const double r0 = all_intra / all_total;
  auto step = [](int q) { return av2_dc_quant_QTX(q, 0, 0, AVM_BITS_12); };
  for (int sy = 0; sy < sb_rows; sy++)
    for (int sx = 0; sx < sb_cols; sx++) {
      double intra = 0, total = 0;
      for (int y = sy * per; y < std::min(s.rows, (sy + 1) * per); y++)
        for (int x = sx * per; x < std::min(s.cols, (sx + 1) * per); x++)
          intra += s.intra[static_cast<size_t>(y) * s.cols + x], total += s.total[static_cast<size_t>(y) * s.cols + x];
      if (intra <= 0 || total <= 0) continue;
      const double beta = r0 / (intra / total);
      // av2_get_deltaq_offset(): the qindex whose step is nearest step / sqrt(beta), from the base.
      const int want = static_cast<int>(std::lrint(step(base) / std::sqrt(beta)));
      int q = base;
      if (want < step(base))
        while (q > 1 && step(q) > want) q--;
      else
        while (q < max_q && step(q) < want) q++;
      out[static_cast<size_t>(sy) * sb_cols + sx] = std::clamp(q - base, -max_offset, max_offset);
    }
  return out;
}

}  // namespace celv2
