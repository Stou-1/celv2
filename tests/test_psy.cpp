#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "check.h"
#include "core/perceptual_aq.h"
#include "core/psy.h"
#include "core/quant.h"
#include "io/y4m.h"

namespace {

// A luma plane from v(x, y) (clamped to 0..255), chroma 128.
template <class F>
Frame420 frame(int w, int h, F v) {
  Frame420 f = make_frame(w, h);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) f.y[y * f.stride_y() + x] = static_cast<uint8_t>(std::clamp(v(x, y), 0, 255));
  std::fill(f.u.begin(), f.u.end(), 128), std::fill(f.v.begin(), f.v.end(), 128);
  pad_frame(f);
  return f;
}

uint32_t mix(int x, int y) {
  uint32_t h = static_cast<uint32_t>(x) * 0x9e3779b1u + static_cast<uint32_t>(y) * 0x85ebca77u;
  h ^= h >> 16, h *= 0x85ebca6bu, h ^= h >> 13, h *= 0xc2b2ae35u, h ^= h >> 16;
  return h;
}

}  // namespace

// A reconstruction equal to the source (every residual coefficient kept exactly): no psy cost.
static int exact_reconstruction_is_zero() {
  std::vector<int32_t> s(64), c(64);
  for (int i = 0; i < 64; i++) s[i] = (i * 37) % 23 - 11, c[i] = (i * 13) % 9 - 4;
  CHECK(celv2::psy_term(s.data(), c.data(), c.data(), 64, 4.0) == 0);
  return 0;
}

// Prediction 0 (residual = source) with every level zeroed: the reconstruction has no AC energy, so the
// term is the source's AC sum (in steps; the DC left out).
static int zeroed_levels_cost_the_ac_sum() {
  std::vector<int32_t> s(16), zero(16, 0);
  double ac = 0;
  for (int i = 0; i < 16; i++) {
    s[i] = (i * 29) % 17 - 8;
    if (i) ac += std::abs(s[i]);
  }
  s[0] = 1000;  // the DC never counts
  CHECK(std::abs(celv2::psy_term(s.data(), s.data(), zero.data(), 16, 2.0) - ac / 2) < 1e-9);
  return 0;
}

// The weights from the --paq map: flat 0 under both rules; noise near 1 under the texture rule and 0
// under the line rule; a drawn edge above 0.5 under the line rule.
static int weights_by_rule() {
  const celv2::AqMap m = celv2::aq_map(frame(192, 64, [](int x, int y) {
    if (x < 64) return 128;                                                  // flat
    if (x < 128) return 128 + static_cast<int>(mix(x, y) % 81) - 40;        // texture
    return x < 168 ? 40 : 220;                                              // a vertical edge at 168
  }), 1);
  using celv2::PsyRule;
  CHECK(celv2::psy_weight(m, 0, 0, 64, 64, PsyRule::Texture) == 0);
  CHECK(celv2::psy_weight(m, 0, 0, 64, 64, PsyRule::Line) == 0);
  CHECK(celv2::psy_weight(m, 64, 0, 64, 64, PsyRule::Texture) > 0.7);
  CHECK(celv2::psy_weight(m, 64, 0, 64, 64, PsyRule::Line) < 0.2);
  CHECK(celv2::psy_weight(m, 160, 0, 16, 64, PsyRule::Line) > 0.5);  // the 16x16s the edge crosses
  return 0;
}

// --psy-trellis: an AC coefficient just under what strength 0 keeps is kept at strength 2 (both
// quantizers, the cost-only path's end of block with them); DC is left alone.
static int trellis_keeps_small_ac() {
  for (const bool tcq : {false, true}) {
    tran_low_t coeff[64] = {}, q[64], dq[64];
    auto eob = [&](double psy) { return quantize(coeff, q, dq, TX_8X8, 130, 0, DCT_DCT, nullptr, tcq, 8, psy); };
    int v = 1;
    for (coeff[1] = v; !eob(0); coeff[1] = ++v) {
    }
    coeff[1] = v - 1;
    CHECK(eob(0) == 0);
    CHECK(eob(2) > 0);
    CHECK(q[1] != 0);
    int fast = -1;
    quantize_cost(coeff, TX_8X8, 130, DCT_DCT, &fast, tcq, 2);
    CHECK(fast > 0);
    coeff[1] = 0, v = 1;
    for (coeff[0] = v; !eob(0); coeff[0] = ++v) {
    }
    coeff[0] = v - 1;
    CHECK(eob(2) == 0);
  }
  return 0;
}

int main() {
  int fails = 0;
  fails += RUN(exact_reconstruction_is_zero);
  fails += RUN(zeroed_levels_cost_the_ac_sum);
  fails += RUN(weights_by_rule);
  fails += RUN(trellis_keeps_small_ac);
  return fails ? 1 : 0;
}
