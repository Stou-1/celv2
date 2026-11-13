#include <cstdint>
#include <cstdio>
#include <vector>

#include "check.h"
#include "core/cdef_search.h"

extern "C" {
#include "av2/common/cdef_block.h"
#include "config/av2_rtcd.h"
}

namespace {

uint32_t g_rng = 777;
int rnd(int lo, int hi) {  // [lo, hi]
  g_rng = g_rng * 1664525u + 1013904223u;
  return lo + static_cast<int>((g_rng >> 8) % static_cast<uint32_t>(hi - lo + 1));
}

// One block's errors by av2_cdef_filter_fb() at each strength, as CodecState::cdef() sums them.
void reference(uint16_t* blk, bool luma, int dir, int var, int damping, int cs, const uint16_t* org16,
               const uint8_t* org8, int os, int vw, int vh, int64_t err[celv2::kCdefStrengths]) {
  const int bs = luma ? 8 : 4, ss = luma ? 0 : 1;
  cdef_list dl[1] = {};
  int dirs[CDEF_NBLOCKS][CDEF_NBLOCKS] = {}, vars[CDEF_NBLOCKS][CDEF_NBLOCKS] = {};
  dirs[0][0] = dir, vars[0][0] = var;
  std::vector<uint16_t> out(64);
  auto error = [&](const uint16_t* f, int stride) {
    int64_t e = 0;
    for (int r = 0; r < vh; r++)
      for (int c = 0; c < vw; c++) {
        const int d = f[r * stride + c] - (org16 ? org16[r * os + c] : org8[r * os + c]);
        e += d * d;
      }
    return e;
  };
  err[0] += error(blk, CDEF_BSTRIDE);
  for (int s = 1; s < celv2::kCdefStrengths; s++) {
    int dirinit = 1;
    const int pri = s / 4, sec = s % 4 + (s % 4 == 3);
    av2_cdef_filter_fb(nullptr, out.data(), 0, blk, ss, ss, dirs, &dirinit, vars, luma ? 0 : 1, dl, 1, pri, sec,
                       damping, cs);
    err[s] += error(out.data(), bs);
  }
}

// pattern 0: noise; 1: an edge with noise (directional); 2: the extremes; 3: flat; 4: a flat block, its taps a little brighter.
int one(bool luma, int depth, int pattern, bool edges, int vw, int vh, int damping) {
  const int cs = depth - 8, peak = (1 << depth) - 1, bs = luma ? 8 : 4;
  std::vector<uint16_t> in(CDEF_INBUF_SIZE);
  const int a = rnd(0, 7);
  for (int r = 0; r < bs + 2 * CDEF_VBORDER; r++)
    for (int c = 0; c < bs + 2 * CDEF_HBORDER; c++) {
      int v = pattern == 0   ? rnd(0, peak)
              : pattern == 1 ? ((r * (a - 3) + c * 2 > 8 ? peak * 3 / 4 : peak / 5) + rnd(-peak / 40, peak / 40))
              : pattern == 2 ? (rnd(0, 1) ? peak : 0)
              : pattern == 3 ? peak / 2
                             // a flat block among taps a strength or so above: the sums overshoot the taps' range,
                             // the clamp decides
                             : (r >= CDEF_VBORDER && r < CDEF_VBORDER + bs && c >= CDEF_HBORDER && c < CDEF_HBORDER + bs
                                    ? peak / 3
                                    : peak / 3 + (rnd(2, 14) << cs));
      v = v < 0 ? 0 : v > peak ? peak : v;
      // the frame's edge: CDEF_VERY_LARGE outside it
      if (edges && (r < CDEF_VBORDER || c < CDEF_HBORDER - 1)) v = CDEF_VERY_LARGE;
      in[r * CDEF_BSTRIDE + c] = static_cast<uint16_t>(v);
    }
  uint16_t* const blk = &in[CDEF_VBORDER * CDEF_BSTRIDE + CDEF_HBORDER];
  int var = 0, dir = rnd(0, 7);
  if (luma) dir = av2_cdef_find_dir(blk, CDEF_BSTRIDE, &var, cs);
  const int os = 11;
  std::vector<uint16_t> o16(os * bs);
  std::vector<uint8_t> o8(os * bs);
  for (int i = 0; i < os * bs; i++) o16[i] = static_cast<uint16_t>(rnd(0, peak)), o8[i] = static_cast<uint8_t>(rnd(0, 255));
  const uint16_t* const org16 = depth > 8 ? o16.data() : nullptr;
  int64_t got[celv2::kCdefStrengths] = {}, want[celv2::kCdefStrengths] = {};
  celv2::cdef_block_errors(blk, luma, dir, var, damping, cs, org16, o8.data(), os, vw, vh, got);
  reference(blk, luma, dir, var, damping, cs, org16, o8.data(), os, vw, vh, want);
  for (int s = 0; s < celv2::kCdefStrengths; s++) {
    if (got[s] != want[s])
      std::fprintf(stderr, "luma %d depth %d pattern %d edges %d %dx%d damping %d dir %d var %d: err[%d] %lld, want %lld\n",
                   luma, depth, pattern, edges, vw, vh, damping, dir, var, s, static_cast<long long>(got[s]),
                   static_cast<long long>(want[s]));
    CHECK_EQ(got[s], want[s]);
  }
  return 0;
}

int errors_match_avm() {
  av2_rtcd();
  for (int trial = 0; trial < 20000; trial++) {
    const bool luma = trial % 3 != 2;
    const int bs = luma ? 8 : 4;
    const int depth = (trial / 3) % 3 == 0 ? 8 : (trial / 3) % 3 == 1 ? 10 : 12;
    const bool part = trial % 7 == 0;
    if (one(luma, depth, trial % 5, trial % 4 == 0, part ? rnd(1, bs) : bs, part ? rnd(1, bs) : bs, rnd(3, 6))) return 1;
  }
  return 0;
}

// Two groups whose best luma strengths differ (5 and 20; chroma best unfiltered): two pairs where the second's
// saving (1000) is large against the bits, one where it is small, off where the first's is too.
int set_search() {
  constexpr int K = celv2::kCdefStrengths;
  std::vector<int64_t> err(4 * K, 10000);
  for (int g = 0; g < 2; g++)
    for (int s = 0; s < K; s++) err[(g * 2 + 1) * K + s] = 500;
  err[0 * K + 5] = 0;
  err[2 * K + 20] = 9000;
  std::vector<int8_t> index(2, -1);
  std::vector<celv2::CdefPair> set = celv2::cdef_choose_set(err.data(), 2, 1.0, &index);
  CHECK_EQ(set.size(), 2u);
  CHECK(index[0] != index[1]);
  CHECK_EQ(set[index[0]].luma, 5);
  CHECK_EQ(set[index[1]].luma, 20);
  CHECK_EQ(set[0].chroma, 0);
  set = celv2::cdef_choose_set(err.data(), 2, 100.0, &index);
  CHECK_EQ(set.size(), 1u);
  CHECK_EQ(set[0].luma, 5);
  CHECK(index[0] == 0 && index[1] == 0);
  CHECK(celv2::cdef_choose_set(err.data(), 2, 1000.0, &index).empty());
  return 0;
}

}  // namespace

#define RUN(fn) run_test(#fn, fn)

int main() {
  int fails = 0;
  fails += RUN(errors_match_avm);
  fails += RUN(set_search);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
