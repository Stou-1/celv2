#include <cstdio>
#include <vector>

#include "check.h"
#include "core/tpl.h"

// A frame of w x h whose left half is a fixed texture and right half fresh noise (seed): later frames
// can predict the left half from it, never the right.
static Frame420 half_still(int w, int h, unsigned seed) {
  Frame420 f = make_frame(w, h);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      unsigned v = (x * 73856093u) ^ (y * 19349663u) ^ (x < w / 2 ? 0u : (seed + 1) * 83492791u);
      v ^= v >> 13, v *= 0x5bd1e995u, v ^= v >> 15;
      f.y[static_cast<size_t>(y) * f.stride_y() + x] = static_cast<uint8_t>(v >> 24);
    }
  return f;
}

// The still half of the first frame collects the later frames' dependence; its superblocks get a
// finer qindex than the noisy half's.
static int still_half_gets_finer_qindex() {
  const int w = 256, h = 128;
  std::vector<Frame420> fs;
  for (unsigned s = 0; s < 4; s++) fs.push_back(half_still(w, h, s));
  std::vector<celv2::TplFrame> fr(4);
  for (int i = 0; i < 4; i++) {
    fr[i].src = &fs[i];
    if (i) fr[i].refs = {i - 1};
  }
  const auto st = celv2::tpl_model(fr, 4);
  CHECK_EQ(st.size(), 4u);
  CHECK_EQ(st[0].cols, w / celv2::kTplBlock);
  const size_t left = 1, right = st[0].cols - 2;  // blocks of the first row
  CHECK(st[0].total[left] > 2 * st[0].intra[left]);       // three frames copy it
  CHECK(st[0].total[right] < 1.01 * st[0].intra[right]);  // nothing predicts from noise
  CHECK(st[3].total[left] == st[3].intra[left]);          // the last frame feeds nothing
  const std::vector<int> dq = celv2::tpl_sb_offsets(st[0], w, h, 150, 255, 35);
  CHECK_EQ(dq.size(), 8u);  // 4 x 2 superblocks
  CHECK(dq[0] < 0);
  CHECK(dq[3] > 0);
  CHECK(dq[0] >= -35 && dq[3] <= 35);
  // The frame-level share: what depends on the first frame lowers its r0; the last keeps 1.
  CHECK(celv2::tpl_r0(st[0]) < 0.9);
  CHECK(celv2::tpl_r0(st[3]) == 1.0);
  return 0;
}

// qindex_for_step_ratio(): ratio 1 keeps the base, a smaller step a lower qindex, within 1..max_q.
static int step_ratio_qindex() {
  CHECK_EQ(celv2::qindex_for_step_ratio(150, 1.0, 255), 150);
  const int half = celv2::qindex_for_step_ratio(150, 0.5, 255);
  CHECK(half < 150 && half > 50);
  CHECK(celv2::qindex_for_step_ratio(150, 2.0, 255) > 150);
  CHECK_EQ(celv2::qindex_for_step_ratio(10, 1e-6, 255), 1);
  CHECK_EQ(celv2::qindex_for_step_ratio(250, 1e6, 255), 255);
  return 0;
}

int main() {
  int fails = 0;
  fails += RUN(still_half_gets_finer_qindex);
  fails += RUN(step_ratio_qindex);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
