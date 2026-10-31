#include <cstdio>
#include <random>
#include <vector>

#include "check.h"
#include "core/ccso_apply.h"

extern "C" {
#include "config/av2_rtcd.h"
#include "av2/common/ccso.h"
}

// celv2::ccso_apply_row() against AVM's C filter (ccso.c), on random planes and every filter shape:
// luma or 4:2:0 chroma, 8 / 10 bits, band only or with edges (3 or 2 classes), 1 to 64 bands, every
// neighbour pair, and blocks skipped at random.
static int matches_avm_filter() {
  std::mt19937 rng(7);
  const int lw = 136, lh = 72, pad = CCSO_PADDING_SIZE, ls = lw + 2 * pad;  // not multiples of 32
  int cases = 0;
  for (int bd : {8, 10})
    for (int sub = 0; sub < 2; sub++)
      for (int bo = 0; bo < 2; bo++)
        for (int clf = 0; clf < 2; clf++)
          for (int bands : {0, 1, 3, 6})
            for (int sup = 0; sup < 7; sup++) {
              const int maxv = (1 << bd) - 1, w = lw >> sub, h = lh >> sub;
              std::vector<uint16_t> luma(static_cast<size_t>(ls) * (lh + 2 * pad)), a(static_cast<size_t>(w) * h);
              // a smooth luma with noise, so all classes occur
              for (size_t k = 0; k < luma.size(); k++)
                luma[k] = static_cast<uint16_t>(std::min<int>(maxv, (k % ls) * maxv / ls / 2 + rng() % (maxv / 8)));
              for (auto& v : a) v = static_cast<uint16_t>(rng() % (maxv + 1));
              std::vector<uint16_t> b = a;
              int8_t offset[CCSO_BAND_NUM * 16];
              for (auto& o : offset) o = static_cast<int8_t>(static_cast<int>(rng() % 15) - 7);
              const int thr = quant_sz[rng() % 4][rng() % 3];
              const int cols = (w + 31) / 32, rows = (h + 31) / 32;
              std::vector<uint8_t> on(static_cast<size_t>(cols) * rows);
              for (auto& o : on) o = rng() % 4 != 0;

              celv2::CcsoPlane p;
              p.luma = luma.data() + pad * ls + pad;
              p.luma_stride = ls;
              p.dst = a.data();
              p.dst_stride = w;
              p.w = w, p.h = h, p.sx = sub, p.sy = sub, p.bit_depth = bd;
              p.bo_only = bo, p.thr = thr, p.edge_clf = clf, p.max_band_log2 = bands, p.offset = offset;
              celv2::ccso_neighbours(sup, ls, p.loc);
              p.on = on.data(), p.on_stride = cols;
              for (int r = 0; r < rows; r++) celv2::ccso_apply_row(p, r);

              int src_cls[2], loc[2];
              derive_ccso_sample_pos(loc, ls, static_cast<uint8_t>(sup));
              for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++) {
                  if (!on[static_cast<size_t>(r) * cols + c]) continue;
                  const uint16_t* src = p.luma + static_cast<size_t>(r * 32 << sub) * ls;
                  uint16_t* dst = b.data() + static_cast<size_t>(r * 32) * w;
                  if (bo)
                    ccso_filter_block_hbd_wo_buf_bo_only_c(src, dst, c * 32, r * 32, w, h, offset, ls, w, sub, sub, maxv, 32,
                                                           32, bands == 0, static_cast<uint8_t>(bd - bands));
                  else
                    ccso_filter_block_hbd_wo_buf_c(src, dst, c * 32, r * 32, w, h, src_cls, offset, ls, w, sub, sub, thr, -thr,
                                                   loc, maxv, 32, 32, bands == 0, static_cast<uint8_t>(bd - bands), clf, 0);
                }
              if (a != b) {
                std::printf("  mismatch: bd %d sub %d bo %d clf %d bands %d sup %d\n", bd, sub, bo, clf, bands, sup);
                return 1;
              }
              cases++;
            }
  CHECK_EQ(cases, 2 * 2 * 2 * 2 * 4 * 7);
  return 0;
}

int main() {
  int fails = 0;
  fails += RUN(matches_avm_filter);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
