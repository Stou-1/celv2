// C17 step 0b, throwaway: c17_tx.slang on residuals dumped from real encodes (CELV2_TX_DUMP), timed per size on one
// GPU. The projection: us a block x the encode's calls of that size (<dump>.counts: every call, since the GPU has no
// memo) / frames = ms a frame of GPU time.
// usage: c17_gpu <dump> <frames> [device index] [rounds] [mode: 1 = no Viterbi]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/quant.h"
#include "vk/vk_context.h"
#include "vk_shaders/c17_tx.h"

extern "C" {
#include "av2/common/common_data.h"
#include "av2/common/idct.h"
#include "av2/common/quant_common.h"
#include "av2/common/scan.h"
#include "av2/common/txb_common.h"
}

namespace {

struct Rec {
  int t, tcq, qindex;
  size_t at;
};
struct Block {
  int at, dcq, acq, pad;
};
struct Push {
  uint64_t res, blocks, mat, scan, bits, out;
  int n, gx, lw, lh, s0, s1, sqrt2, qshift, first_q8, wb_num, mode, pad;
};
static_assert(sizeof(Push) == 96);

const int* dct_matrix(int n) {
  return n == 4 ? tx_kernel_dct2_size4[FWD_TXFM][0] : n == 8 ? tx_kernel_dct2_size8[FWD_TXFM][0]
       : n == 16 ? tx_kernel_dct2_size16[FWD_TXFM][0] : n == 32 ? tx_kernel_dct2_size32[FWD_TXFM][0]
       : tx_kernel_dct2_size64[FWD_TXFM][0];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: c17_gpu <dump> <frames> [device index] [rounds]\n");
    return 2;
  }
  const int frames = std::atoi(argv[2]);
  const int dev = argc > 3 ? std::atoi(argv[3]) : 0;
  const int rounds = argc > 4 ? std::atoi(argv[4]) : 5;
  const int mode = argc > 5 ? std::atoi(argv[5]) : 0;  // 1: no Viterbi
  std::vector<int16_t> pool;
  std::vector<Rec> recs;
  {
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) return std::perror(argv[1]), 1;
    for (;;) {
      uint8_t head[2];
      int16_t q;
      double sc;
      if (std::fread(head, 1, 2, f) != 2 || std::fread(&q, 2, 1, f) != 1 || std::fread(&sc, 8, 1, f) != 1) break;
      const int n = tx_size_wide[head[0]] * tx_size_high[head[0]];
      const size_t at = pool.size();
      pool.resize(at + n);
      if (std::fread(pool.data() + at, 2, n, f) != static_cast<size_t>(n)) break;
      recs.push_back({head[0], head[1], q, at});
    }
    std::fclose(f);
  }
  unsigned long long calls[TX_SIZES_ALL] = {}, unc[TX_SIZES_ALL] = {};
  if (FILE* c = std::fopen((std::string(argv[1]) + ".counts").c_str(), "r")) {
    unsigned every;
    if (std::fscanf(c, "every %u", &every) == 1) {
      int t;
      unsigned long long a, b;
      while (std::fscanf(c, "%d %llu %llu", &t, &a, &b) == 3) calls[t] = a, unc[t] = b;
    }
    std::fclose(c);
  }
  if (pool.size() >= (1ull << 31)) return std::fprintf(stderr, "dump too large\n"), 1;
  int tcq = 0;
  for (const Rec& r : recs) tcq += r.tcq;

  celv2::vk::Context ctx(dev);
  std::printf("%s: %zu records (%d with TCQ), %d frames\n", ctx.name().c_str(), recs.size(), tcq, frames);
  celv2::vk::Kernel k = ctx.kernel(kSpv_c17_tx, sizeof(Push));
  celv2::vk::Buffer res = ctx.buffer(pool.size() * 2);
  ctx.upload(res, pool.data(), pool.size() * 2);
  std::vector<int> mats;
  for (int n = 4; n <= 64; n *= 2) mats.insert(mats.end(), dct_matrix(n), dct_matrix(n) + n * n);
  celv2::vk::Buffer mat = ctx.buffer(mats.size() * 4);
  ctx.upload(mat, mats.data(), mats.size() * 4);
  std::vector<int> bits(256);
  for (int l = 0; l < 256; l++) bits[l] = static_cast<int>(std::lround(coeff_level_bits(l) * 512));
  celv2::vk::Buffer bitsb = ctx.buffer(bits.size() * 4);
  ctx.upload(bitsb, bits.data(), bits.size() * 4);
  const double first = 0.5 + kTcqBitWeight * (coeff_level_bits(1) - coeff_level_bits(0)) / 2;

  std::printf("%-8s %8s %10s %9s %9s %9s\n", "size", "blocks", "calls/f", "us/block", "ms/f", "mean eob");
  double total = 0;
  for (int t = 0; t < TX_SIZES_ALL; t++) {
    std::vector<Block> blocks;
    for (const Rec& r : recs)
      if (r.t == t)
        blocks.push_back({static_cast<int>(r.at), av2_dc_quant_QTX(r.qindex, 0, 0, kQDepth),
                          av2_ac_quant_QTX(r.qindex, 0, 0, kQDepth), 0});
    if (blocks.empty()) continue;
    const TX_SIZE tx = static_cast<TX_SIZE>(t);
    const int nc = std::min(tx_size_wide[tx], 32) * std::min(tx_size_high[tx], 32);
    const SCAN_ORDER* so = get_scan(tx, DCT_DCT);
    std::vector<int> sc(2 * nc);
    for (int i = 0; i < nc; i++) sc[i] = so->scan[i], sc[nc + i] = so->iscan[i];
    celv2::vk::Buffer bb = ctx.buffer(blocks.size() * sizeof(Block)), scb = ctx.buffer(sc.size() * 4),
                      out = ctx.buffer(blocks.size() * 16);
    ctx.upload(bb, blocks.data(), blocks.size() * sizeof(Block));
    ctx.upload(scb, sc.data(), sc.size() * 4);
    const int n = static_cast<int>(blocks.size()), gx = std::min(n, 65535), gy = (n + gx - 1) / gx;
    const Push p{res.addr, bb.addr, mat.addr, scb.addr, bitsb.addr, out.addr, n, gx, tx_size_wide_log2[tx],
                 tx_size_high_log2[tx], fwd_tx_shift[tx][0], fwd_tx_shift[tx][1],
                 (tx_size_wide_log2[tx] + tx_size_high_log2[tx]) & 1, QUANT_TABLE_BITS + av2_get_tx_scale(tx),
                 static_cast<int>(std::lround(first * 256)), static_cast<int>(std::lround(kTcqBitWeight * 65536)), mode, 0};
    ctx.dispatch(k, &p, gx, gy);  // warm up
    double best = 1e300;
    for (int r = 0; r < rounds; r++) {
      const auto t0 = std::chrono::steady_clock::now();
      ctx.begin_batch();
      for (int i = 0; i < 4; i++) ctx.dispatch(k, &p, gx, gy);
      ctx.wait(ctx.submit());
      best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 4);
    }
    std::vector<int64_t> o(blocks.size() * 2);
    ctx.download(o.data(), out, o.size() * 8);
    double eobs = 0;
    for (size_t i = 0; i < blocks.size(); i++) eobs += static_cast<double>(o[i * 2 + 1]);
    const double us = best / n, per_frame = frames ? static_cast<double>(calls[t]) / frames : 0, ms = us * per_frame / 1e3;
    total += ms;
    std::printf("%2dx%-5d %8d %10.0f %9.3f %9.2f %9.1f\n", tx_size_wide[tx], tx_size_high[tx], n, per_frame, us, ms,
                eobs / n);
    ctx.free(bb), ctx.free(scb), ctx.free(out);
  }
  std::printf("total GPU ms a frame (every call): %.1f\n", total);
  ctx.free(res), ctx.free(mat), ctx.free(bitsb);
  return 0;
}
