#include <cstdio>
#include <string>

#include "check.h"
#include "core/encoder.h"
#include "gpu/engine.h"
#include "test_frames.h"
#include "verify/avm_decode.h"

struct Coded {
  size_t bytes = 0;
  double psnr = 0;
  std::string diff;  // empty when bit-exact
};

static Coded code(const Frame420& src, const DecisionMap& dm, int q) {
  SeqConfig c{src.width, src.height, 30, 1, q};
  Encoder enc(c);
  const auto tu = enc.encode(src, dm, true);
  const auto dec = decode_frames({tu});
  Coded r;
  r.bytes = tu.size();
  r.psnr = psnr_y(src, enc.last_recon());
  r.diff = dec.size() == 1 ? first_diff(dec[0], enc.last_recon()) : "frame count";
  return r;
}

static int keyframe_gpu_bitexact() {
  const int sizes[][2] = {{640, 360}, {1920, 1080}};
  for (const auto& s : sizes) {
    const Frame420 src = scene_frame(s[0], s[1]);
    GpuEngine gpu(s[0], s[1], 64);
    const Coded r = code(src, gpu.decide_intra(src, 120), 120);
    std::fprintf(stderr, "%dx%d gpu: %zu bytes, %.2f dB\n", s[0], s[1], r.bytes, r.psnr);
    CHECK_EQ(r.diff, std::string());
  }
  return 0;
}

static int keyframe_gpu_better_than_uniform() {
  const Frame420 src = scene_frame(640, 360);
  GpuEngine gpu(640, 360, 64);
  const Coded g = code(src, gpu.decide_intra(src, 120), 120);
  CHECK_EQ(g.diff, std::string());
  // Rate-matched comparison: lower the uniform encode's qindex until it spends at least as many
  // bytes as the GPU-decided one; the GPU decision must still give the higher PSNR.
  Coded u;
  int q = 120;
  for (; q >= 10; q -= 10) {
    u = code(src, uniform_decision(640, 360, 64), q);
    CHECK_EQ(u.diff, std::string());
    if (u.bytes >= g.bytes) break;
  }
  std::fprintf(stderr, "gpu q120 %zu bytes %.2f dB, uniform q%d %zu bytes %.2f dB\n", g.bytes,
               g.psnr, q, u.bytes, u.psnr);
  CHECK(u.bytes >= g.bytes);
  CHECK(g.psnr > u.psnr);
  return 0;
}

int main() {
  int fails = 0;
  fails += RUN(keyframe_gpu_bitexact);
  fails += RUN(keyframe_gpu_better_than_uniform);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
