#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

#include "check.h"
#include "core/encoder.h"
#include "gpu/decision_map.h"
#include "io/y4m.h"
#include "verify/avm_decode.h"

// Diagonal gradient plus fixed-seed noise.
static Frame420 test_frame(int w, int h, uint32_t seed) {
  Frame420 f = make_frame(w, h);
  uint32_t s = seed;
  auto noise = [&s]() {
    s = s * 1664525u + 1013904223u;
    return static_cast<int>((s >> 24) % 21) - 10;
  };
  auto fill = [&](std::vector<uint8_t>& p, int pw, int pw_vis, int ph_vis, int scale) {
    for (int r = 0; r < ph_vis; r++)
      for (int c = 0; c < pw_vis; c++) {
        int v = 40 + (r + c) * scale / 4 % 170 + noise();
        p[r * pw + c] = static_cast<uint8_t>(v < 0 ? 0 : v > 255 ? 255 : v);
      }
  };
  fill(f.y, f.stride_y(), w, h, 2);
  fill(f.u, f.stride_uv(), (w + 1) / 2, (h + 1) / 2, 3);
  fill(f.v, f.stride_uv(), (w + 1) / 2, (h + 1) / 2, 5);
  pad_frame(f);
  return f;
}

static double psnr_y(const Frame420& a, const Frame420& b) {
  double se = 0;
  for (int r = 0; r < a.height; r++)
    for (int c = 0; c < a.width; c++) {
      const double d = a.y[r * a.stride_y() + c] - b.y[r * b.stride_y() + c];
      se += d * d;
    }
  const double mse = se / (static_cast<double>(a.width) * a.height);
  return mse == 0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
}

struct Result {
  int ok = 0;  // 0 on success
  size_t bytes = 0;
  double psnr = 0;
};

static Result encode_check(int w, int h, int q) {
  Result res;
  SeqConfig c{w, h, 30, 1, q};
  const Frame420 src = test_frame(w, h, 7);
  Encoder enc(c);
  const auto tu = enc.encode(src, uniform_decision(w, h, 64), true);
  const auto dec = decode_frames({tu});
  res.bytes = tu.size();
  res.psnr = psnr_y(src, enc.last_recon());
  std::fprintf(stderr, "%dx%d q%d: %zu bytes, PSNR-Y %.2f dB\n", w, h, q, res.bytes, res.psnr);
  if (dec.size() != 1) {
    std::fprintf(stderr, "%dx%d q%d: %zu frames decoded\n", w, h, q, dec.size());
    res.ok = 1;
    return res;
  }
  const std::string diff = first_diff(dec[0], enc.last_recon());
  if (!diff.empty()) {
    std::fprintf(stderr, "%dx%d q%d: %s\n", w, h, q, diff.c_str());
    res.ok = 1;
  }
  return res;
}

static int keyframe_residual_bitexact() {
  const Result r = encode_check(640, 360, 120);
  CHECK_EQ(r.ok, 0);
  return 0;
}

static int keyframe_quality_monotonic() {
  const Result lo = encode_check(640, 360, 60), hi = encode_check(640, 360, 160);
  CHECK_EQ(lo.ok, 0);
  CHECK_EQ(hi.ok, 0);
  CHECK(lo.psnr > hi.psnr);
  CHECK(lo.bytes > hi.bytes);
  return 0;
}

// Fixed-seed random partition trees down to 4x4 with random intra modes.
static DecisionMap random_decision(int w, int h, uint32_t seed) {
  DecisionMap dm = uniform_decision(w, h, 64);
  uint32_t s = seed;
  auto rnd = [&s](int n) {
    s = s * 1664525u + 1013904223u;
    return static_cast<int>((s >> 16) % n);
  };
  for (auto& sb : dm.sbs) {
    sb.tree.clear();
    sb.blocks.clear();
    auto rec = [&](auto&& self, int bw, int bh) -> void {
      Part p = static_cast<Part>(rnd(4));
      if ((p == Part::Horz || p == Part::Split) && bh <= 4) p = Part::None;
      if ((p == Part::Vert || p == Part::Split) && bw <= 4) p = Part::None;
      sb.tree.push_back(p);
      if (p == Part::None) {
        BlockDecision b;
        b.w4 = static_cast<uint8_t>(bw / 4);
        b.h4 = static_cast<uint8_t>(bh / 4);
        b.intra = static_cast<IntraMode>(rnd(kIntraModes));
        sb.blocks.push_back(b);
      } else if (p == Part::Horz) {
        self(self, bw, bh / 2), self(self, bw, bh / 2);
      } else if (p == Part::Vert) {
        self(self, bw / 2, bh), self(self, bw / 2, bh);
      } else {
        for (int i = 0; i < 4; i++) self(self, bw / 2, bh / 2);
      }
    };
    rec(rec, 64, 64);
  }
  return dm;
}

static int keyframe_random_partitions_bitexact() {
  for (int q : {40, 120, 200}) {
    SeqConfig c{640, 360, 30, 1, q};
    const Frame420 src = test_frame(640, 360, 3);
    Encoder enc(c);
    const auto tu = enc.encode(src, random_decision(640, 360, 11 + q), true);
    const auto dec = decode_frames({tu});
    CHECK_EQ(dec.size(), 1u);
    CHECK_EQ(first_diff(dec[0], enc.last_recon()), std::string());
    std::fprintf(stderr, "random partitions q%d: %zu bytes, PSNR-Y %.2f dB\n", q, tu.size(),
                 psnr_y(src, enc.last_recon()));
  }
  return 0;
}

static int keyframe_edge_sizes_bitexact() {
  const int sizes[][2] = {{1920, 1080}, {202, 122}, {201, 121}, {16, 16}};
  for (const auto& s : sizes) CHECK_EQ(encode_check(s[0], s[1], 120).ok, 0);
  return 0;
}

int main() {
  int fails = 0;
  fails += RUN(keyframe_residual_bitexact);
  fails += RUN(keyframe_quality_monotonic);
  fails += RUN(keyframe_random_partitions_bitexact);
  fails += RUN(keyframe_edge_sizes_bitexact);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
