#include <cstdio>
#include <string>
#include <vector>

#include "check.h"
#include "core/encoder.h"
#include "test_frames.h"
#include "verify/avm_decode.h"

// KEY + INTER frames whose blocks are all DC intra: checks the inter frame header, reference
// handling and CDF carry-over without inter prediction.
static int inter_header_parses() {
  const int w = 202, h = 122;
  SeqConfig c{w, h, 30, 1, 120};
  Encoder enc(c);
  std::vector<std::vector<uint8_t>> tus;
  std::vector<Frame420> recons;
  for (int i = 0; i < 4; i++) {
    const Frame420 src = scene_frame(w, h, 3 * i, i);
    tus.push_back(enc.encode(src, uniform_decision(w, h, 64), i == 0));
    recons.push_back(enc.last_recon());
  }
  const auto dec = decode_frames(tus);
  CHECK_EQ(dec.size(), recons.size());
  for (size_t i = 0; i < dec.size(); i++) {
    const std::string d = first_diff(dec[i], recons[i]);
    if (!d.empty()) std::fprintf(stderr, "frame %zu: %s\n", i, d.c_str());
    CHECK_EQ(d, std::string());
  }
  return 0;
}

int main() {
  int fails = 0;
  fails += RUN(inter_header_parses);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
