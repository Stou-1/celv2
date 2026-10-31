#include <cstdio>
#include <string>

#include "check.h"
#include "core/encoder.h"
#include "gpu/decision_map.h"
#include "io/y4m.h"
#include "verify/avm_decode.h"

static int keyframe_matches(int w, int h) {
  SeqConfig c{w, h, 30, 1, 120};
  Frame420 src = make_frame(w, h);
  for (auto* p : {&src.y, &src.u, &src.v})
    for (auto& s : *p) s = 128;
  Encoder enc(c);
  auto tu = enc.encode(src, uniform_decision(w, h, 64), true);
  auto dec = decode_frames({tu});
  CHECK_EQ(dec.size(), 1u);
  CHECK_EQ(first_diff(dec[0], enc.last_recon()), std::string());
  CHECK_EQ(frame_md5(dec[0]), frame_md5(enc.last_recon()));
  return 0;
}

static int keyframe_skip_dc_bitexact() { return keyframe_matches(128, 128); }
static int keyframe_skip_dc_640x360() { return keyframe_matches(640, 360); }

int main() {
  int fails = 0;
  fails += RUN(keyframe_skip_dc_bitexact);
  fails += RUN(keyframe_skip_dc_640x360);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
