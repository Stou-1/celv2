#include "check.h"
#include "io/ivf.h"
#include "verify/avm_decode.h"

int main() {
  // ffmpeg testsrc2 (352x288, 16 frames) encoded by avmenc --good --cpu-used=6 --qp=150.
  auto tus = read_ivf(CELV2_TEST_DATA "/avm_testsrc2.ivf");
  auto md5s = decode_frame_md5s(tus);
  CHECK_EQ(md5s.size(), 16u);
  CHECK_EQ(md5s[0].size(), 32u);
  CHECK(md5s[0] != md5s[1]);
  return 0;
}
