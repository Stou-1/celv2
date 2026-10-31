// satd_kernel (gpu/kernels.cu): SATD of n block pairs, each a 64x64 tile (stride 64) of w4 x h4 4x4s.
import common;

struct Params {
  uint8_t* a;
  uint8_t* b;
  int* out;
  int n, w4, h4;
};
[vk::push_constant] ConstantBuffer<Params> pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
  const int i = int(id.x);
  if (i >= pc.n) return;
  uint8_t* pa = pc.a + uint64_t(i) * 4096;
  uint8_t* pb = pc.b + uint64_t(i) * 4096;
  int total = 0;
  for (int ty = 0; ty < pc.h4 * 4; ty += 4)
    for (int tx = 0; tx < pc.w4 * 4; tx += 4) {
      int d[16];
      for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) d[r * 4 + c] = int(pa[(ty + r) * 64 + tx + c]) - int(pb[(ty + r) * 64 + tx + c]);
      total += hadamard4x4(d);
    }
  pc.out[i] = total;
}
