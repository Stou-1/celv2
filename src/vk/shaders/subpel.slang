// subpel_kernel (gpu/kernels.cu): quarter-pel phases 1..15 of src (phase 0 is src) at dst + k * phase.
import common;

struct Params {
  Plane src;
  uint8_t* dst;
  uint64_t phase;
  int half;  // only the half-pel phases 2, 8, 10, at slots 1..3 (Plane::kPhasesHalf)
};
[vk::push_constant] ConstantBuffer<Params> pc;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id: SV_DispatchThreadID) {
  const int x = int(id.x), y = int(id.y);
  const int k = pc.half != 0 ? (id.z == 0 ? 2 : id.z == 1 ? 8 : 10) : int(id.z) + 1;
  const int slot = pc.half != 0 ? int(id.z) + 1 : k;
  if (x < pc.src.w && y < pc.src.h)
    pc.dst[uint64_t(slot) * pc.phase + uint64_t(y) * pc.src.stride + x] = uint8_t(subpel_px(pc.src, y, x, k / 4, k % 4));
}
