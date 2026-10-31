// downsample_kernel (gpu/kernels.cu): dst (w/2 x h/2) = 2x2 averages of src.
import common;

struct Params {
  Plane src;
  uint8_t* dst;
};
[vk::push_constant] ConstantBuffer<Params> pc;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id: SV_DispatchThreadID) {
  const int x = int(id.x), y = int(id.y), w = pc.src.w / 2, h = pc.src.h / 2;
  if (x < w && y < h) pc.dst[y * w + x] = uint8_t(downsample_px(pc.src, y, x));
}
