// Probe for the B50's lost device on several queues (tests/test_vk_kernels.cpp, probes_on_queues): 256
// invocations a workgroup, arithmetic only.
struct Params {
  int* out;
  int iters;
};
[vk::push_constant] ConstantBuffer<Params> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid: SV_GroupID, uint tid: SV_GroupIndex) {
  uint v = gid.x * 256 + tid;
  for (int i = 0; i < pc.iters; i++) v = v * 1664525u + 1013904223u + uint(i);
  pc.out[gid.x * 256 + tid] = int(v);
}
