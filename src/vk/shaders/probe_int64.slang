// Probe (probes_on_queues): probe_plain in 64-bit integers, stored as such (as cost's costs).
struct Params {
  int* out;
  int iters;
};
[vk::push_constant] ConstantBuffer<Params> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid: SV_GroupID, uint tid: SV_GroupIndex) {
  uint64_t v = gid.x * 256 + tid;
  for (int i = 0; i < pc.iters; i++) v = v * 6364136223846793005ull + 1442695040888963407ull + uint64_t(i);
  ((int64_t*)pc.out)[gid.x * 256 + tid] = int64_t(v);
}
