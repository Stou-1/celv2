// Probe (probes_on_queues): probe_plain with a subgroup shuffle, as cost's reduction.
struct Params {
  int* out;
  int iters;
};
[vk::push_constant] ConstantBuffer<Params> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid: SV_GroupID, uint tid: SV_GroupIndex) {
  uint v = gid.x * 256 + tid;
  const uint next = (WaveGetLaneIndex() + 1) & (WaveGetLaneCount() - 1);
  for (int i = 0; i < pc.iters; i++) v = v * 1664525u + WaveReadLaneAt(v, next) + uint(i);
  pc.out[gid.x * 256 + tid] = int(v);
}
