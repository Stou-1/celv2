// Probe (probes_on_queues): probe_plain with workgroup memory and workgroup barriers, as cost and finalize.
struct Params {
  int* out;
  int iters;
};
[vk::push_constant] ConstantBuffer<Params> pc;
groupshared uint s[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid: SV_GroupID, uint tid: SV_GroupIndex) {
  uint v = gid.x * 256 + tid;
  for (int i = 0; i < pc.iters; i++) {
    s[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    v = v * 1664525u + s[(tid + 1) & 255u] + uint(i);
    GroupMemoryBarrierWithGroupSync();
  }
  pc.out[gid.x * 256 + tid] = int(v);
}
