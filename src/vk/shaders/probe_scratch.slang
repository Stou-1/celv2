// Probe (probes_on_queues): probe_plain with a private array too large for registers, indexed at run
// time, so it lives in scratch memory (as the cost kernel's spills may).
struct Params {
  int* out;
  int iters;
};
[vk::push_constant] ConstantBuffer<Params> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid: SV_GroupID, uint tid: SV_GroupIndex) {
  uint a[512];
  for (int k = 0; k < 512; k++) a[k] = tid + uint(k);
  uint v = gid.x * 256 + tid;
  for (int i = 0; i < pc.iters; i++) {
    const uint j = (v >> 7) & 511u;
    v = v * 1664525u + a[j] + uint(i);
    a[(j * 5u + 1u) & 511u] = v;
  }
  pc.out[gid.x * 256 + tid] = int(v);
}
