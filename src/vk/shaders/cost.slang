// cost_kernel (gpu/kernels.cu): the costs of every candidate of one superblock, shared by k
// workgroups. Subgroups of 32 (the pipeline requires them) stand in for CUDA's warps.
import common;

struct Params {
  Plane s;
  Tables* tab;
  KRefs* refs;   // intra_part: unused
  int64_t* intra;
  int64_t* costs;
  int64_t lam;
  int sb_cols, row0, k, intra_part;
};
[vk::push_constant] ConstantBuffer<Params> pc;

static const int kThreads = 256;
groupshared int dcv[kSbBlocks];

// One 4x4 term of block_mode_satd() / block_inter_satd() (the CUDA fast paths read the same samples).
int sub_satd(Plane s, Plane rp, int x, int y, int w, int h, int mode, int dc, bool inter, int mvx, int mvy, int tx, int ty) {
  int d[16];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) {
      const int p = inter ? rp.at_q(y + ty + i, x + tx + j, mvy, mvx) : mode == 0 ? dc : intra_pred(s, x, y, w, h, mode, ty + i, tx + j);
      d[i * 4 + j] = s.at(y + ty + i, x + tx + j) - p;
    }
  return hadamard4x4(d);
}

[shader("compute")]
[numthreads(kThreads, 1, 1)]
void main(uint3 gid: SV_GroupID, uint tid: SV_GroupIndex) {
  const bool intra_part = pc.intra_part != 0;
  const int sbx = int(gid.x), sby = pc.row0 + int(gid.y) / pc.k, share = int(gid.y) % pc.k;
  const int ox = sbx * 64, oy = sby * 64;
  if (intra_part) {
    for (int b = int(tid); b < kSbBlocks; b += kThreads) {
      int x, y, w, h;
      sb_block_geom(b, x, y, w, h);
      dcv[b] = dc_value(pc.s, x + ox, y + oy, w, h);
    }
  }
  GroupMemoryBarrierWithGroupSync();
  const uint64_t sb = uint64_t(sby) * pc.sb_cols + sbx;
  const int nrefs = intra_part ? 0 : pc.refs.n;
  // The inter costs from row0's first superblock on (a row's worth is enough, row by row).
  int64_t* cost = intra_part ? pc.intra + sb * kNumIntra
                             : pc.costs + (sb - uint64_t(pc.row0) * pc.sb_cols) * uint64_t(nrefs * kInterSlots);
  // shortcut: assumes subgroup w holds invocations 32w..32w+31 (true on NVIDIA and Intel with full
  // subgroups of 32); test_vk_kernels would catch a driver that maps them otherwise.
  const int lane = int(WaveGetLaneIndex()), warp = int(tid) >> 5, nwarps = kThreads >> 5;
  const int n_ic = pc.tab.n_intra_chunks, n_xc = pc.tab.n_inter_chunks;
  const int n_units = intra_part ? n_ic : nrefs * n_xc;
  for (int u = share * nwarps + warp; u < n_units; u += pc.k * nwarps) {
    const bool inter = !intra_part;
    const int r = inter ? u / n_xc : 0;
    const int c4 = (inter ? u % n_xc : u) * 4;
    const int b = inter ? pc.tab.inter_chunks[c4] : pc.tab.intra_chunks[c4];
    const int first = inter ? pc.tab.inter_chunks[c4 + 1] : pc.tab.intra_chunks[c4 + 1];
    const int count = inter ? pc.tab.inter_chunks[c4 + 2] : pc.tab.intra_chunks[c4 + 2];
    const int lanes = inter ? pc.tab.inter_chunks[c4 + 3] : pc.tab.intra_chunks[c4 + 3];
    const int slot = lane / lanes, q0 = lane % lanes;
    const bool active = slot < count;
    const int task = first + slot;
    int x, y, w, h;
    sb_block_geom(b, x, y, w, h);
    x += ox;
    y += oy;
    int mx = 0, my = 0;
    bool repeated = false;
    Plane rp = pc.s;
    if (inter) rp = pc.refs.p[r];
    if (inter && active) {
      const int kk = pc.tab.inter_k[task];
      repeated = repeated_candidate(pc.refs.mvx[r], pc.refs.mvy[r], pc.refs.mv_cols, pc.refs.mv_rows, x, y, w, h, kk);
      const int i = candidate_mv_index(x, y, w, h, kk, pc.refs.mv_cols, pc.refs.mv_rows);
      mx = pc.refs.mvx[r][i];
      my = pc.refs.mvy[r][i];
    }
    const int sw = w / 4, nsub = sw * (h / 4);
    // A lane's share is under 2^31 (a 64x64 block's whole SATD is), so 32 bits are exact.
    int sum = 0;
    if (active && !repeated)
      for (int q = q0; q < nsub; q += lanes)
        sum += sub_satd(pc.s, rp, x, y, w, h, inter ? 0 : task, intra_part ? dcv[b] : 0, inter, mx, my, (q % sw) * 4, (q / sw) * 4);
    // __shfl_down_sync(..., lanes): the segment's first lane ends with its sum (integers: any order).
    for (int off = lanes / 2; off > 0; off >>= 1) sum += WaveReadLaneAt(sum, min(lane + off, 31));
    if (active && q0 == 0) {
      if (inter) cost[r * kInterSlots + task] = repeated ? int64_t(0x7fffffffffffffffLL) : int64_t(sum) * 256 + pc.lam * inter_bits(r);
      else cost[b * kNumModes + task] = int64_t(sum) * 256;
    }
  }
}
