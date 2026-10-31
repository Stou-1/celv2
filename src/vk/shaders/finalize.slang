// finalize_kernel (gpu/kernels.cu): per superblock, each block's best candidate from cost's costs,
// then the bottom-up partition choice and the SbOut emission.
import common;

static const int kSbOutCap = 256;
// gpu/kernels.h's SbOut, field for field.
struct SbOut {
  int ntree, nblocks;
  uint8_t tree[kSbOutCap];
  uint8_t w4[kSbOutCap], h4[kSbOutCap], mode[kSbOutCap];
  int16_t mvx[kSbOutCap], mvy[kSbOutCap];
  uint8_t ref[kSbOutCap];
  uint8_t ref2[kSbOutCap];
  int16_t mv2x[kSbOutCap], mv2y[kSbOutCap];
  int64_t c_inter_cost[kSbBlocks], c_intra_cost[kSbBlocks];
  int16_t c_mvx[kSbBlocks], c_mvy[kSbBlocks], c_mv2x[kSbBlocks], c_mv2y[kSbBlocks];
  uint8_t c_ref[kSbBlocks], c_ref2[kSbBlocks], c_intra[kSbBlocks];
  uint32_t c_rcost[7][kSbBlocks];
  int16_t c_rmvx[7][kSbBlocks], c_rmvy[7][kSbBlocks];
};

struct Params {
  Plane s;
  Tables* tab;
  KRefs* refs;
  int64_t* intra;
  int64_t* costs;
  SbOut* out;
  int64_t lam;
  int sb_cols, row0;
};
[vk::push_constant] ConstantBuffer<Params> pc;

static const int kThreads = 256;
static const int64_t kMax = 0x7fffffffffffffffLL;
groupshared int64_t best_cost[kSbBlocks];
groupshared int best_mode[kSbBlocks], best_ref[kSbBlocks], best_ref2[kSbBlocks];
groupshared int best_mv[kSbBlocks][2], best_mv2[kSbBlocks][2];
groupshared int64_t sq_cost[kSbSquares];
groupshared int sq_choice[kSbSquares];

void leaf(SbOut* o, int b, inout int ntree, inout int nblocks) {
  int x, y, w, h;
  sb_block_geom(b, x, y, w, h);
  const bool inter = best_mode[b] == kNumModes;
  o.tree[ntree++] = uint8_t(0);
  o.w4[nblocks] = uint8_t(w / 4);
  o.h4[nblocks] = uint8_t(h / 4);
  o.mode[nblocks] = uint8_t(best_mode[b]);
  o.mvx[nblocks] = int16_t(inter ? best_mv[b][0] : 0);
  o.ref[nblocks] = uint8_t(inter ? best_ref[b] : 0);
  o.ref2[nblocks] = uint8_t(inter ? best_ref2[b] : 255);
  o.mv2x[nblocks] = int16_t(inter ? best_mv2[b][0] : 0);
  o.mv2y[nblocks] = int16_t(inter ? best_mv2[b][1] : 0);
  o.mvy[nblocks++] = int16_t(inter ? best_mv[b][1] : 0);
}

[shader("compute")]
[numthreads(kThreads, 1, 1)]
void main(uint3 gid: SV_GroupID, uint tid: SV_GroupIndex) {
  const int sbx = int(gid.x), sby = pc.row0 + int(gid.y);
  const int ox = sbx * 64, oy = sby * 64;
  const uint64_t sb = uint64_t(sby) * pc.sb_cols + sbx;
  const int nrefs = pc.refs.n;
  int64_t* cost = pc.costs + (sb - uint64_t(pc.row0) * pc.sb_cols) * uint64_t(nrefs * kInterSlots);  // from row0 on, as cost's
  int64_t* icost = pc.intra + sb * kNumIntra;
  SbOut* oc = pc.out + sb;
  for (int b = int(tid); b < kSbBlocks; b += kThreads) {
    int64_t ci = kMax;
    int bx = 0, by = 0, br = 0, br2 = 255, bx2 = 0, by2 = 0;
    int x, y, w, h;
    sb_block_geom(b, x, y, w, h);
    int64_t rc[kMaxRefs];
    int rmx[kMaxRefs], rmy[kMaxRefs];
    int a = -1, second = -1;
    for (int r = 0; r < nrefs; r++) {
      rc[r] = kMax;
      rmx[r] = 0;
      rmy[r] = 0;
      for (int it = pc.tab.inter_first[b]; it < pc.tab.inter_first[b + 1]; it++)
        if (cost[r * kInterSlots + it] < rc[r]) {
          rc[r] = cost[r * kInterSlots + it];
          const int i = candidate_mv_index(x + ox, y + oy, w, h, pc.tab.inter_k[it], pc.refs.mv_cols, pc.refs.mv_rows);
          rmx[r] = pc.refs.mvx[r][i];
          rmy[r] = pc.refs.mvy[r][i];
        }
      if (a < 0 || rc[r] < rc[a]) a = r;
    }
    for (int r = 0; r < nrefs; r++)
      if (r != a && (second < 0 || rc[r] < rc[second])) second = r;
    if (a >= 0) { ci = rc[a]; bx = rmx[a]; by = rmy[a]; br = a; }
    if (second >= 0 && compound_pair_ok(a, second, nrefs)) {
      const int64_t cc = block_compound_satd(pc.s, pc.refs.p[a], rmx[a], rmy[a], pc.refs.p[second], rmx[second], rmy[second],
                                             x + ox, y + oy, w, h) * 256 + pc.lam * compound_bits(a, second);
      if (cc < ci) { ci = cc; br2 = second; bx2 = rmx[second]; by2 = rmy[second]; }
    }
    int64_t c = kMax;  // the 64x64 level: inter only (kMaxBlockSize)
    int m = 0;
    for (int k = 0; k < kNumModes && b >= 5; k++) {
      const int64_t ck = icost[b * kNumModes + k] + pc.lam * (mode_bits(k) + (nrefs > 0 ? kIntraInInterBits : 0));
      if (ck < c) { c = ck; m = k; }
    }
    for (int r = 0; r < kMaxRefs; r++) {
      const bool has = r < nrefs && rc[r] != kMax;
      oc.c_rcost[r][b] = has ? uint32_t(min(rc[r] >> 8, int64_t(0xfffffffe))) : 0xffffffffu;
      oc.c_rmvx[r][b] = int16_t(has ? rmx[r] : 0);
      oc.c_rmvy[r][b] = int16_t(has ? rmy[r] : 0);
    }
    oc.c_inter_cost[b] = ci;
    oc.c_intra_cost[b] = c;
    oc.c_intra[b] = uint8_t(m);
    oc.c_mvx[b] = int16_t(bx);
    oc.c_mvy[b] = int16_t(by);
    oc.c_ref[b] = uint8_t(br);
    oc.c_ref2[b] = uint8_t(br2);
    oc.c_mv2x[b] = int16_t(bx2);
    oc.c_mv2y[b] = int16_t(by2);
    if (ci < c) { c = ci; m = kNumModes; }
    best_cost[b] = c;
    best_mode[b] = m;
    best_mv[b][0] = bx;
    best_mv[b][1] = by;
    best_ref[b] = br;
    best_ref2[b] = br2;
    best_mv2[b][0] = bx2;
    best_mv2[b][1] = by2;
  }
  GroupMemoryBarrierWithGroupSync();
  if (tid != 0) return;
  const int64_t part = pc.lam * kPartBits;
  for (int level = 3; level >= 0; level--) {
    const int n = 1 << level;
    for (int i = 0; i < n * n; i++) {
      const int sq = sb_level_offset(level) + i, b = sq * 5;
      int64_t best = kMax;
      int choice = 3;
      if ((64 >> level) <= kMaxBlockSize || (nrefs > 0 && (64 >> level) <= kMaxInterBlockSize)) {
        const int64_t c_horz = part + best_cost[b + 1] + best_cost[b + 2];
        const int64_t c_vert = part + best_cost[b + 3] + best_cost[b + 4];
        best = best_cost[b];
        choice = 0;
        if (c_horz < best) { best = c_horz; choice = 1; }
        if (c_vert < best) { best = c_vert; choice = 2; }
      }
      if (level < 3) {
        const int qx = i % n, qy = i / n;
        int64_t c_split = part;
        for (int q = 0; q < 4; q++) c_split += sq_cost[sb_square_id(level + 1, 2 * qx + (q & 1), 2 * qy + (q >> 1))];
        if (c_split < best) { best = c_split; choice = 3; }
      }
      sq_cost[sq] = best;
      sq_choice[sq] = choice;
    }
  }
  int ntree = 0, nblocks = 0;
  int stack_level[16], stack_qx[16], stack_qy[16];
  stack_level[0] = 0;
  stack_qx[0] = 0;
  stack_qy[0] = 0;
  int sp = 1;
  while (sp > 0) {
    sp--;
    const int level = stack_level[sp], qx = stack_qx[sp], qy = stack_qy[sp];
    const int sq = sb_square_id(level, qx, qy), b = sq * 5;
    const int choice = sq_choice[sq];
    if (choice == 0) {
      leaf(oc, b, ntree, nblocks);
    } else if (choice == 1) {
      oc.tree[ntree++] = uint8_t(1);
      leaf(oc, b + 1, ntree, nblocks);
      leaf(oc, b + 2, ntree, nblocks);
    } else if (choice == 2) {
      oc.tree[ntree++] = uint8_t(2);
      leaf(oc, b + 3, ntree, nblocks);
      leaf(oc, b + 4, ntree, nblocks);
    } else {
      oc.tree[ntree++] = uint8_t(3);
      for (int q = 3; q >= 0; q--) {  // push in reverse so quadrant 0 is emitted first
        stack_level[sp] = level + 1;
        stack_qx[sp] = 2 * qx + (q & 1);
        stack_qy[sp] = 2 * qy + (q >> 1);
        sp++;
      }
    }
  }
  oc.ntree = ntree;
  oc.nblocks = nblocks;
}
