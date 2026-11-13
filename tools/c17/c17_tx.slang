// C17 step 0b, throwaway: the RD pass's transform and TCQ of dumped residuals, one workgroup a block, every block of
// a dispatch one size. Integer DCT2 (AVM's matrices and shifts, full products), magnitudes in 1/256 steps, an
// eight-state Viterbi in int64 on lanes 0-7 (a lane a state, the predecessors' costs by subgroup reads), the
// backtrack and the dropped coefficients' squares. For timing only: close to the CPU's work, not its results.
struct Block {
  int at;   // the residual's first sample in res
  int dcq, acq;
  int pad;
};
struct Params {
  int16_t* res;
  Block* blocks;
  int* mat;    // DCT2 matrices 4, 8, 16, 32, 64 (rows: frequency), at 0, 16, 80, 336, 1360
  int* scan;   // this size's scan (DCT_DCT), then its iscan
  int* bits;   // level bits in 1/512 bit, levels 0-255
  int64_t* out;
  int n, gx;
  int lw, lh;  // log2 width, height
  int s0, s1, sqrt2, qshift;
  int first_q8, wb_num;  // Q0's threshold of level 1; a bit's weight: wb = bits * wb_num >> 9 (1/65536 step^2)
  int mode, pad;         // 1: the transform and magnitudes only
};
[vk::push_constant] ConstantBuffer<Params> pc;

static const int kInfHi = 0x10000000;  // 2^60 as the high half
groupshared int R[4096];  // the residual, then the Viterbi's history (two entries a word)
groupshared int Y[2048];  // the column pass, then the magnitudes
groupshared int C[1024];  // the coefficients
groupshared int g_eob;
groupshared int64_t g_part[64];

int mat_at(int n) { return n == 4 ? 0 : n == 8 ? 16 : n == 16 ? 80 : n == 32 ? 336 : 1360; }

int64_t sq(int64_t x) { return x * x; }

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gid: SV_GroupID, uint tid: SV_GroupIndex) {
  const int b = int(gid.y) * pc.gx + int(gid.x);
  if (b >= pc.n) return;
  const int t = int(tid);
  const int w = 1 << pc.lw, h = 1 << pc.lh, cw = min(w, 32), ch = min(h, 32), nc = cw * ch;
  const Block blk = pc.blocks[b];
  for (int i = t; i < w * h; i += 64) R[i] = int(pc.res[blk.at + i]);
  if (t == 0) g_eob = 0;
  GroupMemoryBarrierWithGroupSync();
  // Columns: Y[v][x], v < ch.
  {
    int* m = pc.mat + mat_at(h);
    const int off = pc.s0 > 0 ? 1 << (pc.s0 - 1) : 0;
    for (int i = t; i < ch * w; i += 64) {
      const int v = i >> pc.lw, x = i & (w - 1);
      int s = 0;
      for (int k = 0; k < h; k++) s += m[v * h + k] * R[k * w + x];
      Y[i] = (s + off) >> pc.s0;
    }
  }
  GroupMemoryBarrierWithGroupSync();
  // Rows: C[v][u], u < cw.
  {
    int* m = pc.mat + mat_at(w);
    const int off = 1 << (pc.s1 - 1);
    for (int i = t; i < nc; i += 64) {
      const int v = i / cw, u = i - v * cw;
      int s = 0;
      for (int k = 0; k < w; k++) s += m[u * w + k] * Y[v * w + k];
      int x = (s + off) >> pc.s1;
      if (pc.sqrt2 != 0) x = int((int64_t(x) * 5793 + 2048) >> 12);
      C[i] = x;
    }
  }
  GroupMemoryBarrierWithGroupSync();
  // Magnitudes in 1/256 steps (into Y), the end of block by Q0's threshold.
  int* scan = pc.scan;
  int* iscan = pc.scan + nc;
  for (int i = t; i < nc; i += 64) {
    const uint a = uint(abs(C[i]));
    const uint e = (a << (8 + pc.qshift)) / uint(i == 0 ? blk.dcq : blk.acq);
    Y[i] = int(e);
    if (int(e) >= pc.first_q8) InterlockedMax(g_eob, iscan[i] + 1);
  }
  GroupMemoryBarrierWithGroupSync();
  const int eob = g_eob;
  // The dropped coefficients' squares, every thread.
  int64_t dropped = 0;
  for (int c = eob + t; c < nc; c += 64) dropped += sq(int64_t(Y[scan[c]]));
  g_part[t] = dropped;
  int64_t best = 0;
  if (t < 8 && eob > 0 && pc.mode == 0) {
    // Predecessors of state t (tcq_levels()'s next[][]): p0 by an even level, p1 by an odd one.
    const int p0 = t == 0 ? 0 : t == 1 ? 2 : t == 2 ? 5 : t == 3 ? 7 : t == 4 ? 1 : t == 5 ? 3 : t == 6 ? 4 : 6;
    const int p1 = t == 0 ? 1 : t == 1 ? 3 : t == 2 ? 4 : t == 3 ? 6 : t == 4 ? 0 : t == 5 ? 2 : t == 6 ? 5 : 7;
    int64_t cost = t == 0 ? 0 : (int64_t(kInfHi) << 32);
    for (int c = eob - 1; c >= 0; c--) {
      const int pos = scan[c];
      const int e = Y[pos];
      const int maxl = 0xffffff / (2 * (pos == 0 ? blk.dcq : blk.acq));
      const bool last = c == eob - 1;
      // Candidates of each quantizer by parity: the least of d2 + wb.
      int64_t ce[2], co[2];
      int le[2], lo[2];
      for (int q1 = 0; q1 < 2; q1++) {
        ce[q1] = int64_t(kInfHi) << 32, co[q1] = int64_t(kInfHi) << 32, le[q1] = 0, lo[q1] = 0;
        const int base = q1 != 0 ? (e + 128) >> 8 : e >> 8;
        for (int k = 0; k < 3; k++) {
          const int l = min(k == 2 ? 0 : base + k, maxl);
          if ((l == 0 && last) || (k == 2 && base == 0)) continue;
          const int rec = l == 0 ? 0 : q1 != 0 ? (l << 8) - 128 : l << 8;
          const int64_t j = sq(int64_t(e - rec)) + ((int64_t(pc.bits[min(l, 255)]) * pc.wb_num) >> 9);
          if ((l & 1) != 0) {
            if (j < co[q1]) co[q1] = j, lo[q1] = l;
          } else if (j < ce[q1]) ce[q1] = j, le[q1] = l;
        }
      }
      const uint c0lo = WaveReadLaneAt(uint(cost), p0), c0hi = WaveReadLaneAt(uint(cost >> 32), p0);
      const uint c1lo = WaveReadLaneAt(uint(cost), p1), c1hi = WaveReadLaneAt(uint(cost >> 32), p1);
      const int64_t a0 = (int64_t(c0hi) << 32) | int64_t(c0lo), a1 = (int64_t(c1hi) << 32) | int64_t(c1lo);
      const int q0 = (p0 & 2) != 0 ? 1 : 0, qq1 = (p1 & 2) != 0 ? 1 : 0;
      const int64_t j0 = a0 + ce[q0], j1 = a1 + co[qq1];
      const bool take1 = j1 < j0;
      cost = min(take1 ? j1 : j0, int64_t(kInfHi) << 32);
      const int l = take1 ? lo[qq1] : le[q0];
      const uint entry = uint(min(l, 4095)) | (uint(take1 ? p1 : p0) << 12);
      const uint pair = entry | (WaveReadLaneAt(entry, t ^ 1) << 16);
      if ((t & 1) == 0) R[c * 4 + (t >> 1)] = int(pair);
    }
    // The least path, and the levels back from the DC end (their bits, as the CPU's sums).
    best = cost;
    for (int k = 0; k < 8; k++) {
      const uint lo2 = WaveReadLaneAt(uint(cost), k), hi2 = WaveReadLaneAt(uint(cost >> 32), k);
      best = min(best, (int64_t(hi2) << 32) | int64_t(lo2));
    }
    if (t == 0) {
      int s = 0;
      int64_t bitsum = 0;
      for (int c = 0; c < eob; c++) {
        const uint e = uint(R[c * 4 + (s >> 1)]) >> ((s & 1) * 16);
        bitsum += pc.bits[min(int(e & 4095), 255)];
        s = int(e >> 12) & 7;
      }
      best += bitsum & 1;  // keeps the walk
    }
  }
  GroupMemoryBarrierWithGroupSync();
  if (t == 0) {
    int64_t d = 0;
    for (int k = 0; k < 64; k++) d += g_part[k];
    pc.out[b * 2] = best + d;
    pc.out[b * 2 + 1] = eob;
  }
}
