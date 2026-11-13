// me.slang's motion search with one workgroup of 64 threads per block (levels of at most kMeWideBlocks blocks): common.slang's me_cell() and me_refine() with each centre's 289 positions,
// and each refinement step's 9, split over the threads, and the first least cost found by a reduction over (cost,
// position in their order): the same field. One thread a block (until 2026-11-09) left most of the GPU idle at the
// coarse levels and read scattered windows at level 0: me -56% on the Arc Pro B50, -58% on the RTX 5090.
import common;

struct Params {
  Plane cur, ref;
  int* pred_x;  // coarser level's field (null at the top level)
  int* pred_y;
  int* mvx;
  int* mvy;
  int pred_cols, pred_rows, cols, rows, refine;
};
[vk::push_constant] ConstantBuffer<Params> pc;

static const int kThreads = 64;
static const int kMePenalty = 4;       // common.slang's (private there)
static const int kSubpelPenalty = 16;
groupshared int g_cost[kThreads];
groupshared int g_idx[kThreads];
groupshared int g_sad[6];  // the predictor centres' SADs

// The group's least (cost, idx), idx breaking ties: the first least of a search in idx order.
int2 least(int cost, int idx, int t) {
  g_cost[t] = cost;
  g_idx[t] = idx;
  GroupMemoryBarrierWithGroupSync();
  for (int s = kThreads / 2; s > 0; s >>= 1) {
    if (t < s) {
      const int c = g_cost[t + s], i = g_idx[t + s];
      if (c < g_cost[t] || (c == g_cost[t] && i < g_idx[t])) {
        g_cost[t] = c;
        g_idx[t] = i;
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }
  const int2 r = int2(g_cost[0], g_idx[0]);
  GroupMemoryBarrierWithGroupSync();  // every thread has read them before they are written again
  return r;
}

// me_search() around (px, py): (its least cost, the position's index dy-major from (-range, -range)).
int2 search(int bx, int by, int px, int py, int t) {
  const int n = 2 * kMeRange + 1;
  int best = 0x7fffffff, at = 0x7fffffff;
  for (int i = t; i < n * n; i += kThreads) {
    const int dy = i / n - kMeRange, dx = i % n - kMeRange;
    const int s = sad8(pc.cur, bx * 8, by * 8, pc.ref, px + dx, py + dy) +
                  kMePenalty * ((dx < 0 ? -dx : dx) + (dy < 0 ? -dy : dy));
    if (s < best) {
      best = s;
      at = i;
    }
  }
  return least(best, at, t);
}

// The k-th predictor centre (k == 5: zero), as me_cell().
int2 centre(int bx, int by, int k) {
  if (k == 5) return int2(0, 0);
  const int ox = k == 1 ? -1 : k == 2 ? 1 : 0, oy = k == 3 ? -1 : k == 4 ? 1 : 0;
  int qx = bx / 2 + ox, qy = by / 2 + oy;
  qx = qx < 0 ? 0 : qx >= pc.pred_cols ? pc.pred_cols - 1 : qx;
  qy = qy < 0 ? 0 : qy >= pc.pred_rows ? pc.pred_rows - 1 : qy;
  return int2(2 * pc.pred_x[qy * pc.pred_cols + qx], 2 * pc.pred_y[qy * pc.pred_cols + qx]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gid: SV_GroupID, uint3 tid: SV_GroupThreadID) {
  const int bx = int(gid.x), by = int(gid.y), t = int(tid.x);
  int px = 0, py = 0, px2 = 0, py2 = 0;
  bool two = false;
  if (pc.pred_x != nullptr) {
    // The six centres' SADs at once, then me_cell()'s choice among them in its order (a skipped centre's SAD unused).
    if (t < 6) {
      const int2 c = centre(bx, by, t);
      g_sad[t] = sad8(pc.cur, bx * 8, by * 8, pc.ref, c.x, c.y);
    }
    GroupMemoryBarrierWithGroupSync();
    int best = 0x7fffffff, second = 0x7fffffff;
    for (int k = 0; k < 6; k++) {
      const int2 c = centre(bx, by, k);
      if ((k > 0 && c.x == px && c.y == py) || (two && c.x == px2 && c.y == py2)) continue;  // each centre once
      const int s = g_sad[k];
      if (s < best) {
        if (k > 0) {
          second = best;
          px2 = px;
          py2 = py;
          two = true;
        }
        best = s;
        px = c.x;
        py = c.y;
      } else if (s < second) {
        second = s;
        px2 = c.x;
        py2 = c.y;
        two = true;
      }
    }
  }
  const int n = 2 * kMeRange + 1;
  const int2 a = search(bx, by, px, py, t);
  int mvx = px + a.y % n - kMeRange, mvy = py + a.y / n - kMeRange;
  if (two) {
    const int2 b = search(bx, by, px2, py2, t);
    if (b.x < a.x) mvx = px2 + b.y % n - kMeRange, mvy = py2 + b.y / n - kMeRange;
  }
  if (pc.refine != 0) {
    // me_refine(): per step the centre (index 0, no penalty) and its 8 neighbours in me_refine()'s order, the first
    // least by the reduction (the centre wins a tie, as there).
    int cx = mvx * 4, cy = mvy * 4;
    for (int step = 2; step >= 1; step--) {
      int cost = 0x7fffffff;
      if (t < 9) {
        const int k = t == 0 ? 4 : t <= 4 ? t - 1 : t;  // index t of the order centre, (-1,-1), (0,-1) ... (1,1)
        const int dx = k % 3 - 1, dy = k / 3 - 1;
        cost = satd8_q(pc.cur, bx * 8, by * 8, pc.ref, cx + dx * step, cy + dy * step) + (t == 0 ? 0 : kSubpelPenalty);
      }
      const int2 r = least(cost, t, t);
      const int k = r.y == 0 ? 4 : r.y <= 4 ? r.y - 1 : r.y;
      cx += (k % 3 - 1) * step, cy += (k / 3 - 1) * step;
    }
    mvx = cx, mvy = cy;
  }
  if (t == 0) {
    pc.mvx[by * pc.cols + bx] = mvx;
    pc.mvy[by * pc.cols + bx] = mvy;
  }
}
