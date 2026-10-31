// me_kernel (gpu/kernels.cu): the 8x8 motion search of one pyramid level, quarter-pel refined at level 0.
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

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID) {
  const int bx = int(id.x), by = int(id.y);
  if (bx >= pc.cols || by >= pc.rows) return;
  int mx, my;
  me_cell(pc.cur, pc.ref, bx, by, pc.pred_x, pc.pred_y, pc.pred_cols, pc.pred_rows, mx, my, pc.refine != 0);
  if (pc.refine != 0) me_refine(pc.cur, pc.ref, bx, by, mx, my);
  pc.mvx[by * pc.cols + bx] = mx;
  pc.mvy[by * pc.cols + bx] = my;
}
