#include "gpu/kernels_ref.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "gpu/inter_common.h"
#include "gpu/sb_blocks.h"

extern "C" {
#include "av2/common/quant_common.h"
}

int satd_ref(const uint8_t* a, int as, const uint8_t* b, int bs, int w4, int h4) {
  int total = 0;
  for (int ty = 0; ty < h4 * 4; ty += 4)
    for (int tx = 0; tx < w4 * 4; tx += 4) {
      int d[16];
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) d[i * 4 + j] = a[(ty + i) * as + tx + j] - b[(ty + i) * bs + tx + j];
      total += celv2::hadamard4x4(d);
    }
  return total;
}

namespace {

struct Best {
  int64_t cost;
  int mode;  // intra modes 0..kNumModes - 1 (IntraMode order), kNumModes = inter
  int mvx, mvy, ref;
  int ref2 = BlockDecision::kSingleRef, mvx2 = 0, mvy2 = 0;  // compound
};

struct Both {
  Best intra{INT64_MAX, 0, 0, 0, 0}, inter{INT64_MAX, celv2::kNumModes, 0, 0, 0};
  Best per_ref[celv2::kMaxRefs];  // the first nrefs: each reference's best
  int nrefs = 0;
};

// The intra modes in order, and apart each reference's candidates in order, then the compound of the
// two best references (first strict minimum wins in each).
Both best_both(const celv2::Plane& s, const RefSet& rs, int x, int y, int w, int h, int64_t lam, bool intra) {
  Both both;
  Best& b = both.intra;
  for (int m = 0; m < celv2::kNumModes && intra; m++) {
    const int64_t c = celv2::block_mode_satd(s, x, y, w, h, m) * 256 +
                      lam * (celv2::mode_bits(m) + (rs.size() > 0 ? celv2::kIntraInInterBits : 0));
    if (c < b.cost) b = Best{c, m, 0, 0, 0};
  }
  // Each reference's first strict minimum, then the best of them (first strict minimum).
  Best per_ref[celv2::kMaxRefs];
  int a = -1, second = -1;
  for (int r = 0; r < rs.size(); r++) {
    const RefMvLevel* mf = rs.field[r];
    per_ref[r] = Best{INT64_MAX, celv2::kNumModes, 0, 0, r};
    for (int k = 0; k < celv2::mv_candidates(x, y, w, h); k++) {
      if (celv2::repeated_candidate(mf->mvx.data(), mf->mvy.data(), mf->cols, mf->rows, x, y, w, h, k)) continue;
      const int i = celv2::candidate_mv_index(x, y, w, h, k, mf->cols, mf->rows);
      const int64_t c =
          celv2::block_inter_satd(s, *rs.plane[r], x, y, w, h, mf->mvx[i], mf->mvy[i]) * 256 +
          lam * celv2::inter_bits(r);
      if (c < per_ref[r].cost) per_ref[r] = Best{c, celv2::kNumModes, mf->mvx[i], mf->mvy[i], r};
    }
    if (a < 0 || per_ref[r].cost < per_ref[a].cost) a = r;
  }
  for (int r = 0; r < rs.size(); r++)
    if (r != a && (second < 0 || per_ref[r].cost < per_ref[second].cost)) second = r;
  if (a >= 0) both.inter = per_ref[a];
  std::copy(per_ref, per_ref + rs.size(), both.per_ref);
  both.nrefs = rs.size();
  if (second >= 0 && celv2::compound_pair_ok(a, second, rs.size())) {
    const Best &p = per_ref[a], &q = per_ref[second];
    const int64_t c = celv2::block_compound_satd(s, *rs.plane[a], p.mvx, p.mvy, *rs.plane[second], q.mvx, q.mvy, x, y,
                                                w, h) * 256 + lam * celv2::compound_bits(a, second);
    if (c < both.inter.cost) both.inter = Best{c, celv2::kNumModes, p.mvx, p.mvy, a, second, q.mvx, q.mvy};
  }
  return both;
}

// Where best_mode() records each block's candidates: the superblock's list and origin.
struct Cands {
  std::vector<BlockCandidates>* v;
  int ox, oy;
};

// The block's best choice (inter where it costs strictly less than the best intra mode), its
// candidates recorded.
Best best_mode(const celv2::Plane& s, const RefSet& rs, int x, int y, int w, int h, int64_t lam, Cands& rec,
               bool intra = true) {
  const Both b = best_both(s, rs, x, y, w, h, lam, intra);
  if (!rec.v) return b.inter.cost < b.intra.cost ? b.inter : b.intra;
  BlockCandidates& c = (*rec.v)[celv2::sb_block_index(x - rec.ox, y - rec.oy, w, h)];
  c.inter_cost = b.inter.cost, c.intra_cost = b.intra.cost, c.intra = static_cast<IntraMode>(b.intra.mode);
  c.inter.mv_col = static_cast<int16_t>(b.inter.mvx * 2), c.inter.mv_row = static_cast<int16_t>(b.inter.mvy * 2);
  c.inter.ref = static_cast<uint8_t>(b.inter.ref), c.inter.ref2 = static_cast<uint8_t>(b.inter.ref2);
  c.inter.mv2_col = static_cast<int16_t>(b.inter.mvx2 * 2), c.inter.mv2_row = static_cast<int16_t>(b.inter.mvy2 * 2);
  for (int r = 0; r < b.nrefs; r++)  // as finalize_kernel
    if (b.per_ref[r].cost != INT64_MAX) {
      c.ref_cost[r] = static_cast<uint32_t>(std::min(b.per_ref[r].cost >> 8, static_cast<int64_t>(UINT32_MAX - 1)));
      c.ref_mv_col[r] = static_cast<int16_t>(b.per_ref[r].mvx * 2), c.ref_mv_row[r] = static_cast<int16_t>(b.per_ref[r].mvy * 2);
    }
  return b.inter.cost < b.intra.cost ? b.inter : b.intra;
}

void leaf(SbDecision& sb, int w, int h, const Best& b) {
  sb.tree.push_back(Part::None);
  BlockDecision d;
  d.w4 = static_cast<uint8_t>(w / 4);
  d.h4 = static_cast<uint8_t>(h / 4);
  d.inter = b.mode == celv2::kNumModes;
  d.intra = d.inter ? IntraMode::DC : static_cast<IntraMode>(b.mode);
  d.mv_col = static_cast<int16_t>(d.inter ? b.mvx * 2 : 0);  // quarter pels to 1/8
  d.mv_row = static_cast<int16_t>(d.inter ? b.mvy * 2 : 0);
  d.ref = static_cast<uint8_t>(d.inter ? b.ref : 0);
  if (d.inter && b.ref2 != BlockDecision::kSingleRef) {
    d.ref2 = static_cast<uint8_t>(b.ref2);
    d.mv2_col = static_cast<int16_t>(b.mvx2 * 2);
    d.mv2_row = static_cast<int16_t>(b.mvy2 * 2);
  }
  sb.blocks.push_back(d);
}

// Bottom-up decision for the square at (x, y); appends its subtree to `out`, returns its cost.
int64_t decide_square(const celv2::Plane& s, const RefSet& rs, int x, int y, int size,
                      int64_t lam, SbDecision& out, Cands& rec) {
  const int half = size / 2;
  const int64_t part = lam * celv2::kPartBits;
  if (size > celv2::kMaxBlockSize) {  // SPLIT, or (inter frames, up to kMaxInterBlockSize) one inter block or two
    // NONE, HORZ, VERT, then SPLIT (first strict minimum, as finalize_kernel).
    SbDecision split;
    int64_t c_split = part;
    for (int q = 0; q < 4; q++)
      c_split += decide_square(s, rs, x + (q & 1) * half, y + (q >> 1) * half, half, lam, split, rec);
    int64_t c = INT64_MAX;
    int choice = 3;
    Best none{}, h0{}, h1{}, v0{}, v1{};
    if (rs.size() > 0 && size <= celv2::kMaxInterBlockSize) {
      none = best_mode(s, rs, x, y, size, size, lam, rec, false);
      h0 = best_mode(s, rs, x, y, size, half, lam, rec, false);
      h1 = best_mode(s, rs, x, y + half, size, half, lam, rec, false);
      v0 = best_mode(s, rs, x, y, half, size, lam, rec, false);
      v1 = best_mode(s, rs, x + half, y, half, size, lam, rec, false);
      const int64_t c_horz = part + h0.cost + h1.cost, c_vert = part + v0.cost + v1.cost;
      c = none.cost, choice = 0;
      if (c_horz < c) c = c_horz, choice = 1;
      if (c_vert < c) c = c_vert, choice = 2;
    }
    if (c_split < c) c = c_split, choice = 3;
    switch (choice) {
      case 0: leaf(out, size, size, none); break;
      case 1: out.tree.push_back(Part::Horz), leaf(out, size, half, h0), leaf(out, size, half, h1); break;
      case 2: out.tree.push_back(Part::Vert), leaf(out, half, size, v0), leaf(out, half, size, v1); break;
      default:
        out.tree.push_back(Part::Split);
        out.tree.insert(out.tree.end(), split.tree.begin(), split.tree.end());
        out.blocks.insert(out.blocks.end(), split.blocks.begin(), split.blocks.end());
    }
    return c;
  }
  const Best none = best_mode(s, rs, x, y, size, size, lam, rec);
  const Best h0 = best_mode(s, rs, x, y, size, half, lam, rec);
  const Best h1 = best_mode(s, rs, x, y + half, size, half, lam, rec);
  const Best v0 = best_mode(s, rs, x, y, half, size, lam, rec);
  const Best v1 = best_mode(s, rs, x + half, y, half, size, lam, rec);
  const int64_t c_horz = part + h0.cost + h1.cost, c_vert = part + v0.cost + v1.cost;
  SbDecision split;
  int64_t c_split = INT64_MAX;
  if (size > 8) {
    c_split = part;
    for (int q = 0; q < 4; q++)
      c_split += decide_square(s, rs, x + (q & 1) * half, y + (q >> 1) * half, half, lam, split, rec);
  }
  int64_t best = none.cost;
  int choice = 0;
  if (c_horz < best) best = c_horz, choice = 1;
  if (c_vert < best) best = c_vert, choice = 2;
  if (c_split < best) best = c_split, choice = 3;
  switch (choice) {
    case 0: leaf(out, size, size, none); break;
    case 1:
      out.tree.push_back(Part::Horz);
      leaf(out, size, half, h0);
      leaf(out, size, half, h1);
      break;
    case 2:
      out.tree.push_back(Part::Vert);
      leaf(out, half, size, v0);
      leaf(out, half, size, v1);
      break;
    default:
      out.tree.push_back(Part::Split);
      out.tree.insert(out.tree.end(), split.tree.begin(), split.tree.end());
      out.blocks.insert(out.blocks.end(), split.blocks.begin(), split.blocks.end());
      break;
  }
  return best;
}

DecisionMap decide_all(const Frame420& src, const std::vector<const RefPyramid*>& refs,
                       const std::vector<const RefMvLevel*>& level0, int sb_size, int qindex) {
  DecisionMap dm;
  dm.sb_size = sb_size;
  dm.sb_cols = (src.width + sb_size - 1) / sb_size;
  dm.sb_rows = (src.height + sb_size - 1) / sb_size;
  for (int r = 0; r < dm.sb_rows; r++)
    for (int c = 0; c < dm.sb_cols; c++) dm.sbs.push_back(decide_sb_ref(src, refs, level0, c, r, sb_size, qindex));
  return dm;
}

}  // namespace

RefPyramid build_pyramid_ref(const Frame420& f, bool phases) {
  RefPyramid pyr;
  pyr.px.resize(celv2::kMeLevels);
  pyr.plane.resize(celv2::kMeLevels);
  const size_t n = f.y.size();
  pyr.px[0].assign(f.y.begin(), f.y.end());
  if (phases) {
    pyr.px[0].resize(n * celv2::kSubpelPhases);
    const celv2::Plane src{f.y.data(), f.stride_y(), f.pw, f.ph};
    for (int k = 1; k < celv2::kSubpelPhases; k++)
      for (int y = 0; y < f.ph; y++)
        for (int x = 0; x < f.pw; x++)
          pyr.px[0][k * n + static_cast<size_t>(y) * f.stride_y() + x] =
              static_cast<uint8_t>(celv2::subpel_px(src, y, x, k / 4, k % 4));
  }
  pyr.plane[0] = celv2::Plane{pyr.px[0].data(), f.stride_y(), f.pw, f.ph, phases ? n : 0};
  for (int l = 1; l < celv2::kMeLevels; l++) {
    const celv2::Plane& up = pyr.plane[l - 1];
    const int w = up.w / 2, h = up.h / 2;
    pyr.px[l].resize(static_cast<size_t>(w) * h);
    pyr.plane[l] = celv2::Plane{pyr.px[l].data(), w, w, h};
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++) pyr.px[l][y * w + x] = static_cast<uint8_t>(celv2::downsample_px(up, y, x));
  }
  return pyr;
}

void me_level_rows_ref(const RefPyramid& cur, const RefPyramid& ref, int l, const RefMvLevel* coarser, int row0,
                       int row1, RefMvLevel& out) {
  for (int by = row0; by < row1; by++)
    for (int bx = 0; bx < out.cols; bx++) {
      int* mx = &out.mvx[by * out.cols + bx];
      int* my = &out.mvy[by * out.cols + bx];
      celv2::me_cell(cur.plane[l], ref.plane[l], bx, by, coarser ? coarser->mvx.data() : nullptr,
                    coarser ? coarser->mvy.data() : nullptr, coarser ? coarser->cols : 0,
                    coarser ? coarser->rows : 0, mx, my, l == 0);
      if (l == 0) celv2::me_refine(cur.plane[0], ref.plane[0], bx, by, mx, my);
    }
}

SbDecision decide_sb_ref(const Frame420& src, const std::vector<const RefPyramid*>& refs,
                         const std::vector<const RefMvLevel*>& level0, int sb_col, int sb_row, int sb_size,
                         int qindex) {
  const celv2::Plane s{src.y.data(), src.stride_y(), src.pw, src.ph};
  RefSet rs;
  for (size_t r = 0; r < level0.size(); r++) rs.plane.push_back(&refs[r]->plane[0]), rs.field.push_back(level0[r]);
  SbDecision sb;
  if (sb_size == 64) sb.cands.resize(celv2::kSbBlocks);  // the candidate blocks are a 64x64 superblock's
  for (int k = 0; k < static_cast<int>(sb.cands.size()); k++) {
    int x, y, w, h;
    celv2::sb_block_geom(k, &x, &y, &w, &h);
    sb.cands[k].inter.w4 = static_cast<uint8_t>(w / 4), sb.cands[k].inter.h4 = static_cast<uint8_t>(h / 4);
    sb.cands[k].inter.inter = true;
  }
  Cands rec{sb.cands.empty() ? nullptr : &sb.cands, sb_col * sb_size, sb_row * sb_size};
  decide_square(s, rs, sb_col * sb_size, sb_row * sb_size, sb_size, lambda_q8(qindex), sb, rec);
  return sb;
}

SbDecision decide_sb_ref(const Frame420& src, const RefPyramid* ref, const RefMvLevel* level0, int sb_col,
                         int sb_row, int sb_size, int qindex) {
  if (!level0) return decide_sb_ref(src, std::vector<const RefPyramid*>{}, {}, sb_col, sb_row, sb_size, qindex);
  return decide_sb_ref(src, std::vector<const RefPyramid*>{ref}, {level0}, sb_col, sb_row, sb_size, qindex);
}

DecisionMap decide_intra_ref(const Frame420& src, int sb_size, int qindex) {
  return decide_all(src, {}, {}, sb_size, qindex);
}

DecisionMap decide_inter_ref(const Frame420& src, const std::vector<const Frame420*>& refs, int sb_size,
                             int qindex) {
  const RefPyramid cur = build_pyramid_ref(src);
  std::vector<RefPyramid> pyr;
  std::vector<std::vector<RefMvLevel>> lv(refs.size(), std::vector<RefMvLevel>(celv2::kMeLevels));
  for (const Frame420* f : refs) pyr.push_back(build_pyramid_ref(*f, true));
  std::vector<const RefPyramid*> pp;
  std::vector<const RefMvLevel*> fields;
  for (size_t r = 0; r < refs.size(); r++) {
    for (int l = celv2::kMeLevels - 1; l >= 0; l--) {
      RefMvLevel& o = lv[r][l];
      o.cols = (cur.plane[l].w + 7) / 8, o.rows = (cur.plane[l].h + 7) / 8;
      o.mvx.assign(static_cast<size_t>(o.cols) * o.rows, 0), o.mvy = o.mvx;
      me_level_rows_ref(cur, pyr[r], l, l == celv2::kMeLevels - 1 ? nullptr : &lv[r][l + 1], 0, o.rows, o);
    }
    pp.push_back(&pyr[r]);
    fields.push_back(&lv[r][0]);
  }
  return decide_all(src, pp, fields, sb_size, qindex);
}

DecisionMap decide_inter_ref(const Frame420& src, const Frame420& prev_src, int sb_size, int qindex) {
  return decide_inter_ref(src, std::vector<const Frame420*>{&prev_src}, sb_size, qindex);
}
