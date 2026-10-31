#include "gpu/decision_map.h"

#include "gpu/sb_out.h"

DecisionMap uniform_decision(int width, int height, int sb_size) {
  DecisionMap dm;
  dm.sb_size = sb_size;
  dm.sb_cols = (width + sb_size - 1) / sb_size;
  dm.sb_rows = (height + sb_size - 1) / sb_size;
  SbDecision sb;
  sb.tree = {Part::None};
  BlockDecision b;
  b.w4 = b.h4 = static_cast<uint8_t>(sb_size / 4);
  sb.blocks = {b};
  dm.sbs.assign(static_cast<size_t>(dm.sb_cols) * dm.sb_rows, sb);
  return dm;
}

void RowGate::publish(int row) {
  {
    std::lock_guard<std::mutex> lk(m_);
    if (row + 1 > ready_) ready_ = row + 1;
  }
  cv_.notify_all();
}

void RowGate::publish_one(int row) {
  {
    std::lock_guard<std::mutex> lk(m_);
    if (one_.size() <= static_cast<size_t>(row)) one_.resize(row + 1, 0);
    one_[row] = 1;
    while (static_cast<size_t>(ready_) < one_.size() && one_[ready_]) ready_++;
  }
  cv_.notify_all();
}

void RowGate::wait_row(int row) const {
  std::unique_lock<std::mutex> lk(m_);
  cv_.wait(lk, [&] { return error_ || ready_ > row || (static_cast<size_t>(row) < one_.size() && one_[row]); });
  if (error_) std::rethrow_exception(error_);
}

void RowGate::fail(std::exception_ptr e) {
  {
    std::lock_guard<std::mutex> lk(m_);
    if (!error_) error_ = e;
  }
  cv_.notify_all();
}

int RowGate::ready_rows() const {
  std::lock_guard<std::mutex> lk(m_);
  return ready_;
}

namespace celv2 {

void sb_out_to_decision(const SbOut& o, SbDecision& d) {
  for (int k = 0; k < o.ntree; k++) d.tree.push_back(static_cast<Part>(o.tree[k]));
  for (int k = 0; k < o.nblocks; k++) {
    BlockDecision b;
    b.w4 = o.w4[k];
    b.h4 = o.h4[k];
    b.inter = o.mode[k] == kNumModes;
    b.intra = b.inter ? IntraMode::DC : static_cast<IntraMode>(o.mode[k]);
    b.mv_col = static_cast<int16_t>(o.mvx[k] * 2);  // quarter pels to 1/8
    b.mv_row = static_cast<int16_t>(o.mvy[k] * 2);
    b.ref = o.ref[k];
    if (b.inter && o.ref2[k] != BlockDecision::kSingleRef) {
      b.ref2 = o.ref2[k];
      b.mv2_col = static_cast<int16_t>(o.mv2x[k] * 2);
      b.mv2_row = static_cast<int16_t>(o.mv2y[k] * 2);
    }
    d.blocks.push_back(b);
  }
  std::vector<BlockCandidates>& cands = d.cands;
  cands.resize(kSbBlocks);
  for (int k = 0; k < kSbBlocks; k++) {
    BlockCandidates& c = cands[k];
    int x, y, w, h;
    sb_block_geom(k, &x, &y, &w, &h);
    c.inter_cost = o.c_inter_cost[k], c.intra_cost = o.c_intra_cost[k];
    c.intra = static_cast<IntraMode>(o.c_intra[k]);
    c.inter.w4 = static_cast<uint8_t>(w / 4), c.inter.h4 = static_cast<uint8_t>(h / 4);
    c.inter.inter = true;
    c.inter.mv_col = static_cast<int16_t>(o.c_mvx[k] * 2), c.inter.mv_row = static_cast<int16_t>(o.c_mvy[k] * 2);
    c.inter.ref = o.c_ref[k];
    for (int r = 0; r < BlockCandidates::kRefs; r++) {
      c.ref_cost[r] = o.c_rcost[r][k];
      c.ref_mv_col[r] = static_cast<int16_t>(o.c_rmvx[r][k] * 2), c.ref_mv_row[r] = static_cast<int16_t>(o.c_rmvy[r][k] * 2);
    }
    if (o.c_ref2[k] != BlockDecision::kSingleRef) {
      c.inter.ref2 = o.c_ref2[k];
      c.inter.mv2_col = static_cast<int16_t>(o.c_mv2x[k] * 2);
      c.inter.mv2_row = static_cast<int16_t>(o.c_mv2y[k] * 2);
    }
  }
}

}  // namespace celv2
