#include "gpu/cpu_engine.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

int decision_threads(int cores, int tile_threads) {
  return std::max(1, (cores > 0 ? cores : 1) - tile_threads);
}

CpuEngine::CpuEngine(int width, int height, int sb_size, int threads, std::function<void(int row)> row_hook)
    : width_(width),
      height_(height),
      sb_size_(sb_size),
      threads_(std::max(1, threads)),
      sb_cols_((width + sb_size - 1) / sb_size),
      sb_rows_((height + sb_size - 1) / sb_size),
      row_hook_(std::move(row_hook)) {}

CpuEngine::~CpuEngine() { stop_and_join(); }

void CpuEngine::launch(const Frame420& src, const std::vector<const Frame420*>& refs, int qindex,
                       DecisionMap& dm, const std::vector<long long>&, long long) {  // nothing to reuse: the pyramids are rebuilt
  if (src.width != width_ || src.height != height_)
    throw std::runtime_error("CpuEngine: source size does not match the engine");
  for (const Frame420* r : refs)
    if (r->width != width_ || r->height != height_)
      throw std::runtime_error("CpuEngine: reference size does not match the engine");
  if (refs.size() > static_cast<size_t>(celv2::kMaxRefs)) throw std::runtime_error("CpuEngine: too many references");
  stop_and_join();  // a previous launch that was never collected
  // No worker runs here, and starting a thread publishes everything written before it.
  src_ = src;
  qindex_ = qindex;
  nrefs_ = static_cast<int>(refs.size());
  if (nrefs_ > 0) {
    cur_ = build_pyramid_ref(src_);
    ref_.clear();
    for (const Frame420* r : refs) ref_.push_back(build_pyramid_ref(*r, true));
    lv_.assign(nrefs_, std::vector<RefMvLevel>(celv2::kMeLevels));
    for (int l = 0; l < celv2::kMeLevels; l++) {
      for (auto& per_ref : lv_) {
        RefMvLevel& o = per_ref[l];
        o.cols = (cur_.plane[l].w + 7) / 8, o.rows = (cur_.plane[l].h + 7) / 8;
        o.mvx.assign(static_cast<size_t>(o.cols) * o.rows, 0), o.mvy = o.mvx;
      }
      next_me_row_[l] = 0;
      arrived_[l] = 0;
    }
  }
  const size_t nsb = static_cast<size_t>(sb_cols_) * sb_rows_;
  out_.assign(nsb, SbDecision{});
  row_done_.assign(sb_rows_, 0);
  next_sb_row_ = 0;
  error_ = nullptr;
  abort_ = false;
  dm.sb_size = sb_size_;
  dm.sb_cols = sb_cols_;
  dm.sb_rows = sb_rows_;
  dm.sbs.assign(nsb, SbDecision{});
  try {
    for (int i = 0; i < threads_; i++) workers_.emplace_back([this] { work(); });
  } catch (...) {
    stop_and_join();  // a thread that failed to start must not leave the others running
    throw;
  }
}

void CpuEngine::work() {
  try {
    if (nrefs_ > 0)
      for (int l = celv2::kMeLevels - 1; l >= 0; l--) {
        const int rows = lv_[0][l].rows;
        for (int t; !abort_ && (t = next_me_row_[l]++) < rows * nrefs_;) {
          const int ref = t / rows, r = t % rows;
          const RefMvLevel* coarser = l == celv2::kMeLevels - 1 ? nullptr : &lv_[ref][l + 1];
          me_level_rows_ref(cur_, ref_[ref], l, coarser, r, r + 1, lv_[ref][l]);
        }
        // Level l - 1 reads all of level l; abort_ releases the wait so no worker hangs here.
        std::unique_lock<std::mutex> lk(m_);
        if (++arrived_[l] == threads_) cv_.notify_all();
        cv_.wait(lk, [&] { return arrived_[l] == threads_ || abort_; });
        if (abort_) return;
      }
    std::vector<const RefPyramid*> refs;
    std::vector<const RefMvLevel*> fields;
    for (int i = 0; i < nrefs_; i++) refs.push_back(&ref_[i]), fields.push_back(&lv_[i][0]);
    for (int r; !abort_ && (r = next_sb_row_++) < sb_rows_;) {
      if (row_hook_) row_hook_(r);
      for (int c = 0; c < sb_cols_; c++)
        out_[static_cast<size_t>(r) * sb_cols_ + c] = decide_sb_ref(src_, refs, fields, c, r, sb_size_, qindex_);
      std::lock_guard<std::mutex> lk(m_);  // orders the row's contents before collect sees it done
      row_done_[r] = 1;
      cv_.notify_all();
    }
  } catch (...) {
    std::lock_guard<std::mutex> lk(m_);
    if (!error_) error_ = std::current_exception();
    abort_ = true;
    cv_.notify_all();
  }
}

void CpuEngine::collect(DecisionMap& dm, RowGate& gate) {
  std::exception_ptr err;
  try {
    for (int next = 0; next < sb_rows_;) {
      int end = next;
      {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return row_done_[next] || error_; });
        if (error_) {
          err = error_;
          break;
        }
        while (end < sb_rows_ && row_done_[end]) end++;  // rows finish out of order
      }
      for (size_t i = static_cast<size_t>(next) * sb_cols_; i < static_cast<size_t>(end) * sb_cols_; i++)
        dm.sbs[i] = std::move(out_[i]);
      gate.publish(end - 1);
      next = end;
    }
  } catch (...) {
    err = std::current_exception();
  }
  if (err) gate.fail(err);
  stop_and_join();
  if (err) std::rethrow_exception(err);
}

void CpuEngine::stop_and_join() {
  {
    std::lock_guard<std::mutex> lk(m_);  // under m_ so a waiting worker cannot miss it
    abort_ = true;
  }
  cv_.notify_all();
  for (std::thread& t : workers_) t.join();
  workers_.clear();
}
