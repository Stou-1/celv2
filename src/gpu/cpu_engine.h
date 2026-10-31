// CPU decision engine: the GPU's decision rules (kernels_ref) on worker threads, used when no
// usable GPU exists. Decisions are integer-only, so the result equals GpuEngine's exactly.
#pragma once
#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "gpu/decision_engine.h"
#include "gpu/decision_map.h"
#include "gpu/inter_common.h"
#include "gpu/kernels_ref.h"
#include "io/y4m.h"

// Decision worker threads that, with the tile threads, fit in the cores.
int decision_threads(int cores, int tile_threads);  // max(1, (cores > 0 ? cores : 1) - tile_threads)

class CpuEngine : public DecisionEngine {
 public:
  // row_hook is called by a worker right before it decides a superblock row (tests only).
  CpuEngine(int width, int height, int sb_size, int threads, std::function<void(int row)> row_hook = {});
  ~CpuEngine() override;  // stops and joins the workers of a launch that was never collected
  CpuEngine(const CpuEngine&) = delete;
  CpuEngine& operator=(const CpuEngine&) = delete;

  // Copies src and the references, sizes dm, starts the workers and returns.
  using DecisionEngine::launch;
  void launch(const Frame420& src, const std::vector<const Frame420*>& refs, int qindex,
              DecisionMap& dm, const std::vector<long long>& ref_keys = {}, long long src_key = -1) override;
  // Moves finished rows into dm and publishes them top to bottom; joins every worker before it
  // returns. A worker's first exception goes to gate.fail and is rethrown.
  void collect(DecisionMap& dm, RowGate& gate) override;

 private:
  void work();
  void stop_and_join();  // aborts the remaining work and joins every worker

  int width_, height_, sb_size_, threads_, sb_cols_, sb_rows_;
  std::function<void(int)> row_hook_;
  int qindex_ = 0;
  int nrefs_ = 0;  // 0: KEY frame
  Frame420 src_;
  RefPyramid cur_;
  std::vector<RefPyramid> ref_;               // per reference
  std::vector<std::vector<RefMvLevel>> lv_;   // per reference, motion field per pyramid level
  std::vector<SbDecision> out_;  // raster order; a row is written only by the worker that took it
  std::vector<std::thread> workers_;
  std::atomic<int> next_me_row_[celv2::kMeLevels];  // next (reference, 8x8 block row) to search, per level
  std::atomic<int> next_sb_row_{0};
  std::atomic<bool> abort_{false};
  std::mutex m_;  // guards the members below; cv_ signals each change
  std::condition_variable cv_;
  int arrived_[celv2::kMeLevels] = {};  // workers done with a level (barrier before the next one)
  std::vector<char> row_done_;
  std::exception_ptr error_;  // first worker exception
};
