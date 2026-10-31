// GPU decision engine: analyses source frames with CUDA and returns only a DecisionMap.
// CUDA errors and unsupported inputs throw std::runtime_error. Without a usable GPU, make_engine
// (pipeline.h) picks CpuEngine at startup; a GPU failure mid-encode is not switched to the CPU.
#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include "gpu/decision_engine.h"
#include "gpu/decision_map.h"
#include "io/y4m.h"

class GpuEngine : public DecisionEngine {
 public:
  GpuEngine(int width, int height, int sb_size);
  ~GpuEngine() override;
  GpuEngine(const GpuEngine&) = delete;
  GpuEngine& operator=(const GpuEngine&) = delete;

  DecisionMap decide_intra(const Frame420& src, int qindex);
  // Hierarchical motion search against the previous source frame, then intra/inter decision.
  DecisionMap decide_inter(const Frame420& src, const Frame420& prev_src, int qindex);
  // The same against each reference (nearest first).
  DecisionMap decide_inter(const Frame420& src, const std::vector<const Frame420*>& refs, int qindex);

  // Queues uploads, motion search (per reference) and every superblock row's decision, then
  // returns; dm is sized for all superblocks. src and the references may change once this returns.
  // The previous launch must have been collected first.
  using DecisionEngine::launch;
  void launch(const Frame420& src, const std::vector<const Frame420*>& refs, int qindex,
              DecisionMap& dm, const std::vector<long long>& ref_keys = {}, long long src_key = -1) override;
  void prepare(const Frame420& next, long long key) override;
  // Waits for each row in the order launched, fills dm's superblocks of that row, publishes it on
  // gate. On a CUDA error calls gate.fail() and rethrows. No GPU work is pending on return.
  void collect(DecisionMap& dm, RowGate& gate) override;
  void set_row_streams(int n) override;
  // Test hook: superblock sb's intra costs of the last collected decision (block x mode).
  std::vector<int64_t> intra_costs(int sb) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Checks device 0 in order (never throws): a CUDA device exists, it is Turing (7.5) or newer, and
// the one-time kernel init succeeds (fails with no kernel image for this GPU's architecture).
ProbeResult probe_gpu();

// Test hook: SATD of n block pairs (each 64x64 bytes, stride 64), w4/h4 in 4-pixel units.
std::vector<int> gpu_satd_batch(const uint8_t* a, const uint8_t* b, int n, int w4, int h4);
