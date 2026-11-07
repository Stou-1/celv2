// Common entry of the decision engines (GpuEngine, CpuEngine). No CUDA, no AVM dependency.
#pragma once
#include <string>
#include <vector>

#include "gpu/decision_map.h"
#include "io/y4m.h"

class DecisionEngine {
 public:
  virtual ~DecisionEngine() = default;
  // Queues (GPU) or starts (CPU) the decision of every superblock row; sizes dm. src and the
  // references may change once this returns. refs: nearest first (refs[r] is AV2 reference r, at
  // most celv2::kMaxRefs); empty: KEY frame (intra only). src_key: see prepare(). ref_keys (empty, or one per reference):
  // a reference with the key of one in an earlier launch is the same picture, so the engine may
  // reuse what it built from it (GpuEngine: the upload, pyramid and quarter-pel phases).
  virtual void launch(const Frame420& src, const std::vector<const Frame420*>& refs, int qindex,
                      DecisionMap& dm, const std::vector<long long>& ref_keys = {}, long long src_key = -1) = 0;
  // Starts on the next frame to be launched (with src_key key) what needs no references (the GPU:
  // its upload and intra costs), behind the work queued so far; next may change once this returns.
  // Call after a launch, before its collect. Nothing by default.
  virtual void prepare(const Frame420& next, long long key) { (void)next, (void)key; }
  // One reference, or none (ref == nullptr).
  void launch(const Frame420& src, const Frame420* ref, int qindex, DecisionMap& dm) {
    launch(src, ref ? std::vector<const Frame420*>{ref} : std::vector<const Frame420*>{}, qindex, dm);
  }
  // Publishes rows on gate as they complete; on failure calls gate.fail and rethrows. Returns
  // with no work pending. Must follow each launch.
  virtual void collect(DecisionMap& dm, RowGate& gate) = 0;
  // The frame's superblock rows fall into n equal runs (the tile rows), each coded by its own
  // threads: an engine may decide the runs' first rows first, then their second ones, and so on,
  // so every tile can start at once. 1 (the default): top to bottom.
  virtual void set_row_streams(int n) { (void)n; }
#ifdef CELV2_ORACLE
  // L1: the tree 64x64 superblock sb must keep (tools/oracle/oracle.h), nullptr for the RD pass's own.
  virtual const celv2::oracle::Sb* forced(size_t sb) const { (void)sb; return nullptr; }
#endif

  // A launch and its collect, for the tests.
  DecisionMap decide_inter(const Frame420& src, const std::vector<const Frame420*>& refs, int qindex) {
    DecisionMap dm;
    RowGate gate;
    launch(src, refs, qindex, dm);
    collect(dm, gate);
    return dm;
  }
  DecisionMap decide_inter(const Frame420& src, const Frame420& prev, int qindex) {
    return decide_inter(src, std::vector<const Frame420*>{&prev}, qindex);
  }
  DecisionMap decide_intra(const Frame420& src, int qindex) { return decide_inter(src, std::vector<const Frame420*>{}, qindex); }
};

// Result of probe_gpu(): the device name and compute capability, or why it is unusable.
struct ProbeResult {
  bool ok = false;
  std::string name;
  int cc_major = 0, cc_minor = 0;
  std::string reason;
};
