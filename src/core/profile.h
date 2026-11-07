// CELV2_PROFILE=1: CPU time per encoding stage, printed at the end (docs/superpowers/specs/
// 2026-11-04-speed-budget-design.md). Each thread's CPU cycles (QueryThreadCycleTime: waiting costs
// none) go to the innermost stage open on it, so the stages add up without double counting; work a
// stage hands to the thread pool counts as that stage. Off (the default), nothing is timed.
#pragma once
#include <cstdint>
#include <cstdio>

namespace celv2 {

enum class Stage : int {
  None,
  Read,
  Denoise,
  Analysis,  // scene cuts, still share, the --paq map
  Tpl,
  Decision,  // the decision engine's CPU side (all of it with --device cpu)
  GlobalMotion,
  RdPass,
  Coder,  // the coder's own work, less the stages below
  Transform,
  Quant,
  Entropy,
  Deblock,
  Cdef,
  Ccso,
  Restoration,
  Count
};

bool profiling();  // CELV2_PROFILE set (read once)
Stage current_stage();  // the innermost stage open on this thread

// Opens `s` on this thread until destroyed (nothing when not profiling).
class StageScope {
 public:
  explicit StageScope(Stage s);
  ~StageScope();
  StageScope(const StageScope&) = delete;
  StageScope& operator=(const StageScope&) = delete;

 private:
  bool on_ = false;
  Stage prev_ = Stage::None;
};

// For stages that never wait and run millions of times (quantization, transforms): timed by the
// time-stamp counter, which costs next to nothing, and taken from the stage open on the thread. Only
// the outermost such scope on a thread counts.
class HotScope {
 public:
  explicit HotScope(Stage s);
  ~HotScope();
  HotScope(const HotScope&) = delete;
  HotScope& operator=(const HotScope&) = delete;

 private:
  Stage s_ = Stage::None;
  uint64_t t0_ = 0;
};

// Adds the CPU cycles other threads (AVM's filter workers) spent since `before` (as from
// thread_cycles()) to a stage.
uint64_t thread_cycles(void* thread_handle);
void add_cycles(Stage s, uint64_t cycles);

// The per-stage table, against the process's CPU time and the frames coded.
void print_profile(FILE* out, int frames);

}  // namespace celv2
