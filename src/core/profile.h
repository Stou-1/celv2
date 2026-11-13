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
  CdefSearch,  // the filters' searches, less their stages above
  CcsoSearch,
  LrFit,
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

// The decision engine's wall times for one launch (the Vulkan wait spec's V0): launch() itself, and
// the end of launch() to row 0 ready. Printed as means over the launches.
void add_engine_wall(double launch_ms, double to_row0_ms);

// One frame's wall times, ms (where the 48-thread fps cap sits): Pipeline::code() from its start to
// launch() done, to the RD pass done, and in all; Encoder::code() from its start to the rows coded,
// then the in-loop filters and the range coding. Printed as means.
enum class Wall : int { Begin, Launch, RdDone, Coded, Total, Gm, Rows, Filters, Entropy, Header,
  // the in-loop filters' parts (CodecState::end_frame())
  FDeblock, FPad, FCdefSearch, FCdefApply, FCcsoSearch, FCcsoApply, FLrFit, FLrApply, FLrDecide, FCopy, Count };
void add_wall(Wall w, double ms);

// Adds the wall time from construction to destruction to `w` (nothing when not profiling).
class WallScope {
 public:
  explicit WallScope(Wall w);
  ~WallScope();
  WallScope(const WallScope&) = delete;
  WallScope& operator=(const WallScope&) = delete;

 private:
  Wall w_;
  long long t0_ = 0;
};

// The per-stage table, against the process's CPU time and the frames coded.
void print_profile(FILE* out, int frames);

// CELV2_RD_STATS=1 (the speed-first spec's step 0): repeated work in the RD pass and the coder's NEWMV search,
// and how often the 64-size intra mode the RD pass picks falls outside the SATD order's best 3 or 4. Printed
// at the end; off (the default), nothing is counted. Measurement only: it slows the encode.
enum class Count : int {
  TxCalls,           // tx_cost() calls of the RD pass (luma and chroma)
  TxRepeats,         // ... whose size and residual repeat an earlier call's in the same superblock
  TxSamples,         // samples transformed by them
  TxRepeatSamples,   // ... by the repeats
  TxWeighted,        // calls under --psy-rd / --psy-trellis (their cost reads more than the residual)
  Tx32Calls,         // 32x32 calls
  Tx32Repeats,       // ... repeating
  InterCalls,        // Pass::inter_cost() calls
  InterRepeats,      // ... with a block, references and MVs costed before in the same superblock
  NewmvBlocks,       // the coder's NEWMV searches with an RD cost
  NewmvCalls,        // their rd_cost_ calls
  NewmvRepeats,      // ... on an MV the same search costed before
  Intra64Blocks,     // the RD pass's 64-size intra blocks scoring all modes
  Intra64Miss3,      // ... whose pick is outside the SATD order's best 3
  Intra64Miss4,      // ... best 4
  RdIntraBlocks,     // the --rd-intra loop's blocks
  RdIntraMiss3,
  RdIntraMiss4,
  Count
};
bool rd_stats();
void count(Count c, uint64_t n = 1);
void print_rd_stats(FILE* out);

// CELV2_RD_TIMES=1 (the speed-first spec's step 3): time-stamp ticks of the RD pass's parts, each inclusive (a part
// inside another counts in both), printed at the end as ms a frame. Off, nothing is timed.
enum class RdPart : int {
  Sb,          // rd_pass_sb(), all of it
  InterCost,   // Pass::inter_cost()
  InterPred,   //   its predict() calls
  InterRes,    //   its residual loops
  IntraCost,   // Pass::intra_cost()
  IntraPred,   //   its predictions and residuals
  PsyOf,       // psy_of() and trellis_of()
  TxCost,      // tx_cost(): the memo's lookup and what it calls
  TxHash,      //   the residual's hash and the lookup
  TxUncached,  //   tx_cost_uncached()
  Fdct,        //     its fwd_dct()
  Quant,       //     its quantize()
  Sums,        //     its error and bit sums
  Dropped,     //     its dropped_cost()
  TcqPrep,     // tcq_levels(), any caller: tcq_prep() and the fills
  TcqViterbi,  //   the Viterbi pass
  TcqBack,     //   the backtrack and the dequantization
  Count
};
bool rd_times();
class RdTimer {
 public:
  explicit RdTimer(RdPart p);
  ~RdTimer();
  RdTimer(const RdTimer&) = delete;
  RdTimer& operator=(const RdTimer&) = delete;

 private:
  int p_ = -1;
  uint64_t t0_ = 0;
};
void print_rd_times(FILE* out, int frames);

}  // namespace celv2
