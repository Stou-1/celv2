#include "core/profile.h"

#include <atomic>
#include <mutex>
#include <set>
#include <chrono>
#include <cstdlib>
#include <intrin.h>

#include <windows.h>

namespace celv2 {

namespace {

constexpr int kStages = static_cast<int>(Stage::Count);

// Per thread (shared counters made the many threads' quantization calls fight over one cache line:
// 12% slower): thread cycles per stage, and HotScope's time-stamp counter ticks spent in a stage and
// taken from the stage around it. A thread's counts go to the totals when it ends; print_profile()
// adds the live threads'.
// under[h][s]: HotScope h's ticks spent while stage s was open (which caller the hot stages served).
struct Counts {
  uint64_t cycles[kStages] = {}, ticks[kStages] = {}, taken[kStages] = {};
  uint64_t under[kStages][kStages] = {};
  void add_to(Counts& o) const {
    for (int s = 0; s < kStages; s++) {
      o.cycles[s] += cycles[s], o.ticks[s] += ticks[s], o.taken[s] += taken[s];
      for (int e = 0; e < kStages; e++) o.under[s][e] += under[s][e];
    }
  }
};
std::mutex g_mu;
Counts g_done;               // of the threads that ended (and add_cycles())
std::set<Counts*> g_live;    // the running threads'

struct ThreadCounts {
  Counts c;
  ThreadCounts() {
    std::lock_guard<std::mutex> lock(g_mu);
    g_live.insert(&c);
  }
  ~ThreadCounts() {
    std::lock_guard<std::mutex> lock(g_mu);
    c.add_to(g_done);
    g_live.erase(&c);
  }
};
Counts& mine() {
  thread_local ThreadCounts t;
  return t.c;
}

thread_local Stage t_stage = Stage::None;
thread_local uint64_t t_since = 0;
thread_local bool t_hot = false;

uint64_t now() {
  ULONG64 c = 0;
  QueryThreadCycleTime(GetCurrentThread(), &c);
  return c;
}

const char* name(Stage s) {
  static const char* const kNames[] = {"none",     "reading", "denoising",  "analysis",       "TPL",     "decision",
                                       "global motion", "RD pass", "coder", "transforms", "quantization", "entropy coding",
                                       "deblocking", "CDEF",    "CCSO",     "loop restoration", "CDEF search", "CCSO search", "LR fit"};
  return kNames[static_cast<int>(s)];
}

// Thread cycles and time-stamp counter ticks per second: a 50 ms busy loop timed three ways.
void rates(double& cycles_hz, double& ticks_hz) {
  const auto t0 = std::chrono::steady_clock::now();
  const uint64_t c0 = now(), k0 = __rdtsc();
  while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(50)) {
  }
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  cycles_hz = static_cast<double>(now() - c0) / secs;
  ticks_hz = static_cast<double>(__rdtsc() - k0) / secs;
}

}  // namespace

bool profiling() {
  static const bool on = [] {
    const char* e = std::getenv("CELV2_PROFILE");
    return e && *e && *e != '0';
  }();
  return on;
}

Stage current_stage() { return t_stage; }

StageScope::StageScope(Stage s) {
  if (!profiling()) return;
  on_ = true;
  const uint64_t t = now();
  if (t_stage != Stage::None) mine().cycles[static_cast<int>(t_stage)] += t - t_since;
  prev_ = t_stage;
  t_stage = s, t_since = t;
}

StageScope::~StageScope() {
  if (!on_) return;
  const uint64_t t = now();
  if (t_stage != Stage::None) mine().cycles[static_cast<int>(t_stage)] += t - t_since;
  t_stage = prev_, t_since = t;
}

HotScope::HotScope(Stage s) {
  if (!profiling() || t_hot) return;
  t_hot = true;
  s_ = s;
  t0_ = __rdtsc();
}

HotScope::~HotScope() {
  if (s_ == Stage::None) return;
  const uint64_t d = __rdtsc() - t0_;
  Counts& c = mine();
  c.ticks[static_cast<int>(s_)] += d;
  c.under[static_cast<int>(s_)][static_cast<int>(t_stage)] += d;
  if (t_stage != Stage::None) c.taken[static_cast<int>(t_stage)] += d;
  t_hot = false;
}

uint64_t thread_cycles(void* thread_handle) {
  ULONG64 c = 0;
  if (thread_handle) QueryThreadCycleTime(static_cast<HANDLE>(thread_handle), &c);
  return c;
}

void add_cycles(Stage s, uint64_t cycles) { mine().cycles[static_cast<int>(s)] += cycles; }

namespace {
std::mutex g_wall_mu;
double g_launch_ms = 0, g_row0_ms = 0;
long long g_launches = 0;
}  // namespace

void add_engine_wall(double launch_ms, double to_row0_ms) {
  std::lock_guard<std::mutex> lock(g_wall_mu);
  g_launch_ms += launch_ms, g_row0_ms += to_row0_ms, g_launches++;
}

namespace {
constexpr int kWalls = static_cast<int>(Wall::Count);
double g_walls[kWalls] = {};
long long g_wall_n[kWalls] = {};
}  // namespace

void add_wall(Wall w, double ms) {
  if (!profiling()) return;
  std::lock_guard<std::mutex> lock(g_wall_mu);
  g_walls[static_cast<int>(w)] += ms, g_wall_n[static_cast<int>(w)]++;
}

WallScope::WallScope(Wall w) : w_(w) {
  if (profiling()) t0_ = std::chrono::steady_clock::now().time_since_epoch().count();
}

WallScope::~WallScope() {
  if (!profiling()) return;
  const long long t1 = std::chrono::steady_clock::now().time_since_epoch().count();
  add_wall(w_, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::duration(t1 - t0_)).count());
}

void print_profile(FILE* out, int frames) {
  if (!profiling()) return;
  double hz, tick_hz;
  rates(hz, tick_hz);
  Counts all;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    g_done.add_to(all);
    for (const Counts* c : g_live) c->add_to(all);
  }
  auto stage_secs = [&](int s) {
    return all.cycles[s] / hz + (static_cast<double>(all.ticks[s]) - static_cast<double>(all.taken[s])) / tick_hz;
  };
  FILETIME created, exited, kernel, user;
  GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user);
  auto secs = [](const FILETIME& f) { return (static_cast<double>(f.dwHighDateTime) * 4294967296.0 + f.dwLowDateTime) * 1e-7; };
  const double process = secs(kernel) + secs(user);
  double staged = 0;
  for (int s = 1; s < static_cast<int>(Stage::Count); s++) staged += stage_secs(s);
  std::fprintf(out, "celv2enc: profile: CPU %.1f s (%.1f ms a frame), %.1f s in the stages below\n", process,
               frames > 0 ? 1000 * process / frames : 0.0, staged);
  for (int s = 1; s < static_cast<int>(Stage::Count); s++) {
    const double t = stage_secs(s);
    if (t <= 0) continue;
    std::fprintf(out, "celv2enc: profile: %-16s %8.2f s %8.1f ms a frame %5.1f%%\n", name(static_cast<Stage>(s)), t,
                 frames > 0 ? 1000 * t / frames : 0.0, process > 0 ? 100 * t / process : 0.0);
  }
  // The hot stages by the stage open around them, and the RD pass with what it called.
  auto per_frame = [&](double t) { return frames > 0 ? 1000 * t / frames : 0.0; };
  for (const Stage h : {Stage::Transform, Stage::Quant})
    for (int e = 0; e < kStages; e++) {
      const double t = all.under[static_cast<int>(h)][e] / tick_hz;
      if (t > 0)
        std::fprintf(out, "celv2enc: profile:   %s under %s %8.2f s %8.1f ms a frame %5.1f%%\n", name(h),
                     name(static_cast<Stage>(e)), t, per_frame(t), process > 0 ? 100 * t / process : 0.0);
    }
  const int rd = static_cast<int>(Stage::RdPass);
  const double rd_all = stage_secs(rd) + (all.under[static_cast<int>(Stage::Transform)][rd] +
                                          all.under[static_cast<int>(Stage::Quant)][rd]) / tick_hz;
  std::fprintf(out, "celv2enc: profile: RD pass with its transforms and quantization %8.2f s %8.1f ms a frame %5.1f%%\n",
               rd_all, per_frame(rd_all), process > 0 ? 100 * rd_all / process : 0.0);
  std::lock_guard<std::mutex> lock(g_wall_mu);
  if (g_launches > 0)
    std::fprintf(out, "celv2enc: profile: engine wall: launch() %.2f ms, %.2f ms to row 0 (means of %lld launches)\n",
                 g_launch_ms / g_launches, g_row0_ms / g_launches, g_launches);
  auto mean = [](Wall w) {
    const int i = static_cast<int>(w);
    return g_wall_n[i] ? g_walls[i] / g_wall_n[i] : 0.0;
  };
  if (g_wall_n[static_cast<int>(Wall::Total)] > 0)
    std::fprintf(out, "celv2enc: profile: frame wall (means of %lld): begun %.1f ms, launched %.1f, RD pass done %.1f, coded %.1f,"
                 " total %.1f; coder (all frames): global motion %.1f ms, rows %.1f, filters %.1f, range coding %.1f, header %.1f\n",
                 g_wall_n[static_cast<int>(Wall::Total)], mean(Wall::Begin), mean(Wall::Launch), mean(Wall::RdDone),
                 mean(Wall::Coded), mean(Wall::Total), mean(Wall::Gm), mean(Wall::Rows), mean(Wall::Filters),
                 mean(Wall::Entropy), mean(Wall::Header));
  if (g_wall_n[static_cast<int>(Wall::FCopy)] > 0)
    std::fprintf(out, "celv2enc: profile: filter wall: deblocking %.1f ms, CCSO pad %.1f, CDEF search %.1f, CDEF %.1f, CCSO search %.1f,"
                 " CCSO %.1f, LR fit %.1f, LR %.1f, LR decide %.1f, copy %.1f\n",
                 mean(Wall::FDeblock), mean(Wall::FPad), mean(Wall::FCdefSearch), mean(Wall::FCdefApply), mean(Wall::FCcsoSearch),
                 mean(Wall::FCcsoApply), mean(Wall::FLrFit), mean(Wall::FLrApply), mean(Wall::FLrDecide), mean(Wall::FCopy));
}

namespace {
std::atomic<uint64_t> g_counts[static_cast<int>(Count::Count)];
}  // namespace

bool rd_stats() {
  static const bool on = [] {
    const char* e = std::getenv("CELV2_RD_STATS");
    return e && *e && *e != '0';
  }();
  return on;
}

void count(Count c, uint64_t n) { g_counts[static_cast<int>(c)].fetch_add(n, std::memory_order_relaxed); }

void print_rd_stats(FILE* out) {
  if (!rd_stats()) return;
  auto v = [](Count c) { return static_cast<double>(g_counts[static_cast<int>(c)].load()); };
  auto pct = [](double a, double b) { return b > 0 ? 100 * a / b : 0.0; };
  std::fprintf(out, "celv2enc: rd-stats: tx_cost %.0f calls, %.1f%% repeats (%.1f%% of the samples), %.1f%% weighted"
               " (--psy-*: not counted as repeats)\n", v(Count::TxCalls), pct(v(Count::TxRepeats), v(Count::TxCalls)),
               pct(v(Count::TxRepeatSamples), v(Count::TxSamples)), pct(v(Count::TxWeighted), v(Count::TxCalls)));
  std::fprintf(out, "celv2enc: rd-stats: tx_cost 32x32 %.0f calls, %.1f%% repeats\n", v(Count::Tx32Calls),
               pct(v(Count::Tx32Repeats), v(Count::Tx32Calls)));
  std::fprintf(out, "celv2enc: rd-stats: inter_cost %.0f calls, %.1f%% repeats\n", v(Count::InterCalls),
               pct(v(Count::InterRepeats), v(Count::InterCalls)));
  std::fprintf(out, "celv2enc: rd-stats: coder NEWMV %.0f searches, %.1f rd_cost_ calls each, %.1f%% repeats\n",
               v(Count::NewmvBlocks), v(Count::NewmvBlocks) > 0 ? v(Count::NewmvCalls) / v(Count::NewmvBlocks) : 0.0,
               pct(v(Count::NewmvRepeats), v(Count::NewmvCalls)));
  std::fprintf(out, "celv2enc: rd-stats: intra64 %.0f blocks, %.1f%% outside SATD top 3, %.1f%% outside top 4\n",
               v(Count::Intra64Blocks), pct(v(Count::Intra64Miss3), v(Count::Intra64Blocks)),
               pct(v(Count::Intra64Miss4), v(Count::Intra64Blocks)));
  std::fprintf(out, "celv2enc: rd-stats: rd-intra %.0f blocks, %.1f%% outside SATD top 3, %.1f%% outside top 4\n",
               v(Count::RdIntraBlocks), pct(v(Count::RdIntraMiss3), v(Count::RdIntraBlocks)),
               pct(v(Count::RdIntraMiss4), v(Count::RdIntraBlocks)));
}

namespace {
constexpr int kParts = static_cast<int>(RdPart::Count);
std::mutex g_parts_mu;
uint64_t g_parts[kParts] = {}, g_part_calls[kParts] = {};
// A thread's ticks, added to the totals when it ends (the RD pass's threads end every frame).
struct PartTicks {
  uint64_t t[kParts] = {}, n[kParts] = {};
  ~PartTicks() {
    const std::lock_guard<std::mutex> lock(g_parts_mu);
    for (int p = 0; p < kParts; p++) g_parts[p] += t[p], g_part_calls[p] += n[p];
  }
};
thread_local PartTicks t_parts;
}  // namespace

bool rd_times() {
  static const bool on = [] {
    const char* e = std::getenv("CELV2_RD_TIMES");
    return e && *e && *e != '0';
  }();
  return on;
}

RdTimer::RdTimer(RdPart p) {
  if (!rd_times()) return;
  p_ = static_cast<int>(p);
  t0_ = __rdtsc();
}

RdTimer::~RdTimer() {
  if (p_ < 0) return;
  t_parts.t[p_] += __rdtsc() - t0_;
  t_parts.n[p_]++;
}

void print_rd_times(FILE* out, int frames) {
  if (!rd_times()) return;
  double hz, tick_hz;
  rates(hz, tick_hz);
  static const char* const kNames[] = {"rd_pass_sb",    "inter_cost", "  predict",    "  residual", "intra_cost",
                                       "  pred+res",    "psy_of etc", "tx_cost",      "  hash+look", "  uncached",
                                       "    fwd_dct",   "    quantize", "    sums",   "    dropped",
                                       "tcq prep",      "tcq viterbi", "tcq back"};
  const std::lock_guard<std::mutex> lock(g_parts_mu);
  for (int p = 0; p < kParts; p++) {
    const uint64_t t = g_parts[p] + t_parts.t[p], n = g_part_calls[p] + t_parts.n[p];
    std::fprintf(out, "celv2enc: rd-times: %-14s %10.1f ms a frame %12llu calls %8.0f ns each\n", kNames[p],
                 frames > 0 ? 1000.0 * t / tick_hz / frames : 0.0, static_cast<unsigned long long>(n),
                 n ? 1e9 * t / tick_hz / n : 0.0);
  }
}

}  // namespace celv2
