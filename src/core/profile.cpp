#include "core/profile.h"

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
struct Counts {
  uint64_t cycles[kStages] = {}, ticks[kStages] = {}, taken[kStages] = {};
  void add_to(Counts& o) const {
    for (int s = 0; s < kStages; s++) o.cycles[s] += cycles[s], o.ticks[s] += ticks[s], o.taken[s] += taken[s];
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
                                       "deblocking", "CDEF",    "CCSO",     "loop restoration"};
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
  if (t_stage != Stage::None) c.taken[static_cast<int>(t_stage)] += d;
  t_hot = false;
}

uint64_t thread_cycles(void* thread_handle) {
  ULONG64 c = 0;
  if (thread_handle) QueryThreadCycleTime(static_cast<HANDLE>(thread_handle), &c);
  return c;
}

void add_cycles(Stage s, uint64_t cycles) { mine().cycles[static_cast<int>(s)] += cycles; }

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
}

}  // namespace celv2
