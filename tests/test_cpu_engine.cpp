#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include "check.h"
#include "gpu/cpu_engine.h"
#include "gpu/decision_map.h"
#include "pipeline.h"
#include "test_frames.h"

namespace {
// Aborts the process if not destroyed within 30 s: a hang cannot be released reliably and the
// engine's threads use stack objects, so the test must not outlive them.
class Watchdog {
 public:
  explicit Watchdog(const char* test) : t_([this, test] {
    std::unique_lock<std::mutex> lk(m_);
    if (!cv_.wait_for(lk, std::chrono::seconds(30), [this] { return done_; })) {
      std::fprintf(stderr, "%s: not done within 30 s\n", test);
      std::abort();
    }
  }) {}
  ~Watchdog() {
    {
      std::lock_guard<std::mutex> lk(m_);
      done_ = true;
    }
    cv_.notify_all();
    t_.join();
  }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  bool done_ = false;
  std::thread t_;  // last: starts after the members it uses
};
}  // namespace

static int decision_threads_values() {
  CHECK_EQ(decision_threads(48, 2), 46);
  CHECK_EQ(decision_threads(4, 4), 1);
  CHECK_EQ(decision_threads(1, 1), 1);
  CHECK_EQ(decision_threads(0, 1), 1);
  return 0;
}

static int cpu_engine_publishes_in_order() {
  const Frame420 prev = scene_frame(640, 360), src = scene_frame(640, 360, 3, -2);
  CpuEngine cpu(640, 360, 64, 8);
  DecisionMap dm;
  RowGate gate;
  std::atomic<bool> finished{false};
  std::exception_ptr err;
  bool monotonic = true;
  {
    Watchdog wd("cpu_engine_publishes_in_order");
    cpu.launch(src, &prev, 120, dm);
    std::thread collector([&] {
      try {
        cpu.collect(dm, gate);
      } catch (...) {
        err = std::current_exception();
      }
      finished = true;
    });
    int last = 0;
    while (!finished) {
      const int now = gate.ready_rows();
      if (now < last) monotonic = false;
      last = now;
      std::this_thread::yield();
    }
    collector.join();
  }
  if (err) std::rethrow_exception(err);
  CHECK(monotonic);
  CHECK_EQ(dm.sb_rows, 6);
  CHECK_EQ(gate.ready_rows(), dm.sb_rows);
  return 0;
}

static int cpu_engine_fails_cleanly() {
  const Frame420 prev = scene_frame(640, 360), src = scene_frame(640, 360, 3, -2);
  CpuEngine cpu(640, 360, 64, 8, [](int row) {
    if (row == 2) throw std::runtime_error("hook");
  });
  DecisionMap dm;
  RowGate gate;
  std::string what;
  bool waiter_threw = false;
  {
    Watchdog wd("cpu_engine_fails_cleanly");
    cpu.launch(src, &prev, 120, dm);
    try {
      cpu.collect(dm, gate);
    } catch (const std::runtime_error& e) {
      what = e.what();
    }
    try {
      gate.wait_row(5);
    } catch (const std::runtime_error&) {
      waiter_threw = true;
    }
  }
  CHECK_EQ(what, std::string("hook"));
  CHECK(waiter_threw);
  return 0;
}

static int cpu_engine_tiny_frames() {
  for (int s : {32, 16}) {
    const Frame420 prev = scene_frame(s, s), src = scene_frame(s, s, 1, 1);
    CpuEngine cpu(s, s, 64, 46);
    for (bool key : {true, false}) {
      DecisionMap dm;
      RowGate gate;
      Watchdog wd("cpu_engine_tiny_frames");
      cpu.launch(src, key ? nullptr : &prev, 120, dm);
      cpu.collect(dm, gate);
      CHECK_EQ(gate.ready_rows(), 1);
      CHECK_EQ(dm.sbs.size(), size_t{1});
      CHECK(!dm.sbs[0].blocks.empty());
    }
  }
  return 0;
}

static int cpu_engine_many_frames() {
  CpuEngine cpu(64, 64, 64, 8);
  Watchdog wd("cpu_engine_many_frames");
  Frame420 prev = scene_frame(64, 64);
  for (int i = 0; i < 30; i++) {
    const Frame420 src = scene_frame(64, 64, i, i / 2);
    DecisionMap dm;
    RowGate gate;
    cpu.launch(src, i == 0 ? nullptr : &prev, 120, dm);
    cpu.collect(dm, gate);
    CHECK_EQ(gate.ready_rows(), 1);
    prev = src;
  }
  return 0;
}

// A fake failed probe: only Cuda minds (it throws); Auto takes Vulkan (in a build and on a machine with
// it) or the CPU, and Cpu never probes.
static int make_engine_selection() {
  const auto failed = [] {
    ProbeResult p;
    p.reason = "fake";
    return p;
  };
  std::string desc;
  const auto autoe = make_engine(Device::Auto, 640, 360, 1, failed, &desc);
  const bool vulkan = desc.find("decisions on Vulkan") != std::string::npos;
  CHECK(vulkan || dynamic_cast<CpuEngine*>(autoe.get()) != nullptr);
  CHECK(vulkan || desc.find("decisions on CPU") != std::string::npos);
  bool threw = false;
  try {
    make_engine(Device::Cuda, 640, 360, 1, failed, &desc);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw);
  bool probed = false;
  const auto cpue = make_engine(Device::Cpu, 640, 360, 1, [&] {
    probed = true;
    return failed();
  }, &desc);
  CHECK(dynamic_cast<CpuEngine*>(cpue.get()) != nullptr);
  CHECK(!probed);
  CHECK(desc.find("decisions on CPU") != std::string::npos);
  return 0;
}

int main() {
  int fails = 0;
  fails += RUN(decision_threads_values);
  fails += RUN(cpu_engine_publishes_in_order);
  fails += RUN(cpu_engine_fails_cleanly);
  fails += RUN(cpu_engine_tiny_frames);
  fails += RUN(cpu_engine_many_frames);
  fails += RUN(make_engine_selection);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
