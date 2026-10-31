#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>

#include "check.h"
#include "gpu/decision_map.h"

namespace {
// Polls so a hang fails the test instead of hanging CTest.
bool wait_true(const std::atomic<bool>& f) {
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (!f && std::chrono::steady_clock::now() < end) std::this_thread::yield();
  return f;
}

// A stuck waiter cannot be released reliably and must not outlive the stack objects it uses.
[[noreturn]] void die_on_timeout(const char* test) {
  std::fprintf(stderr, "%s: waiter not woken within 1 s\n", test);
  std::abort();
}
}  // namespace

int publish_releases_waiters() {
  RowGate gate;
  std::atomic<bool> done{false};
  std::thread t([&] {
    gate.wait_row(2);
    done = true;
  });
  gate.publish(0);
  gate.publish(1);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  const bool early = done;
  gate.publish(2);
  const bool ok = wait_true(done);
  if (!ok) die_on_timeout("publish_releases_waiters");
  t.join();
  CHECK(!early);
  CHECK(ok);
  CHECK_EQ(gate.ready_rows(), 3);
  return 0;
}

int publish_out_of_order_is_monotonic() {
  RowGate gate;
  gate.publish(3);
  gate.publish(1);
  CHECK_EQ(gate.ready_rows(), 4);
  gate.wait_row(3);
  return 0;
}

int fail_wakes_and_rethrows() {
  RowGate gate;
  std::atomic<bool> done[2] = {false, false};
  std::atomic<int> got[2] = {0, 0};
  auto waiter = [&](int i) {
    try {
      gate.wait_row(5);
    } catch (const std::runtime_error& e) {
      got[i] = std::string(e.what()) == "gpu";
    }
    done[i] = true;
  };
  std::thread a(waiter, 0), b(waiter, 1);
  gate.fail(std::make_exception_ptr(std::runtime_error("gpu")));
  const bool ok = wait_true(done[0]) && wait_true(done[1]);
  if (!ok) die_on_timeout("fail_wakes_and_rethrows");
  a.join();
  b.join();
  CHECK(ok);
  CHECK_EQ(got[0].load(), 1);
  CHECK_EQ(got[1].load(), 1);
  bool threw = false;
  try {
    gate.wait_row(0);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw);
  return 0;
}

int main() {
  int fails = 0;
  fails += RUN(publish_releases_waiters);
  fails += RUN(publish_out_of_order_is_monotonic);
  fails += RUN(fail_wakes_and_rethrows);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
