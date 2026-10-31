#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "check.h"
#include "core/threads.h"

// Every k runs once, also when the tasks themselves call parallel() (nested, as the CCSO planes'
// searches do), and from several callers at once.
static int each_k_once_nested_and_concurrent() {
  std::vector<std::atomic<int>> hits(64 * 32);
  auto outer = [&](int a) {
    celv2::parallel(32, [&](int b) { hits[a * 32 + b]++; });
  };
  std::thread other([&] { celv2::parallel(32, outer); });
  celv2::parallel(32, [&](int a) { outer(a + 32); });
  other.join();
  for (auto& h : hits) CHECK_EQ(h.load(), 1);
  return 0;
}

// An exception from a task comes back to the caller once all tasks are done; the pool still works.
static int exception_reaches_the_caller() {
  std::atomic<int> ran{0};
  bool caught = false;
  try {
    celv2::parallel(100, [&](int k) {
      ran++;
      if (k == 37) throw std::runtime_error("task 37");
    });
  } catch (const std::runtime_error&) {
    caught = true;
  }
  CHECK(caught);
  CHECK_EQ(ran.load(), 100);
  std::atomic<int> after{0};
  celv2::parallel(10, [&](int) { after++; });
  CHECK_EQ(after.load(), 10);
  return 0;
}

int main() {
  int fails = 0;
  fails += RUN(each_k_once_nested_and_concurrent);
  fails += RUN(exception_reaches_the_caller);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
