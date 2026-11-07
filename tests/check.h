// Minimal assertion macros: each test is its own executable registered with CTest.
#pragma once
#include <cstdio>
#include <iostream>

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, \
                   #cond);                                                 \
      return 1;                                                            \
    }                                                                      \
  } while (0)

#define CHECK_EQ(a, b)                                                          \
  do {                                                                          \
    const auto& va_ = (a);                                                      \
    const auto& vb_ = (b);                                                      \
    if (!(va_ == vb_)) {                                                        \
      std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK_EQ failed: " #a " == " \
                << #b << " (" << va_ << " vs " << vb_ << ")\n";                 \
      return 1;                                                                 \
    }                                                                           \
  } while (0)

#include <cstdlib>
#include <cstring>
#include <exception>

// Runs one test case; an escaping exception counts as a failure and prints its message.
// CHECK_ONLY=<name> runs only the case of that name.
inline int run_test(const char* name, int (*fn)()) {
  if (const char* only = std::getenv("CHECK_ONLY"); only && std::strcmp(only, name) != 0) return 0;
  try {
    const int r = fn();
    if (r) std::fprintf(stderr, "FAILED: %s\n", name);
    return r;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAILED: %s threw: %s\n", name, e.what());
    return 1;
  }
}
#define RUN(fn) run_test(#fn, fn)
