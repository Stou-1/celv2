#include "io/memory.h"

#include <cstdlib>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

uint64_t installed_memory_bytes() {
  if (const char* t = std::getenv("CELV2_TEST_TOTAL_RAM_MB"); t && *t) return std::strtoull(t, nullptr, 10) << 20;  // tests/test_cli.cpp
  MEMORYSTATUSEX m{};
  m.dwLength = sizeof(m);
  return GlobalMemoryStatusEx(&m) ? m.ullTotalPhys : 0;
}
