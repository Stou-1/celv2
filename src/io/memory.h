// The machine's memory, for celv2enc's --memory-limit.
#pragma once
#include <cstdint>

// The installed RAM in bytes (GlobalMemoryStatusEx), or CELV2_TEST_TOTAL_RAM_MB's when set (tests).
uint64_t installed_memory_bytes();
