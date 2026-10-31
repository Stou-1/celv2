// celv2enc --pass 1 statistics: each IVF frame's payload in bytes, for --pass 2 --target-size.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace celv2 {

struct PassStats {
  int width = 0, height = 0, qindex = 0;
  std::vector<uint64_t> bytes;  // bytes[i]: the i-th IVF frame's payload (without its 12-byte header)
};

// Text: "celv2-stats 1 <width> <height> <qindex> <frames>", then one line of bytes a frame.
void write_stats(const std::string& path, const PassStats& s);  // std::runtime_error on failure
PassStats read_stats(const std::string& path);                  // std::runtime_error, with the reason

}  // namespace celv2
