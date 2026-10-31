// IVF container: 32-byte file header ("DKIF", FourCC "AV02") + 12-byte frame headers.
#pragma once
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

class IvfWriter {
 public:
  IvfWriter(const std::string& path, int w, int h, int fps_num, int fps_den);
  ~IvfWriter();
  void write(const std::vector<uint8_t>& tu, int64_t pts);
  void close();  // rewrites the frame count in the header

 private:
  std::ofstream out_;
  uint32_t frames_ = 0;
};

std::vector<std::vector<uint8_t>> read_ivf(const std::string& path);
