// 4:2:0 y4m input, 8 or 10 bits. Frames are padded to a multiple of 8 by edge replication.
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

struct Frame420 {
  int width = 0, height = 0;     // visible size (may be odd)
  int pw = 0, ph = 0;            // padded luma size (multiple of 8)
  std::vector<uint8_t> y, u, v;  // y: pw*ph, u/v: (pw/2)*(ph/2)
  // Above 8 bits the samples are in hy / hu / hv (same layout), and y / u / v hold them rounded to
  // 8 bits for the decisions (the engines, the RD pass, scene cuts). At 8 bits hy..hv are empty.
  int bit_depth = 8;
  std::vector<uint16_t> hy, hu, hv;
  int stride_y() const { return pw; }
  int stride_uv() const { return pw / 2; }
};

// Allocates planes for a visible size, padded to a multiple of 8 (contents zero).
Frame420 make_frame(int width, int height, int bit_depth = 8);
// y / u / v from hy / hu / hv: each sample rounded to 8 bits (nothing at 8 bits).
void round_to_8bit(Frame420& f);
// Re-applies edge replication from the visible area into the padding.
void pad_frame(Frame420& f);

class Y4mReader {
 public:
  // path "-" reads standard input (switched to binary mode). Throws std::runtime_error unless
  // 4:2:0 at 8 or 10 bits (C420p10: 16-bit little-endian samples).
  explicit Y4mReader(const std::string& path);
  ~Y4mReader();
  Y4mReader(const Y4mReader&) = delete;
  Y4mReader& operator=(const Y4mReader&) = delete;
  int width() const { return w_; }
  int height() const { return h_; }
  int fps_num() const { return fps_num_; }
  int fps_den() const { return fps_den_; }
  // The depth of the frames read(): the file's, unless set_bit_depth() asked for another (8 -> 10
  // shifts every sample left by 2; 10 -> 8 rounds).
  int bit_depth() const { return out_depth_; }
  void set_bit_depth(int d);
  bool read(Frame420& out);  // false at end of file

 private:
  bool read_line(std::string& line);  // false at end of file before any byte

  std::FILE* f_ = nullptr;
  bool owned_ = false;  // false for standard input
  int w_ = 0, h_ = 0, fps_num_ = 30, fps_den_ = 1, bit_depth_ = 8, out_depth_ = 8;
};

// Visible area of a frame as one y4m frame payload (without the "FRAME\n" line); 16-bit
// little-endian samples above 8 bits.
std::vector<uint8_t> write_y4m_frame_bytes(const Frame420& f);
