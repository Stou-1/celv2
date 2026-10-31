#include "io/ivf.h"

#include <stdexcept>

static void put_le(std::ofstream& o, uint64_t v, int bytes) {
  for (int i = 0; i < bytes; i++) o.put(static_cast<char>((v >> (8 * i)) & 0xff));
}

IvfWriter::IvfWriter(const std::string& path, int w, int h, int fps_num, int fps_den)
    : out_(path, std::ios::binary) {
  if (!out_) throw std::runtime_error("cannot create " + path);
  out_.write("DKIF", 4);
  put_le(out_, 0, 2);   // version
  put_le(out_, 32, 2);  // header size
  out_.write("AV02", 4);
  put_le(out_, w, 2);
  put_le(out_, h, 2);
  put_le(out_, fps_num, 4);  // timebase denominator (rate)
  put_le(out_, fps_den, 4);  // timebase numerator (scale)
  put_le(out_, 0, 4);        // frame count, patched in close()
  put_le(out_, 0, 4);
}

IvfWriter::~IvfWriter() {
  try {
    if (out_.is_open()) close();
  } catch (const std::exception&) {  // a destructor must not throw; close() reports it when called
  }
}

void IvfWriter::write(const std::vector<uint8_t>& tu, int64_t pts) {
  put_le(out_, tu.size(), 4);
  put_le(out_, static_cast<uint64_t>(pts), 8);
  out_.write(reinterpret_cast<const char*>(tu.data()), static_cast<std::streamsize>(tu.size()));
  if (!out_) throw std::runtime_error("writing the IVF file failed (disk full?)");
  frames_++;
}

void IvfWriter::close() {
  out_.seekp(24);
  put_le(out_, frames_, 4);
  out_.close();
  if (out_.fail()) throw std::runtime_error("finishing the IVF file failed (disk full?)");
}

std::vector<std::vector<uint8_t>> read_ivf(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  char hdr[32];
  if (!f.read(hdr, 32) || std::string(hdr, 4) != "DKIF") throw std::runtime_error("not an IVF file: " + path);
  std::vector<std::vector<uint8_t>> tus;
  for (;;) {
    unsigned char fh[12];
    if (!f.read(reinterpret_cast<char*>(fh), 12)) break;
    const uint32_t sz = fh[0] | (fh[1] << 8) | (fh[2] << 16) | (uint32_t(fh[3]) << 24);
    std::vector<uint8_t> tu(sz);
    if (!f.read(reinterpret_cast<char*>(tu.data()), sz)) throw std::runtime_error("truncated IVF frame");
    tus.push_back(std::move(tu));
  }
  return tus;
}
