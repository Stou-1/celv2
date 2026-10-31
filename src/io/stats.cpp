#include "io/stats.h"

#include <fstream>
#include <stdexcept>

namespace celv2 {

void write_stats(const std::string& path, const PassStats& s) {
  std::ofstream out(path, std::ios::binary);
  out << "celv2-stats 1 " << s.width << ' ' << s.height << ' ' << s.qindex << ' ' << s.bytes.size() << '\n';
  for (uint64_t b : s.bytes) out << b << '\n';
  out.close();
  if (!out) throw std::runtime_error("cannot write --stats " + path);
}

PassStats read_stats(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open --stats " + path);
  std::string magic;
  int version = 0;
  size_t frames = 0;
  PassStats s;
  if (!(in >> magic >> version >> s.width >> s.height >> s.qindex >> frames) || magic != "celv2-stats")
    throw std::runtime_error("--stats " + path + " is not a celv2enc --pass 1 statistics file");
  if (version != 1) throw std::runtime_error("--stats " + path + " is version " + std::to_string(version) + ", this celv2enc reads 1");
  s.bytes.resize(frames);
  for (uint64_t& b : s.bytes)
    if (!(in >> b)) throw std::runtime_error("--stats " + path + " ends before its " + std::to_string(frames) + " frames");
  return s;
}

}  // namespace celv2
