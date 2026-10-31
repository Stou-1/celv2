#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "check.h"
#include "io/ivf.h"
#include "io/memory.h"
#include "io/y4m.h"

namespace fs = std::filesystem;

static std::string tmp(const char* name) { return (fs::temp_directory_path() / name).string(); }

static void write_raw_y4m(const std::string& path, const std::string& header,
                          const std::string& frame_payload) {
  std::ofstream f(path, std::ios::binary);
  f << header << "\n" << "FRAME\n" << frame_payload;
}

// The installed RAM, or CELV2_TEST_TOTAL_RAM_MB's (tests of the memory cap).
static int installed_memory() {
  CHECK(installed_memory_bytes() > (1ull << 30));
  _putenv_s("CELV2_TEST_TOTAL_RAM_MB", "1024");
  const uint64_t m = installed_memory_bytes();
  _putenv_s("CELV2_TEST_TOTAL_RAM_MB", "");
  CHECK_EQ(m, 1ull << 30);
  return 0;
}

static int y4m_roundtrip() {
  Frame420 f;
  f.width = 16; f.height = 8; f.pw = 16; f.ph = 8;
  f.y.resize(16 * 8); f.u.resize(8 * 4); f.v.resize(8 * 4);
  for (size_t i = 0; i < f.y.size(); i++) f.y[i] = static_cast<uint8_t>(i * 3);
  for (size_t i = 0; i < f.u.size(); i++) { f.u[i] = static_cast<uint8_t>(100 + i); f.v[i] = static_cast<uint8_t>(200 - i); }
  const auto bytes = write_y4m_frame_bytes(f);
  write_raw_y4m(tmp("celv2_rt.y4m"), "YUV4MPEG2 W16 H8 F30:1 Ip A1:1 C420jpeg",
                std::string(bytes.begin(), bytes.end()));
  Y4mReader r(tmp("celv2_rt.y4m"));
  CHECK_EQ(r.width(), 16); CHECK_EQ(r.height(), 8);
  CHECK_EQ(r.fps_num(), 30); CHECK_EQ(r.fps_den(), 1);
  Frame420 g;
  CHECK(r.read(g));
  CHECK_EQ(g.width, 16); CHECK_EQ(g.height, 8); CHECK_EQ(g.pw, 16); CHECK_EQ(g.ph, 8);
  CHECK(g.y == f.y); CHECK(g.u == f.u); CHECK(g.v == f.v);
  CHECK(!r.read(g));
  return 0;
}

static int y4m_odd_size_chroma() {
  const int w = 201, h = 121, cw = 101, ch = 61;
  std::string payload(w * h + 2 * cw * ch, '\0');
  for (int i = 0; i < w * h; i++) payload[i] = static_cast<char>(i % 251);
  for (int i = 0; i < cw * ch; i++) {
    payload[w * h + i] = static_cast<char>(i % 199);
    payload[w * h + cw * ch + i] = static_cast<char>(i % 97);
  }
  write_raw_y4m(tmp("celv2_odd.y4m"), "YUV4MPEG2 W201 H121 F25:1 Ip A1:1 C420jpeg", payload);
  Y4mReader r(tmp("celv2_odd.y4m"));
  Frame420 g;
  CHECK(r.read(g));
  CHECK_EQ(g.pw, 208); CHECK_EQ(g.ph, 128);
  const int s = g.stride_uv();
  CHECK_EQ(s, 104);
  CHECK_EQ(int(g.u[60 * s + 100]), int(static_cast<uint8_t>(payload[w * h + 60 * cw + 100])));
  CHECK_EQ(g.u[60 * s + 101], g.u[60 * s + 100]);   // right padding replicates the edge
  CHECK_EQ(g.u[63 * s + 5], g.u[60 * s + 5]);       // bottom padding replicates the edge
  CHECK_EQ(g.y[120 * 208 + 207], g.y[120 * 208 + 200]);
  return 0;
}

static int y4m_rejects_unsupported() {
  // (10 bits are read: test_cli)
  for (const char* hdr : {"YUV4MPEG2 W16 H8 F30:1 C444", "YUV4MPEG2 W16 H8 F30:1 C420p12"}) {
    write_raw_y4m(tmp("celv2_bad.y4m"), hdr, std::string(16 * 8 * 3, '\0'));
    bool threw = false;
    try { Y4mReader r(tmp("celv2_bad.y4m")); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
  }
  return 0;
}

static int ivf_roundtrip() {
  const std::string p = tmp("celv2_rt.ivf");
  {
    IvfWriter w(p, 64, 32, 30, 1);
    w.write({1, 2, 3}, 0);
    w.write({4, 5, 6, 7, 8}, 1);
    w.close();
  }
  std::ifstream f(p, std::ios::binary);
  char hdr[32];
  f.read(hdr, 32);
  CHECK(std::string(hdr, 4) == "DKIF");
  CHECK(std::string(hdr + 8, 4) == "AV02");
  CHECK_EQ(int(static_cast<unsigned char>(hdr[24])), 2);
  const auto tus = read_ivf(p);
  CHECK_EQ(tus.size(), 2u);
  CHECK(tus[0] == std::vector<uint8_t>({1, 2, 3}));
  CHECK(tus[1] == std::vector<uint8_t>({4, 5, 6, 7, 8}));
  return 0;
}

int main() {
  if (int r = y4m_roundtrip()) return r;
  if (int r = y4m_odd_size_chroma()) return r;
  if (int r = y4m_rejects_unsupported()) return r;
  if (int r = ivf_roundtrip()) return r;
  if (int r = installed_memory()) return r;
  return 0;
}
