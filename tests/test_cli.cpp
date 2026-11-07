#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "bitstream/obu.h"
#include "check.h"
#include "io/ivf.h"
#include "io/stats.h"
#include "io/y4m.h"
#include "test_frames.h"
#include "verify/avm_decode.h"

namespace fs = std::filesystem;

// Runs celv2enc with `args`; returns the exit code and fills `err` with its stderr.
static int run_cli(const std::string& args, std::string& err) {
  const fs::path errfile = fs::temp_directory_path() / "celv2_cli_stderr.txt";
  // cmd.exe strips one pair of outer quotes, hence the extra pair.
  const std::string cmd =
      "\"\"" CELV2ENC_PATH "\" " + args + " 2> \"" + errfile.string() + "\"\"";
  const int rc = std::system(cmd.c_str());
  std::ifstream in(errfile);
  err.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  return rc;
}

static std::string read_bytes(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// Sets an environment variable that std::system's children inherit; restores it on destruction,
// so a failed CHECK cannot leave later tests without the GPU.
class EnvGuard {
 public:
  EnvGuard(const char* name, const char* value) : name_(name) {
    const char* old = std::getenv(name);
    old_ = old ? old : "";
    _putenv_s(name, value);
  }
  ~EnvGuard() { _putenv_s(name_.c_str(), old_.c_str()); }  // "" removes the variable

 private:
  std::string name_, old_;
};

// The 128x64, 3-frame input of the --device tests.
static fs::path device_input() {
  const fs::path y4m = fs::temp_directory_path() / "celv2_cli_dev_in.y4m";
  std::ofstream out(y4m, std::ios::binary);
  out << "YUV4MPEG2 W128 H64 F30:1 Ip C420jpeg\n";
  for (int i = 0; i < 3; i++) {
    const auto bytes = write_y4m_frame_bytes(scene_frame(128, 64, 3 * i, 2 * i));
    out << "FRAME\n";
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }
  return y4m;
}

static const fs::path kDevGpuIvf = fs::temp_directory_path() / "celv2_cli_dev_gpu.ivf";

static int cli_encodes_and_decodes() {
  const fs::path y4m = fs::temp_directory_path() / "celv2_cli_in.y4m";
  const fs::path ivf = fs::temp_directory_path() / "celv2_cli_out.ivf";
  {
    std::ofstream out(y4m, std::ios::binary);
    out << "YUV4MPEG2 W32 H32 F30:1 Ip C420jpeg\n";
    for (int i = 0; i < 3; i++) {
      const auto bytes = write_y4m_frame_bytes(scene_frame(32, 32, 2 * i, i));
      out << "FRAME\n";
      out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
  }
  std::string err;
  CHECK_EQ(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --qindex 120", err), 0);
  CHECK(err.find("speed: 3 frames") != std::string::npos);
  const auto tus = read_ivf(ivf.string());
  CHECK_EQ(tus.size(), 3u);
  CHECK_EQ(decode_frame_md5s(tus).size(), 3u);
  // --frames limits the count, --keyint 1 makes every frame a keyframe, --recon matches decoding.
  const fs::path recon = fs::temp_directory_path() / "celv2_cli_recon.yuv";
  CHECK_EQ(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() +
                       "\" --frames 2 --keyint 1 --recon \"" + recon.string() + "\"",
                   err),
           0);
  const auto tus2 = read_ivf(ivf.string());
  CHECK_EQ(tus2.size(), 2u);
  const auto dec = decode_frames(tus2);
  CHECK_EQ(dec.size(), 2u);
  std::ifstream rin(recon, std::ios::binary);
  const std::string rbytes((std::istreambuf_iterator<char>(rin)), std::istreambuf_iterator<char>());
  std::string expect;
  for (const auto& f : dec) {
    const auto b = write_y4m_frame_bytes(f);
    expect.append(b.begin(), b.end());
  }
  CHECK(rbytes == expect);
  return 0;
}

// --bframes 3: still one temporal unit per frame, decoded in display order and equal to --recon.
// A 10-bit input (C420p10, its low 2 bits used) codes at 10 bits: AVM decodes 10-bit frames equal to
// --recon, over keyframes, B frames and hidden frames.
static int cli_encodes_10bit() {
  const fs::path y4m = fs::temp_directory_path() / "celv2_cli_in10.y4m";
  const fs::path ivf = fs::temp_directory_path() / "celv2_cli_out10.ivf";
  const fs::path recon = fs::temp_directory_path() / "celv2_cli_recon10.yuv";
  {
    std::ofstream out(y4m, std::ios::binary);
    out << "YUV4MPEG2 W64 H48 F30:1 Ip C420p10\n";
    for (int i = 0; i < 9; i++) {
      const Frame420 s = scene_frame(64, 48, 2 * i, i);
      Frame420 f = make_frame(64, 48, 10);
      for (auto [p8, p16] : {std::pair{&s.y, &f.hy}, std::pair{&s.u, &f.hu}, std::pair{&s.v, &f.hv}})
        for (size_t k = 0; k < p8->size(); k++) (*p16)[k] = static_cast<uint16_t>((*p8)[k] * 4 + (k * 7 + i) % 4);
      const auto bytes = write_y4m_frame_bytes(f);
      out << "FRAME\n";
      out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
  }
  std::string err;
  CHECK_EQ(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --qindex 120 --recon \"" + recon.string() + "\"",
                   err),
           0);
  const auto dec = decode_frames(read_ivf(ivf.string()));
  CHECK_EQ(dec.size(), 9u);
  CHECK_EQ(dec[0].bit_depth, 10);
  std::ifstream rin(recon, std::ios::binary);
  const std::string rbytes((std::istreambuf_iterator<char>(rin)), std::istreambuf_iterator<char>());
  std::string expect;
  for (const auto& f : dec) {
    const auto b = write_y4m_frame_bytes(f);
    expect.append(b.begin(), b.end());
  }
  CHECK_EQ(rbytes.size(), 9u * 64 * 48 * 3);  // 1.5 samples a pixel, 2 bytes each
  CHECK(rbytes == expect);
  return 0;
}

// A grainy source: slow shading with fresh noise (about +-10) in every frame, and a moving bar.
static Frame420 grainy_frame(int w, int h, int i, int bit_depth) {
  Frame420 f = make_frame(w, h, bit_depth);
  uint32_t s = 12345u + 977u * static_cast<uint32_t>(i);
  auto noise = [&] {  // the sum of 4 uniform values: roughly Gaussian, sd about 6
    int n = 0;
    for (int k = 0; k < 4; k++) s = s * 1664525u + 1013904223u, n += static_cast<int>(s >> 27) - 16;
    return n / 2;
  };
  const int scale = bit_depth > 8 ? 4 : 1, top = (1 << bit_depth) - 1;
  auto put = [&](std::vector<uint8_t>& p8, std::vector<uint16_t>& p16, int stride, int pw, int ph, int base, bool bar) {
    for (int r = 0; r < ph; r++)
      for (int c = 0; c < pw; c++) {
        int v = (base + (c + r) / 8 + (bar && std::abs(c - 3 * i * pw / w - pw / 4) < pw / 16 ? 60 : 0)) * scale + noise() * scale;
        v = std::clamp(v, 0, top);
        if (bit_depth > 8) p16[r * stride + c] = static_cast<uint16_t>(v);
        else p8[r * stride + c] = static_cast<uint8_t>(v);
      }
  };
  put(f.y, f.hy, f.stride_y(), w, h, 90, true);
  put(f.u, f.hu, f.stride_uv(), (w + 1) / 2, (h + 1) / 2, 120, false);
  put(f.v, f.hv, f.stride_uv(), (w + 1) / 2, (h + 1) / 2, 135, false);
  pad_frame(f);
  round_to_8bit(f);
  return f;
}

// OBUs of a type in a temporal unit (uleb128 size, then the header byte: type in bits 6..2).
static int count_obus(const std::vector<uint8_t>& tu, int type) {
  int n = 0;
  for (size_t p = 0; p < tu.size();) {
    uint64_t size = 0;
    for (int shift = 0;; shift += 7) {
      const uint8_t b = tu[p++];
      size |= static_cast<uint64_t>(b & 0x7f) << shift;
      if (!(b & 0x80)) break;
    }
    n += ((tu[p] >> 2) & 0x1f) == type;
    p += size;
  }
  return n;
}

// --grain 25 on a grainy source, with B frames (hidden frames shown later): AVM decodes the stream (its
// film grain model OBUs, the frame headers' grain fields) equal to --recon with the grain skipped, and
// adds grain to it otherwise, at 8 and 10 bits.
static int cli_grain() {
  const fs::path y4m = fs::temp_directory_path() / "celv2_cli_grain.y4m";
  const fs::path ivf = fs::temp_directory_path() / "celv2_cli_grain.ivf";
  const fs::path recon = fs::temp_directory_path() / "celv2_cli_grain_recon.yuv";
  const int w = 256, h = 128, n = 11;
  for (const int depth : {8, 10}) {
    {
      std::ofstream out(y4m, std::ios::binary);
      out << "YUV4MPEG2 W" << w << " H" << h << " F30:1 Ip " << (depth > 8 ? "C420p10" : "C420jpeg") << "\n";
      for (int i = 0; i < n; i++) {
        const auto bytes = write_y4m_frame_bytes(grainy_frame(w, h, i, depth));
        out << "FRAME\n";
        out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
      }
    }
    std::string err;
    CHECK_EQ(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --bframes 3 --grain 25 --recon \"" +
                         recon.string() + "\"",
                     err),
             0);
    const auto tus = read_ivf(ivf.string());
    CHECK_EQ(tus.size(), static_cast<size_t>(n));
    int models = 0;
    for (const auto& tu : tus) models += count_obus(tu, OBU_FILM_GRAIN_MODEL);
    CHECK(models >= 1);
    const auto dec = decode_frames(tus);
    CHECK_EQ(dec.size(), static_cast<size_t>(n));
    std::string expect;
    for (const auto& f : dec) {
      const auto b = write_y4m_frame_bytes(f);
      expect.append(b.begin(), b.end());
    }
    CHECK(read_bytes(recon) == expect);
    const auto grained = decode_frames(tus, true);
    CHECK_EQ(grained.size(), static_cast<size_t>(n));
    int differ = 0;
    for (int i = 0; i < n; i++) differ += !first_diff(grained[i], dec[i]).empty();
    CHECK_EQ(differ, n);
    // Without --grain: no model OBUs.
    CHECK_EQ(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --bframes 3", err), 0);
    models = 0;
    for (const auto& tu : read_ivf(ivf.string())) models += count_obus(tu, OBU_FILM_GRAIN_MODEL);
    CHECK_EQ(models, 0);
  }
  std::string err;
  // --tune film: --grain 25 unless given; not with --target-vmaf, which scores the frames without the grain.
  auto coded = [&](const std::string& opts) {
    return run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --bframes 3" + opts, err) == 0 ? read_bytes(ivf) : std::string();
  };
  auto models = [&] {
    int m = 0;
    for (const auto& tu : read_ivf(ivf.string())) m += count_obus(tu, OBU_FILM_GRAIN_MODEL);
    return m;
  };
  const std::string film = coded(" --tune film");
  CHECK(!film.empty());
  CHECK(models() >= 1);
  CHECK(film == coded(" --tune film --grain 25"));
  CHECK(!coded(" --tune film --grain 0").empty());
  CHECK_EQ(models(), 0);
  _putenv_s("CELV2_TEST_VMAF", "3,40");
  const std::string vmaf = coded(" --tune film --target-vmaf 90");
  _putenv_s("CELV2_TEST_VMAF", "");
  CHECK(!vmaf.empty());
  CHECK_EQ(models(), 0);
  CHECK(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --grain 51", err) != 0);
  return 0;
}

// --paq: 0..2; outside refused, naming the option.
static int cli_paq_args() {
  const fs::path y4m = device_input();
  const fs::path ivf = fs::temp_directory_path() / "celv2_cli_paq_args.ivf";
  for (const char* bad : {"2.5", "-1", "x"}) {
    std::string err;
    CHECK(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --paq " + bad, err) != 0);
    CHECK(err.find("--paq") != std::string::npos);
  }
  std::string err;
  CHECK_EQ(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --paq 1", err), 0);
  return 0;
}

// --paq on frames half detailed (scene_frame) and half flat, at 8 or 10 bits.
static void write_paq_clip(const fs::path& y4m, int w, int h, int frames, int scene, int depth) {
  std::ofstream out(y4m, std::ios::binary);
  out << "YUV4MPEG2 W" << w << " H" << h << " F30:1 Ip " << (depth > 8 ? "C420p10" : "C420jpeg") << "\n";
  for (int i = 0; i < frames; i++) {
    Frame420 s = scene_frame(w, h, i % scene, (i % scene) / 2, 1 + static_cast<uint32_t>(i / scene));
    for (int r = 0; r < h; r++)
      for (int c = w / 2; c < w; c++) s.y[r * s.stride_y() + c] = static_cast<uint8_t>(150 + (r + c) / 32);
    pad_frame(s);
    Frame420 f = s;
    if (depth > 8) {
      f = make_frame(w, h, 10);
      for (auto [p8, p16] : {std::pair{&s.y, &f.hy}, std::pair{&s.u, &f.hu}, std::pair{&s.v, &f.hv}})
        for (size_t k = 0; k < p8->size(); k++) (*p16)[k] = static_cast<uint16_t>((*p8)[k] * 4 + (k * 7 + i) % 4);
    }
    const auto bytes = write_y4m_frame_bytes(f);
    out << "FRAME\n";
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }
}

// --paq: AVM decodes the stream equal to --recon on every frame type (anchors, mids and non-reference B
// frames all carry offsets), at 8 and 10 bits and at the qindex ends; 0.25 is the default; any worker
// count gives the same bytes.
static int cli_paq() {
  const fs::path dir = fs::temp_directory_path();
  const fs::path y4m = dir / "celv2_cli_paq.y4m", ivf = dir / "celv2_cli_paq.ivf", recon = dir / "celv2_cli_paq_recon.yuv";
  std::string err;
  auto run = [&](const std::string& opts) {
    return run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\"" + opts, err) == 0 ? read_bytes(ivf) : std::string("failed");
  };
  auto decodes_as_recon = [&](const std::string& opts) {
    if (run(opts + " --recon \"" + recon.string() + "\"") == "failed") return false;
    std::string expect;
    for (const auto& f : decode_frames(read_ivf(ivf.string()))) {
      const auto b = write_y4m_frame_bytes(f);
      expect.append(b.begin(), b.end());
    }
    return read_bytes(recon) == expect;
  };
  for (const int depth : {8, 10}) {
    write_paq_clip(y4m, 256, 128, 11, 1000, depth);
    CHECK(decodes_as_recon(" --bframes 3 --paq 1"));
    const std::string with = run(" --bframes 3 --paq 1"), without = run(" --bframes 3 --paq 0");
    CHECK(with != "failed" && without != "failed");
    CHECK(with != without);
    CHECK(run(" --bframes 3 --paq 0.25") == run(" --bframes 3"));  // the default
  }
  CHECK(decodes_as_recon(" --bframes 3 --qindex 250 --paq 2"));
  CHECK(decodes_as_recon(" --bframes 3 --qindex 5 --paq 2"));
  write_paq_clip(y4m, 128, 64, 120, 60, 8);  // a scene cut at 60: two chunks
  const std::string one = run(" --paq 1 --workers 1");
  CHECK(one != "failed");
  CHECK(run(" --paq 1 --workers 2") == one);
  return 0;
}

// --psy-rd: AVM decodes the stream equal to --recon (texture and line rules, 8 and 10 bits, with --paq
// 0 too: the map is built for it alone); the bytes differ from off; any worker count gives the same
// bytes; out-of-range strengths are refused.
static int cli_psy_rd() {
  const fs::path dir = fs::temp_directory_path();
  const fs::path y4m = dir / "celv2_cli_psy.y4m", ivf = dir / "celv2_cli_psy.ivf", recon = dir / "celv2_cli_psy_recon.yuv";
  std::string err;
  auto run = [&](const std::string& opts) {
    return run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\"" + opts, err) == 0 ? read_bytes(ivf) : std::string("failed");
  };
  auto decodes_as_recon = [&](const std::string& opts) {
    if (run(opts + " --recon \"" + recon.string() + "\"") == "failed") return false;
    std::string expect;
    for (const auto& f : decode_frames(read_ivf(ivf.string()))) {
      const auto b = write_y4m_frame_bytes(f);
      expect.append(b.begin(), b.end());
    }
    return read_bytes(recon) == expect;
  };
  for (const int depth : {8, 10}) {
    write_paq_clip(y4m, 256, 128, 11, 1000, depth);
    CHECK(decodes_as_recon(" --bframes 3 --psy-rd 1"));
    CHECK(decodes_as_recon(" --bframes 3 --psy-rd 2 --paq 0 --tune animation"));
    const std::string with = run(" --bframes 3 --psy-rd 1"), without = run(" --bframes 3");
    CHECK(with != "failed" && without != "failed");
    CHECK(with != without);
    // The coder's own term: without the RD pass, only it can change the bytes.
    CHECK(run(" --bframes 3 --psy-rd 1 --rd-pass 0") != run(" --bframes 3 --rd-pass 0"));
  }
  write_paq_clip(y4m, 128, 64, 120, 60, 8);  // a scene cut at 60: two chunks
  const std::string one = run(" --psy-rd 1 --workers 1");
  CHECK(one != "failed");
  CHECK(run(" --psy-rd 1 --workers 2") == one);
  // --psy-trellis: decodes, changes the bytes, any worker count.
  CHECK(decodes_as_recon(" --psy-trellis 1"));
  CHECK(decodes_as_recon(" --psy-rd 1 --psy-trellis 1 --tune animation"));
  CHECK(run(" --psy-trellis 1 --workers 1") != run(" --workers 1"));
  CHECK(run(" --psy-trellis 1 --workers 2") == run(" --psy-trellis 1 --workers 1"));
  CHECK(run(" --psy-rd 1 --psy-trellis 1 --device cpu") == run(" --psy-rd 1 --psy-trellis 1"));
  CHECK(run(" --psy-trellis 2.5") == "failed");
  CHECK(run(" --psy-rd 2.5") == "failed");
  CHECK(run(" --psy-rd -1") == "failed");
  return 0;
}

// A textured picture rotating and zooming about its centre, 1 degree and 1.5% a frame (analytic, so
// each frame is exact), at 8 or 10 bits.
static void write_spin_clip(const fs::path& y4m, int w, int h, int frames, int depth) {
  std::ofstream out(y4m, std::ios::binary);
  out << "YUV4MPEG2 W" << w << " H" << h << " F30:1 Ip " << (depth > 8 ? "C420p10" : "C420jpeg") << "\n";
  for (int i = 0; i < frames; i++) {
    const double t = i * 3.14159265358979 / 180, s = std::pow(1.015, i), cs = s * std::cos(t), sn = s * std::sin(t);
    Frame420 f = make_frame(w, h, depth);
    auto level = [&](int x, int y) {
      const double u = (x - w / 2) * cs + (y - h / 2) * sn, v = -(x - w / 2) * sn + (y - h / 2) * cs;
      return 128 + 60 * std::sin(u / 7) * std::cos(v / 9) + 40 * std::sin((u + 2 * v) / 13);
    };
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++) {
        if (depth > 8) f.hy[y * f.stride_y() + x] = static_cast<uint16_t>(level(x, y) * 4);
        else f.y[y * f.stride_y() + x] = static_cast<uint8_t>(level(x, y));
      }
    if (depth > 8) std::fill(f.hu.begin(), f.hu.end(), uint16_t(512)), std::fill(f.hv.begin(), f.hv.end(), uint16_t(512));
    else std::fill(f.u.begin(), f.u.end(), uint8_t(128)), std::fill(f.v.begin(), f.v.end(), uint8_t(128));
    const auto bytes = write_y4m_frame_bytes(f);
    out << "FRAME\n";
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }
}

// --warp: on a rotating, zooming picture AVM decodes the stream equal to --recon (8 and 10 bits) and it
// is smaller than without; any worker count and either device give the same bytes.
static int cli_warp() {
  const fs::path dir = fs::temp_directory_path();
  const fs::path y4m = dir / "celv2_cli_warp.y4m", ivf = dir / "celv2_cli_warp.ivf", recon = dir / "celv2_cli_warp_recon.yuv";
  std::string err;
  auto run = [&](const std::string& opts) {
    return run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\"" + opts, err) == 0 ? read_bytes(ivf) : std::string("failed");
  };
  auto decodes_as_recon = [&](const std::string& opts) {
    if (run(opts + " --recon \"" + recon.string() + "\"") == "failed") return false;
    std::string expect;
    for (const auto& f : decode_frames(read_ivf(ivf.string()))) {
      const auto b = write_y4m_frame_bytes(f);
      expect.append(b.begin(), b.end());
    }
    return read_bytes(recon) == expect;
  };
  for (const int depth : {8, 10}) {
    write_spin_clip(y4m, 256, 192, 6, depth);
    CHECK(decodes_as_recon(" --warp 1"));
    CHECK(run(" --warp 1").size() < run(" --warp 0").size());
  }
  write_spin_clip(y4m, 256, 192, 6, 8);
  const std::string one = run(" --warp 1 --workers 1");
  CHECK(one != "failed");
  CHECK(run(" --warp 1 --workers 2") == one);
  CHECK(run(" --warp 1 --device cpu") == run(" --warp 1"));
  return 0;
}

// glyph_frame() clips (screen content) at 8 or 10 bits (10: the 8-bit values times 4). changes: frame
// 1 the glyphs inverted and moved, frames 2 on flat (new content the inter frames code as intra:
// palette blocks, then flat DC blocks where they were).
// changes 2: frames 1 on are 16x16 tiles of 8-level stripes (palettes of 8) between flat tiles, each
// at its own level (intra DC, no palette), in inter frames where frame 0's blocks took palettes of 2:
// the mi grid's stale palette size must not reach the flat tiles (it wrote the stripes' colour
// indices against 2 colours; Flash at q190 decoded corrupt).
// changes 3: every frame one 32x32 tile of noise, repeated (new each frame): what only IntraBC codes
// cheaply. changes 4: noise, rows 256 on repeating rows 0 on: out of local IntraBC's reach (global's).
static void write_glyph_clip(const fs::path& y4m, int w, int h, int frames, int depth, int changes = 0) {
  std::ofstream out(y4m, std::ios::binary);
  out << "YUV4MPEG2 W" << w << " H" << h << " F30:1 Ip " << (depth > 8 ? "C420p10" : "C420jpeg") << "\n";
  for (int i = 0; i < frames; i++) {
    Frame420 s = glyph_frame(w, h, changes == 1 && i ? 7 : changes ? 0 : i);
    if (changes == 1 && i == 1)
      for (auto& v : s.y) v = static_cast<uint8_t>(255 - v);
    if (changes == 1 && i > 1) std::fill(s.y.begin(), s.y.end(), static_cast<uint8_t>(90));
    if (changes == 2 && i) {
      for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++)
          s.y[r * s.stride_y() + c] = static_cast<uint8_t>((r / 16 + c / 16 + i) % 2 ? 30 + 25 * ((r + c + i) % 8)
                                                                                : 40 + (r / 16 * 7 + c / 16 * 13 + i * 29) % 160);
      pad_frame(s);
    }
    if (changes == 3) {
      for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++) {
          uint32_t n = static_cast<uint32_t>((r % 32) * 32 + c % 32 + i * 4096 + 1) * 2654435761u;
          n ^= n >> 15, n *= 2246822519u, n ^= n >> 13;
          s.y[r * s.stride_y() + c] = static_cast<uint8_t>(n >> 24);
        }
      pad_frame(s);
    }
    if (changes == 4) {
      for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++) {
          uint32_t n = static_cast<uint32_t>(r % 256 * w + c + 1) * 2654435761u;
          n ^= n >> 15, n *= 2246822519u, n ^= n >> 13;
          s.y[r * s.stride_y() + c] = static_cast<uint8_t>(n >> 24);
        }
      pad_frame(s);
    }
    Frame420 f = s;
    if (depth > 8) {
      f = make_frame(w, h, 10);
      for (auto [p8, p16] : {std::pair{&s.y, &f.hy}, std::pair{&s.u, &f.hu}, std::pair{&s.v, &f.hv}})
        for (size_t k = 0; k < p8->size(); k++) (*p16)[k] = static_cast<uint16_t>((*p8)[k] * 4);
    }
    const auto bytes = write_y4m_frame_bytes(f);
    out << "FRAME\n";
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }
}

// --screen: palettes on glyphs decode equal to --recon (key and inter frames, 8 and 10 bits) and
// code them smaller than without; auto turns them on there (the same bytes as 1) and gives any
// worker count the same bytes; 0 is the default.
static int cli_screen() {
  const fs::path dir = fs::temp_directory_path();
  const fs::path y4m = dir / "celv2_cli_screen.y4m", ivf = dir / "celv2_cli_screen.ivf", recon = dir / "celv2_cli_screen_recon.yuv";
  std::string err;
  auto run = [&](const std::string& opts) {
    return run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\"" + opts, err) == 0 ? read_bytes(ivf) : std::string("failed");
  };
  auto decodes_as_recon = [&](const std::string& opts) {
    if (run(opts + " --recon \"" + recon.string() + "\"") == "failed") return false;
    std::string expect;
    for (const auto& f : decode_frames(read_ivf(ivf.string()))) {
      const auto b = write_y4m_frame_bytes(f);
      expect.append(b.begin(), b.end());
    }
    return read_bytes(recon) == expect;
  };
  for (const int depth : {8, 10}) {
    write_glyph_clip(y4m, 256, 128, 6, depth);
    CHECK(decodes_as_recon(" --bframes 3 --screen 1"));
    CHECK(decodes_as_recon(" --bframes 3 --screen 1 --qindex 190"));
    // Intra blocks in inter frames where the last frame's block at the same place took a palette.
    write_glyph_clip(y4m, 256, 128, 4, depth, 1);
    CHECK(decodes_as_recon(" --bframes 0 --screen 1"));
    write_glyph_clip(y4m, 256, 128, 4, depth, 2);
    CHECK(decodes_as_recon(" --bframes 0 --screen 1"));
    // IntraBC: repeated tiles copied, on key and inter frames, by any worker count.
    write_glyph_clip(y4m, 256, 128, 4, depth, 3);
    CHECK(decodes_as_recon(" --bframes 0 --screen 1"));
    const std::string tiles = run(" --bframes 0 --screen 1 --workers 1"), plain = run(" --bframes 0 --screen 0");
    CHECK(tiles != "failed" && plain != "failed");
    CHECK(tiles.size() * 10 < plain.size() * 9);
    CHECK(run(" --bframes 0 --screen 1 --workers 2") == tiles);
    // Global IntraBC: a key frame's far repeats.
    write_glyph_clip(y4m, 256, 384, 1, depth, 4);
    CHECK(decodes_as_recon(" --screen 1"));
    const std::string far_on = run(" --screen 1"), far_off = run(" --screen 0");
    CHECK(far_on != "failed" && far_off != "failed");
    CHECK(far_on.size() * 10 < far_off.size() * 9);
    write_glyph_clip(y4m, 256, 128, 6, depth);
    const std::string on = run(" --bframes 3 --screen 1"), off = run(" --bframes 3 --screen 0");
    CHECK(on != "failed" && off != "failed");
    CHECK(on.size() < off.size());
    CHECK(run(" --bframes 3 --screen auto") == on);
    CHECK(run(" --bframes 3") == on);  // auto, the default, opens on glyphs
  }
  CHECK(decodes_as_recon(" --bframes 3 --screen 1 --qindex 250"));
  CHECK(decodes_as_recon(" --bframes 3 --screen 1 --qindex 5"));
  // Near copies: a dot screen with noise, no exact repeats; on key frames, auto takes the tools for it.
  {
    std::ofstream o(y4m, std::ios::binary);
    o << "YUV4MPEG2 W256 H128 F30:1 Ip A1:1 C420mpeg2\nFRAME\n";
    uint32_t n = 1;
    std::string plane;
    for (int y = 0; y < 128; y++)
      for (int x = 0; x < 256; x++) {
        n = n * 1664525u + 1013904223u;
        plane.push_back(static_cast<char>(((x % 6 < 3) != (y % 6 < 3) ? 60 : 190) + static_cast<int>(n >> 29)));
      }
    o << plane << std::string(256 * 128 / 2, static_cast<char>(128));
  }
  CHECK(decodes_as_recon(" --screen 1"));
  const std::string dots = run(" --screen 1"), no_dots = run(" --screen 0");
  CHECK(dots != "failed" && no_dots != "failed");
  CHECK(dots.size() < no_dots.size());
  CHECK(run(" --screen auto") == dots);
  write_paq_clip(y4m, 128, 64, 120, 60, 8);  // natural content, a scene cut at 60: two chunks
  const std::string one = run(" --screen auto --workers 1");
  CHECK(one != "failed");
  CHECK(run(" --screen auto --workers 2") == one);
  CHECK(decodes_as_recon(" --screen auto"));
  CHECK(run(" --screen 2") == "failed");
  return 0;
}

// MHCCP past the frame's edge: what earlier blocks left in AVM's luma buffer changes nothing (it reaches the
// residual of the rows not shown, so it once made the bytes depend on which worker coded a chunk). Only a value
// like the picture's shows it: anything else, and MHCCP just loses those blocks.
static int cli_mhccp_past_the_edge() {
  const fs::path dir = fs::temp_directory_path();
  const fs::path y4m = dir / "celv2_cli_mhccp.y4m", ivf = dir / "celv2_cli_mhccp.ivf";
  std::string err;
  auto run = [&](const char* stale) {
    EnvGuard g("CELV2_TEST_MHCCP_STALE", stale);
    return run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --qindex 170 --tune animation", err) == 0 ? read_bytes(ivf) : std::string("failed");
  };
  // Blocks past the bottom, chroma that follows the luma so MHCCP wins some.
  for (const auto& [w, h] : {std::pair{192, 88}, std::pair{256, 120}}) {
    std::ofstream o(y4m, std::ios::binary);
    o << "YUV4MPEG2 W" << w << " H" << h << " F30:1 Ip A1:1 C420mpeg2\nFRAME\n";
    auto luma = [](int x, int y) { return 128 + 60 * std::sin(x * 0.11 + y * 0.07) + 39 * std::sin(x * 0.05 - y * 0.13); };
    auto pel = [](double v) { return static_cast<char>(std::clamp(static_cast<int>(v), 0, 255)); };
    uint32_t n = 1;
    std::string yp, up, vp;
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++) {
        n = n * 1664525u + 1013904223u;
        yp.push_back(pel(static_cast<int>(luma(x, y)) + static_cast<int>(n >> 29) * 2));
      }
    for (int y = 0; y < h / 2; y++)
      for (int x = 0; x < w / 2; x++) {
        const double l = luma(2 * x, 2 * y) - 128;
        up.push_back(pel(128 + l + l * l / 40));
        vp.push_back(pel(128 - l + l * l / 60));
      }
    o << yp << up << vp;
    o.close();
    const std::string a = run("3600");
    CHECK(a != "failed");
    CHECK(run("4096") == a);
    CHECK(run("4600") == a);
  }
  return 0;
}

// CELV2_WIDE W3: trial-coded 64x64s put everything back, so the decoder still follows; any worker count, the same bytes.
static int cli_wide_real_cost() {
  const fs::path dir = fs::temp_directory_path();
  const fs::path y4m = dir / "celv2_cli_wide.y4m", ivf = dir / "celv2_cli_wide.ivf", recon = dir / "celv2_cli_wide_recon.yuv";
  std::string err;
  auto run = [&](const std::string& opts) {
    return run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --qindex 130 --bframes 3" + opts, err) == 0 ? read_bytes(ivf) : std::string("failed");
  };
  write_glyph_clip(y4m, 256, 128, 6, 8);
  const std::string off = run("");
  CHECK(off != "failed");
  EnvGuard g("CELV2_WIDE", "3");
  const std::string on = run(" --recon \"" + recon.string() + "\"");
  CHECK(on != "failed" && on != off);  // alternatives were taken
  std::string expect;
  for (const auto& f : decode_frames(read_ivf(ivf.string()))) {
    const auto b = write_y4m_frame_bytes(f);
    expect.append(b.begin(), b.end());
  }
  CHECK(read_bytes(recon) == expect);
  CHECK(run(" --workers 1") == run(" --workers 2"));
  return 0;
}

// X_L0: a frame list in place of the mini-GOPs, leaves kept in slots: it decodes to its --recon, in display order.
static int cli_frame_list() {
  const fs::path dir = fs::temp_directory_path();
  const fs::path y4m = dir / "celv2_cli_l0.y4m", ivf = dir / "celv2_cli_l0.ivf", recon = dir / "celv2_cli_l0_recon.yuv", list = dir / "celv2_cli_l0.txt";
  std::string err;
  auto run = [&](const std::string& opts) {
    return run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --qindex 130 --refs 7 --bframes 8 --scenecut 0" + opts, err) == 0 ? read_bytes(ivf) : std::string("failed");
  };
  write_glyph_clip(y4m, 256, 128, 6, 8);
  std::ofstream(list) << "0 1 255 100\n5 0 2 110\n2 0 4 120\n1 1 8 130\n3 1 16 130\n4 1 0 130\n";
  const std::string off = run("");
  CHECK(off != "failed");
  EnvGuard g("X_L0", list.string().c_str());
  const std::string on = run(" --recon \"" + recon.string() + "\"");
  CHECK(on != "failed" && on != off);
  const auto frames = decode_frames(read_ivf(ivf.string()));
  CHECK_EQ(frames.size(), size_t{6});
  std::string expect;
  for (const auto& f : frames) {
    const auto b = write_y4m_frame_bytes(f);
    expect.append(b.begin(), b.end());
  }
  CHECK(read_bytes(recon) == expect);
  return 0;
}

#ifdef CELV2_ORACLE
// tools/oracle: a stream's decisions read back and coded again. Every level decodes to its --recon, L2's stream
// holds the first one's frames, trees, modes and MVs, and L4's and L6's are the first one byte for byte.
static int cli_oracle_replay() {
  const fs::path dir = fs::temp_directory_path();
  const fs::path y4m = dir / "celv2_cli_or.y4m", first = dir / "celv2_cli_or1.ivf", again = dir / "celv2_cli_or2.ivf", recon = dir / "celv2_cli_or.yuv",
                 list = dir / "celv2_cli_or.txt", dump1 = dir / "celv2_cli_or1.bin", dump2 = dir / "celv2_cli_or2.bin";
  std::string err;
  auto dump = [](const fs::path& ivf, const fs::path& out) {
    return std::system(("\"\"" ORACLE_DUMP_PATH "\" \"" + ivf.string() + "\" \"" + out.string() + "\"\"").c_str());
  };
  write_glyph_clip(y4m, 256, 128, 6, 8);
  std::ofstream(list) << "0 1 255 100\n5 0 2 110\n2 0 4 120\n1 1 8 130\n3 1 16 130\n4 1 0 130\n";
  EnvGuard g("X_L0", list.string().c_str());
  const std::string args = " --qindex 130 --refs 7 --bframes 8 --scenecut 0 --warp 0 --interintra 0 --screen 0";
  CHECK_EQ(run_cli("-i \"" + y4m.string() + "\" -o \"" + first.string() + "\"" + args, err), 0);
  CHECK_EQ(dump(first, dump1), 0);
  EnvGuard o("X_ORACLE", dump1.string().c_str());
  for (const char* level : {"1", "3", "5", "4", "6", "2"}) {  // 2 last: its dump is compared below
    EnvGuard l("X_ORACLE_LEVEL", level);
    CHECK_EQ(run_cli("-i \"" + y4m.string() + "\" -o \"" + again.string() + "\" --recon \"" + recon.string() + "\"" + args, err), 0);
    std::string expect;
    for (const auto& f : decode_frames(read_ivf(again.string()))) {
      const auto b = write_y4m_frame_bytes(f);
      expect.append(b.begin(), b.end());
    }
    CHECK(read_bytes(recon) == expect);
    if (*level == '4' || *level == '6') CHECK(read_bytes(again) == read_bytes(first));
  }
  CHECK_EQ(dump(again, dump2), 0);
  CHECK(read_bytes(dump1) == read_bytes(dump2));
  // X_ORACLE_Q at L4: on fixed CDFs a path's rate is the bits of the symbols written for its levels (X_ORACLE_QCHECK fails the
  // run otherwise), over every transform block here and the same past the range; and the stream decodes to its --recon.
  EnvGuard l4("X_ORACLE_LEVEL", "4"), check("X_ORACLE_QCHECK", "1");
  for (const char* q : {"a", "abd", "acd"}) {
    EnvGuard mode("X_ORACLE_Q", q);
    CHECK_EQ(run_cli("-i \"" + y4m.string() + "\" -o \"" + again.string() + "\" --recon \"" + recon.string() + "\"" + args, err), 0);
    std::string expect;
    for (const auto& f : decode_frames(read_ivf(again.string()))) {
      const auto b = write_y4m_frame_bytes(f);
      expect.append(b.begin(), b.end());
    }
    CHECK(read_bytes(recon) == expect);
    CHECK(read_bytes(again) != read_bytes(first));
  }
  return 0;
}
#endif

// CELV2_WIDE's other switches: each changes the stream, which still decodes to its --recon.
static int cli_wide_switches() {
  const fs::path dir = fs::temp_directory_path();
  const fs::path y4m = dir / "celv2_cli_wide2.y4m", ivf = dir / "celv2_cli_wide2.ivf", recon = dir / "celv2_cli_wide2_recon.yuv";
  std::string err;
  auto run = [&](const std::string& opts) {
    return run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --qindex 130 --refs 7 --bframes 7" + opts, err) == 0 ? read_bytes(ivf) : std::string("failed");
  };
  write_glyph_clip(y4m, 256, 128, 12, 8);
  const std::string off = run("");
  CHECK(off != "failed");
  for (const char* w : {"5", "4", "6", "1", "2", "124563"}) {
    EnvGuard g("CELV2_WIDE", w);
    const std::string on = run(" --recon \"" + recon.string() + "\"");
    CHECK(on != "failed" && (on != off || w[0] == '6'));  // this clip takes no further pair
    std::string expect;
    for (const auto& f : decode_frames(read_ivf(ivf.string()))) {
      const auto bytes = write_y4m_frame_bytes(f);
      expect.append(bytes.begin(), bytes.end());
    }
    CHECK(read_bytes(recon) == expect);
    CHECK(on == run(" --workers 2"));
  }
  return 0;
}

// --rd-zero: changes the stream, which still decodes to its --recon, luma and chroma, at any worker count, on the plain path
// and in --pass 1. Off under every tune until the default changes.
static int cli_rd_zero() {
  const fs::path dir = fs::temp_directory_path();
  const fs::path y4m = dir / "celv2_cli_rdzero.y4m", ivf = dir / "celv2_cli_rdzero.ivf", recon = dir / "celv2_cli_rdzero_recon.yuv";
  const std::string stats = " --pass 1 --stats \"" + (dir / "celv2_cli_rdzero.stats").string() + "\"";
  std::string err;
  auto run = [&](const std::string& opts) {
    return run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --qindex 130 --refs 7 --bframes 7 --screen 0" + opts, err) == 0 ? read_bytes(ivf) : std::string("failed");
  };
  auto decodes_as_recon = [&](const std::string& opts) {
    if (run(opts + " --recon \"" + recon.string() + "\"") == "failed") return false;
    std::string expect;
    for (const auto& f : decode_frames(read_ivf(ivf.string()))) {
      const auto bytes = write_y4m_frame_bytes(f);
      expect.append(bytes.begin(), bytes.end());
    }
    return read_bytes(recon) == expect;
  };
  write_glyph_clip(y4m, 256, 128, 12, 8);
  const std::string off = run("");
  CHECK(off != "failed");
  CHECK(run(" --rd-zero 0") == off);
  for (const char* t : {" --tune film", " --tune still"}) CHECK(run(t) == run(std::string(t) + " --rd-zero 0"));
  CHECK(run(" --tune animation") == run(" --tune animation --rd-zero 1"));
  CHECK(run(" --tune animation") != run(" --tune animation --rd-zero 0"));
  const std::string on = run(" --rd-zero 1");
  CHECK(on != "failed" && on != off);
  CHECK(decodes_as_recon(" --rd-zero 1"));
  CHECK(on == run(" --rd-zero 1 --workers 1") && on == run(" --rd-zero 1 --workers 2"));
  CHECK(decodes_as_recon(" --rd-zero 1 --tcq 0"));
  CHECK(run(" --rd-zero 1 --tcq 0") != run(" --tcq 0"));
  CHECK(run(" --rd-zero 1" + stats) != run(stats));
  write_paq_clip(y4m, 128, 64, 120, 60, 8);  // a scene cut at 60: two chunks
  const std::string chunks = run(" --rd-zero 1 --workers 1");
  CHECK(chunks != "failed" && chunks != run(" --workers 1"));
  CHECK(run(" --rd-zero 1 --workers 2") == chunks);
  return 0;
}

// CELV2_PROFILE=1: per-stage CPU time lines at the end, the same bytes as without.
static int cli_profile() {
  const fs::path y4m = device_input();
  const fs::path ivf = fs::temp_directory_path() / "celv2_cli_profile.ivf";
  std::string err;
  CHECK_EQ(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --bframes 1", err), 0);
  CHECK(err.find("profile:") == std::string::npos);
  const std::string plain = read_bytes(ivf);
  {
    EnvGuard on("CELV2_PROFILE", "1");
    CHECK_EQ(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --bframes 1", err), 0);
  }
  CHECK(read_bytes(ivf) == plain);
  for (const char* stage : {"profile: CPU", "profile: reading", "profile: decision", "profile: RD pass", "profile: coder",
                            "profile: quantization", "profile: entropy coding", "profile: deblocking", "profile: CDEF"})
    CHECK(err.find(stage) != std::string::npos);
  return 0;
}

// CELV2_GOP_LOG=1: a scene-cut line per frame and a line per mini-GOP, the same bytes as without.
static int cli_gop_log() {
  const fs::path y4m = fs::temp_directory_path() / "celv2_cli_goplog.y4m";
  const fs::path ivf = fs::temp_directory_path() / "celv2_cli_goplog.ivf";
  {
    std::ofstream out(y4m, std::ios::binary);
    out << "YUV4MPEG2 W64 H48 F30:1 Ip C420jpeg\n";
    for (int i = 0; i < 11; i++) {
      const auto bytes = write_y4m_frame_bytes(scene_frame(64, 48, 2 * i, i));
      out << "FRAME\n";
      out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
  }
  std::string err;
  CHECK_EQ(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --bframes 3", err), 0);
  CHECK(err.find("celv2enc: gop") == std::string::npos);
  const std::string plain = read_bytes(ivf);
  {
    EnvGuard on("CELV2_GOP_LOG", "1");
    CHECK_EQ(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --bframes 3", err), 0);
  }
  CHECK(read_bytes(ivf) == plain);
  CHECK(err.find("celv2enc: gop frame 10: mad ") != std::string::npos);
  CHECK(err.find(" unchanged ") != std::string::npos && err.find(" hf ") != std::string::npos);
  CHECK(err.find("celv2enc: gop group 1 len 4 still ") != std::string::npos);
  CHECK(err.find(" anchor_q ") != std::string::npos && err.find(" r0 ") != std::string::npos);
  return 0;
}

// --gop-change 0.3: a new picture revealed by an iris wipe (a circle opening over frames 10-15) ends a
// mini-GOP inside the wipe, that frame coded as an anchor; off, the anchors stay at 8 and 16.
static int cli_gop_change() {
  const fs::path y4m = fs::temp_directory_path() / "celv2_cli_wipe.y4m";
  const fs::path ivf = fs::temp_directory_path() / "celv2_cli_wipe.ivf";
  const int w = 128, h = 96;
  {
    std::ofstream out(y4m, std::ios::binary);
    out << "YUV4MPEG2 W" << w << " H" << h << " F30:1 Ip C420jpeg\n";
    const Frame420 a = scene_frame(w, h, 0, 0, 1), b = scene_frame(w, h, 0, 0, 7);
    for (int i = 0; i < 20; i++) {
      // Ordinary motion (a 48x48 square, 16 px a frame: about a third of the blocks change), then
      // a new picture all at once at frame 14, and another at 15.
      Frame420 f = a;
      if (i >= 15) {
        f = b;
      } else if (i == 14) {
        for (int y = 0; y < h; y++)
          for (int x = 0; x < w; x++) f.y[y * f.stride_y() + x] = 255 - b.y[y * b.stride_y() + x];
      } else {
        const int x0 = i * 16 % (w - 48);
        for (int y = 24; y < 72; y++)
          for (int x = x0; x < x0 + 48; x++) f.y[y * f.stride_y() + x] = 250;
      }
      pad_frame(f);
      const auto bytes = write_y4m_frame_bytes(f);
      out << "FRAME\n";
      out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
  }
  auto anchors = [&](const std::string& opts) {
    std::string err;
    std::vector<int> out;
    if (run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --bframes 7 --scenecut 0 --progress" + opts, err) != 0) return out;
    for (size_t p = 0; (p = err.find("progress frame=", p)) != std::string::npos; p++) {
      const int n = std::atoi(err.c_str() + p + 15);
      if (err.compare(err.find("type=", p) + 5, 1, "A") == 0) out.push_back(n - 1);
    }
    return out;
  };
  const std::vector<int> off = anchors(""), on = anchors(" --gop-change 0.9");
  CHECK(!off.empty() && !on.empty());
  for (const int d : off) CHECK(d < 9 || d > 15);
  bool at14 = false;
  for (const int d : on) {
    CHECK(d < 9 || d >= 14);  // the motion before it does not end a group
    at14 = at14 || d == 14;
  }
  CHECK(at14);
  // The new picture is no coarser than the base qindex: the next group refers to it, which the TPL
  // of its own group does not see.
  std::string err;
  CHECK_EQ(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --bframes 7 --scenecut 0 --progress --qindex 120 --gop-change 0.9", err), 0);
  const size_t at = err.find("progress frame=15 ");
  CHECK(at != std::string::npos);
  CHECK(std::atoi(err.c_str() + err.find("qindex=", at) + 7) <= 120);
  // The same where --gop-still shortens the group after the cut was found (frames 1 .. 14 to 1 .. 8):
  // frame 14 still ends the next one.
  bool still14 = false;
  for (const int d : anchors(" --bframes 15 --gop-still 0.99 --gop-change 0.9")) still14 = still14 || d == 14;
  CHECK(still14);
  // Frame 15, one after that anchor, is not one: a cut needs 4 frames from the last one.
  const size_t next = err.find("progress frame=16 ");
  CHECK(next != std::string::npos);
  CHECK(err.compare(err.find("type=", next) + 5, 1, "B") == 0);
  CHECK(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --gop-change 1.5", err) != 0);
  return 0;
}

// --colour / --mdcv / --cll add a content interpretation OBU and two HDR metadata OBUs after the
// sequence header of a keyframe; AVM still decodes the stream.
static int cli_hdr_signalling() {
  const fs::path y4m = device_input();
  const fs::path ivf = fs::temp_directory_path() / "celv2_cli_hdr.ivf";
  std::string err;
  CHECK_EQ(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() +
                       "\" --colour pq --mdcv G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)"
                       " --cll 1000,400",
                   err),
           0);
  const auto tus = read_ivf(ivf.string());
  CHECK_EQ(decode_frames(tus).size(), 3u);
  std::vector<int> types;  // the first unit's OBU types
  for (size_t i = 0; i < tus[0].size();) {
    uint64_t size = 0;
    for (int shift = 0;; shift += 7) {
      const uint8_t b = tus[0][i++];
      size |= static_cast<uint64_t>(b & 0x7f) << shift;
      if (!(b & 0x80)) break;
    }
    types.push_back((tus[0][i] >> 2) & 0x1f);
    i += size;
  }
  const std::vector<int> expect = {OBU_TEMPORAL_DELIMITER, OBU_SEQUENCE_HEADER, OBU_CONTENT_INTERPRETATION,
                                   OBU_METADATA_SHORT, OBU_METADATA_SHORT, OBU_CLOSED_LOOP_KEY};
  CHECK(types == expect);
  CHECK(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --cll 70000,1", err) != 0);
  return 0;
}

static int cli_bframes() {
  const fs::path y4m = fs::temp_directory_path() / "celv2_cli_bf.y4m";
  const fs::path ivf = fs::temp_directory_path() / "celv2_cli_bf.ivf";
  const fs::path recon = fs::temp_directory_path() / "celv2_cli_bf_recon.yuv";
  {
    std::ofstream out(y4m, std::ios::binary);
    out << "YUV4MPEG2 W64 H48 F30:1 Ip C420jpeg\n";
    for (int i = 0; i < 11; i++) {
      const auto bytes = write_y4m_frame_bytes(scene_frame(64, 48, 2 * i, i));
      out << "FRAME\n";
      out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
  }
  // The default references (pyramid: a hidden middle frame), 2 (no pyramid) and 1 (no B frames).
  for (const char* refs : {"", " --refs 2", " --refs 1"}) {
    std::string err;
    CHECK_EQ(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --bframes 3" + refs + " --recon \"" +
                         recon.string() + "\"",
                     err),
             0);
    CHECK(err.find("speed: 11 frames") != std::string::npos);
    const auto tus = read_ivf(ivf.string());
    CHECK_EQ(tus.size(), 11u);
    const auto dec = decode_frames(tus);
    CHECK_EQ(dec.size(), 11u);
    std::ifstream rin(recon, std::ios::binary);
    const std::string rbytes((std::istreambuf_iterator<char>(rin)), std::istreambuf_iterator<char>());
    std::string expect;
    for (const auto& f : dec) {
      const auto b = write_y4m_frame_bytes(f);
      expect.append(b.begin(), b.end());
    }
    CHECK(rbytes == expect);
  }
  return 0;
}

// --list-devices: "<--device value>\t<name>" lines, values the parser takes, CPU last; no input needed.
static int cli_lists_devices() {
  const fs::path out = fs::temp_directory_path() / "celv2_cli_devices.txt";
  std::string err;
  CHECK_EQ(run_cli("--list-devices > \"" + out.string() + "\"", err), 0);
  std::ifstream in(out);
  std::string line, last;
  int n = 0;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const size_t tab = line.find('\t');
    CHECK(tab != std::string::npos && tab + 1 < line.size());
    const std::string v = line.substr(0, tab);
    CHECK(v == "vulkan" || v == "vulkan2" || v == "cpu");
    last = line, n++;
  }
  CHECK(n >= 1);
  CHECK(last == "cpu\tCPU");
  return 0;
}

static int cli_rejects_bad_args() {
  const fs::path y4m = fs::temp_directory_path() / "celv2_cli_in.y4m";
  const fs::path ivf = fs::temp_directory_path() / "celv2_cli_bad.ivf";
  std::string err;
  CHECK(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --qindex 0", err) != 0);
  CHECK(!err.empty());
  CHECK(run_cli("-i \"" + (fs::temp_directory_path() / "no_such_file.y4m").string() + "\" -o \"" +
                    ivf.string() + "\"",
                err) != 0);
  CHECK(!err.empty());
  CHECK(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --tune grain", err) != 0);
  CHECK(err.find("--tune must be") != std::string::npos);
  CHECK(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --gop-still 1.5", err) != 0);
  CHECK(err.find("--gop-still must be") != std::string::npos);
  CHECK(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --preset fast", err) != 0);
  CHECK(err.find("--preset must be") != std::string::npos);
  return 0;
}

// --tune with 7 references: 16-frame mini-GOPs where the picture holds still (here all but a
// slow pan), each stream decoded equal to --recon.
static int cli_tunes() {
  const fs::path y4m = fs::temp_directory_path() / "celv2_cli_tune.y4m";
  const fs::path ivf = fs::temp_directory_path() / "celv2_cli_tune.ivf";
  const fs::path recon = fs::temp_directory_path() / "celv2_cli_tune_recon.yuv";
  {
    std::ofstream out(y4m, std::ios::binary);
    out << "YUV4MPEG2 W64 H48 F30:1 Ip C420jpeg\n";
    for (int i = 0; i < 20; i++) {
      const auto bytes = write_y4m_frame_bytes(scene_frame(64, 48, i / 8, 0));
      out << "FRAME\n";
      out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
  }
  for (const char* tune : {"animation", "film", "still"}) {
    std::string err;
    CHECK_EQ(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --refs 7 --qindex 120 --tune " + tune +
                         " --grain 0 --recon \"" + recon.string() + "\"",
                     err),
             0);
    const auto dec = decode_frames(read_ivf(ivf.string()));
    CHECK_EQ(dec.size(), 20u);
    std::ifstream rin(recon, std::ios::binary);
    const std::string rbytes((std::istreambuf_iterator<char>(rin)), std::istreambuf_iterator<char>());
    std::string expect;
    for (const auto& f : dec) {
      const auto b = write_y4m_frame_bytes(f);
      expect.append(b.begin(), b.end());
    }
    CHECK(rbytes == expect);
  }
  return 0;
}

// The anchors' qindex comes from TPL by default (--tplq 3): the same bytes as given, others without.
static int cli_tplq_default() {
  const fs::path y4m = fs::temp_directory_path() / "celv2_cli_tplq.y4m";
  const fs::path ivf = fs::temp_directory_path() / "celv2_cli_tplq.ivf";
  {
    std::ofstream out(y4m, std::ios::binary);
    out << "YUV4MPEG2 W128 H96 F30:1 Ip C420jpeg\n";
    for (int i = 0; i < 20; i++) {
      const auto bytes = write_y4m_frame_bytes(scene_frame(128, 96, i / 2, 0));
      out << "FRAME\n";
      out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
  }
  std::string err;
  auto bytes_of = [&](const std::string& opts) {
    return run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --refs 7 --qindex 120" + opts, err) == 0 ? read_bytes(ivf) : std::string("failed");
  };
  const std::string def = bytes_of("");
  CHECK(def != "failed");
  CHECK(def == bytes_of(" --tplq 3"));
  CHECK(def != bytes_of(" --tplq 0"));
  return 0;
}

// --tune animation turns on temporal MVs and the keyframe's temporal filter (each can still be
// set), and its streams decode equal to --recon (cli_tunes).
static Frame420 blob_appears(int w, int h, int i);
static int cli_anime_tune_tools() {
  const fs::path y4m = fs::temp_directory_path() / "celv2_cli_anime.y4m";
  const fs::path ivf = fs::temp_directory_path() / "celv2_cli_anime.ivf";
  {
    std::ofstream out(y4m, std::ios::binary);
    out << "YUV4MPEG2 W64 H48 F30:1 Ip C420jpeg\n";
    for (int i = 0; i < 12; i++) {
      const auto bytes = write_y4m_frame_bytes(blob_appears(64, 48, i));
      out << "FRAME\n";
      out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
  }
  std::string err;
  auto bytes_of = [&](const std::string& opts) {
    return run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --refs 7 --qindex 120" + opts, err) == 0 ? read_bytes(ivf) : std::string("failed");
  };
  const std::string tune = bytes_of(" --tune animation");
  CHECK(tune != "failed");
  CHECK(tune == bytes_of(" --tune animation --tmv 1 --key-filter 1 --rd-zero 1"));
  CHECK(tune != bytes_of(" --tune animation --tmv 0 --key-filter 1"));
  CHECK(tune != bytes_of(" --tune animation --tmv 1 --key-filter 0"));
  CHECK(tune != bytes_of(" --tune animation --rd-zero 0"));
  CHECK(bytes_of("") == bytes_of(" --tmv 0 --key-filter 0 --rd-zero 0"));
  return 0;
}

// Noisy diagonal ramps with chroma linear in luma (what cross-component prediction fits).
static Frame420 chroma_follows_luma(int w, int h, int i) {
  Frame420 f = make_frame(w, h);
  uint32_t n = 1;
  for (int r = 0; r < h; r++)
    for (int c = 0; c < w; c++) {
      n = n * 1103515245u + 12345u;
      const int v = 128 + 60 * ((c * 7 + r * 3 + i) % 37 - 18) / 18 + static_cast<int>(n >> 16) % 41 - 20;
      f.y[r * f.stride_y() + c] = static_cast<uint8_t>(v < 0 ? 0 : v > 255 ? 255 : v);
    }
  for (int r = 0; r < h / 2; r++)
    for (int c = 0; c < w / 2; c++) {
      const int l = f.y[2 * r * f.stride_y() + 2 * c];
      f.u[r * f.stride_uv() + c] = static_cast<uint8_t>(64 + l / 2);
      f.v[r * f.stride_uv() + c] = static_cast<uint8_t>(200 - l / 3);
    }
  pad_frame(f);
  return f;
}

// A noisy wave texture moving 2 pixels a frame, a bright blob appearing in its middle from frame 1 on
// (what an inter prediction blended with an intra one fits).
static Frame420 blob_appears(int w, int h, int i) {
  Frame420 f = make_frame(w, h);
  uint32_t n = static_cast<uint32_t>(i) + 1;
  for (int r = 0; r < h; r++)
    for (int c = 0; c < w; c++) {
      n = n * 1103515245u + 12345u;
      const double d2 = (c - w / 2) * (c - w / 2) + (r - h / 2) * (r - h / 2);
      const int v = static_cast<int>(100 + 50 * std::sin((c + 2 * i) / 5.0) * std::cos(r / 7.0) +
                                     (i ? 80 * std::exp(-d2 / 1600) : 0)) +
                    static_cast<int>(n >> 16) % 13 - 6;
      f.y[r * f.stride_y() + c] = static_cast<uint8_t>(v < 0 ? 0 : v > 255 ? 255 : v);
    }
  for (int r = 0; r < h / 2; r++)
    for (int c = 0; c < w / 2; c++) f.u[r * f.stride_uv() + c] = f.v[r * f.stride_uv() + c] = 128;
  pad_frame(f);
  return f;
}

// --<tool> 1 (cfl / mhccp: some chroma block takes CfL / MHCCP; cctx: some chroma pair is rotated; lr-uv: some
// chroma restoration unit keeps the frame's Wiener filter; interintra: some inter block blends in an intra
// prediction): the reconstruction differs from --<tool> 0's, and both streams decode equal to their --recon.
static int cli_tool_used(const std::string& tool, Frame420 (*pattern)(int w, int h, int i) = nullptr, int qindex = 120,
                         const std::string& extra = "") {
  const int w = pattern ? 320 : 128, h = pattern ? 192 : 96;
  const fs::path y4m = fs::temp_directory_path() / ("celv2_cli_" + tool + ".y4m");
  const fs::path ivf = fs::temp_directory_path() / ("celv2_cli_" + tool + ".ivf");
  const fs::path recon = fs::temp_directory_path() / ("celv2_cli_" + tool + "_recon.yuv");
  {
    std::ofstream out(y4m, std::ios::binary);
    out << "YUV4MPEG2 W" << w << " H" << h << " F30:1 Ip C420jpeg\n";
    for (int i = 0; i < 3; i++) {
      const auto bytes = write_y4m_frame_bytes(pattern ? pattern(w, h, i) : scene_frame(w, h, i, i));
      out << "FRAME\n";
      out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
  }
  std::string recons[2];
  for (int on = 0; on < 2; on++) {
    std::string err;
    CHECK_EQ(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --qindex " + std::to_string(qindex) + " --" + tool + " " +
                         std::to_string(on) + extra + " --recon \"" + recon.string() + "\"",
                     err),
             0);
    const auto dec = decode_frames(read_ivf(ivf.string()));
    CHECK_EQ(dec.size(), 3u);
    std::ifstream rin(recon, std::ios::binary);
    recons[on].assign((std::istreambuf_iterator<char>(rin)), std::istreambuf_iterator<char>());
    std::string expect;
    for (const auto& f : dec) {
      const auto b = write_y4m_frame_bytes(f);
      expect.append(b.begin(), b.end());
    }
    CHECK(recons[on] == expect);
  }
  CHECK(recons[0] != recons[1]);
  return 0;
}
// On the small scene picture no chroma block takes CfL once the RD pass scores DC blocks right (its
// partitions change); chroma that follows luma is what CfL is for.
static int cli_cfl() { return cli_tool_used("cfl", chroma_follows_luma); }
// qindex 90: at 120, with the chroma modes chosen by quantized cost (--cfl-rd), no chroma pair of the
// small scene picture rotates.
static int cli_cctx() { return cli_tool_used("cctx", nullptr, 90); }
static int cli_mhccp() { return cli_tool_used("mhccp", chroma_follows_luma); }
// --cfl-rd with CfL on: the signalled alpha and the chroma RD shortlist decode as --recon says.
static int cli_cfl_rd() { return cli_tool_used("cfl-rd", chroma_follows_luma, 120, " --cfl 1"); }
static int cli_lr_uv() { return cli_tool_used("lr-uv", chroma_follows_luma); }
// qindex 90: at 120 the intra RD mode search codes the new blob's blocks intra, none inter-intra.
static int cli_interintra() { return cli_tool_used("interintra", blob_appears, 90); }
// intra-txp: some intra block splits its luma transform.
static int cli_intra_txp() { return cli_tool_used("intra-txp"); }

// --preset max: a 16-frame mini-GOP over the still first frames is coded both ways and the better
// kept; the output decodes to its --recon and equals one of the two fixed choices byte for byte.
static int cli_preset_max() {
  const fs::path y4m = fs::temp_directory_path() / "celv2_cli_max.y4m";
  const fs::path ivf = fs::temp_directory_path() / "celv2_cli_max.ivf";
  const fs::path recon = fs::temp_directory_path() / "celv2_cli_max_recon.yuv";
  {
    std::ofstream out(y4m, std::ios::binary);
    out << "YUV4MPEG2 W64 H48 F30:1 Ip C420jpeg\n";
    for (int i = 0; i < 20; i++) {
      const auto bytes = write_y4m_frame_bytes(scene_frame(64, 48, i < 17 ? 0 : i, 0));
      out << "FRAME\n";
      out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
  }
  const std::string io = "-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --refs 7 --qindex 120 --preset max";
  std::string err;
  CHECK_EQ(run_cli(io + " --recon \"" + recon.string() + "\"", err), 0);
  const auto dec = decode_frames(read_ivf(ivf.string()));
  CHECK_EQ(dec.size(), 20u);
  std::string expect;
  for (const auto& f : dec) {
    const auto b = write_y4m_frame_bytes(f);
    expect.append(b.begin(), b.end());
  }
  CHECK(read_bytes(recon) == expect);
  const std::string tried = read_bytes(ivf);
  CHECK_EQ(run_cli(io + " --gop-trial 0", err), 0);
  const std::string sixteen = read_bytes(ivf);
  CHECK_EQ(run_cli(io + " --bframes 7", err), 0);
  const std::string eights = read_bytes(ivf);
  CHECK(sixteen != eights);  // the clip has a mini-GOP the trial decides
  CHECK(tried == sixteen);  // 16 still frames: one mini-GOP costs less than two
  // A trial that fails while coding (GPU memory runs out, say) costs only the trial: a warning, and
  // the main pipeline's 8 + 8 kept.
  _putenv_s("CELV2_TEST_TRIAL_FAIL", "1");
  const int failed_rc = run_cli(io, err);
  _putenv_s("CELV2_TEST_TRIAL_FAIL", "");
  CHECK_EQ(failed_rc, 0);
  CHECK(err.find("--gop-trial off") != std::string::npos);
  CHECK(read_bytes(ivf) == eights);
  // Per-frame deblocking strength alone: still decodes to the recon.
  CHECK_EQ(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --refs 7 --qindex 120 --deblock-search 1 --recon \"" +
                       recon.string() + "\"",
                   err),
           0);
  expect.clear();
  for (const auto& f : decode_frames(read_ivf(ivf.string()))) {
    const auto b = write_y4m_frame_bytes(f);
    expect.append(b.begin(), b.end());
  }
  CHECK(read_bytes(recon) == expect);
  // By content: animation tries the GOPs without the deblocking search, film the other way round.
  auto bytes_of = [&](const std::string& opts) {
    return run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --refs 7 --qindex 120" + opts, err) == 0 ? read_bytes(ivf) : std::string("failed");
  };
  CHECK(bytes_of(" --preset max --tune animation") == bytes_of(" --tune animation --gop-still 0.5 --gop-trial 1 --rd-intra 1"));
  CHECK(bytes_of(" --preset max --tune film") == bytes_of(" --tune film --deblock-search 1 --rd-intra 1"));
  // The video ends in a still mini-GOP of 9 (8 + the lone frame) or 13 frames: tried both ways too,
  // and the output decodes to its --recon and is one of the two fixed choices.
  for (int frames : {10, 14}) {
    {
      std::ofstream out(y4m, std::ios::binary);
      out << "YUV4MPEG2 W64 H48 F30:1 Ip C420jpeg\n";
      for (int i = 0; i < frames; i++) {
        const auto bytes = write_y4m_frame_bytes(scene_frame(64, 48, 0, 0));
        out << "FRAME\n";
        out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
      }
    }
    CHECK_EQ(run_cli(io + " --recon \"" + recon.string() + "\"", err), 0);
    const auto frames_out = decode_frames(read_ivf(ivf.string()));
    CHECK_EQ(frames_out.size(), static_cast<size_t>(frames));
    expect.clear();
    for (const auto& f : frames_out) {
      const auto b = write_y4m_frame_bytes(f);
      expect.append(b.begin(), b.end());
    }
    CHECK(read_bytes(recon) == expect);
    const std::string t = read_bytes(ivf);
    CHECK(t == bytes_of(" --preset max --gop-trial 0") || t == bytes_of(" --preset max --bframes 7"));
  }
  return 0;
}

// --pass 1, then --pass 2 --target-size B: the file fits B and decodes to its --recon (with --preset
// max too); a budget nothing fits exits 3; bad combinations and statistics of another video are
// refused, with the reason.
static int cli_target_size() {
  const fs::path dir = fs::temp_directory_path();
  const fs::path y4m = dir / "celv2_cli_ts.y4m", other = dir / "celv2_cli_ts_other.y4m", ivf = dir / "celv2_cli_ts.ivf";
  const fs::path stats = dir / "celv2_cli_ts_stats.txt", recon = dir / "celv2_cli_ts_recon.yuv";
  auto write_clip = [](const fs::path& p, int w, int h, int frames) {  // 10 still frames, then moving on from there
    std::ofstream out(p, std::ios::binary);
    out << "YUV4MPEG2 W" << w << " H" << h << " F30:1 Ip C420jpeg\n";
    for (int i = 0; i < frames; i++) {
      const auto bytes = write_y4m_frame_bytes(scene_frame(w, h, i < 10 ? 0 : 2 * (i - 10), i < 10 ? 0 : i - 10));
      out << "FRAME\n";
      out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
  };
  auto decodes_to_recon = [&] {
    std::string expect;
    for (const auto& f : decode_frames(read_ivf(ivf.string()))) {
      const auto b = write_y4m_frame_bytes(f);
      expect.append(b.begin(), b.end());
    }
    return read_bytes(recon) == expect;
  };
  auto io = [&](const fs::path& in) { return "-i \"" + in.string() + "\" -o \"" + ivf.string() + "\" --refs 7 --stats \"" + stats.string() + "\""; };
  std::string err;
  write_clip(y4m, 256, 144, 40);  // smaller frames are mostly headers, which no qindex shrinks
  for (const std::string opts : {"", " --preset max --tune animation"}) {
    CHECK_EQ(run_cli(io(y4m) + opts + " --pass 1 --qindex 120", err), 0);
    const celv2::PassStats st = celv2::read_stats(stats.string());
    // the quick look: the GOP trial, the deblocking search and the RD pass off unless given
    const std::string quick = read_bytes(stats);
    CHECK_EQ(run_cli(io(y4m) + opts + " --pass 1 --qindex 120 --gop-trial 0 --deblock-search 0 --rd-pass 0", err), 0);
    CHECK(read_bytes(stats) == quick);
    CHECK_EQ(st.bytes.size(), 40u);
    CHECK_EQ(st.width, 256);
    CHECK_EQ(st.height, 144);
    const uint64_t budget = fs::file_size(ivf) * 9 / 10;  // its keyframe is half the file and shrinks slowly with qindex
    CHECK_EQ(run_cli(io(y4m) + opts + " --pass 2 --target-size " + std::to_string(budget) + " --recon \"" + recon.string() + "\"", err), 0);
    CHECK(err.find("target-size: ") != std::string::npos);
    CHECK(fs::file_size(ivf) <= budget);
    CHECK(fs::file_size(ivf) >= budget * 7 / 10);  // a small clip with few mini-GOPs: not the 97% aim
    CHECK(decodes_to_recon());
  }
  CHECK_EQ(run_cli(io(y4m) + " --pass 2 --target-size 200", err), 3);  // not even at qindex 255
  CHECK(err.find("over --target-size") != std::string::npos);
  CHECK(err.find("even at qindex 255") != std::string::npos);
  // Statistics of another video: other sizes, one frame less, one frame more.
  write_clip(other, 128, 144, 40);
  CHECK(run_cli(io(other) + " --pass 2 --target-size 100000", err) != 0);
  CHECK(err.find("--stats") != std::string::npos);
  write_clip(other, 256, 144, 39);
  CHECK(run_cli(io(other) + " --pass 2 --target-size 100000", err) != 0);
  CHECK(err.find("frames") != std::string::npos);
  write_clip(other, 256, 144, 41);
  CHECK(run_cli(io(other) + " --pass 2 --target-size 100000", err) != 0);
  CHECK(err.find("frames") != std::string::npos);
  // Keyframes only (no mini-GOP, no update before the end): still fits.
  write_clip(other, 256, 144, 3);
  CHECK_EQ(run_cli(io(other) + " --keyint 1 --pass 1", err), 0);
  const uint64_t small = fs::file_size(ivf) * 9 / 10;
  CHECK_EQ(run_cli(io(other) + " --keyint 1 --pass 2 --target-size " + std::to_string(small), err), 0);
  CHECK(fs::file_size(ivf) <= small);
  // Refused, with the reason.
  const std::string in = "-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\"";
  CHECK(run_cli(in + " --pass 2 --stats \"" + stats.string() + "\"", err) != 0);
  CHECK(err.find("--target-size") != std::string::npos);
  CHECK(run_cli(in + " --pass 2 --target-size 1000", err) != 0);
  CHECK(err.find("--stats") != std::string::npos);
  CHECK(run_cli(io(y4m) + " --pass 2 --target-size 0", err) != 0);
  CHECK(err.find("--target-size") != std::string::npos);
  CHECK(run_cli(io(y4m) + " --pass 3", err) != 0);
  CHECK(err.find("--pass") != std::string::npos);
  CHECK(run_cli(in + " --pass 2 --target-size 1000 --stats \"" + (dir / "celv2_no_such_stats.txt").string() + "\"", err) != 0);
  CHECK(err.find("--stats") != std::string::npos);
  return 0;
}

// --workers: scene chunks coded at once give the bytes of one worker (the default, the anime max
// settings, one scene only, a scene longer than --chunk-memory, --frames ending inside a scene);
// bad values are refused with the reason.
static int cli_workers() {
  const fs::path dir = fs::temp_directory_path();
  const fs::path y4m = dir / "celv2_cli_wk.y4m", ivf = dir / "celv2_cli_wk.ivf", recon = dir / "celv2_cli_wk_recon.yuv";
  // scenes of `scene` frames each, a new pattern (seed) per scene
  auto write_clip = [&](int w, int h, int frames, int scene) {
    std::ofstream out(y4m, std::ios::binary);
    out << "YUV4MPEG2 W" << w << " H" << h << " F30:1 Ip C420jpeg\n";
    for (int i = 0; i < frames; i++) {
      const auto bytes = write_y4m_frame_bytes(scene_frame(w, h, i % scene, (i % scene) / 2, 1 + static_cast<uint32_t>(i / scene)));
      out << "FRAME\n";
      out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
  };
  std::string err;
  auto bytes_of = [&](const std::string& opts) {
    return run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --qindex 120" + opts, err) == 0 ? read_bytes(ivf) : std::string("failed");
  };
  write_clip(128, 64, 200, 50);
  for (const std::string opts : {std::string(" --refs 7"), std::string(" --refs 7 --preset max --tune animation")}) {
    const std::string one = bytes_of(opts);
    CHECK(one != "failed");
    CHECK(bytes_of(opts + " --workers 2") == one);
    CHECK(err.find("celv2: 4 chunks on 2 workers") != std::string::npos);
    CHECK(bytes_of(opts + " --workers 4") == one);
  }
  // decodes to its --recon with --workers 4
  CHECK_EQ(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --qindex 120 --workers 4 --recon \"" + recon.string() + "\"", err), 0);
  std::string expect;
  for (const auto& f : decode_frames(read_ivf(ivf.string()))) {
    const auto b = write_y4m_frame_bytes(f);
    expect.append(b.begin(), b.end());
  }
  CHECK(read_bytes(recon) == expect);
  CHECK(bytes_of(" --frames 120 --workers 2") == bytes_of(" --frames 120"));  // ends inside the third scene
  write_clip(128, 64, 60, 1000);  // one scene
  CHECK(bytes_of(" --workers 4") == bytes_of(""));
  write_clip(512, 288, 120, 1000);  // one scene of 120 frames of ~660 KB (8- and 16-bit planes): over 64 MB
  CHECK(bytes_of(" --workers 2 --chunk-memory 64") == bytes_of(""));
  CHECK(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --workers 0", err) != 0);
  CHECK(err.find("--workers") != std::string::npos);
  CHECK(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --workers 17", err) != 0);
  CHECK(err.find("--workers") != std::string::npos);
  CHECK(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --chunk-memory 10", err) != 0);
  CHECK(err.find("--chunk-memory") != std::string::npos);
  return 0;
}

// --target-vmaf with CELV2_TEST_VMAF (VMAF 100 - 3 x 2^(q / 40), +10 on odd chunks, in place of ffmpeg):
// each scene takes the highest probed qindex that reaches 90 (69 on even chunks, 109 on odd ones), the stream
// decodes equal to its --recon, and two workers give the same bytes; bad uses refused.
static int cli_target_vmaf() {
  const fs::path dir = fs::temp_directory_path();
  const fs::path y4m = dir / "celv2_cli_vmaf.y4m", ivf = dir / "celv2_cli_vmaf.ivf", recon = dir / "celv2_cli_vmaf_recon.yuv";
  {
    std::ofstream out(y4m, std::ios::binary);
    out << "YUV4MPEG2 W256 H144 F30:1 Ip C420jpeg\n";
    for (int i = 0; i < 200; i++) {  // 4 scenes of 50 frames
      const auto bytes = write_y4m_frame_bytes(scene_frame(256, 144, i % 50, (i % 50) / 2, 1 + static_cast<uint32_t>(i / 50)));
      out << "FRAME\n";
      out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
  }
  const std::string io = "-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --refs 7 --target-vmaf 90";
  std::string err;
  _putenv_s("CELV2_TEST_VMAF", "3,40,10");
  const int rc = run_cli(io + " --workers 1 --recon \"" + recon.string() + "\"", err);
  const std::string one_err = err, one = read_bytes(ivf);
  const int rc2 = run_cli(io + " --workers 2", err);
  const std::string two = read_bytes(ivf);
  _putenv_s("CELV2_TEST_VMAF", "");
  CHECK_EQ(rc, 0);
  CHECK_EQ(rc2, 0);
  CHECK(one == two);
  CHECK(one_err.find("scene 0 (frames 0-49): qindex 69 ") != std::string::npos);
  CHECK(one_err.find("scene 1 (frames 50-99): qindex 109 ") != std::string::npos);
  CHECK(one_err.find("scene 2 (frames 100-149): qindex 69 ") != std::string::npos);
  std::string expect;
  for (const auto& f : decode_frames(read_ivf(ivf.string()))) {
    const auto b = write_y4m_frame_bytes(f);
    expect.append(b.begin(), b.end());
  }
  CHECK(read_bytes(recon) == expect);
  CHECK(run_cli(io + " --pass 1 --stats \"" + (dir / "celv2_cli_vmaf.txt").string() + "\"", err) != 0);
  CHECK(err.find("--target-vmaf") != std::string::npos);
  CHECK(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --target-vmaf 101", err) != 0);
  CHECK(run_cli(io + " --ffmpeg \"" + (dir / "no_such_ffmpeg.exe").string() + "\"", err) != 0);
  CHECK(err.find("libvmaf") != std::string::npos);
  return 0;
}

// --target-vmaf with little read-ahead (128x72: --memory-limit 856 about 100 frames, 855 about 27): whole chunks hand their
// frames back, so three scenes pass 100 frames of budget; a scene over half of it, or over all of it, probes what is there.
// A scene no qindex lifts to the target says so.
static int cli_target_vmaf_memory() {
  const fs::path dir = fs::temp_directory_path();
  const fs::path y4m = dir / "celv2_cli_vmafm.y4m", ivf = dir / "celv2_cli_vmafm.ivf";
  auto clip = [&](int frames, int scene) {
    std::ofstream out(y4m, std::ios::binary);
    out << "YUV4MPEG2 W128 H72 F30:1 Ip C420jpeg\n";
    for (int i = 0; i < frames; i++) {
      const auto bytes = write_y4m_frame_bytes(scene_frame(128, 72, i % scene, (i % scene) / 2, 1 + static_cast<uint32_t>(i / scene)));
      out << "FRAME\n";
      out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
  };
  const std::string io = "-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --target-vmaf 90 --device cpu --workers 1 --parallel-b 0 --gop-trial 0";
  std::string err;
  _putenv_s("CELV2_TEST_VMAF", "3,40");
  clip(150, 50);
  const int three = run_cli(io + " --memory-limit 856", err);
  const std::string three_err = err;
  const size_t three_frames = decode_frames(read_ivf(ivf.string())).size();
  clip(60, 60);
  const int over_half = run_cli(io + " --memory-limit 856", err);
  const size_t over_half_frames = decode_frames(read_ivf(ivf.string())).size();
  const int over_all = run_cli(io + " --memory-limit 855", err);
  const size_t over_all_frames = decode_frames(read_ivf(ivf.string())).size();
  _putenv_s("CELV2_TEST_VMAF", "1000,1000");
  const int short_of = run_cli(io, err);
  _putenv_s("CELV2_TEST_VMAF", "");
  CHECK_EQ(three, 0);
  CHECK(three_err.find("scene 2 ") != std::string::npos);
  CHECK_EQ(three_frames, 150u);
  CHECK_EQ(over_half, 0);
  CHECK_EQ(over_half_frames, 60u);
  CHECK_EQ(over_all, 0);
  CHECK_EQ(over_all_frames, 60u);
  CHECK_EQ(short_of, 0);
  CHECK(err.find("target VMAF 90.00 not reached") != std::string::npos);
  return 0;
}

// --target-size with --workers 2: within the budget, decodes to its --recon, the same bytes twice;
// aiming over the budget (CELV2_TEST_AIM) the last chunk is re-coded and the file still fits; a
// budget nothing fits exits 3; --frames ending inside a scene.
static int cli_target_size_workers() {
  const fs::path dir = fs::temp_directory_path();
  const fs::path y4m = dir / "celv2_cli_tsw.y4m", ivf = dir / "celv2_cli_tsw.ivf", stats = dir / "celv2_cli_tsw_stats.txt";
  const fs::path recon = dir / "celv2_cli_tsw_recon.yuv";
  {
    std::ofstream out(y4m, std::ios::binary);
    out << "YUV4MPEG2 W256 H144 F30:1 Ip C420jpeg\n";
    for (int i = 0; i < 200; i++) {  // 4 scenes of 50 frames
      const auto bytes = write_y4m_frame_bytes(scene_frame(256, 144, i % 50, (i % 50) / 2, 1 + static_cast<uint32_t>(i / 50)));
      out << "FRAME\n";
      out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
  }
  const std::string io = "-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --refs 7 --stats \"" + stats.string() + "\"";
  std::string err;
  CHECK_EQ(run_cli(io + " --pass 1 --qindex 120", err), 0);
  const uint64_t budget = fs::file_size(ivf) * 9 / 10;
  const std::string two = io + " --workers 2 --pass 2 --target-size " + std::to_string(budget);
  CHECK_EQ(run_cli(two + " --recon \"" + recon.string() + "\"", err), 0);
  CHECK(err.find("celv2: 4 chunks on 2 workers") != std::string::npos);
  CHECK(fs::file_size(ivf) <= budget);
  std::string expect;
  for (const auto& f : decode_frames(read_ivf(ivf.string()))) {
    const auto b = write_y4m_frame_bytes(f);
    expect.append(b.begin(), b.end());
  }
  CHECK(read_bytes(recon) == expect);
  const std::string once = read_bytes(ivf);
  CHECK_EQ(run_cli(two, err), 0);
  CHECK(read_bytes(ivf) == once);  // deterministic
  _putenv_s("CELV2_TEST_AIM", "104");
  const int aimed_rc = run_cli(two, err);
  _putenv_s("CELV2_TEST_AIM", "");
  CHECK_EQ(aimed_rc, 0);
  CHECK(err.find("re-coded the last chunk") != std::string::npos);
  CHECK(fs::file_size(ivf) <= budget);
  // --workers auto with a target size: one worker (the control measured closest to the target)
  CHECK_EQ(run_cli(io + " --workers auto --pass 2 --target-size " + std::to_string(budget), err), 0);
  CHECK(err.find("chunks on") == std::string::npos);
  CHECK_EQ(run_cli(io + " --workers 2 --pass 2 --target-size 200", err), 3);
  CHECK(err.find("over --target-size") != std::string::npos);
  CHECK_EQ(run_cli(io + " --frames 150 --pass 1 --qindex 120", err), 0);
  CHECK_EQ(run_cli(io + " --frames 150 --workers 2 --pass 2 --target-size " + std::to_string(fs::file_size(ivf) * 9 / 10), err), 0);
  return 0;
}

// --workers auto, and the memory cap (--memory-limit, default 80% of the installed RAM; the RAM
// simulated with CELV2_TEST_TOTAL_RAM_MB): first fewer workers, then no parallel B frames, each with a
// warning and the same bytes; bad values refused.
static int cli_memory_limit() {
  const fs::path dir = fs::temp_directory_path();
  const fs::path y4m = dir / "celv2_cli_mem.y4m", ivf = dir / "celv2_cli_mem.ivf";
  {
    std::ofstream out(y4m, std::ios::binary);
    out << "YUV4MPEG2 W128 H64 F30:1 Ip C420jpeg\n";
    for (int i = 0; i < 200; i++) {
      const auto bytes = write_y4m_frame_bytes(scene_frame(128, 64, i % 50, (i % 50) / 2, 1 + static_cast<uint32_t>(i / 50)));
      out << "FRAME\n";
      out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
  }
  std::string err;
  auto bytes_of = [&](const std::string& opts) {
    return run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --refs 7 --qindex 120" + opts, err) == 0 ? read_bytes(ivf) : std::string("failed");
  };
  const std::string one = bytes_of(" --workers 1");
  CHECK(one != "failed");
  CHECK(bytes_of(" --workers auto") == one);
  _putenv_s("CELV2_TEST_TOTAL_RAM_MB", "1800");  // 1440 MB: one worker fits, two do not
  const std::string fewer = bytes_of(" --workers 4");
  const std::string fewer_err = err;
  _putenv_s("CELV2_TEST_TOTAL_RAM_MB", "1400");  // 1120 MB: one worker only without parallel B frames
  const std::string no_pb = bytes_of("");
  const std::string no_pb_err = err;
  _putenv_s("CELV2_TEST_TOTAL_RAM_MB", "");
  CHECK(fewer == one);
  CHECK(fewer_err.find("--workers 4 -> 1") != std::string::npos && fewer_err.find("--memory-limit") != std::string::npos);
  CHECK(no_pb == bytes_of(" --parallel-b 0"));
  CHECK(no_pb_err.find("--parallel-b off") != std::string::npos);
  CHECK(bytes_of(" --memory-limit 100000") == one);
  CHECK(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --memory-limit 0", err) != 0);
  CHECK(err.find("--memory-limit") != std::string::npos);
  CHECK(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --memory-limit x", err) != 0);
  CHECK(err.find("--memory-limit") != std::string::npos);
  return 0;
}

// 32x32 has one superblock, so 2^6 tile columns cannot be honoured: still encodes, with a warning.
static int cli_warns_on_clamped_tiles() {
  const fs::path y4m = fs::temp_directory_path() / "celv2_cli_in.y4m";
  const fs::path ivf = fs::temp_directory_path() / "celv2_cli_tiles.ivf";
  std::string err;
  CHECK_EQ(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --frames 1 --tile-columns 6", err), 0);
  CHECK(err.find("warning: tile log2 columns/rows 6/0 adjusted to 0/0") != std::string::npos);
  CHECK_EQ(decode_frame_md5s(read_ivf(ivf.string())).size(), 1u);
  return 0;
}

// Tiles coded on parallel threads while the GPU publishes rows: the same bytes in every process.
static int cli_tiles_deterministic() {
  const fs::path y4m = fs::temp_directory_path() / "celv2_cli_tiles_in.y4m";
  {
    std::ofstream out(y4m, std::ios::binary);
    out << "YUV4MPEG2 W256 H128 F30:1 Ip C420jpeg\n";
    for (int i = 0; i < 3; i++) {
      const auto bytes = write_y4m_frame_bytes(scene_frame(256, 128, 3 * i, 2 * i));
      out << "FRAME\n";
      out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
  }
  std::string first;
  for (int run = 0; run < 5; run++) {
    const fs::path ivf =
        fs::temp_directory_path() / ("celv2_cli_det" + std::to_string(run) + ".ivf");
    std::string err;
    CHECK_EQ(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() +
                         "\" --tile-columns 1 --tile-rows 1",
                     err),
             0);
    std::ifstream in(ivf, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    CHECK(!bytes.empty());
    if (run == 0) first = bytes;
    CHECK(bytes == first);
  }
  return 0;
}

static int cli_device_cpu_matches_gpu() {
  const fs::path y4m = device_input();
  const fs::path cpu_ivf = fs::temp_directory_path() / "celv2_cli_dev_cpu.ivf";
  std::string err;
  CHECK_EQ(run_cli("-i \"" + y4m.string() + "\" -o \"" + cpu_ivf.string() + "\" --device cpu", err), 0);
  CHECK(err.find("decisions on CPU") != std::string::npos);
  CHECK_EQ(run_cli("-i \"" + y4m.string() + "\" -o \"" + kDevGpuIvf.string() + "\" --device gpu", err), 0);
  CHECK(err.find("decisions on Vulkan") != std::string::npos);
  const std::string cpu = read_bytes(cpu_ivf);
  CHECK(!cpu.empty());
  CHECK(cpu == read_bytes(kDevGpuIvf));
  return 0;
}

// --device cuda: the CUDA engine, kept for one release to compare with; the same bytes (no CUDA GPU: skipped).
static int cli_device_cuda_matches_gpu() {
  const fs::path ivf = fs::temp_directory_path() / "celv2_cli_dev_cuda.ivf";
  std::string err;
  if (run_cli("-i \"" + device_input().string() + "\" -o \"" + ivf.string() + "\" --device cuda", err) != 0) {
    CHECK(err.find("no usable CUDA GPU") != std::string::npos);
    return 0;
  }
  CHECK(err.find("decisions on CUDA") != std::string::npos);
  CHECK(read_bytes(ivf) == read_bytes(kDevGpuIvf));  // written by cli_device_cpu_matches_gpu
  return 0;
}

// CELV2_VK_PHASES=A / B (the Vulkan engine's other layouts of the quarter-pel phases): the same
// bytes as every phase kept, on each Vulkan device (none: skipped).
static int cli_vk_phase_layouts() {
  const fs::path y4m = device_input();
  const fs::path ivf = fs::temp_directory_path() / "celv2_cli_vk_phases.ivf";
  for (const char* dev : {"vulkan", "vulkan2"}) {
    std::string err;
    if (run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --refs 4 --device " + dev, err) != 0) continue;
    const std::string all = read_bytes(ivf);
    for (const char* layout : {"A", "B"}) {
      EnvGuard g("CELV2_VK_PHASES", layout);
      CHECK_EQ(run_cli("-i \"" + y4m.string() + "\" -o \"" + ivf.string() + "\" --refs 4 --device " + dev, err), 0);
      CHECK(read_bytes(ivf) == all);
    }
  }
  return 0;
}

static int cli_device_bogus_rejected() {
  const fs::path ivf = fs::temp_directory_path() / "celv2_cli_bad.ivf";
  std::string err;
  CHECK(run_cli("-i \"" + device_input().string() + "\" -o \"" + ivf.string() + "\" --device bogus", err) != 0);
  CHECK(!err.empty());
  return 0;
}

// With no Vulkan GPU, auto takes the CPU and still gives the GPU's bytes.
static int cli_auto_falls_back_without_gpu() {
  const fs::path ivf = fs::temp_directory_path() / "celv2_cli_dev_auto.ivf";
  EnvGuard no_gpu("CELV2_NO_VULKAN", "1");
  std::string err;
  CHECK_EQ(run_cli("-i \"" + device_input().string() + "\" -o \"" + ivf.string() + "\" --device auto", err), 0);
  CHECK(err.find("decisions on CPU") != std::string::npos);
  const std::string bytes = read_bytes(ivf);
  CHECK(!bytes.empty());
  CHECK(bytes == read_bytes(kDevGpuIvf));  // written by cli_device_cpu_matches_gpu
  return 0;
}

static int cli_device_gpu_without_gpu_fails() {
  const fs::path ivf = fs::temp_directory_path() / "celv2_cli_dev_nogpu.ivf";
  fs::remove(ivf);  // a stale file would hide a missing "no output" guarantee
  EnvGuard no_gpu("CELV2_NO_VULKAN", "1");
  std::string err;
  CHECK_EQ(run_cli("-i \"" + device_input().string() + "\" -o \"" + ivf.string() + "\" --device gpu", err), 1);
  CHECK(err.find("no usable GPU") != std::string::npos);  // the reason is shown, and no CPU fallback
  CHECK(!fs::exists(ivf));
  return 0;
}

// Runs `type <y4m> | celv2enc -i - <args>` through cmd.exe (whose pipes pass bytes unchanged).
static int run_cli_stdin(const fs::path& y4m, const std::string& args, std::string& err) {
  const fs::path errfile = fs::temp_directory_path() / "celv2_cli_stderr.txt";
  const std::string cmd = "\"type \"" + y4m.string() + "\" | \"" CELV2ENC_PATH "\" -i - " + args +
                          " 2> \"" + errfile.string() + "\"\"";
  const int rc = std::system(cmd.c_str());
  std::ifstream in(errfile);
  err.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  return rc;
}

// The same y4m piped through stdin and read from a file gives the same stream, also with
// --frames stopping before the end of the pipe.
static int cli_reads_stdin() {
  const fs::path y4m = fs::temp_directory_path() / "celv2_cli_stdin_in.y4m";
  {
    std::ofstream out(y4m, std::ios::binary);
    out << "YUV4MPEG2 W136 H72 F30:1 Ip C420jpeg\n";
    for (int i = 0; i < 4; i++) {
      const auto bytes = write_y4m_frame_bytes(scene_frame(136, 72, 3 * i, 2 * i));
      out << "FRAME\n";
      out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
  }
  const fs::path from_file = fs::temp_directory_path() / "celv2_cli_stdin_file.ivf";
  const fs::path from_pipe = fs::temp_directory_path() / "celv2_cli_stdin_pipe.ivf";
  for (const char* frames : {"", " --frames 2"}) {
    fs::remove(from_file);
    fs::remove(from_pipe);
    std::string err;
    CHECK_EQ(run_cli("-i \"" + y4m.string() + "\" -o \"" + from_file.string() + "\"" + frames, err), 0);
    CHECK_EQ(run_cli_stdin(y4m, "-o \"" + from_pipe.string() + "\"" + frames, err), 0);
    const std::string a = read_bytes(from_file), b = read_bytes(from_pipe);
    CHECK(!a.empty());
    CHECK(a == b);
    CHECK_EQ(read_ivf(from_pipe.string()).size(), frames[0] ? 2u : 4u);
  }
  return 0;
}

// --progress prints "progress frame=N type=T qindex=Q bytes=B" once per coded frame, in order: the
// keyframe, then the mini-GOP's B frame and its anchor, each with its qindex and its temporal unit's
// bytes as written; without it nothing.
static int cli_reports_progress() {
  const fs::path ivf = fs::temp_directory_path() / "celv2_cli_progress.ivf";
  std::string err;
  CHECK_EQ(run_cli("-i \"" + device_input().string() + "\" -o \"" + ivf.string() + "\" --progress", err), 0);
  const auto tus = read_ivf(ivf.string());
  CHECK_EQ(tus.size(), 3u);
  const char types[] = {'K', 'B', 'A'};
  size_t at = 0, sum = 0, tu_sum = 0, own[3] = {0, 0, 0};
  for (int n = 1; n <= 3; n++) {
    const std::string line = "progress frame=" + std::to_string(n) + " ";
    const size_t p = err.find(line, at);
    CHECK(p != std::string::npos);
    CHECK(err.find(line, p + 1) == std::string::npos);
    char type = 0;
    int qindex = 0;
    size_t bytes = 0;
    CHECK_EQ(std::sscanf(err.c_str() + p + line.size(), "type=%c qindex=%d bytes=%zu", &type, &qindex, &bytes), 3);
    CHECK_EQ(type, types[n - 1]);
    CHECK(qindex >= 1 && qindex <= 255);
    own[n - 1] = bytes, sum += bytes, tu_sum += tus[n - 1].size();
    at = p;
  }
  // Each frame's own bytes: the hidden anchor counts at its showing, not in the B frame's unit it
  // rides in, and the rows still add up to the stream.
  CHECK_EQ(own[0], tus[0].size());
  CHECK(own[1] < tus[1].size());
  CHECK(own[2] > tus[2].size());
  CHECK_EQ(sum, tu_sum);
  CHECK(err.find("progress frame=4") == std::string::npos);
  CHECK_EQ(run_cli("-i \"" + device_input().string() + "\" -o \"" + ivf.string() + "\"", err), 0);
  CHECK(err.find("progress frame=") == std::string::npos);
  return 0;
}

int main() {
  int fails = 0;
  fails += RUN(cli_reports_progress);
  fails += RUN(cli_reads_stdin);
  fails += RUN(cli_encodes_and_decodes);
  fails += RUN(cli_bframes);
  fails += RUN(cli_tunes);
  fails += RUN(cli_tplq_default);
  fails += RUN(cli_anime_tune_tools);
  fails += RUN(cli_cfl);
  fails += RUN(cli_cctx);
  fails += RUN(cli_mhccp);
  fails += RUN(cli_lr_uv);
  fails += RUN(cli_interintra);
  fails += RUN(cli_intra_txp);
  fails += RUN(cli_preset_max);
  fails += RUN(cli_target_size);
  fails += RUN(cli_workers);
  fails += RUN(cli_target_size_workers);
  fails += RUN(cli_target_vmaf);
  fails += RUN(cli_target_vmaf_memory);
  fails += RUN(cli_memory_limit);
  fails += RUN(cli_encodes_10bit);
  fails += RUN(cli_hdr_signalling);
  fails += RUN(cli_warns_on_clamped_tiles);
  fails += RUN(cli_lists_devices);
  fails += RUN(cli_rejects_bad_args);
  fails += RUN(cli_tiles_deterministic);
  fails += RUN(cli_device_cpu_matches_gpu);
  fails += RUN(cli_device_cuda_matches_gpu);
  fails += RUN(cli_device_bogus_rejected);
  fails += RUN(cli_auto_falls_back_without_gpu);
  fails += RUN(cli_device_gpu_without_gpu_fails);
  fails += RUN(cli_grain);
  fails += RUN(cli_paq_args);
  fails += RUN(cli_paq);
  fails += RUN(cli_profile);
  fails += RUN(cli_vk_phase_layouts);
  fails += RUN(cli_gop_log);
  fails += RUN(cli_gop_change);
  fails += RUN(cli_cfl_rd);
  fails += RUN(cli_psy_rd);
  fails += RUN(cli_screen);
  fails += RUN(cli_mhccp_past_the_edge);
  fails += RUN(cli_wide_real_cost);
  fails += RUN(cli_wide_switches);
  fails += RUN(cli_rd_zero);
  fails += RUN(cli_frame_list);
#ifdef CELV2_ORACLE
  fails += RUN(cli_oracle_replay);
#endif
  fails += RUN(cli_warp);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
