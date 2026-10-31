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
    CHECK(v == "gpu" || v == "vulkan" || v == "vulkan2" || v == "cpu");
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
                         " --recon \"" + recon.string() + "\"",
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
  CHECK(bytes_of(" --preset max --tune animation") == bytes_of(" --tune animation --gop-still 0.5 --gop-trial 1"));
  CHECK(bytes_of(" --preset max --tune film") == bytes_of(" --tune film --deblock-search 1"));
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
  _putenv_s("CELV2_TEST_TOTAL_RAM_MB", "2600");  // 2080 MB: one worker fits, two do not
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
  CHECK(err.find("decisions on GPU") != std::string::npos);
  const std::string cpu = read_bytes(cpu_ivf);
  CHECK(!cpu.empty());
  CHECK(cpu == read_bytes(kDevGpuIvf));
  return 0;
}

static int cli_device_bogus_rejected() {
  const fs::path ivf = fs::temp_directory_path() / "celv2_cli_bad.ivf";
  std::string err;
  CHECK(run_cli("-i \"" + device_input().string() + "\" -o \"" + ivf.string() + "\" --device bogus", err) != 0);
  CHECK(!err.empty());
  return 0;
}

// With no visible CUDA device, auto falls back to Vulkan (where there is one) or the CPU and still
// gives the GPU's bytes.
static int cli_auto_falls_back_without_gpu() {
  const fs::path ivf = fs::temp_directory_path() / "celv2_cli_dev_auto.ivf";
  EnvGuard no_gpu("CUDA_VISIBLE_DEVICES", "-1");
  std::string err;
  CHECK_EQ(run_cli("-i \"" + device_input().string() + "\" -o \"" + ivf.string() + "\" --device auto", err), 0);
  CHECK(err.find("decisions on CPU") != std::string::npos || err.find("decisions on Vulkan") != std::string::npos);
  const std::string bytes = read_bytes(ivf);
  CHECK(!bytes.empty());
  CHECK(bytes == read_bytes(kDevGpuIvf));  // written by cli_device_cpu_matches_gpu
  return 0;
}

static int cli_device_gpu_without_gpu_fails() {
  const fs::path ivf = fs::temp_directory_path() / "celv2_cli_dev_nogpu.ivf";
  fs::remove(ivf);  // a stale file would hide a missing "no output" guarantee
  EnvGuard no_gpu("CUDA_VISIBLE_DEVICES", "-1");
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

// --progress prints "progress frame=N" once per coded frame, in order; without it nothing.
static int cli_reports_progress() {
  const fs::path ivf = fs::temp_directory_path() / "celv2_cli_progress.ivf";
  std::string err;
  CHECK_EQ(run_cli("-i \"" + device_input().string() + "\" -o \"" + ivf.string() + "\" --progress", err), 0);
  size_t at = 0;
  for (int n = 1; n <= 3; n++) {
    const std::string line = "progress frame=" + std::to_string(n);
    const size_t p = err.find(line, at);
    CHECK(p != std::string::npos);
    CHECK(err.find(line, p + 1) == std::string::npos);
    at = p;
  }
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
  fails += RUN(cli_preset_max);
  fails += RUN(cli_target_size);
  fails += RUN(cli_workers);
  fails += RUN(cli_target_size_workers);
  fails += RUN(cli_memory_limit);
  fails += RUN(cli_encodes_10bit);
  fails += RUN(cli_hdr_signalling);
  fails += RUN(cli_warns_on_clamped_tiles);
  fails += RUN(cli_lists_devices);
  fails += RUN(cli_rejects_bad_args);
  fails += RUN(cli_tiles_deterministic);
  fails += RUN(cli_device_cpu_matches_gpu);
  fails += RUN(cli_device_bogus_rejected);
  fails += RUN(cli_auto_falls_back_without_gpu);
  fails += RUN(cli_device_gpu_without_gpu_fails);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
