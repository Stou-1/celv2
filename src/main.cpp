// celv2enc: y4m (8-bit 4:2:0) -> IVF with GPU- or CPU-decided partitions/modes/motion and CPU
// reconstruction.
// Usage: celv2enc -i in.y4m -o out.ivf [--qindex 120] [--frames N] [--keyint 0] [--recon r.yuv]
//                [--tile-columns L] [--tile-rows L] [--threads N] [--device auto|gpu|cpu|vulkan|vulkan2]
//                [--refs N] [--scenecut 0|1] [--bframes N] [--bq D] [--aq D] [--pyramid 0|1] [--progress] [--psnr]
//                [--tune film|animation|still]
//   -i - reads the y4m from standard input (e.g. ffmpeg ... -f yuv4mpegpipe - | celv2enc -i - ...).
//   --progress prints "progress frame=N" (N from 1) on stderr after each coded frame is written.
//   --psnr prints "psnr: P" at the end: the mean over the shown frames of their luma PSNR against
//   the source (at most 99 dB); then "psnr uv: U V", the same for chroma.
//   --keyint N: at most N frames from one keyframe to the next (0: no limit); --scenecut 1 (the
//   default) also makes the first frame of each new shot a keyframe (SceneCutDetector).
//   --recon writes the reconstruction (I420; 16-bit little-endian samples at 10 bits).
//   --colour 709|pq|hlg [--range limited|full]: the colour description sent with keyframes
//   (BT.709 SDR, BT.2100 PQ or HLG). --mdcv "G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min)" (chromaticity in
//   0.00002, luminance in 0.0001 cd/m2, as x265's --master-display) and --cll MaxCLL,MaxFALL (cd/m2)
//   send HDR10 static metadata with keyframes.
//   --bit-depth 8|10: the coded depth (default 10, 8-bit inputs too: their samples shifted left by 2).
//   8-bit sources coded at 10 bits against 8, 33 frames 1080p q100-190: anime A / C / B Y -20.1 /
//   -12.5 / -4.3%, live action Y -7.7% BD-rate, at the same speed. 10-bit y4m is C420p10.
//   --tile-columns/--tile-rows: log2 of the tile counts (default: halve the width until under 1280
//   and the height until under 2560, at most 16 tiles: 2 at 1080p, 4 at 4K, 16 at 8K and 16K); tiles are coded on --threads
//   CPU threads (default 0 = one per tile); never more threads than tiles or CPU cores. With
//   --device cpu, --threads limits only these tile threads; the decision threads take the other cores.
//   --device: where the decisions run (default auto = the CUDA GPU if usable, else the first Vulkan
//   GPU, else the CPU; gpu fails without a usable CUDA GPU; vulkan / vulkan2 take the first / second
//   discrete Vulkan GPU). All give the same output; the choice is printed once on stderr.
//   --list-devices: prints the usable devices, one "<--device value><TAB><name>" line each, and exits.
//   --refs: past frames an inter frame may predict from (1..7, default SeqConfig's).
//   --bframes N: up to N frames between reference frames are coded after the next one (which is
//   coded hidden and shown later), predicting from both sides (default 7 with --refs 5 or more and
//   --qindex 100 or more, else 3); 0 codes every frame in order.
//   --bq / --aq: qindex offsets of those B frames (default 16) and hidden frames (default -16).
//   --pyramid 0|1: of 3 or more B frames, code the middle one right after the hidden frame, as a
//   hidden reference the others also predict from, and so on in each half (default 1; 3 B frames
//   with it against 2 without: -8.5% BD-rate on anime A 1080p, -6.0% / -3.1% on anime clips at
//   qindex 45-180 / 30-75). The middle frame has qindex offset --mq, each level below it 8 more.
//   --mq D: qindex offset of that middle frame (default 0).
//   --deltaq 0|1: a qindex offset per superblock of the reference frames (keyframes, hidden ones)
//   from a temporal dependency model of their mini-GOP's source frames (core/tpl.h): finer where
//   later frames predict from them (default 0: 33 frames 1080p q100-190, live action Y -1.4%,
//   U / V -6%, but anime +0.1 to +2.3%; docs/design/tried-not-kept.md).
//   --tplq 0|1|2: the qindex of keyframes (1) or of keyframes and anchors (2, the mids then between
//   the anchor's and --bq's by level) from TPL instead of the offsets and still-share boosts:
//   quantizer step = the B frames' (--bq) * r0^0.7, r0 = the frame's own share of its weight in the
//   model of the mini-GOP after it (libaom's constant-q GOPs take r0^0.5). Default 0: 120 frames
//   1080p q100-190, 1: AM -3.5%, Dragon Maid -1.1%, AIR -0.5%, live action +0.7%, kaguya +0.9%;
//   2: +0.1% on average (the model sees one mini-GOP, not the anchors' chain after it).
//   --deblock-delta D: deblocking strength offset (-16..15, steps of 8 qindex; default 2).
//   --deblock-search 0|1: per frame, the offset in {-4, -2, 0, 2, 4} whose deblocked frame is nearest
//   the source (default 0; 1 with --preset max).
//   --cdef 0|1: CDEF after deblocking, one strength per frame (default 1).
//   --ccso 0|1: cross-component sample offset after CDEF (default 1).
//   --mrls 0|1: intra prediction from reference lines further out (default 1).
//   --amvd 0|1: adaptive MVD, NEWMV differences from a short set of values (default 1).
//   --tmv 0|1: temporal MVs, the references' motion projected into the MV candidates (default 0).
//   --flexmv 0|1: NEWMV blocks may code their MV at 1, 4 or 8 pel instead of 1/4 (default 1).
//   --kq D: qindex offset of keyframes (default 0).
//   --key-boost N / --anchor-boost N: a keyframe's / an anchor's qindex goes down by up to N as the
//   still share (to the frame up to 8 ahead / through its mini-GOP) goes from 1/2 to all (default 48 / 24).
//   --level-step N: the qindex step from one pyramid level to the next (default 8).
//   --level-step-16 N: the same in mini-GOPs over 8 frames (default: --level-step).
//   --gop-still T: with mini-GOPs over 8 frames, one takes 16 only where the still share to its last
//   frame reaches T, else 8 (0..1; default 0: always the full length).
//   --preset max: speed ignored for quality and size (docs/superpowers/specs/2026-10-31-celv2-preset-max-design.md):
//   16-frame mini-GOPs where --bframes is not given (as --tune), --gop-still 0.5, --gop-trial 1,
//   --deblock-search 1; with --tune animation no --deblock-search, with --tune film no --gop-trial
//   (nor its --gop-still).
//   --gop-trial 0|1: a mini-GOP over 8 frames is coded both as it is and as 8 + the rest, and the one
//   with the lower squared error + sse_per_bit x bits kept (default 0; 1 with --preset max).
//   --pass 1 --stats F: writes each IVF frame's bytes to F, with --gop-trial 0 --deblock-search 0
//   --rd-pass 0 unless given (docs/superpowers/specs/2026-10-31-target-size-design.md).
//   --pass 2 --stats F --target-size B: the same input and settings again, the IVF file at most B
//   bytes: one qindex for the whole video from F (--qindex is not used), moved by at most 2 a
//   mini-GOP as the bytes come in. Ends with "target-size: <bytes> of <B> bytes"; over B: exit 3.
//   --workers N: scene chunks (a keyframe to the next, 48 frames at least) coded at once on N
//   pipelines, written in order: the same bytes as 1 (the default). --chunk-memory MB: the frames
//   read ahead for them (default 4096; the reader waits beyond it). --workers auto: one per 16
//   hardware threads, 1..4 (provisional). --memory-limit MB: at most this (default 80% of the
//   installed RAM): fewer workers, then no parallel B frames (the same bytes), then no GOP trial
//   (unless given), each with a warning.
//   --tune film|animation|still: settings by content (x264's --tune); options given on the command
//   line override them. All take 16-frame mini-GOPs where --bframes is not given (with --refs 5 or
//   more and --qindex 100 or more). film: --gop-still 0.7 --deltaq 1. animation: --gop-still 0.85
//   --kq -8 --key-boost 64 --level-step-16 2 --deblock-delta 0. still: 16 frames throughout. 120 frames 1080p
//   q100-190 against no tune: animation on six anime sources -0.13 to -13.5% (AM -2.2%, kaguya
//   -13.5%, Dragon Maid -0.3%, AIR -0.8%, chika -0.8%, Angel Beats -0.1%); docs/design/2026-10-30-tunes.md.
//   --tcq 0|1: trellis coded quantization of luma (default 1).
//   --rd-pass 0|1: re-pick the decision engine's partitions and modes on the CPU by estimated
//   rate and distortion (default 1; -10 to -18% BD-rate on 1080p clips, 25% slower at 1080p).
//   --parallel-b auto|0|1: code a mini-GOP's B frames at once, all but the last then not referencing
//   each other (+1.4% / +3.2% BD-rate at qindex 45-180 / 30-75 on anime clips). auto (the default):
//   up to 2 x 4K pixels, where it is faster (4K: 8.2 -> 8.9 fps; 8K: no gain; 16K: 1.55 -> 1.44 fps).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <optional>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <functional>
#include <map>
#include <mutex>
#include <thread>
#include <deque>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bitstream/obu.h"
#include "core/scene_cut.h"
#include "chunk_coder.h"
#include "core/size_control.h"
#include "core/tpl.h"
#include "gpu/engine.h"
#include "io/ivf.h"
#include "io/memory.h"
#include "io/stats.h"
#include "io/y4m.h"
#include "pipeline.h"

namespace {

void put_be(std::vector<uint8_t>& out, uint32_t v, int bytes) {
  for (int i = bytes - 1; i >= 0; i--) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

// x265's --master-display string to the HDR MDCV metadata payload: primaries R, G, B then the white
// point, x / y in 0.16 fixed point, then max / min luminance in 24.8 / 18.14 (cd/m2).
std::vector<uint8_t> parse_mdcv(const std::string& s) {
  long long gx, gy, bx, by, rx, ry, wx, wy, lmax, lmin;
  if (std::sscanf(s.c_str(), "G(%lld,%lld)B(%lld,%lld)R(%lld,%lld)WP(%lld,%lld)L(%lld,%lld)", &gx, &gy, &bx, &by, &rx,
                  &ry, &wx, &wy, &lmax, &lmin) != 10)
    throw std::invalid_argument("--mdcv must look like G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min)");
  std::vector<uint8_t> out;
  for (long long c : {rx, ry, gx, gy, bx, by, wx, wy}) {
    if (c < 0 || c > 50000) throw std::invalid_argument("--mdcv chromaticity must be 0..50000 (units of 0.00002)");
    put_be(out, static_cast<uint32_t>(std::llround(c * 0.00002 * 65536)), 2);
  }
  if (lmin < 0 || lmax <= lmin || lmax > 100000000) throw std::invalid_argument("--mdcv luminance out of range");
  put_be(out, static_cast<uint32_t>(std::llround(lmax * 0.0001 * 256)), 4);
  put_be(out, static_cast<uint32_t>(std::llround(lmin * 0.0001 * 16384)), 4);
  return out;
}

// "MaxCLL,MaxFALL" (cd/m2) to the HDR CLL metadata payload.
std::vector<uint8_t> parse_cll(const std::string& s) {
  long long cll, fall;
  if (std::sscanf(s.c_str(), "%lld,%lld", &cll, &fall) != 2 || cll < 0 || fall < 0 || cll > 65535 || fall > 65535)
    throw std::invalid_argument("--cll must be MaxCLL,MaxFALL (0..65535)");
  std::vector<uint8_t> out;
  put_be(out, static_cast<uint32_t>(cll), 2);
  put_be(out, static_cast<uint32_t>(fall), 2);
  return out;
}

struct Args {
  std::string in, out, recon;
  int qindex = 120, frames = 0, keyint = 0;
  int tile_cols_log2 = -1, tile_rows_log2 = -1, threads = 0;  // -1: from the frame size
  int refs = 0;  // 0: SeqConfig's default
  // Coded depth: 10 by default, 8-bit inputs too (33 frames 1080p q100-190 against 8: anime A Y -20.1%,
  // anime C -12.5%, anime B -4.3%, live action -7.7% BD-rate, the same speed).
  int bit_depth = 10;
  int colour = 0;  // color_description_idc to send (0: none)
  bool full_range = false;
  std::vector<uint8_t> mdcv, cll;
  bool scenecut = true;
  // On anime clips 2 B frames at qindex offsets +16 (B) / -16 (hidden reference frames) gave
  // -13% BD-rate at qindex 45-180 and -6% at 30-75 (offsets of +-12% of the qindex were worse).
  int bframes = -1;  // -1: 7 with 5 or more references, else 3
  bool long_gop = false;  // --tune: bframes' default 15 instead of 7
  double gop_still = 0;   // a mini-GOP over 8 frames only where its still share reaches this
  int key_boost = 48, anchor_boost = 24;  // qindex down by up to these with the still share
  int level_step = 8;  // qindex step between pyramid levels
  int level_step16 = -1;  // the same in mini-GOPs over 8 frames (-1: level_step)
  std::string tune;
  std::string preset;  // "max": speed ignored (docs/superpowers/specs/2026-10-31-celv2-preset-max-design.md)
  bool gop_trial = false;  // a mini-GOP over 8 frames coded both as it is and as 8 + the rest; the better kept
  int pass = 0;  // --pass 1 / 2 (0: one pass)
  std::string stats;  // --stats: written by --pass 1, read by --pass 2
  long long target_size = 0;  // --target-size: the IVF file's bytes at most (--pass 2)
  int bq = 16, aq = -16;
  bool pyramid = true;
  int mq = 0;
  bool deltaq = false;
  int tplq = 0;  // 1: keyframes' qindex from TPL; 2: anchors' too
  bool list_devices = false;
  int kq = 0;  // keyframe qindex offset
  bool deblock_search = false;  // per frame, the deblocking offset of the lowest squared error
  int deblock_delta = 2;  // with CDEF: -0.7% / -0.1% BD-rate at qindex 45-180 / 30-75 on anime clips
  bool cdef = true;
  bool ccso = true;
  bool mrls = true;
  bool amvd = true;
  bool tmv = false;
  bool flex_mv = true;
  bool tcq = true;
  bool rd_pass = true;
  Device device = Device::Auto;
  bool progress = false;
  bool psnr = false;
  int parallel_b = -1;  // -1: auto
  int workers = 1;  // --workers: scene chunks coded at once (0: auto)
  int memory_limit = 0;  // --memory-limit MB (0: 80% of the installed RAM)
  bool gop_trial_given = false;  // --gop-trial on the command line (the memory cap leaves it alone)
  int chunk_memory = 4096;  // --chunk-memory: MB of frames read ahead for the workers
};

// Reads frames on its own thread, up to `depth` ahead of take(): a mini-GOP takes bframes + 1
// frames at once, and a 16K frame took 0.06 s to read.
class ReadAhead {
 public:
  ReadAhead(Y4mReader& r, int limit, size_t depth) : r_(r), limit_(limit), depth_(depth), t_([this] { run(); }) {}
  ~ReadAhead() {
    {
      std::lock_guard<std::mutex> lk(m_);
      stop_ = true;
    }
    cv_.notify_all();
    t_.join();
  }
  // The next frame; false at the end of the input (or of --frames). Rethrows a read error.
  bool take(Frame420& out) {
    std::unique_lock<std::mutex> lk(m_);
    cv_.wait(lk, [&] { return !q_.empty() || done_; });
    if (q_.empty()) {
      if (err_) std::rethrow_exception(err_);
      return false;
    }
    out = std::move(q_.front());
    q_.pop_front();
    cv_.notify_all();
    return true;
  }

 private:
  void run() {
    try {
      for (int k = 0; limit_ == 0 || k < limit_; k++) {
        Frame420 f;
        if (!r_.read(f)) break;
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return q_.size() < depth_ || stop_; });
        if (stop_) return;
        q_.push_back(std::move(f));
        cv_.notify_all();
      }
    } catch (...) {
      std::lock_guard<std::mutex> lk(m_);
      err_ = std::current_exception();
    }
    std::lock_guard<std::mutex> lk(m_);
    done_ = true;
    cv_.notify_all();
  }

  Y4mReader& r_;
  const int limit_;  // 0: no limit
  const size_t depth_;
  std::mutex m_;
  std::condition_variable cv_;
  std::deque<Frame420> q_;
  bool done_ = false, stop_ = false;
  std::exception_ptr err_;
  std::thread t_;  // last: starts once the rest is constructed
};

int to_int(const char* s, const char* name) {
  char* end = nullptr;
  const long v = std::strtol(s, &end, 10);
  if (!*s || *end) throw std::invalid_argument(std::string("bad value for ") + name + ": " + s);
  return static_cast<int>(v);
}

Args parse(int argc, char** argv) {
  Args a;
  std::vector<std::string> given;  // the options the command line sets, which --tune leaves alone
  for (int i = 1; i < argc; i++) {
    const std::string k = argv[i];
    given.push_back(k);
    if (k == "--psnr") {
      a.psnr = true;
      continue;
    }
    if (k == "--progress") {
      a.progress = true;
      continue;
    }
    if (k == "--list-devices") {
      a.list_devices = true;
      continue;
    }
    if (i + 1 >= argc) throw std::invalid_argument("missing value for " + k);
    const char* v = argv[++i];
    if (k == "-i") a.in = v;
    else if (k == "-o") a.out = v;
    else if (k == "--qindex") a.qindex = to_int(v, "--qindex");
    else if (k == "--frames") a.frames = to_int(v, "--frames");
    else if (k == "--keyint") a.keyint = to_int(v, "--keyint");
    else if (k == "--recon") a.recon = v;
    else if (k == "--tile-columns") a.tile_cols_log2 = to_int(v, "--tile-columns");
    else if (k == "--tile-rows") a.tile_rows_log2 = to_int(v, "--tile-rows");
    else if (k == "--threads") a.threads = to_int(v, "--threads");
    else if (k == "--refs") a.refs = to_int(v, "--refs");
    else if (k == "--colour") {
      const std::string c = v;
      a.colour = c == "709" ? 1 : c == "pq" ? 2 : c == "hlg" ? 3 : 0;
      if (!a.colour) throw std::invalid_argument("--colour must be 709, pq or hlg");
    }
    else if (k == "--range") {
      const std::string r = v;
      if (r != "limited" && r != "full") throw std::invalid_argument("--range must be limited or full");
      a.full_range = r == "full";
    }
    else if (k == "--mdcv") a.mdcv = parse_mdcv(v);
    else if (k == "--cll") a.cll = parse_cll(v);
    else if (k == "--bit-depth") {
      a.bit_depth = to_int(v, "--bit-depth");
      if (a.bit_depth != 8 && a.bit_depth != 10) throw std::invalid_argument("--bit-depth must be 8 or 10");
    }
    else if (k == "--scenecut") a.scenecut = to_int(v, "--scenecut") != 0;
    else if (k == "--bframes") a.bframes = to_int(v, "--bframes");
    else if (k == "--gop-still") {
      char* end = nullptr;
      a.gop_still = std::strtod(v, &end);
      if (!*v || *end || a.gop_still < 0 || a.gop_still > 1) throw std::invalid_argument("--gop-still must be 0..1");
    }
    else if (k == "--preset") {
      a.preset = v;
      if (a.preset != "max") throw std::invalid_argument("--preset must be max");
    }
    else if (k == "--gop-trial") a.gop_trial = to_int(v, "--gop-trial") != 0, a.gop_trial_given = true;
    else if (k == "--pass") {
      a.pass = to_int(v, "--pass");
      if (a.pass != 1 && a.pass != 2) throw std::invalid_argument("--pass must be 1 or 2");
    }
    else if (k == "--stats") a.stats = v;
    else if (k == "--workers") {
      a.workers = std::string(v) == "auto" ? 0 : to_int(v, "--workers");
      if (a.workers < 0 || a.workers > 16 || std::string(v) == "0") throw std::invalid_argument("--workers must be 1..16 or auto");
    }
    else if (k == "--memory-limit") {
      a.memory_limit = std::string(v) == "auto" ? 0 : to_int(v, "--memory-limit");
      if (a.memory_limit < 0 || std::string(v) == "0") throw std::invalid_argument("--memory-limit must be MB above 0, or auto");
    }
    else if (k == "--chunk-memory") {
      a.chunk_memory = to_int(v, "--chunk-memory");
      if (a.chunk_memory < 64) throw std::invalid_argument("--chunk-memory must be 64 (MB) or more");
    }
    else if (k == "--target-size") {
      char* end = nullptr;
      a.target_size = std::strtoll(v, &end, 10);
      if (!*v || *end || a.target_size <= 0) throw std::invalid_argument(std::string("--target-size must be a number of bytes above 0: ") + v);
    }
    else if (k == "--tune") {
      a.tune = v;
      if (a.tune != "film" && a.tune != "animation" && a.tune != "still")
        throw std::invalid_argument("--tune must be film, animation or still");
    }
    else if (k == "--bq") a.bq = to_int(v, "--bq");
    else if (k == "--pyramid") a.pyramid = to_int(v, "--pyramid") != 0;
    else if (k == "--deltaq") a.deltaq = to_int(v, "--deltaq") != 0;
    else if (k == "--tplq") a.tplq = to_int(v, "--tplq");
    else if (k == "--mq") a.mq = to_int(v, "--mq");
    else if (k == "--aq") a.aq = to_int(v, "--aq");
    else if (k == "--kq") a.kq = to_int(v, "--kq");
    else if (k == "--key-boost") a.key_boost = to_int(v, "--key-boost");
    else if (k == "--level-step") a.level_step = to_int(v, "--level-step");
    else if (k == "--level-step-16") a.level_step16 = to_int(v, "--level-step-16");
    else if (k == "--anchor-boost") a.anchor_boost = to_int(v, "--anchor-boost");
    else if (k == "--cdef") a.cdef = to_int(v, "--cdef") != 0;
    else if (k == "--ccso") a.ccso = to_int(v, "--ccso") != 0;
    else if (k == "--mrls") a.mrls = to_int(v, "--mrls") != 0;
    else if (k == "--amvd") a.amvd = to_int(v, "--amvd") != 0;
    else if (k == "--tmv") a.tmv = to_int(v, "--tmv") != 0;
    else if (k == "--flexmv") a.flex_mv = to_int(v, "--flexmv") != 0;
    else if (k == "--tcq") a.tcq = to_int(v, "--tcq") != 0;
    else if (k == "--rd-pass") a.rd_pass = to_int(v, "--rd-pass") != 0;
    else if (k == "--deblock-delta") a.deblock_delta = to_int(v, "--deblock-delta");
    else if (k == "--deblock-search") a.deblock_search = to_int(v, "--deblock-search") != 0;
    else if (k == "--parallel-b") a.parallel_b = std::string(v) == "auto" ? -1 : to_int(v, "--parallel-b") != 0;
    else if (k == "--device") {
      const std::string d = v;
      if (d == "auto") a.device = Device::Auto;
      else if (d == "gpu") a.device = Device::Gpu;
      else if (d == "cpu") a.device = Device::Cpu;
      else if (d == "vulkan") a.device = Device::Vulkan;
      else if (d == "vulkan2") a.device = Device::Vulkan2;
      else throw std::invalid_argument("bad value for --device: " + d + " (auto, gpu, cpu, vulkan or vulkan2)");
    }
    else throw std::invalid_argument("unknown option " + k);
  }
  if (!a.list_devices && (a.in.empty() || a.out.empty())) throw std::invalid_argument("-i and -o are required");
  if (a.frames < 0 || a.keyint < 0) throw std::invalid_argument("--frames/--keyint must be >= 0");
  if (a.bframes < -1 || a.bframes > 15) throw std::invalid_argument("--bframes must be 0..15");
  if (a.tplq < 0 || a.tplq > 2) throw std::invalid_argument("--tplq must be 0, 1 or 2");
  // --tune: settings by content, each only where the command line does not set it.
  auto unset = [&](const char* opt) { return std::find(given.begin(), given.end(), opt) == given.end(); };
  if (!a.tune.empty() || a.preset == "max") a.long_gop = true;
  if (a.tune == "film" || a.tune == "animation")
    if (unset("--gop-still")) a.gop_still = a.tune == "film" ? 0.7 : 0.85;
  if (a.tune == "film" && unset("--deltaq")) a.deltaq = true;
  if (a.tune == "animation") {
    if (unset("--kq")) a.kq = -8;
    if (unset("--key-boost")) a.key_boost = 64;
    if (unset("--level-step-16")) a.level_step16 = 2;
    if (unset("--deblock-delta")) a.deblock_delta = 0;
  }
  // After the tune: its --gop-still gives way to the trial's wider one. By content (120 / 600 frames
  // 1080p q100-190, docs/design/2026-10-31-preset-max.md): the trial gains on anime (PSNR -1.9 to
  // -2.4% on the mixed stretches) but loses on live action (吉村 PSNR +1.1%, VMAF +2.4%; bits weighed
  // x4 PSNR -0.8% but VMAF +2.5%); the deblocking search gains on live action (VMAF -0.3 to -4.1%)
  // but loses on anime (VMAF up to +4.2%).
  if (a.preset == "max") {
    const bool trial = a.tune != "film", search = a.tune != "animation";
    if (trial && unset("--gop-still")) a.gop_still = 0.5;
    if (trial && unset("--gop-trial")) a.gop_trial = true;
    if (search && unset("--deblock-search")) a.deblock_search = true;
  }
  if (a.pass == 1 && a.stats.empty()) throw std::invalid_argument("--pass 1 needs --stats");
  if (a.pass == 2 && (a.stats.empty() || !a.target_size)) throw std::invalid_argument("--pass 2 needs --stats and --target-size");
  if (a.pass != 2 && a.target_size) throw std::invalid_argument("--target-size needs --pass 2 (and --pass 1 before it)");
  // The quick look: with the RD pass off too it varies more against the real pass (1.23 against 1.16 at
  // most over 120-frame stretches of mixed anime, q150) but is twice as fast, and the target size comes
  // out as close (600-frame mixed stretches: 95.7-99.2% of the target against 96.0-99.1%,
  // docs/design/2026-10-31-target-size.md).
  if (a.pass == 1) {
    if (unset("--gop-trial")) a.gop_trial = false;
    if (unset("--deblock-search")) a.deblock_search = false;
    if (unset("--rd-pass")) a.rd_pass = false;
  }
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args a = parse(argc, argv);
    if (a.list_devices) {  // one "<--device value>\t<name>" line each, then nothing else
      for (const auto& [arg, name] : list_devices(probe_gpu)) std::printf("%s\t%s\n", arg.c_str(), name.c_str());
      return 0;
    }
    Y4mReader reader(a.in);
    celv2::PassStats stats;  // --pass 1: what it writes; --pass 2: what it reads
    std::optional<celv2::SizeControl> size_control;
    int qindex = a.qindex;
    if (a.pass == 2) {
      stats = celv2::read_stats(a.stats);
      if (stats.width != reader.width() || stats.height != reader.height())
        throw std::runtime_error("--stats is for " + std::to_string(stats.width) + "x" + std::to_string(stats.height) +
                                 ", the input is " + std::to_string(reader.width()) + "x" + std::to_string(reader.height()));
      size_control.emplace(stats, static_cast<uint64_t>(a.target_size), a.workers);
      qindex = size_control->qindex();
    }
    SeqConfig cfg{reader.width(), reader.height(), reader.fps_num(), reader.fps_den(), qindex};
    reader.set_bit_depth(a.bit_depth);
    cfg.bit_depth = reader.bit_depth();
    cfg.colour = a.colour, cfg.full_range = a.full_range, cfg.mdcv = a.mdcv, cfg.cll = a.cll;
    // Tiles are coded on separate threads, so a large frame needs many (at 16K the AV2 minimum of 4
    // tiles kept 4 of 48 cores busy); each costs some bits (4K: 2 / 4 / 8 tiles at 2.4 / 2.7 /
    // 3.1 fps, +0.6% / +1.6% bytes), and below 4K the GPU decision is the limit anyway.
    auto auto_log2 = [](int size, int limit) {
      int l = 0;
      while ((size >> l) >= limit && l < 6) l++;
      return l;
    };
    // At most 16 tiles: 16K ran at 0.43 / 0.48 / 0.55 fps with 8 / 16 / 64 tiles for +1.7% / +15%
    // bytes (the GPU decision and the per-frame work between tiles set the pace).
    int c = auto_log2(cfg.width, 1280), r = auto_log2(cfg.height, 2560);
    while (c + r > 4) (c > r ? c : r)--;  // squarer tiles: 16K as 4 x 4 cost 6% less than 8 x 2
    cfg.tile_cols_log2 = a.tile_cols_log2 >= 0 ? a.tile_cols_log2 : c;
    cfg.tile_rows_log2 = a.tile_rows_log2 >= 0 ? a.tile_rows_log2 : r;
    cfg.threads = a.threads;
    if (a.refs) cfg.refs = a.refs;
    cfg.deblock_delta = a.deblock_delta;
    cfg.deblock_search = a.deblock_search;
    cfg.cdef = a.cdef;
    cfg.ccso = a.ccso;
    cfg.mrls = a.mrls;
    cfg.tmv = a.tmv;
    cfg.flex_mv = a.flex_mv;
    cfg.amvd = a.amvd;
    cfg.tcq = a.tcq;
    cfg.rd_pass = a.rd_pass;
    cfg.validate();
    // B frames predict from both sides: with one slot the hidden frame takes it and the decoder no
    // longer holds the frame before (the stream did not decode).
    // 8-frame mini-GOPs need 5 slots for their 4 hidden frames and the last shown one; 120 frames
    // 1080p q100-190 against 4: live action -9.3%, anime B -2.2 / -8.2%, anime C -3.4%,
    // anime D +1.0%, anime A -3.3%, three fast Xiph clips -0.5 / +0.3 / -0.2% BD-rate (16 frames:
    // -13.5 / +2.7 / -5.1 / -4.7 / +5.0 / +11.3%). Below qindex 100 they lose (30 frames live action
    // / anime: q40-100 +1.5 / +2.3%, q70-130 -0.2 / +1.4%). Choosing 4 frames where the motion is
    // fast did not help on 120 frames: it lost anime B's -8.2% for -4.9%.
    const int bframes = cfg.refs < 2 ? 0 : a.bframes >= 0 ? a.bframes : cfg.refs >= 5 && cfg.qindex >= 100 ? (a.long_gop ? 15 : 7) : 3;
    bool parallel_b = a.parallel_b >= 0 ? a.parallel_b != 0 : static_cast<long long>(cfg.width) * cfg.height <= 2LL * 3840 * 2160;
    bool gop_trial = a.gop_trial;
    // Memory: GiB a worker takes (AM 120 frames 1080p / 2160p, --refs 7, peak private bytes; the copies
    // of the encoder for parallel B frames and the GOP trial's second pipeline are most of it), each
    // further worker 0.5 GiB less (the process's own part). Decided before coding, so the same input
    // and settings give the same bytes.
    const double mpx = static_cast<double>(cfg.width) * cfg.height / 1e6;
    auto worker_gib = [&] {
      return parallel_b ? (gop_trial ? 2.46 + 2.84 * mpx : 1.58 + 1.47 * mpx) : (gop_trial ? 0.96 + 1.00 * mpx : 0.83 + 0.50 * mpx);
    };
    auto need_gib = [&](int w) { return worker_gib() + (w - 1) * (worker_gib() - 0.5); };
    const double limit_gib = (a.memory_limit > 0 ? a.memory_limit : installed_memory_bytes() * 0.8 / (1 << 20)) / 1024;
    const int asked = a.workers > 0 ? a.workers : std::clamp(static_cast<int>(std::thread::hardware_concurrency()) / 16, 1, 4);
    // With a target size, auto is one worker: the control shared by chunks came out further from the
    // target (600-frame mixed anime: 82-99%, retries in 7 of 12, up to -1.9 dB) than the one in order
    // (95.7-99.2%, no retries; docs/design/2026-10-31-target-size.md). --workers N still shares it.
    int workers = size_control && a.workers == 0 ? 1 : asked;
    while (workers > 1 && need_gib(workers) > limit_gib) workers--;
    if (workers < asked && a.workers > 0)
      std::fprintf(stderr, "celv2enc: --workers %d -> %d: about %.1f GiB each, --memory-limit %.1f GiB\n", asked, workers, worker_gib(), limit_gib);
    if (need_gib(1) > limit_gib && parallel_b && a.parallel_b < 0) {
      parallel_b = false;
      std::fprintf(stderr, "celv2enc: --parallel-b off: --memory-limit %.1f GiB (the same output)\n", limit_gib);
    }
    if (need_gib(1) > limit_gib && gop_trial && !a.gop_trial_given) {
      gop_trial = false;
      std::fprintf(stderr, "celv2enc: --gop-trial off: --memory-limit %.1f GiB\n", limit_gib);
    }
    if (need_gib(1) > limit_gib)
      std::fprintf(stderr, "celv2enc: warning: about %.1f GiB, over --memory-limit %.1f GiB\n", need_gib(1), limit_gib);
    if (size_control) size_control->set_workers(workers);
    LoopConfig lc;
    lc.bframes = bframes, lc.parallel_b = parallel_b, lc.psnr = a.psnr, lc.recon = !a.recon.empty();
    lc.pyramid = a.pyramid, lc.deltaq = a.deltaq, lc.tplq = a.tplq;
    lc.bq = a.bq, lc.aq = a.aq, lc.mq = a.mq, lc.kq = a.kq;
    lc.key_boost = a.key_boost, lc.anchor_boost = a.anchor_boost, lc.level_step = a.level_step, lc.level_step16 = a.level_step16;
    lc.gop_still = a.gop_still, lc.gop_trial = gop_trial;
    // --target-size with more than one worker: the control is shared by the chunks (SizeControl's chunk
    // calls); with one it runs per mini-GOP in order, as it always did.
    const bool chunked = size_control && workers > 1;
    uint64_t written = 32;  // the IVF file's bytes so far
    std::vector<std::unique_ptr<ChunkCoder>> coders;
    for (int w = 0; w < workers; w++) coders.push_back(std::make_unique<ChunkCoder>(cfg, a.device, lc));
    std::fprintf(stderr, "celv2: %s\n", coders[0]->engine_description().c_str());
    IvfWriter ivf(a.out, cfg.width, cfg.height, cfg.fps_num, cfg.fps_den);
    std::FILE* recon = a.recon.empty() ? nullptr : std::fopen(a.recon.c_str(), "wb");
    if (!a.recon.empty() && !recon) throw std::runtime_error("cannot open " + a.recon);
    const auto t0 = std::chrono::steady_clock::now();
    int n = 0;  // frames output
    double psnr_sum = 0;  // --psnr: per shown frame, luma against its source
    double psnr_uv_sum[2] = {0, 0};  // and chroma
    // Each keyframe's, lone frame's or mini-GOP's frames as they are coded, in display order.
    auto write = [&](std::vector<CodedFrame>& group) {
      for (CodedFrame& c : group) {
        ivf.write(c.tu, c.display);
        written += 12 + c.tu.size();
        if (size_control && !chunked) size_control->coded(c.tu.size());
        if (a.pass == 1) stats.bytes.push_back(c.tu.size());
        if (a.psnr) psnr_sum += c.psnr[0], psnr_uv_sum[0] += c.psnr[1], psnr_uv_sum[1] += c.psnr[2];
        if (recon && std::fwrite(c.recon.data(), 1, c.recon.size(), recon) != c.recon.size())
          throw std::runtime_error("writing " + a.recon + " failed (disk full?)");
        n++;
        if (a.progress) {
          std::fprintf(stderr, "progress frame=%d\n", n);
          std::fflush(stderr);
        }
      }
      // --target-size: the qindex for what is coded next, moved after each keyframe, lone frame or mini-GOP.
      if (size_control && !chunked) size_control->update(), coders[0]->set_qindex_offset(size_control->qindex() - cfg.qindex);
    };
    // The reader: frames in display order, each classified (keyframe or not) once the frame after it
    // is read, as the scene-cut test looks one frame ahead. A chunk starts at a keyframe once the one
    // before holds 48 frames (shorter scenes join it). It holds one frame back, so with one worker the
    // coder sees bframes + 2 frames at most of the bframes + 3 read ahead; more workers share
    // --chunk-memory of frames.
    const size_t frame_bytes = static_cast<size_t>(cfg.width + 7) / 8 * 8 * ((cfg.height + 7) / 8 * 8) * 3 / 2 * (cfg.bit_depth > 8 ? 3 : 1);
    const double frames_gib = std::min(static_cast<double>(a.chunk_memory) / 1024, std::max(0.0, limit_gib - need_gib(workers)));
    const size_t read_ahead = workers == 1 ? static_cast<size_t>(bframes) + 3
                                           : std::max(static_cast<size_t>(bframes) + 3, static_cast<size_t>(frames_gib * (1 << 30) / frame_bytes));
    constexpr size_t kMinChunk = 48;
    FrameBudget budget(read_ahead);
    std::mutex chunks_mu;
    std::condition_variable chunks_cv;
    std::vector<std::shared_ptr<FrameQueue>> chunks;
    std::vector<char> chunk_done;  // coded and handed to the writer
    size_t next_chunk = 0;  // the chunk the next free worker takes
    bool read_done = false;
    std::exception_ptr read_err, code_err;
    std::atomic<bool> failed{false};
    // The first error stops everything: no more reading, every chunk closed so no coder waits.
    auto fail = [&](std::exception_ptr e) {
      {
        std::lock_guard<std::mutex> l(chunks_mu);
        if (!code_err) code_err = e;
        for (auto& c : chunks) c->close();
        failed = true;  // under the lock: a worker between its wait's check and its sleep would miss the notify
      }
      budget.cancel();
      chunks_cv.notify_all();
    };
    ReadAhead frames(reader, a.frames, static_cast<size_t>(bframes) + 2);
    std::thread reader_thread([&] {
      std::shared_ptr<FrameQueue> cur;
      size_t cur_frames = 0;
      try {
        SceneCutDetector cuts;
        int read_count = 0, last_key = 0;
        ChunkFrame f;
        bool have = budget.take() && frames.take(f.f);
        f.display = read_count++;
        while (have) {
          ChunkFrame next;
          const bool room = budget.take();
          const bool more = room && frames.take(next.f);
          if (room && !more) budget.give(1);
          next.display = read_count++;
          const bool cut = cuts.push(f.f, more ? &next.f : nullptr);  // every frame, in order: it keeps history
          f.key = f.display == 0 || (a.keyint > 0 && f.display - last_key >= a.keyint) || (a.scenecut && cut);
          if (f.key) last_key = f.display;
          if (f.key && (!cur || cur_frames >= kMinChunk)) {
            if (cur) cur->close();
            cur = std::make_shared<FrameQueue>(budget);
            cur_frames = 0;
            {
              std::lock_guard<std::mutex> l(chunks_mu);
              if (failed) cur->close();
              chunks.push_back(cur);
              chunk_done.push_back(0);
            }
            chunks_cv.notify_all();
          }
          cur->push(std::move(f));
          cur_frames++;
          if (chunked) chunks_cv.notify_all();  // a worker may wait for this chunk to fill
          if (!room) break;  // cancelled: a coder failed
          f = std::move(next);
          have = more;
        }
      } catch (...) {
        read_err = std::current_exception();
      }
      if (cur) cur->close();
      {
        std::lock_guard<std::mutex> l(chunks_mu);
        read_done = true;
      }
      chunks_cv.notify_all();
    });
    // Output in chunk order: the chunk being written writes its groups as they are coded; a later one
    // keeps them until the chunks before it are done.
    struct Held {
      std::vector<CodedFrame> frames;
      bool done = false;
    };
    std::mutex out_mu;
    std::map<size_t, Held> held;
    size_t writing = 0;
    auto deliver = [&](size_t k, std::vector<CodedFrame>& group) {
      std::lock_guard<std::mutex> l(out_mu);
      if (k == writing) write(group);
      else for (CodedFrame& c : group) held[k].frames.push_back(std::move(c));
    };
    auto finish = [&](size_t k) {
      std::lock_guard<std::mutex> l(out_mu);
      held[k].done = true;
      while (held.count(writing) && held[writing].done) {
        held.erase(writing++);
        auto it = held.find(writing);
        if (it != held.end() && !it->second.frames.empty()) {
          write(it->second.frames);
          it->second.frames.clear();
        }
      }
    };
    auto work = [&](ChunkCoder& coder) {
      try {
        for (;;) {
          std::shared_ptr<FrameQueue> c;
          size_t k;
          {
            std::unique_lock<std::mutex> l(chunks_mu);
            chunks_cv.wait(l, [&] { return next_chunk < chunks.size() || read_done || failed; });
            if (failed || next_chunk >= chunks.size()) return;
            k = next_chunk++;
            c = chunks[k];
          }
          if (workers > 1) {  // at most 2 x workers chunks on hand: what a later one holds for the writer is bounded
            std::unique_lock<std::mutex> l(chunks_mu);
            const size_t ahead = 2 * static_cast<size_t>(workers);
            chunks_cv.wait(l, [&] {
              return failed || k < ahead || std::all_of(chunk_done.begin(), chunk_done.begin() + static_cast<std::ptrdiff_t>(k - ahead + 1), [](char d) { return d != 0; });
            });
            if (failed) return;
          }
          bool last = false;
          if (chunked) {
            std::unique_lock<std::mutex> l(chunks_mu);
            auto done_below = [&](size_t n) { return std::all_of(chunk_done.begin(), chunk_done.begin() + static_cast<std::ptrdiff_t>(n), [](char d) { return d != 0; }); };
            const size_t w = static_cast<size_t>(workers);
            // what its control sees: the chunks up to k - workers, done
            chunks_cv.wait(l, [&] { return failed || k < w || done_below(k - w + 1); });
            // read to its end (a later chunk, or the input's end), unless it holds over half the read-ahead
            chunks_cv.wait(l, [&] { return failed || k + 1 < chunks.size() || read_done || c->size() > read_ahead / 2; });
            // the last chunk, whole (half the read-ahead at most, so it never waits on frames it holds itself):
            // after all the others, with what is left of the budget. Its length alone decides, not timing.
            last = read_done && k + 1 == chunks.size() && c->size() <= read_ahead / 2;
            if (last) chunks_cv.wait(l, [&] { return failed || done_below(k); });
            if (failed) return;
          }
          if (!chunked) {
            coder.code(*c, [&](std::vector<CodedFrame>& g) { deliver(k, g); });
          } else {
            // over --target-size: the last chunk again from its frames at one qindex, 2 above the highest the
            // try before used (so each try is smaller), 3 times at most
            if (last) c->keep();
            std::vector<CodedFrame> out;
            const size_t first = static_cast<size_t>(c->at(0)->display);
            for (int again = 0, at_least = 1;; again++) {
              int q = again ? at_least : size_control->chunk_start_qindex(k, first, c->size());
              int highest = q;
              coder.set_qindex_offset(q - cfg.qindex);
              coder.code(*c, [&](std::vector<CodedFrame>& g) {
                for (const CodedFrame& f : g) size_control->coded(k, static_cast<size_t>(f.display), f.tu.size(), q);
                if (again) return void(out.insert(out.end(), std::make_move_iterator(g.begin()), std::make_move_iterator(g.end())));
                q = size_control->chunk_update(k);
                highest = std::max(highest, q);
                coder.set_qindex_offset(q - cfg.qindex);
                if (last) for (CodedFrame& f : g) out.push_back(std::move(f));
                else deliver(k, g);
              });
              if (!last) break;
              uint64_t total = 0;
              {
                std::lock_guard<std::mutex> l(out_mu);
                total = written;
              }
              for (const CodedFrame& f : out) total += 12 + f.tu.size();
              if (total <= static_cast<uint64_t>(a.target_size) || again == 3 || highest >= 255) break;
              at_least = std::min(255, highest + 2);
              std::fprintf(stderr, "celv2enc: re-coded the last chunk at qindex %d (it was over by %llu bytes)\n", at_least,
                           static_cast<unsigned long long>(total - a.target_size));
              size_control->forget(k);
              c->rewind();
              out.clear();
            }
            if (last) deliver(k, out);
          }
          if (failed) return;
          finish(k);
          {
            std::lock_guard<std::mutex> l(chunks_mu);
            chunk_done[k] = 1;
          }
          chunks_cv.notify_all();
        }
      } catch (...) {
        fail(std::current_exception());
      }
    };
    std::vector<std::thread> worker_threads;
    for (int w = 1; w < workers; w++) worker_threads.emplace_back(work, std::ref(*coders[w]));
    work(*coders[0]);
    for (auto& t : worker_threads) t.join();
    reader_thread.join();
    if (code_err) std::rethrow_exception(code_err);
    if (read_err) std::rethrow_exception(read_err);
    if (workers > 1) std::fprintf(stderr, "celv2: %zu chunks on %d workers\n", chunks.size(), workers);
    ivf.close();
    if (a.pass == 1) {
      stats.width = cfg.width, stats.height = cfg.height, stats.qindex = cfg.qindex;
      celv2::write_stats(a.stats, stats);
    }
    if (size_control && size_control->frames() < stats.bytes.size())
      throw std::runtime_error("the input has fewer frames (" + std::to_string(size_control->frames()) + ") than --stats (" +
                               std::to_string(stats.bytes.size()) + ")");
    if (recon && std::fclose(recon) != 0) throw std::runtime_error("writing " + a.recon + " failed (disk full?)");
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    // Same format as av2enc.bat.
    if (a.psnr) {
      std::fprintf(stderr, "psnr: %.3f\n", n > 0 ? psnr_sum / n : 0.0);
      std::fprintf(stderr, "psnr uv: %.3f %.3f\n", n > 0 ? psnr_uv_sum[0] / n : 0.0, n > 0 ? psnr_uv_sum[1] / n : 0.0);
    }
    std::fprintf(stderr, "speed: %d frames / %.0f s = %.1f s/frame, %.3f fps\n", n, s,
                 s / (n > 0 ? n : 1), s > 0 ? n / s : 0.0);
    if (size_control) {
      const unsigned long long size = std::filesystem::file_size(std::filesystem::u8path(a.out));
      std::fprintf(stderr, "target-size: %llu of %lld bytes\n", size, a.target_size);
      if (size > static_cast<unsigned long long>(a.target_size)) {
        std::fprintf(stderr, "celv2enc: over --target-size by %llu bytes%s\n", size - a.target_size,
                     size_control->coarsest() ? " even at qindex 255" : "");
        return 3;
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "celv2enc: error: %s\n", e.what());
    return 1;
  }
}
