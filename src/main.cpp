// celv2enc: y4m (8-bit 4:2:0) -> IVF with GPU- or CPU-decided partitions/modes/motion and CPU
// reconstruction.
// Usage: celv2enc -i in.y4m -o out.ivf [--qindex 120] [--frames N] [--keyint 0] [--recon r.yuv]
//                [--tile-columns L] [--tile-rows L] [--threads N] [--device auto|gpu|cuda|cpu|vulkan|vulkan2]
//                [--refs N] [--scenecut 0|1] [--bframes N] [--bq D] [--aq D] [--pyramid 0|1] [--progress] [--psnr]
//                [--tune film|animation|still]
//   -i - reads the y4m from standard input (e.g. ffmpeg ... -f yuv4mpegpipe - | celv2enc -i - ...).
//   --progress prints "progress frame=N type=T qindex=Q bytes=B" (N from 1) on stderr after each coded
//   frame is written: T K keyframe, P lone frame, A anchor, B the others; B the frame's own bytes
//   (a hidden anchor's or mid's counted where it is shown, not in the unit that carries it).
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
//   --device: where the decisions run (default auto = the first discrete Vulkan GPU, else the CPU; gpu
//   is that GPU and fails without one; vulkan / vulkan2 take the first / second one; cuda the CUDA
//   engine, for one release). All give the same output; the choice is printed once on stderr.
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
//   --gop-change S: a mini-GOP ends at a frame whose share of 16x16 blocks changed from the frame
//   before (by average or high-frequency energy) passes S, 4 frames at least after the last one, that
//   frame its anchor (0..1; 0 off, the default: at 0.9, 120 frames 1080p q100-190 against 0,
//   SSIMULACRA2 / PSNR-Y, Angel Beats -5.7 / +2.6%, Dragon Maid +2.7 / +0.2%, 8 other sources the same
//   bytes; halftone analysis spec, branch 1; docs/reports/2026-11-04-halftone-gop.md).
//   --psy-rd S: psychovisual RD strength (0 off, the default; 2 at most): the RD pass's and the coder's
//   luma decisions also cost the AC energy a candidate loses or adds against the source, weighted per
//   block by the importance map: texture (live action) or, with --tune animation, drawn lines
//   (docs/superpowers/specs/2026-11-04-psy-rd-design.md).
//   --psy-trellis T: the same map's strength in the level choice (0 off, the default; 2 at most): an AC
//   level costs T * weight * | |coefficient| - |level| |, so small levels stop dropping to 0 as readily.
//   --paq S: perceptual AQ strength (0 off; 0.25 the default; 2 at most): each 64x64's qindex by
//   a 16x16 importance map of the source luma (variance, edges, halftone, darkness), on every coded
//   frame, added to --deltaq's (docs/superpowers/specs/2026-11-04-perceptual-aq-design.md). 120 frames
//   1080p q100-190, 8 sources, BD-rate SSIMULACRA2 / XPSNR: 0.25 -0.86 / -0.91% (each source within
//   +1%; park_joy -3.6 / -4.1%), 1 -1.6 / -3.4% but AIR +5.0 / +5.3%; 1.03-1.14x the time
//   (docs/reports/2026-11-04-perceptual-aq.md).
//   --tplq 0|1|2|3: the qindex of keyframes (1), of keyframes and anchors (2, the mids then between
//   the anchor's and --bq's by level) or of anchors only (3, the default; keyframes and mids as
//   without) from TPL instead of the offsets and still-share boosts: quantizer step = the B frames'
//   (--bq) * r0^0.7, r0 = the frame's own share of its weight in the model of the mini-GOP after it
//   (libaom's constant-q GOPs take r0^0.5). 120 frames 1080p q100-190 against 0, global PSNR-Y
//   (2026-11-02, 5-level motion search): 3: Dragon Maid -4.0%, Angel Beats -2.6%, chika -2.1%,
//   蜷画搗 -1.5%, AM -0.4%, park_joy -0.3%, AIR -0.1%, crowd_run +0.3% (-1.3% on average);
//   2: -1.1% on average, Angel Beats +1.7%; 1: Angel Beats +3.2% (its scene-cut keyframes).
//   --deblock-delta D: deblocking strength offset (-16..15, steps of 8 qindex; default 2).
//   --deblock-search 0|1: per frame, the offset in {-4, -2, 0, 2, 4} whose deblocked frame is nearest
//   the source (default 0; 1 with --preset max).
//   --cdef 0|1: CDEF after deblocking, one strength per frame (default 1).
//   --ccso 0|1: cross-component sample offset after CDEF (default 1).
//   --mrls 0|1: intra prediction from reference lines further out (default 1).
//   --lr 0|1: loop restoration, luma PC_WIENER switched per 128x128 unit (default 1).
//   --lr-uv 0|1: with --lr, chroma loop restoration, a Wiener filter fitted per frame and switched
//       per 64x64 unit (default 1).
//   --ist 0|1: secondary transform of luma low frequencies, per transform block (default 1).
//   --cctx 0|1: cross-chroma transform (U and V rotated together), per chroma transform block (default 1).
//   --mhccp 0|1: multi-hypothesis cross-component chroma prediction, searched per block (default 1 without
//   CfL or with --tune animation, 0 otherwise).
//   --cfl-rd 0|1: chroma modes chosen with CfL / MHCCP predicted from the block's reconstructed luma,
//   a shortlist (the SATD's 4 best, the CfL types, the signalled alpha around the derived one) by
//   quantized U + V cost (default 1; intra gap spec, fix 1).
//   --intra-txp 0|1: an intra block may split its luma transform into 2 or 4, chosen by the residual
//       cost of each (default 1).
//   --warp 0|1: local warp, rotation / zoom models coded per block, fitted to the engine's MVs (default 0).
//   --interintra 0|1: inter-intra, a single-reference prediction blended with an intra one, per block
//       (default 0).
//   --cfl 0|1: chroma predicted from the reconstructed luma, scaling derived from the neighbours or
//   signalled (default 1; live action with --cfl-rd, 120 frames 1080p q100-190 against 0, Y / YUV
//   6:1:1: 吉村 +0.10 / -0.66%, crowd_run +0.04 / -0.13%, park_joy +0.03 / -0.09%).
//   --amvd 0|1: adaptive MVD, NEWMV differences from a short set of values (default 1).
//   --tmv 0|1: temporal MVs, the references' motion projected into the MV candidates (default 0, 1 with
//   --tune animation).
//   --flexmv 0|1: NEWMV blocks may code their MV at 1, 4 or 8 pel instead of 1/4 (default 1).
//   --kq D: qindex offset of keyframes (default 0).
//   --key-filter 0|1: a keyframe is coded temporally filtered with the 2 frames after it (AVM's
//   method; default 0, 1 with --tune animation).
//   --key-boost N / --anchor-boost N: a keyframe's / an anchor's qindex goes down by up to N as the
//   still share (to the frame up to 8 ahead / through its mini-GOP) goes from 1/2 to all (default 48 / 0).
//   --level-step N: the qindex step from one pyramid level to the next (default 8).
//   --level-step-16 N: the same in mini-GOPs over 8 frames (default: --level-step).
//   --gop-still T: with mini-GOPs over 8 frames, one takes 16 only where the still share to its last
//   frame reaches T, else 8 (0..1; default 0: always the full length).
//   --preset max: speed ignored for quality and size (docs/superpowers/specs/2026-10-31-celv2-preset-max-design.md):
//   16-frame mini-GOPs where --bframes is not given (as --tune), --gop-still 0.5, --gop-trial 1,
//   --deblock-search 1, --rd-intra 1; with --tune animation no --deblock-search, with --tune film no --gop-trial
//   (nor its --gop-still).
//   --gop-trial 0|1: a mini-GOP over 8 frames is coded both as it is and as 8 + the rest, and the one
//   with the lower squared error + sse_per_bit x bits kept (default 0; 1 with --preset max).
//   --pass 1 --stats F: writes each IVF frame's bytes to F, with --gop-trial 0 --deblock-search 0
//   --rd-pass 0 unless given (docs/superpowers/specs/2026-10-31-target-size-design.md).
//   --pass 2 --stats F --target-size B: the same input and settings again, the IVF file at most B
//   bytes: one qindex for the whole video from F (--qindex is not used), moved by at most 2 a
//   mini-GOP as the bytes come in. Ends with "target-size: <bytes> of <B> bytes"; over B: exit 3.
//   --target-vmaf V [--ffmpeg PATH]: each scene chunk at the highest qindex whose probe reaches VMAF V
//   (the mean, vmaf_v0.6.1): up to 4 probes of the chunk (128 frames at most, else its first 96) with
//   these settings, scored by the libvmaf of PATH (default ffmpeg on the PATH); the first at --qindex.
//   About 3.5-4x the time. Not with --pass (docs/superpowers/specs/2026-11-03-target-vmaf-design.md).
//   --grain N (1..50, 0 off): each frame denoised by AVM's noise model at noise level N / 10 (aomenc's
//   --denoise-noise-level), coded so, and its fitted film grain sent for the decoder to add back
//   (docs/superpowers/specs/2026-11-03-film-grain-design.md). --recon and --psnr see the denoised frames.
//   --workers N: scene chunks (a keyframe to the next, 48 frames at least) coded at once on N
//   pipelines, written in order: the same bytes as 1 (the default). --chunk-memory MB: the frames
//   read ahead for them (default 4096; the reader waits beyond it). --workers auto: one per 16
//   hardware threads, 1..4 (provisional). --memory-limit MB: at most this (default 80% of the
//   installed RAM): fewer workers, then no parallel B frames (the same bytes), then no GOP trial
//   (unless given), each with a warning.
//   --tune film|animation|still: settings by content (x264's --tune); options given on the command
//   line override them. All take 16-frame mini-GOPs where --bframes is not given (with --refs 5 or
//   more and --qindex 100 or more). film: --gop-still 0.7 --deltaq 1 --grain 25 (not with --target-vmaf). animation: --gop-still 0.85
//   --kq -8 --key-boost 64 --level-step-16 2 --deblock-delta 0 --tmv 1 --key-filter 1 --rd-zero 1. still: 16 frames throughout. 120 frames 1080p
//   q100-190 against no tune: animation on six anime sources -0.13 to -13.5% (AM -2.2%, kaguya
//   -13.5%, Dragon Maid -0.3%, AIR -0.8%, chika -0.8%, Angel Beats -0.1%); docs/design/2026-10-30-tunes.md.
//   --tcq 0|1: trellis coded quantization of luma (default 1).
//   --rd-pass 0|1: re-pick the decision engine's partitions and modes on the CPU by estimated
//   rate and distortion (default 1; -10 to -18% BD-rate on 1080p clips, 25% slower at 1080p).
//   --rd-intra 0|1: the RD pass tries every intra mode, not only the engine's best (default 0, 1 with
//   --preset max; 120 frames 1080p q100-190: -0.18% BD-rate on average, about 60% slower).
//   --rd-zero 0|1: zero residual by RD: the trellis takes its end of block or all zero by J, the plain path weighs a block's
//   levels against none (default 0, 1 with --tune animation). 120 frames 1080p q100-190 at --paq 0, PSNR-Y / SSIMULACRA2 / XPSNR BD-rate: five anime
//   -0.84 / -1.88 / -0.51%, three live action -0.56 / +0.37 / +4.76%, which lose detail
//   (docs/superpowers/specs/2026-11-06-rd-zeroing-design.md).
//   --screen 0|1|auto: palettes and IntraBC for screen content and halftone: off, on, or frame by frame by a gate (default
//   auto; regular sources stay shut, at the speed of 0; sources that open it, such as Flash, run 1.12-1.18x slower;
//   docs/superpowers/specs/2026-11-04-intrabc-palette-design.md).
//   --intra64 0|1: intra frames also take 64x64, 64x32 and 32x64 intra blocks (default 1; one frame
//   q100-190: AIR OP -12.3%, Dragon Maid -9.3%, AM -1.2%, Flash +1.3 / -1.3% BD-rate).
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
#include <future>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bitstream/obu.h"
#include "core/film_grain.h"
#include "core/profile.h"
#include "core/scene_cut.h"
#include "chunk_coder.h"
#include "core/size_control.h"
#include "core/vmaf_probe.h"
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
  bool key_filter = false;
  int key_boost = 48, anchor_boost = 0;  // qindex down by up to these with the still share
  int level_step = 8;  // qindex step between pyramid levels
  int level_step16 = -1;  // the same in mini-GOPs over 8 frames (-1: level_step)
  std::string tune;
  std::string preset;  // "max": speed ignored (docs/superpowers/specs/2026-10-31-celv2-preset-max-design.md)
  bool gop_trial = false;  // a mini-GOP over 8 frames coded both as it is and as 8 + the rest; the better kept
  int pass = 0;  // --pass 1 / 2 (0: one pass)
  std::string stats;  // --stats: written by --pass 1, read by --pass 2
  double target_vmaf = 0;  // --target-vmaf (0: off)
  int grain = 0;  // --grain (0: off)
  // --paq (0: off). 0.25 by default: the one strength with no source past +1% (report), and the
  // user's viewing on 2026-11-04 (吉村 better, AIR and Dragon Maid unchanged).
  double paq = 0.25;
  double gop_change = 0;  // --gop-change (0: off)
  double psy_rd = 0;  // --psy-rd (0: off)
  double psy_trellis = 0;  // --psy-trellis (0: off)
  int screen = 2;  // --screen: 0, 1, 2 (auto)
  std::string ffmpeg = "ffmpeg";  // --ffmpeg: the one --target-vmaf's probes run
  long long target_size = 0;  // --target-size: the IVF file's bytes at most (--pass 2)
  int bq = 16, aq = -16;
  bool pyramid = true;
  int mq = 0;
  bool deltaq = false;
  int tplq = 3;  // 1: keyframes' qindex from TPL; 2: anchors' too; 3: anchors' only
  bool list_devices = false;
  int kq = 0;  // keyframe qindex offset
  bool deblock_search = false;  // per frame, the deblocking offset of the lowest squared error
  int deblock_delta = 2;  // with CDEF: -0.7% / -0.1% BD-rate at qindex 45-180 / 30-75 on anime clips
  bool cdef = true;
  bool ccso = true;
  bool mrls = true;
  bool ist = true;
  bool cctx = true;
  bool mhccp = true;
  bool lr = true;
  bool lr_uv = true;
  bool interintra = false;
  bool warp = false;
  bool intra_txp = true;
  // --cfl-rd: 120 frames 1080p q100-190 against 0, Y / YUV 6:1:1: mean -0.02 / -0.84%, every
  // source's YUV negative (吉村 -2.20%), Y within +0.17% (AIR).
  bool cfl_rd = true;
  bool cfl = true;
  bool amvd = true;
  bool tmv = false;
  bool flex_mv = true;
  bool tcq = true;
  bool rd_pass = true;
  bool rd_intra = false;
  bool rd_zero = false;
  bool intra64 = true;
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
  // grain: --grain's model, run on each frame as it is read (nullptr: none).
  ReadAhead(Y4mReader& r, int limit, size_t depth, GrainModel* grain = nullptr)
      : r_(r), limit_(limit), depth_(depth), grain_(grain), t_([this] { run(); }) {}
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
    // --grain: frames denoised several at once, each on its own thread (one at a time made 1080p
    // 0.37 s a frame, the coder's 0.27), the model updated in display order; at most ~2 GB of their
    // 16-bit buffers (6 bytes a pixel), 8 frames.
    struct Pending {
      std::unique_ptr<Frame420> f;
      std::future<std::shared_ptr<GrainModel::Denoised>> d;
    };
    std::deque<Pending> pending;
    auto push = [&](Frame420&& f) {
      std::unique_lock<std::mutex> lk(m_);
      cv_.wait(lk, [&] { return q_.size() < depth_ || stop_; });
      if (stop_) return false;
      q_.push_back(std::move(f));
      cv_.notify_all();
      return true;
    };
    auto finish_front = [&] {
      Pending p = std::move(pending.front());
      pending.pop_front();
      celv2::StageScope prof(celv2::Stage::Denoise);
      grain_->finish(*p.f, *p.d.get());
      return push(std::move(*p.f));
    };
    try {
      size_t in_flight = 1;
      for (int k = 0; limit_ == 0 || k < limit_; k++) {
        auto f = std::make_unique<Frame420>();
        bool got;
        {
          celv2::StageScope prof(celv2::Stage::Read);
          got = r_.read(*f);
        }
        if (!got) break;
        if (!grain_) {
          if (!push(std::move(*f))) return;
          continue;
        }
        if (k == 0) in_flight = std::clamp<size_t>((size_t{2} << 30) / (static_cast<size_t>(f->pw) * f->ph * 6), 1, 8);
        const Frame420* const fp = f.get();
        pending.push_back(Pending{std::move(f), std::async(std::launch::async, [this, fp] {
                                            celv2::StageScope prof(celv2::Stage::Denoise);
                                            return grain_->denoise(*fp);
                                          })});
        if (pending.size() >= in_flight && !finish_front()) return;
      }
      while (!pending.empty())
        if (!finish_front()) return;
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
  GrainModel* const grain_;
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
    else if (k == "--target-vmaf") {
      char* end = nullptr;
      a.target_vmaf = std::strtod(v, &end);
      if (!*v || *end || !(a.target_vmaf > 0 && a.target_vmaf <= 100)) throw std::invalid_argument(std::string("--target-vmaf must be above 0, 100 at most: ") + v);
    }
    else if (k == "--ffmpeg") a.ffmpeg = v;
    else if (k == "--grain") {
      char* end = nullptr;
      const long n = std::strtol(v, &end, 10);
      if (!*v || *end || n < 0 || n > 50) throw std::invalid_argument(std::string("--grain must be 0..50: ") + v);
      a.grain = static_cast<int>(n);
    }
    else if (k == "--tune") {
      a.tune = v;
      if (a.tune != "film" && a.tune != "animation" && a.tune != "still")
        throw std::invalid_argument("--tune must be film, animation or still");
    }
    else if (k == "--bq") a.bq = to_int(v, "--bq");
    else if (k == "--pyramid") a.pyramid = to_int(v, "--pyramid") != 0;
    else if (k == "--deltaq") a.deltaq = to_int(v, "--deltaq") != 0;
    else if (k == "--gop-change") {
      char* end = nullptr;
      a.gop_change = std::strtod(v, &end);
      if (!*v || *end || !(a.gop_change >= 0 && a.gop_change <= 1)) throw std::invalid_argument(std::string("--gop-change must be 0..1: ") + v);
    }
    else if (k == "--psy-rd") {
      char* end = nullptr;
      a.psy_rd = std::strtod(v, &end);
      if (!*v || *end || !(a.psy_rd >= 0 && a.psy_rd <= 2)) throw std::invalid_argument(std::string("--psy-rd must be 0..2: ") + v);
    }
    else if (k == "--psy-trellis") {
      char* end = nullptr;
      a.psy_trellis = std::strtod(v, &end);
      if (!*v || *end || !(a.psy_trellis >= 0 && a.psy_trellis <= 2)) throw std::invalid_argument(std::string("--psy-trellis must be 0..2: ") + v);
    }
    else if (k == "--screen") {
      const std::string m = v;
      if (m == "0" || m == "1") a.screen = m == "1";
      else if (m == "auto") a.screen = 2;
      else throw std::invalid_argument("--screen must be 0, 1 or auto: " + m);
    }
    else if (k == "--paq") {
      char* end = nullptr;
      a.paq = std::strtod(v, &end);
      if (!*v || *end || !(a.paq >= 0 && a.paq <= 2)) throw std::invalid_argument(std::string("--paq must be 0..2: ") + v);
    }
    else if (k == "--tplq") a.tplq = to_int(v, "--tplq");
    else if (k == "--mq") a.mq = to_int(v, "--mq");
    else if (k == "--aq") a.aq = to_int(v, "--aq");
    else if (k == "--kq") a.kq = to_int(v, "--kq");
    else if (k == "--key-boost") a.key_boost = to_int(v, "--key-boost");
    else if (k == "--level-step") a.level_step = to_int(v, "--level-step");
    else if (k == "--level-step-16") a.level_step16 = to_int(v, "--level-step-16");
    else if (k == "--anchor-boost") a.anchor_boost = to_int(v, "--anchor-boost");
    else if (k == "--key-filter") a.key_filter = to_int(v, "--key-filter") != 0;
    else if (k == "--cdef") a.cdef = to_int(v, "--cdef") != 0;
    else if (k == "--ccso") a.ccso = to_int(v, "--ccso") != 0;
    else if (k == "--mrls") a.mrls = to_int(v, "--mrls") != 0;
    else if (k == "--ist") a.ist = to_int(v, "--ist") != 0;
    else if (k == "--cctx") a.cctx = to_int(v, "--cctx") != 0;
    else if (k == "--mhccp") a.mhccp = to_int(v, "--mhccp") != 0;
    else if (k == "--lr") a.lr = to_int(v, "--lr") != 0;
    else if (k == "--lr-uv") a.lr_uv = to_int(v, "--lr-uv") != 0;
    else if (k == "--interintra") a.interintra = to_int(v, "--interintra") != 0;
    else if (k == "--warp") a.warp = to_int(v, "--warp") != 0;
    else if (k == "--intra-txp") a.intra_txp = to_int(v, "--intra-txp") != 0;
    else if (k == "--cfl-rd") a.cfl_rd = to_int(v, "--cfl-rd") != 0;
    else if (k == "--cfl") a.cfl = to_int(v, "--cfl") != 0;
    else if (k == "--amvd") a.amvd = to_int(v, "--amvd") != 0;
    else if (k == "--tmv") a.tmv = to_int(v, "--tmv") != 0;
    else if (k == "--flexmv") a.flex_mv = to_int(v, "--flexmv") != 0;
    else if (k == "--tcq") a.tcq = to_int(v, "--tcq") != 0;
    else if (k == "--rd-pass") a.rd_pass = to_int(v, "--rd-pass") != 0;
    else if (k == "--rd-intra") a.rd_intra = to_int(v, "--rd-intra") != 0;
    else if (k == "--rd-zero") a.rd_zero = to_int(v, "--rd-zero") != 0;
    else if (k == "--intra64") a.intra64 = to_int(v, "--intra64") != 0;
    else if (k == "--deblock-delta") a.deblock_delta = to_int(v, "--deblock-delta");
    else if (k == "--deblock-search") a.deblock_search = to_int(v, "--deblock-search") != 0;
    else if (k == "--parallel-b") a.parallel_b = std::string(v) == "auto" ? -1 : to_int(v, "--parallel-b") != 0;
    else if (k == "--device") {
      const std::string d = v;
      if (d == "auto") a.device = Device::Auto;
      else if (d == "gpu") a.device = Device::Gpu;
      else if (d == "cuda") a.device = Device::Cuda;
      else if (d == "cpu") a.device = Device::Cpu;
      else if (d == "vulkan") a.device = Device::Vulkan;
      else if (d == "vulkan2") a.device = Device::Vulkan2;
      else throw std::invalid_argument("bad value for --device: " + d + " (auto, gpu, cuda, cpu, vulkan or vulkan2)");
    }
    else throw std::invalid_argument("unknown option " + k);
  }
  if (!a.list_devices && (a.in.empty() || a.out.empty())) throw std::invalid_argument("-i and -o are required");
  if (a.frames < 0 || a.keyint < 0) throw std::invalid_argument("--frames/--keyint must be >= 0");
  if (a.bframes < -1 || a.bframes > 15) throw std::invalid_argument("--bframes must be 0..15");
  if (a.tplq < 0 || a.tplq > 3) throw std::invalid_argument("--tplq must be 0, 1, 2 or 3");
  // --tune: settings by content, each only where the command line does not set it.
  auto unset = [&](const char* opt) { return std::find(given.begin(), given.end(), opt) == given.end(); };
  if (!a.tune.empty() || a.preset == "max") a.long_gop = true;
  if (a.tune == "film" || a.tune == "animation")
    if (unset("--gop-still")) a.gop_still = a.tune == "film" ? 0.7 : 0.85;
  if (a.tune == "film" && unset("--deltaq")) a.deltaq = true;
  // By viewing (吉村, crowd_run q150, docs/reports/2026-11-04-film-grain.md). --target-vmaf scores the
  // frames without the grain, so it would pay to code back what the denoising took out.
  if (a.tune == "film" && unset("--grain") && a.target_vmaf == 0) a.grain = 25;
  if (a.tune == "animation") {
    if (unset("--kq")) a.kq = -8;
    if (unset("--key-boost")) a.key_boost = 64;
    if (unset("--level-step-16")) a.level_step16 = 2;
    if (unset("--deblock-delta")) a.deblock_delta = 0;
    // Re-measured 2026-11-01 by global-MSE PSNR (refs 7, q100-190, against the tune without them):
    // temporal MVs AIR 240 -3.93%, mixed AM 600 -2.59%, AM 60 -0.38%, Angel Beats -0.43%, Dragon
    // Maid +0.06%, chika +0.63% (VMAF NEG -1.45%); the keyframe filter mixed AM 600 -1.69%, Angel
    // Beats -0.55%, AIR 240 -0.07%, Dragon Maid +0.07%, AM 60 +0.14%, chika +0.15%. Live action
    // (33 frames): temporal MVs +0.60%, so not by default. Both against the tune without them: AIR 240
    // -4.06%, mixed AM 600 -4.30%, Angel Beats -0.98%, AM 60 -0.13%, Dragon Maid +0.12%, chika +0.59%
    // (VMAF NEG -4.64 / -6.29 / -2.59 / -1.26 / +0.25 / -0.92%), 0.95-1.01x the speed.
    if (unset("--tmv")) a.tmv = true;
    if (unset("--key-filter")) a.key_filter = true;
    // At the default --paq, 120 frames against --rd-zero 0, PSNR-Y / SSIMULACRA2 / XPSNR-Y mean -0.92 / -1.81 / -0.49%, AIR alone
    // over +1% (+1.04 / +0.61 / +1.25%); +0.5 to +1.0% CPU a frame. The user chose it by viewing, for anime only.
    if (unset("--rd-zero")) a.rd_zero = true;
  }
  // MHCCP where CfL is off: with CfL (--tune animation) it took blocks CfL coded better (AIR OP +0.99%).
  // With CfL too under --tune animation (with --cfl-rd, 120 frames 1080p q100-190 against without, Y /
  // YUV 6:1:1: AIR -0.05 / -0.44%, AM -0.24 / -0.49%, Angel Beats -0.08 / -0.17%, chika and Dragon Maid
  // within 0.1%); not on live action (吉村 -0.11 / +0.52%).
  if (unset("--mhccp")) a.mhccp = !a.cfl || a.tune == "animation";
  // After the tune: its --gop-still gives way to the trial's wider one. By content (120 / 600 frames
  // 1080p q100-190, docs/design/2026-10-31-preset-max.md): the trial gains on anime (PSNR -1.9 to
  // -2.4% on the mixed stretches) but loses on live action (蜷画搗 PSNR +1.1%, VMAF +2.4%; bits weighed
  // x4 PSNR -0.8% but VMAF +2.5%); the deblocking search gains on live action (VMAF -0.3 to -4.1%)
  // but loses on anime (VMAF up to +4.2%).
  if (a.preset == "max") {
    const bool trial = a.tune != "film", search = a.tune != "animation";
    if (trial && unset("--gop-still")) a.gop_still = 0.5;
    if (trial && unset("--gop-trial")) a.gop_trial = true;
    if (search && unset("--deblock-search")) a.deblock_search = true;
    if (unset("--rd-intra")) a.rd_intra = true;
  }
  if (a.pass == 1 && a.stats.empty()) throw std::invalid_argument("--pass 1 needs --stats");
  if (a.pass == 2 && (a.stats.empty() || !a.target_size)) throw std::invalid_argument("--pass 2 needs --stats and --target-size");
  if (a.pass != 2 && a.target_size) throw std::invalid_argument("--target-size needs --pass 2 (and --pass 1 before it)");
  if (a.target_vmaf > 0 && a.pass) throw std::invalid_argument("--target-vmaf is a one-pass mode: not with --pass / --target-size");
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
      for (const auto& [arg, name] : list_devices()) std::printf("%s\t%s\n", arg.c_str(), name.c_str());
      return 0;
    }
    Y4mReader reader(a.in);
    const char* const test_vmaf = std::getenv("CELV2_TEST_VMAF");  // "c,d[,odd]": VMAF 100 - c x 2^(qindex / d) (+ odd on odd chunks), tests/test_cli.cpp
    if (a.target_vmaf > 0 && !test_vmaf && !celv2::ffmpeg_has_libvmaf(a.ffmpeg))
      throw std::runtime_error("--target-vmaf: " + a.ffmpeg + " does not run or has no libvmaf filter (--ffmpeg PATH)");
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
    cfg.film_grain = a.grain > 0;
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
    cfg.ist = a.ist;
    cfg.cctx = a.cctx;
    cfg.mhccp = a.mhccp;
    cfg.lr = a.lr;
    cfg.lr_uv = a.lr_uv;
    cfg.interintra = a.interintra;
    cfg.warp = a.warp;
    cfg.intra_tx_part = a.intra_txp;
    cfg.cfl_rd = a.cfl_rd;
    cfg.cfl = a.cfl;
    cfg.tmv = a.tmv;
    cfg.flex_mv = a.flex_mv;
    cfg.amvd = a.amvd;
    cfg.tcq = a.tcq;
    cfg.rd_pass = a.rd_pass;
    cfg.rd_intra = a.rd_intra;
    cfg.rd_zero = a.rd_zero;
    cfg.intra64 = a.intra64;
    cfg.psy_rd = a.psy_rd, cfg.psy_trellis = a.psy_trellis, cfg.psy_lines = a.tune == "animation";
    cfg.screen = a.screen;
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
    // and settings give the same bytes. Parallel B frames without the trial measured again 2026-11-03
    // (2.90 / 8.18 GiB; it was 1.58 + 1.47 Mpx before the shadows shared their references); the trial's
    // terms are from before (with it the clip measured 2.92 / 8.47 GiB, but it may not have tried much).
    const double mpx = static_cast<double>(cfg.width) * cfg.height / 1e6;
    auto worker_gib = [&] {
      return parallel_b ? (gop_trial ? 2.46 + 2.84 * mpx : 1.14 + 0.85 * mpx) : (gop_trial ? 0.96 + 1.00 * mpx : 0.83 + 0.50 * mpx);
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
    cfg.engines = workers * (gop_trial ? 2 : 1);
    LoopConfig lc;
    lc.bframes = bframes, lc.parallel_b = parallel_b, lc.psnr = a.psnr, lc.recon = !a.recon.empty() || a.target_vmaf > 0;  // --target-vmaf scores the probes' recon
    lc.pyramid = a.pyramid, lc.deltaq = a.deltaq, lc.tplq = a.tplq, lc.paq = a.paq, lc.gop_change = a.gop_change, lc.psy_rd = a.psy_rd, lc.psy_trellis = a.psy_trellis, lc.screen = a.screen;
    lc.bq = a.bq, lc.aq = a.aq, lc.mq = a.mq, lc.kq = a.kq;
    lc.key_filter = a.key_filter, lc.key_boost = a.key_boost, lc.anchor_boost = a.anchor_boost, lc.level_step = a.level_step, lc.level_step16 = a.level_step16;
    lc.gop_still = a.gop_still, lc.gop_trial = gop_trial;
    // --target-size with more than one worker: the control is shared by the chunks (SizeControl's chunk
    // calls); with one it runs per mini-GOP in order, as it always did.
    // --target-vmaf: chunks at any worker count, each at the qindex of its own probes.
    const bool vmaf = a.target_vmaf > 0;
    const bool chunked = (size_control && workers > 1) || vmaf;
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
          std::fprintf(stderr, "progress frame=%d type=%c qindex=%d bytes=%zu\n", n, c.type, c.qindex, c.bytes);
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
    const size_t read_ahead = workers == 1 && !vmaf ? static_cast<size_t>(bframes) + 3
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
    std::unique_ptr<GrainModel> grain;
    if (a.grain > 0) grain = std::make_unique<GrainModel>(cfg.bit_depth, a.grain);
    ReadAhead frames(reader, a.frames, static_cast<size_t>(bframes) + 2, grain.get());
    std::thread reader_thread([&] {
      std::shared_ptr<FrameQueue> cur;
      size_t cur_frames = 0;
      try {
        SceneCutDetector cuts;
        ChangeMeter change;  // --gop-change
        int read_count = 0, last_key = 0;
        ChunkFrame f;
        bool have = budget.take() && frames.take(f.f);
        f.display = read_count++;
        if (have && a.gop_change > 0) f.change = change.push(f.f);
        while (have) {
          ChunkFrame next;
          const bool room = budget.take();
          const bool more = room && frames.take(next.f);
          if (room && !more) budget.give(1);
          next.display = read_count++;
          if (more && a.gop_change > 0) {
            celv2::StageScope prof(celv2::Stage::Analysis);
            next.change = change.push(next.f);
          }
          bool cut;
          {
            celv2::StageScope prof(celv2::Stage::Analysis);
            cut = cuts.push(f.f, more ? &next.f : nullptr);  // every frame, in order: it keeps history
          }
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
    // --target-vmaf: chunk k's qindex from probes on the worker's own coder (the same settings, so the
    // probe at the pick is the chunk's output when it took the whole chunk), scored by ffmpeg's libvmaf
    // (CELV2_TEST_VMAF: by its formula instead). A chunk of 128 frames at most is probed whole (kept and
    // rewound), a longer one by its first 96 and then coded (4 probes of n frames cost less than 4 of 96
    // and n up to 128). AM / Dragon Maid / 蜷画搗 120 frames at VMAF 93: each scene's
    // VMAF 93.13-93.24; every 2nd frame instead (48 at most) 92.3-95.0 (89.9 on a Dragon Maid scene: the
    // gaps cost more bits), runs of 8-16 frames from keyframes 92.9-94.7, pass 1's settings about 1.8 lower.
    struct VmafPick {
      int q;
      std::vector<CodedFrame> frames;  // the chunk's output, when its probe can stand
    };
    auto vmaf_pick = [&](size_t k, FrameQueue& c, ChunkCoder& coder) {
      constexpr size_t kWhole = 128, kProbeFrames = 96;
      const size_t on_hand = c.size();
      const bool whole = on_hand <= kWhole && on_hand <= read_ahead / 2;  // then the chunk is closed
      // Not whole: as many as are sure to come, whatever the chunk's length or the budget, and the same at any worker count.
      const size_t n = whole ? on_hand : std::min(kProbeFrames, read_ahead / 2 + 1);
      std::vector<const Frame420*> src;
      for (size_t i = 0; i < n; i++) src.push_back(&c.at(i)->f);
      const int first = c.at(0)->display;
      double tc = 0, td = 1, odd = 0;
      if (test_vmaf && std::sscanf(test_vmaf, "%lf,%lf,%lf", &tc, &td, &odd) < 2) throw std::invalid_argument("CELV2_TEST_VMAF must be c,d[,odd]");
      std::map<int, std::vector<CodedFrame>> outs;  // whole: each probe's output
      const bool keep_recon = recon != nullptr;  // --recon: the output keeps its recon too
      double code_s = 0, vmaf_s = 0;
      auto score = [&](int q) {
        const auto t0 = std::chrono::steady_clock::now();
        coder.set_qindex_offset(q - cfg.qindex);
        std::vector<std::vector<uint8_t>> recon(src.size());
        std::vector<CodedFrame> out;
        auto take = [&](std::vector<CodedFrame>& g) {
          for (CodedFrame& f : g) {
            std::vector<uint8_t>& r = recon.at(static_cast<size_t>(f.display - first));
            if (keep_recon) r = f.recon;
            else r = std::move(f.recon);
            if (whole) out.push_back(std::move(f));
          }
        };
        if (whole) {
          c.keep();
          coder.code(c, take);
          c.rewind();
          outs[q] = std::move(out);
        } else {
          FrameBudget pb(src.size());
          FrameQueue pq(pb);
          for (size_t j = 0; j < src.size(); j++) pq.push(ChunkFrame{*src[j], first + static_cast<int>(j), j == 0});
          pq.close();
          coder.code(pq, take);
        }
        const auto t1 = std::chrono::steady_clock::now();
        code_s += std::chrono::duration<double>(t1 - t0).count();
        if (test_vmaf) return std::clamp(100 - tc * std::pow(2.0, q / td) + (k % 2 ? odd : 0), 0.0, 100.0);
        const double v = celv2::ffmpeg_vmaf(a.ffmpeg, src, recon, cfg.fps_num, cfg.fps_den);
        vmaf_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        return v;
      };
      int probes = 0;
      VmafPick pick{celv2::vmaf_search(a.target_vmaf, cfg.qindex, score, 4, &probes, whole), {}};
      if (whole) c.release(), c.pop(on_hand), pick.frames = std::move(outs[pick.q]);  // the pop gives the frames' budget back
      std::fprintf(stderr, "celv2enc: scene %zu (frames %d-%d): qindex %d (%d probes of %zu frames%s: coding %.1f s, VMAF %.1f s)\n", k,
                   first, first + static_cast<int>(on_hand) - 1, pick.q, probes, src.size(), whole ? "" : ", then coded", code_s, vmaf_s);
      return pick;
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
            last = size_control && read_done && k + 1 == chunks.size() && c->size() <= read_ahead / 2;
            if (last) chunks_cv.wait(l, [&] { return failed || done_below(k); });
            if (failed) return;
          }
          if (vmaf) {
            VmafPick pick = vmaf_pick(k, *c, coder);
            if (!pick.frames.empty()) {
              deliver(k, pick.frames);
            } else {
              coder.set_qindex_offset(pick.q - cfg.qindex);
              coder.code(*c, [&](std::vector<CodedFrame>& g) { deliver(k, g); });
            }
          } else if (!chunked) {
            coder.code(*c, [&](std::vector<CodedFrame>& g) { deliver(k, g); });
          } else {
            // over --target-size: the last chunk again from its frames at one qindex, 2 above the highest the
            // try before used (so each try is smaller), or from the second retry on as far above as the last two
            // tries' bytes per qindex say the excess needs, 3 times at most
            if (last) c->keep();
            std::vector<CodedFrame> out;
            const size_t first = static_cast<size_t>(c->at(0)->display);
            int prev_q = -1;
            uint64_t prev_total = 0;
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
              int step = 2;
              if (prev_q >= 0 && highest > prev_q && prev_total > total)
                step = std::max(2, static_cast<int>(std::ceil((total - a.target_size) * (highest - prev_q) /
                                                              static_cast<double>(prev_total - total))) + 1);
              prev_q = highest, prev_total = total;
              at_least = std::min(255, highest + step);
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
    celv2::print_profile(stderr, n);
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
