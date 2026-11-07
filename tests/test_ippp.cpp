#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "bitstream/obu.h"
#include "check.h"
#include "core/encoder.h"
#include "pipeline.h"
#include "test_frames.h"
#include "vk/vk_engine.h"
#include "verify/avm_decode.h"

struct Sequence {
  std::vector<std::vector<uint8_t>> tus;
  std::vector<Frame420> recons;
  std::vector<double> psnr;
};

// I + P... with the device's decisions; frame i shows the scene shifted by (i*dx, i*dy). keyint 0
// means only frame 0 is a keyframe.
static Sequence encode_ippp(int w, int h, int frames, int dx, int dy, int q, int tile_cols_log2 = 0,
                            int tile_rows_log2 = 0, int keyint = 0,
                            Device device = Device::Auto, int refs = 0, bool rd_pass = true) {
  SeqConfig c{w, h, 30, 1, q};
  if (refs) c.refs = refs;  // 0: the default
  c.rd_pass = rd_pass;
  c.tile_cols_log2 = tile_cols_log2;
  c.tile_rows_log2 = tile_rows_log2;
  Pipeline pipe(c, device);
  Sequence s;
  for (int i = 0; i < frames; i++) {
    const Frame420 src = scene_frame(w, h, i * dx, i * dy);
    s.tus.push_back(pipe.encode(src, i == 0 || (keyint > 0 && i % keyint == 0)));
    s.recons.push_back(pipe.last_recon());
    s.psnr.push_back(psnr_y(src, pipe.last_recon()));
    std::fprintf(stderr, "  frame %d: %zu bytes, %.2f dB\n", i, s.tus.back().size(), s.psnr.back());
  }
  return s;
}

static int check_decode(const Sequence& s) {
  const auto dec = decode_frames(s.tus);
  CHECK_EQ(dec.size(), s.recons.size());
  for (size_t i = 0; i < dec.size(); i++) {
    const std::string d = first_diff(dec[i], s.recons[i]);
    if (!d.empty()) std::fprintf(stderr, "frame %zu: %s\n", i, d.c_str());
    CHECK_EQ(d, std::string());
  }
  return 0;
}

static int ippp_bitexact() {
  const Sequence s = encode_ippp(640, 360, 8, 3, 0, 120);
  CHECK_EQ(check_decode(s), 0);
  size_t p_total = 0;
  for (size_t i = 1; i < s.tus.size(); i++) p_total += s.tus[i].size();
  const double p_avg = static_cast<double>(p_total) / (s.tus.size() - 1);
  std::fprintf(stderr, "I %zu bytes, P average %.0f bytes\n", s.tus[0].size(), p_avg);
  CHECK(p_avg * 2 < static_cast<double>(s.tus[0].size()));
  return 0;
}

// P-frame quality must not drift away from the first P frame on steadily moving content.
static int ippp_quality_holds() {
  const Sequence s = encode_ippp(640, 360, 8, 3, 0, 120);
  CHECK(s.psnr.back() > s.psnr[1] - 1.5);
  return 0;
}

static int ippp_edge_motion_bitexact() {
  const Sequence s = encode_ippp(202, 122, 4, 40, 24, 120);
  CHECK_EQ(check_decode(s), 0);
  return 0;
}

// 4x2 tiles coded on parallel threads: same decoder match, and the same bytes on a second run.
static int ippp_tiles_bitexact() {
  const Sequence s = encode_ippp(640, 360, 4, 3, 2, 120, 2, 1);
  CHECK_EQ(check_decode(s), 0);
  const Sequence again = encode_ippp(640, 360, 4, 3, 2, 120, 2, 1);
  CHECK(again.tus == s.tus);
  return 0;
}

// The pipelined Pipeline (rows coded as the GPU publishes them) against the whole-frame decision
// followed by Encoder::encode: the same bytes for every frame (without the CPU RD pass, which only
// the Pipeline runs).
static int pipeline_matches_sync() {
  struct Case {
    int w, h, frames, cols_log2, rows_log2, keyint;
  };
  const Case cases[] = {{640, 360, 8, 0, 0, 0}, {640, 360, 8, 1, 0, 0}, {640, 360, 8, 1, 1, 0},
                        {202, 122, 6, 1, 0, 0}, {32, 32, 4, 0, 0, 0},   {640, 360, 7, 1, 0, 3}};
  for (const Case& k : cases) {
    std::fprintf(stderr, "%dx%d tiles %d/%d keyint %d\n", k.w, k.h, k.cols_log2, k.rows_log2,
                 k.keyint);
    const Sequence s =
        encode_ippp(k.w, k.h, k.frames, 3, 2, 120, k.cols_log2, k.rows_log2, k.keyint, Device::Auto, 0, false);
    CHECK_EQ(check_decode(s), 0);
    SeqConfig c{k.w, k.h, 30, 1, 120};
    c.tile_cols_log2 = k.cols_log2;
    c.tile_rows_log2 = k.rows_log2;
    Encoder enc(c);
    VulkanEngine gpu(k.w, k.h, 64);
    std::vector<Frame420> recons;  // the references, nearest first, as Pipeline keeps them
    for (int i = 0; i < k.frames; i++) {
      const Frame420 src = scene_frame(k.w, k.h, i * 3, i * 2);
      const bool key = i == 0 || (k.keyint > 0 && i % k.keyint == 0);
      std::vector<const Frame420*> refs;
      for (const Frame420& r : recons) refs.push_back(&r);
      const DecisionMap dm = key ? gpu.decide_intra(src, 120) : gpu.decide_inter(src, refs, 120);
      CHECK(enc.encode(src, dm, key) == s.tus[i]);
      if (key) recons.clear();
      recons.insert(recons.begin(), enc.last_recon());
      if (static_cast<int>(recons.size()) > c.refs) recons.pop_back();
    }
  }
  return 0;
}

// CPU decisions give the same bytes as GPU decisions, frame by frame, and decode to the recon.
static int cpu_pipeline_matches_gpu() {
  struct Case {
    int w, h, frames, cols_log2, keyint;
  };
  const Case cases[] = {
      {640, 360, 6, 0, 0}, {640, 360, 6, 1, 0}, {202, 122, 4, 1, 0}, {640, 360, 7, 1, 3}};
  for (const Case& k : cases) {
    std::fprintf(stderr, "%dx%d tile cols %d keyint %d\n", k.w, k.h, k.cols_log2, k.keyint);
    const Sequence cpu =
        encode_ippp(k.w, k.h, k.frames, 3, 2, 120, k.cols_log2, 0, k.keyint, Device::Cpu);
    const Sequence gpu =
        encode_ippp(k.w, k.h, k.frames, 3, 2, 120, k.cols_log2, 0, k.keyint, Device::Gpu);
    CHECK_EQ(cpu.tus.size(), gpu.tus.size());
    for (size_t i = 0; i < cpu.tus.size(); i++) CHECK(cpu.tus[i] == gpu.tus[i]);
    CHECK_EQ(check_decode(cpu), 0);
  }
  return 0;
}

// A failed gate (GPU error) makes the tile threads throw instead of waiting forever.
static int encode_fails_when_gate_fails() {
  SeqConfig c{640, 360, 30, 1, 120};
  c.tile_cols_log2 = 1;
  Encoder enc(c);
  const DecisionMap dm = uniform_decision(640, 360, 64);
  RowGate gate;
  gate.fail(std::make_exception_ptr(std::runtime_error("gpu")));
  const Frame420 src = scene_frame(640, 360);
  auto f = std::async(std::launch::async, [&] { return enc.encode(src, dm, true, &gate); });
  if (f.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
    // The future's destructor would block on the stuck encode, so a hang must end the process.
    std::fprintf(stderr, "encode_fails_when_gate_fails: encode did not return within 10 s\n");
    std::abort();
  }
  bool threw = false;
  try {
    f.get();
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw);
  return 0;
}

// tile_threads() is what encode() uses: min(tiles, cores, cfg.threads > 0 ? cfg.threads : tiles).
static int tile_threads_counts() {
  SeqConfig c{640, 360, 30, 1, 120};
  c.tile_cols_log2 = 1;
  if (std::thread::hardware_concurrency() >= 2) CHECK_EQ(Encoder(c).tile_threads(), 2);
  c.threads = 1;
  CHECK_EQ(Encoder(c).tile_threads(), 1);
  SeqConfig one{640, 360, 30, 1, 120};
  CHECK_EQ(Encoder(one).tile_threads(), 1);
  return 0;
}

// Mean luma step across vertical 8-pixel block edges over the mean step 4 pixels inside the
// blocks: blocking shows up as edges that step more than the texture next to them.
static double edge_ratio(const Frame420& f) {
  double edge = 0, inside = 0;
  for (int r = 0; r < f.height; r++) {
    const uint8_t* row = &f.y[static_cast<size_t>(r) * f.stride_y()];
    for (int x = 8; x + 4 < f.width; x += 8) {
      edge += std::abs(row[x] - row[x - 1]);
      inside += std::abs(row[x + 4] - row[x + 3]);
    }
  }
  return edge / (inside + 1);
}

// The same coarse IPPP sequence with and without deblocking: both decode bit-exactly, and the
// filtered reconstruction has softer block edges.
static int deblock_bitexact_and_smoother() {
  const int w = 640, h = 360;
  Sequence s[2];
  for (int on = 0; on < 2; on++) {
    SeqConfig c{w, h, 30, 1, 200};
    c.deblock = on != 0;
    Pipeline pipe(c, Device::Auto);
    for (int i = 0; i < 4; i++) {
      s[on].tus.push_back(pipe.encode(scene_frame(w, h, 3 * i, 0), i == 0));
      s[on].recons.push_back(pipe.last_recon());
    }
    CHECK_EQ(check_decode(s[on]), 0);
  }
  for (int i = 0; i < 4; i++) {
    const double off = edge_ratio(s[0].recons[i]), on = edge_ratio(s[1].recons[i]);
    std::fprintf(stderr, "  frame %d: edge ratio %.3f off, %.3f on\n", i, off, on);
    CHECK(on < off);
  }
  return 0;
}

// Three references through the pipeline (GPU and CPU decisions): decodes bit-exactly, also across
// keyframes, and both devices give the same bytes.
static int ippp_multi_ref_bitexact() {
  const Sequence g = encode_ippp(640, 360, 8, 3, 1, 120, 0, 0, 0, Device::Auto, 3);
  CHECK_EQ(check_decode(g), 0);
  const Sequence c = encode_ippp(640, 360, 8, 3, 1, 120, 0, 0, 0, Device::Cpu, 3);
  CHECK(c.tus == g.tus);
  const Sequence k = encode_ippp(202, 122, 9, 5, 3, 120, 1, 0, 4, Device::Auto, 3);
  CHECK_EQ(check_decode(k), 0);
  return 0;
}

// Mini-GOPs of 3 B frames after a keyframe: frame 4k + 4 coded hidden first, frames 4k + 1..3 from
// both sides (not references), then the hidden one shown by a show-existing-frame OBU. Every
// temporal unit outputs one frame; the decoder outputs them in display order, bit-exactly, and
// GPU and CPU decisions give the same bytes.
static int bframes_bitexact() {
  const int w = 640, h = 360, groups = 3;
  std::vector<std::vector<uint8_t>> by_device[2];
  for (int dev = 0; dev < 2; dev++) {
    SeqConfig c{w, h, 30, 1, 120};
    Pipeline pipe(c, dev ? Device::Cpu : Device::Auto);
    auto src = [&](int d) { return scene_frame(w, h, 3 * d, d); };
    std::vector<std::vector<uint8_t>> tus;
    std::vector<Frame420> recons;
    tus.push_back(pipe.encode(src(0), true));
    recons.push_back(pipe.last_recon());
    for (int g = 0; g < groups; g++) {
      const int base = 4 * g;
      std::vector<uint8_t> prefix = pipe.code(src(base + 4), FrameSpec{false, base + 4, false, true});
      const int hint = pipe.display_order_hint();
      const Frame420 anchor = pipe.last_recon();
      for (int d = base + 1; d < base + 4; d++) {
        std::vector<uint8_t> tu = make_obu(OBU_TEMPORAL_DELIMITER, {});
        tu.insert(tu.end(), prefix.begin(), prefix.end());
        prefix.clear();
        const std::vector<uint8_t> b = pipe.code(src(d), FrameSpec{false, d, true, false});
        tu.insert(tu.end(), b.begin(), b.end());
        tus.push_back(tu);
        recons.push_back(pipe.last_recon());
      }
      std::vector<uint8_t> tu = make_obu(OBU_TEMPORAL_DELIMITER, {});
      const std::vector<uint8_t> sef = pipe.show_existing(hint);
      tu.insert(tu.end(), sef.begin(), sef.end());
      tus.push_back(tu);
      recons.push_back(anchor);
    }
    const auto dec = decode_frames(tus);
    CHECK_EQ(dec.size(), recons.size());
    for (size_t i = 0; i < dec.size(); i++) {
      const std::string d = first_diff(dec[i], recons[i]);
      if (!d.empty()) std::fprintf(stderr, "frame %zu: %s\n", i, d.c_str());
      CHECK_EQ(d, std::string());
      const Frame420 s = src(static_cast<int>(i));
      CHECK(psnr_y(s, dec[i]) > 30);  // and each is the right frame
    }
    by_device[dev] = tus;
  }
  CHECK(by_device[0] == by_device[1]);
  return 0;
}

// Pipeline::code_group (B frames on copies of the state at once) gives the bytes and
// reconstructions of coding the same specs one after the other, and the decoder matches them.
static int bframe_group_matches_sequential() {
  const int w = 640, h = 360, groups = 2, nb = 3;
  auto src = [&](int d) { return scene_frame(w, h, 3 * d, d); };
  std::vector<std::vector<uint8_t>> out[2];
  std::vector<Frame420> recons[2];
  for (int grouped = 0; grouped < 2; grouped++) {
    SeqConfig c{w, h, 30, 1, 120};
    Pipeline pipe(c);
    out[grouped].push_back(pipe.encode(src(0), true));
    recons[grouped].push_back(pipe.last_recon());
    for (int g = 0; g < groups; g++) {
      const int base = (nb + 1) * g;
      std::vector<uint8_t> prefix = pipe.code(src(base + nb + 1), FrameSpec{false, base + nb + 1, false, true});
      const int hint = pipe.display_order_hint();
      const Frame420 anchor = pipe.last_recon();
      std::vector<Frame420> frames;
      std::vector<FrameSpec> specs;
      for (int i = 0; i < nb; i++)
        frames.push_back(src(base + 1 + i)), specs.push_back(FrameSpec{false, base + 1 + i, true, false, 0, i + 1 == nb});
      std::vector<std::vector<uint8_t>> bs;
      std::vector<Frame420> rs;
      if (grouped) {
        std::vector<const Frame420*> ptrs;
        for (const Frame420& f : frames) ptrs.push_back(&f);
        bs = pipe.code_group(ptrs, specs, &rs);
      } else {
        for (int i = 0; i < nb; i++) bs.push_back(pipe.code(frames[i], specs[i])), rs.push_back(pipe.last_recon());
      }
      for (int i = 0; i < nb; i++) {
        std::vector<uint8_t> tu = make_obu(OBU_TEMPORAL_DELIMITER, {});
        tu.insert(tu.end(), prefix.begin(), prefix.end());
        prefix.clear();
        tu.insert(tu.end(), bs[i].begin(), bs[i].end());
        out[grouped].push_back(tu);
        recons[grouped].push_back(rs[i]);
      }
      std::vector<uint8_t> tu = make_obu(OBU_TEMPORAL_DELIMITER, {});
      const std::vector<uint8_t> sef = pipe.show_existing(hint);
      tu.insert(tu.end(), sef.begin(), sef.end());
      out[grouped].push_back(tu);
      recons[grouped].push_back(anchor);
    }
  }
  CHECK(out[0] == out[1]);
  const auto dec = decode_frames(out[1]);
  CHECK_EQ(dec.size(), recons[1].size());
  for (size_t i = 0; i < dec.size(); i++) CHECK_EQ(first_diff(dec[i], recons[1][i]), std::string());
  return 0;
}

int main() {
  int fails = 0;
  fails += RUN(bframes_bitexact);
  fails += RUN(bframe_group_matches_sequential);
  fails += RUN(ippp_multi_ref_bitexact);
  fails += RUN(ippp_bitexact);
  fails += RUN(deblock_bitexact_and_smoother);
  fails += RUN(ippp_quality_holds);
  fails += RUN(ippp_edge_motion_bitexact);
  fails += RUN(ippp_tiles_bitexact);
  fails += RUN(pipeline_matches_sync);
  fails += RUN(cpu_pipeline_matches_gpu);
  fails += RUN(encode_fails_when_gate_fails);
  fails += RUN(tile_threads_counts);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
