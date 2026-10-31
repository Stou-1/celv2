// Regression tests of the decoder mirror that do not depend on the GPU heuristics: fixed-seed
// random partition trees, intra modes, intra/inter choices and integer MVs (pointing outside the
// frame near edges), over odd/tiny sizes, qindex extremes, keyframe intervals and order-hint wrap.
#include <cstdio>
#include <string>
#include <vector>

#include "bitstream/obu.h"
#include "check.h"
#include "core/encoder.h"
#include "test_frames.h"
#include "verify/avm_decode.h"

static uint32_t g_seed = 1;
static int rnd(int n) {
  g_seed = g_seed * 1664525u + 1013904223u;
  return static_cast<int>((g_seed >> 16) % n);
}

static DecisionMap random_dm(int w, int h, bool inter, int mv_range_px, int min_size, int refs = 1) {
  DecisionMap dm = uniform_decision(w, h, 64);
  for (auto& sb : dm.sbs) {
    sb.tree.clear();
    sb.blocks.clear();
    auto rec = [&](auto&& self, int bw, int bh) -> void {
      Part p = static_cast<Part>(rnd(4));
      if ((p == Part::Horz || p == Part::Split) && bh <= min_size) p = Part::None;
      if ((p == Part::Vert || p == Part::Split) && bw <= min_size) p = Part::None;
      sb.tree.push_back(p);
      if (p == Part::None) {
        BlockDecision b;
        b.w4 = static_cast<uint8_t>(bw / 4);
        b.h4 = static_cast<uint8_t>(bh / 4);
        b.intra = static_cast<IntraMode>(rnd(kIntraModes));
        b.inter = inter && rnd(4) != 0;
        if (b.inter) {
          b.mv_col = static_cast<int16_t>((rnd(2 * mv_range_px + 1) - mv_range_px) * 8);
          b.mv_row = static_cast<int16_t>((rnd(2 * mv_range_px + 1) - mv_range_px) * 8);
          b.ref = static_cast<uint8_t>(rnd(refs));  // may exceed the references a frame has yet
          if (refs > 1 && rnd(3) == 0) {  // compound; a pair the frame cannot code falls back to single
            b.ref2 = static_cast<uint8_t>(rnd(refs));
            b.mv2_col = static_cast<int16_t>((rnd(2 * mv_range_px + 1) - mv_range_px) * 8);
            b.mv2_row = static_cast<int16_t>((rnd(2 * mv_range_px + 1) - mv_range_px) * 8);
          }
        }
        sb.blocks.push_back(b);
      } else if (p == Part::Horz) {
        self(self, bw, bh / 2), self(self, bw, bh / 2);
      } else if (p == Part::Vert) {
        self(self, bw / 2, bh), self(self, bw / 2, bh);
      } else {
        for (int i = 0; i < 4; i++) self(self, bw / 2, bh / 2);
      }
    };
    rec(rec, 64, 64);
  }
  return dm;
}

// Encodes `frames` frames with random decisions; every frame must decode bit-exactly.
static int run(int w, int h, int q, int frames, int keyint, bool inter, int mv_range, int min_size,
               int refs = 1, bool mv_tools = false) {
  SeqConfig c{w, h, 30, 1, q};
  c.refs = refs;
  c.tmv = mv_tools;  // temporal MVs (block MV precision is on by default)
  Encoder enc(c);
  std::vector<std::vector<uint8_t>> tus;
  std::vector<std::string> md5;
  for (int i = 0; i < frames; i++) {
    const bool key = i == 0 || (keyint > 0 && i % keyint == 0);
    const Frame420 src = scene_frame(w, h, 3 * i, i, 1 + i % 3);
    tus.push_back(enc.encode(src, random_dm(w, h, !key && inter, mv_range, min_size, refs), key));
    md5.push_back(frame_md5(enc.last_recon()));
  }
  const auto dec = decode_frame_md5s(tus);
  CHECK_EQ(dec.size(), md5.size());
  for (size_t i = 0; i < md5.size(); i++) {
    if (dec[i] != md5[i]) std::fprintf(stderr, "%dx%d q%d: frame %zu mismatch\n", w, h, q, i);
    CHECK_EQ(dec[i], md5[i]);
  }
  return 0;
}

static int random_inter_sizes_bitexact() {
  g_seed = 7;
  CHECK_EQ(run(202, 122, 120, 6, 0, true, 40, 4), 0);
  CHECK_EQ(run(201, 121, 120, 6, 0, true, 40, 4), 0);
  CHECK_EQ(run(16, 16, 120, 6, 0, true, 20, 4), 0);
  CHECK_EQ(run(640, 360, 90, 4, 0, true, 56, 8), 0);
  return 0;
}

static int random_inter_qindex_extremes_bitexact() {
  g_seed = 8;
  CHECK_EQ(run(202, 122, 1, 4, 0, true, 8, 4), 0);
  CHECK_EQ(run(202, 122, 255, 4, 0, true, 8, 4), 0);
  return 0;
}

static int long_sequences_bitexact() {  // order-hint wrap (7 bits), mid-stream keyframes
  g_seed = 9;
  CHECK_EQ(run(32, 32, 120, 300, 0, true, 8, 8), 0);
  CHECK_EQ(run(32, 32, 120, 300, 50, true, 8, 8), 0);
  CHECK_EQ(run(16, 16, 200, 200, 7, false, 0, 8), 0);
  return 0;
}

static int random_multi_ref_bitexact() {
  g_seed = 10;
  CHECK_EQ(run(202, 122, 120, 10, 0, true, 24, 4, 2), 0);
  CHECK_EQ(run(202, 122, 120, 10, 0, true, 24, 4, 3), 0);
  CHECK_EQ(run(201, 121, 90, 12, 5, true, 24, 4, 3), 0);  // references rebuilt after each keyframe
  CHECK_EQ(run(64, 64, 120, 20, 0, true, 16, 8, 7), 0);
  CHECK_EQ(run(32, 32, 120, 300, 0, true, 8, 8, 4), 0);   // order-hint wrap with 4 references
  return 0;
}

// Scenes A, B (A shifted by 37, 23), A, B...: with 2 references, inter blocks at MV 0 predicting
// from reference 1 copy the frame two back, the same scene, so from the third frame on the residual,
// and the frame, is far smaller than with reference 0 (the other scene). (Skip mode, which predicts
// from both references, lets some reference-0 blocks reach the same scene too, so "far" is 1.5x.)
static int second_reference_is_two_frames_back() {
  const int w = 128, h = 96;
  size_t bytes[2][6] = {};
  for (int ref = 0; ref < 2; ref++) {
    SeqConfig c{w, h, 30, 1, 40};
    c.refs = 2;
    Encoder enc(c);
    std::vector<std::vector<uint8_t>> tus;
    std::vector<Frame420> recons;
    for (int i = 0; i < 6; i++) {
      const Frame420 src = i % 2 ? scene_frame(w, h, 37, 23) : scene_frame(w, h);
      DecisionMap dm = uniform_decision(w, h, 64);
      // 32x32 blocks (inter after the keyframe): the reference matters here, not what a 64-point
      // transform leaves out.
      for (auto& sb : dm.sbs) {
        BlockDecision b = sb.blocks[0];
        b.inter = i > 0, b.ref = static_cast<uint8_t>(i > 0 ? ref : 0), b.w4 = 8, b.h4 = 8;
        sb.tree = {Part::Split, Part::None, Part::None, Part::None, Part::None};
        sb.blocks.assign(4, b);
      }
      tus.push_back(enc.encode(src, dm, i == 0));
      recons.push_back(enc.last_recon());
      bytes[ref][i] = tus.back().size();
    }
    const auto dec = decode_frames(tus);
    CHECK_EQ(dec.size(), recons.size());
    for (size_t i = 0; i < dec.size(); i++) CHECK_EQ(first_diff(dec[i], recons[i]), std::string());
  }
  for (int i = 2; i < 6; i++) {
    std::fprintf(stderr, "  frame %d: ref 0 %zu bytes, ref 1 %zu bytes\n", i, bytes[0][i], bytes[1][i]);
    CHECK(bytes[1][i] * 3 < bytes[0][i] * 2);
  }
  return 0;
}

// A compound block's MVs belong to their references, whichever of the pair is d.ref: {ref 1, MV a;
// ref 0, MV b} and {ref 0, MV b; ref 1, MV a} are the same prediction and the same bytes.
static int compound_mvs_follow_their_references() {
  const int w = 128, h = 96;
  std::vector<uint8_t> tu[2];
  for (int order = 0; order < 2; order++) {
    SeqConfig c{w, h, 30, 1, 90};
    c.refs = 2;
    Encoder enc(c);
    std::vector<std::vector<uint8_t>> tus;
    for (int i = 0; i < 3; i++) {
      DecisionMap dm = uniform_decision(w, h, 64);
      if (i == 2)
        for (auto& sb : dm.sbs)
          for (auto& b : sb.blocks) {
            b.inter = true;
            b.ref = order ? 0 : 1, b.ref2 = order ? 1 : 0;
            b.mv_col = order ? -24 : 16, b.mv2_col = order ? 16 : -24;  // ref 1: +2 px, ref 0: -3 px
          }
      else if (i == 1)
        for (auto& sb : dm.sbs)
          for (auto& b : sb.blocks) b.inter = true;
      tus.push_back(enc.encode(scene_frame(w, h, 3 * i, i), dm, i == 0));
    }
    tu[order] = tus[2];
    CHECK_EQ(decode_frames(tus).size(), 3u);
  }
  CHECK(tu[0] == tu[1]);
  return 0;
}

// Mini-GOPs of a hidden reference frame and 2 shown B frames, past the 7-bit order hint (128):
// display order hints keep counting up and each frame's nearest reference is ranked first.
static int bframes_display_hints_past_order_hint_wrap() {
  const int w = 32, h = 32, n = 160;
  SeqConfig c{w, h, 30, 1, 120};
  c.refs = 4;
  Encoder enc(c);
  std::vector<std::vector<uint8_t>> tus;
  int bad = 0;
  auto code = [&](const FrameSpec& f) {
    const std::vector<int> hints = enc.begin(f);
    bool ok = enc.display_order_hint() == f.display;
    for (int hint : hints) ok = ok && (hint <= hints[0] || hint > f.display);  // nearest past one first
    if (!ok && bad++ < 3) std::fprintf(stderr, "  display %d: hint %d, refs from %d\n", f.display, enc.display_order_hint(), hints.empty() ? -1 : hints[0]);
    return enc.code(scene_frame(w, h, f.display, f.display), uniform_decision(w, h, 64));
  };
  auto tu = [](const std::vector<uint8_t>& prefix, const std::vector<uint8_t>& obus) {
    std::vector<uint8_t> t = make_obu(OBU_TEMPORAL_DELIMITER, {});
    t.insert(t.end(), prefix.begin(), prefix.end());
    t.insert(t.end(), obus.begin(), obus.end());
    return t;
  };
  tus.push_back(tu({}, code(FrameSpec{true, 0})));
  for (int d = 3; d < n; d += 3) {
    const std::vector<uint8_t> anchor = code(FrameSpec{false, d, false, true});
    const int hint = enc.display_order_hint();
    tus.push_back(tu(anchor, code(FrameSpec{false, d - 2, true, false})));
    tus.push_back(tu({}, code(FrameSpec{false, d - 1, true, false})));
    tus.push_back(tu({}, enc.show_existing(hint)));
  }
  CHECK_EQ(bad, 0);
  CHECK_EQ(decode_frames(tus).size(), tus.size());
  return 0;
}

// Random qindex offsets per 64x64 superblock (coded as a delta qindex per 128x128 superblock, the
// mean of the four rounded to the step), on keyframes and inter frames, near both qindex ends: every
// frame decodes bit-exactly, and the offsets change the bytes.
static int random_superblock_qindex_bitexact() {
  g_seed = 11;
  const int w = 330, h = 202, cells = ((w + 63) / 64) * ((h + 63) / 64);
  for (int q : {2, 120, 254}) {
    size_t bytes[2] = {};
    for (int with = 0; with < 2; with++) {
      SeqConfig c{w, h, 30, 1, q};
      c.refs = 2;
      Encoder enc(c);
      std::vector<std::vector<uint8_t>> tus;
      std::vector<std::string> md5;
      for (int i = 0; i < 5; i++) {
        FrameSpec f{i == 0};
        if (with)
          for (int k = 0; k < cells; k++) f.sb_dq.push_back(rnd(81) - 40);
        enc.begin(f);
        std::vector<uint8_t> tu = make_obu(OBU_TEMPORAL_DELIMITER, {});
        const std::vector<uint8_t> obus = enc.code(scene_frame(w, h, 3 * i, i, 1 + i % 3), random_dm(w, h, i > 0, 24, 4, 2));
        tu.insert(tu.end(), obus.begin(), obus.end());
        bytes[with] += tu.size();
        tus.push_back(tu);
        md5.push_back(frame_md5(enc.last_recon()));
      }
      const auto dec = decode_frame_md5s(tus);
      CHECK_EQ(dec.size(), md5.size());
      for (size_t i = 0; i < md5.size(); i++) {
        if (dec[i] != md5[i]) std::fprintf(stderr, "q%d offsets %d: frame %zu mismatch\n", q, with, i);
        CHECK_EQ(dec[i], md5[i]);
      }
    }
    CHECK(bytes[0] != bytes[1]);
  }
  return 0;
}

// Temporal MVs (--tmv) on top of per-block MV precision, with random MVs and references.
static int random_mv_tools_bitexact() {
  g_seed = 12;
  CHECK_EQ(run(202, 122, 120, 10, 0, true, 40, 4, 3, true), 0);
  CHECK_EQ(run(201, 121, 30, 8, 5, true, 64, 4, 3, true), 0);
  CHECK_EQ(run(202, 122, 250, 8, 0, true, 64, 4, 2, true), 0);
  return 0;
}

int main() {
  int fails = 0;
  fails += RUN(random_mv_tools_bitexact);
  fails += RUN(random_superblock_qindex_bitexact);
  fails += RUN(compound_mvs_follow_their_references);
  fails += RUN(bframes_display_hints_past_order_hint_wrap);
  fails += RUN(random_inter_sizes_bitexact);
  fails += RUN(random_inter_qindex_extremes_bitexact);
  fails += RUN(long_sequences_bitexact);
  fails += RUN(random_multi_ref_bitexact);
  fails += RUN(second_reference_is_two_frames_back);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
