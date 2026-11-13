#include <cstdio>
#include <vector>

#include "check.h"
#include "core/encoder.h"
#include "core/rd_pass.h"
#include "gpu/sb_blocks.h"
#include "test_frames.h"

// A flat keyframe superblock (its neighbours flat too) whose 32x32 squares have DC as the engine's intra
// mode and whose smaller blocks have V: DC predicts the squares exactly, so each stays one block.
static int flat_dc_squares_stay_whole() {
  Frame420 f = make_frame(128, 128);
  for (auto& p : f.y) p = 100;
  pad_frame(f);
  SbDecision sb;
  sb.cands.resize(celv2::kSbBlocks);
  for (int b = 0; b < celv2::kSbBlocks; b++) {
    sb.cands[b].intra_cost = 1;
    sb.cands[b].intra = IntraMode::V;
  }
  for (int q = 0; q < 4; q++) sb.cands[celv2::sb_square_id(1, q & 1, q >> 1) * 5].intra = IntraMode::DC;
  RdPassParams p;
  p.qindex = 120;
  p.intra64 = false;
  for (double& s : p.scale) s = 1;
  rd_pass_sb(f, {}, 64, 64, p, sb);
  CHECK_EQ(sb.tree.size(), 5u);
  CHECK(sb.tree[0] == Part::Split);
  for (int q = 1; q < 5; q++) CHECK(sb.tree[q] == Part::None);
  for (const BlockDecision& d : sb.blocks) CHECK(d.intra == IntraMode::DC);
  return 0;
}

// Horizontal stripes (each row one value, the left neighbours too) whose blocks all have V as the
// engine's mode: H predicts them exactly, and the pass finds it only when it tries every mode.
static int every_intra_mode_finds_h() {
  for (int all = 0; all < 2; all++) {
    Frame420 f = make_frame(128, 128);
    for (int y = 0; y < 128; y++)
      for (int x = 0; x < 128; x++) f.y[y * f.stride_y() + x] = static_cast<uint8_t>(40 + (y * 37) % 160);
    pad_frame(f);
    SbDecision sb;
    sb.cands.resize(celv2::kSbBlocks);
    for (int b = 0; b < celv2::kSbBlocks; b++) {
      sb.cands[b].intra_cost = 1;
      sb.cands[b].intra = IntraMode::V;
    }
    RdPassParams p;
    p.qindex = 120;
    p.all_intra_modes = all != 0;
    p.intra64 = false;
    for (double& s : p.scale) s = 1;
    rd_pass_sb(f, {}, 64, 64, p, sb);
    for (const BlockDecision& d : sb.blocks) CHECK(d.intra == (all ? IntraMode::H : IntraMode::V));
  }
  return 0;
}

// A flat keyframe superblock but for its first 32x32 square, whose bottom half is two different
// flat 16x16 squares: the cheapest tree halves it, keeps the top half whole and splits the bottom
// half vertically (AV2 partitions each half again), not four quarters nor two whole halves.
static int a_half_splits_again() {
  Frame420 f = make_frame(128, 128);
  for (auto& p : f.y) p = 100;
  for (int y = 80; y < 96; y++)
    for (int x = 64; x < 96; x++) f.y[y * f.stride_y() + x] = static_cast<uint8_t>(x < 80 ? 40 : 200);
  pad_frame(f);
  SbDecision sb;
  sb.cands.resize(celv2::kSbBlocks);
  for (int b = 0; b < celv2::kSbBlocks; b++) {
    sb.cands[b].intra_cost = 1;
    sb.cands[b].intra = IntraMode::DC;
  }
  RdPassParams p;
  p.qindex = 120;
  p.intra64 = false;
  for (double& s : p.scale) s = 1;
  rd_pass_sb(f, {}, 64, 64, p, sb);
  const std::vector<Part> want = {Part::Split, Part::Horz, Part::None, Part::Vert, Part::None, Part::None,
                                  Part::None, Part::None, Part::None};
  CHECK(sb.tree == want);
  CHECK_EQ(sb.blocks.size(), 6u);
  return 0;
}

// A flat keyframe superblock with the engines' candidates (none for the 64x64 level's five blocks):
// with intra64 the pass scores those itself and keeps one 64x64 block, without it splits.
static int flat_superblock_one_intra64_block() {
  for (int on = 0; on < 2; on++) {
    Frame420 f = make_frame(128, 128);
    for (auto& p : f.y) p = 100;
    pad_frame(f);
    SbDecision sb;
    sb.cands.resize(celv2::kSbBlocks);
    for (int b = 5; b < celv2::kSbBlocks; b++) {
      sb.cands[b].intra_cost = 1;
      sb.cands[b].intra = IntraMode::DC;
    }
    RdPassParams p;
    p.qindex = 120;
    p.intra64 = on != 0;
    for (double& s : p.scale) s = 1;
    rd_pass_sb(f, {}, 64, 64, p, sb);
    CHECK(sb.tree[0] == (on ? Part::None : Part::Split));
    CHECK_EQ(sb.blocks.size(), on ? 1u : 4u);
    CHECK_EQ(sb.blocks[0].w4, on ? 16 : 8);
  }
  return 0;
}

// An inter frame's side prices (Encoder::side_costs()): the 8x4 / 4x8 halves of an 8x8 code no partition symbol
// (NONE their only one: 0, where a 16x8's is coded) and an inter block of theirs only skip_txfm 0, less than the
// is_inter and skip_txfm flags of larger ones.
static int side_costs_of_8x4() {
  const int w = 128, h = 128;
  Encoder enc(SeqConfig{w, h, 30, 1, 120});
  enc.encode(scene_frame(w, h), uniform_decision(w, h, 64), true);
  enc.begin(FrameSpec{});
  SideCosts t;
  enc.side_costs(t);
  CHECK_EQ(t.part_half[3][0][0], 0);
  CHECK_EQ(t.part_half[3][1][0], 0);
  CHECK(t.part_half[2][0][0] > 0 && t.part_half[2][1][0] > 0);
  CHECK(t.skip_txfm > 0 && t.skip_txfm < t.inter_block);
  return 0;
}

// The pass prices an inter frame's 8x4 / 4x8 blocks by skip_txfm alone (inter_side()): with an inter block's flags
// at 10000 bits and every other side price 0, a picture its reference predicts exactly takes 8x4s throughout (128
// blocks), not one 64x64 block.
static int inter_8x4_flags_are_skip_txfm() {
  const Frame420 f = scene_frame(128, 128);
  SbDecision sb;
  sb.cands.resize(celv2::kSbBlocks);
  for (int b = 0; b < celv2::kSbBlocks; b++) {
    int x, y, bw, bh;
    celv2::sb_block_geom(b, &x, &y, &bw, &bh);
    sb.cands[b].inter_cost = 1;
    sb.cands[b].inter.inter = true;  // reference 0, MV 0
    sb.cands[b].inter.w4 = static_cast<uint8_t>(bw / 4), sb.cands[b].inter.h4 = static_cast<uint8_t>(bh / 4);
  }
  SideCosts t;
  t.inter_block = 10000 * 512;
  RdPassParams p;
  p.qindex = 120;
  p.block_bits = 0;
  p.side = &t;
  for (double& s : p.scale) s = 1;
  rd_pass_sb(f, {&f}, 64, 64, p, sb);
  CHECK_EQ(sb.blocks.size(), 128u);
  for (const BlockDecision& d : sb.blocks) CHECK(d.inter && d.w4 * d.h4 == 2);
  return 0;
}

int main() {
  int fails = 0;
  fails += RUN(flat_dc_squares_stay_whole);
  fails += RUN(every_intra_mode_finds_h);
  fails += RUN(a_half_splits_again);
  fails += RUN(flat_superblock_one_intra64_block);
  fails += RUN(side_costs_of_8x4);
  fails += RUN(inter_8x4_flags_are_skip_txfm);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
