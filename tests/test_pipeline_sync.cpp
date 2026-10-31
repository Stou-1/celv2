// Pipeline::sync_from: a pipeline that takes another's state codes the next frames as that one would.
#include <cstdio>
#include <string>
#include <vector>

#include "check.h"
#include "pipeline.h"
#include "test_frames.h"
#include "verify/avm_decode.h"

namespace {

struct Coded {
  std::vector<std::vector<uint8_t>> obus;
  std::vector<Frame420> recons;
};

// A mini-GOP after the frame at display `base`: the anchor at base + 4 (hidden, a reference, qindex
// offset anchor_q), the B frames before it shown (with group, at once: Pipeline::code_group, only
// the last keeping a slot), then the anchor shown.
Coded minigop(Pipeline& p, int base, int anchor_q = -16, bool group = false) {
  Coded c;
  auto frame = [&](int d) { return scene_frame(64, 48, 2 * d, d); };
  const Frame420 anchor = frame(base + 4);
  c.obus.push_back(p.code(anchor, FrameSpec{false, base + 4, false, true, anchor_q}));
  c.recons.push_back(p.last_recon());
  const int hint = p.display_order_hint();
  if (group) {
    std::vector<Frame420> srcs;
    std::vector<FrameSpec> specs;
    for (int d = base + 1; d < base + 4; d++) srcs.push_back(frame(d)), specs.push_back(FrameSpec{false, d, true, false, 16, d == base + 3});
    std::vector<const Frame420*> ptrs;
    for (const Frame420& f : srcs) ptrs.push_back(&f);
    std::vector<Frame420> rs;
    for (auto& o : p.code_group(ptrs, specs, &rs)) c.obus.push_back(o);
    for (auto& r : rs) c.recons.push_back(r);
  } else {
    for (int d = base + 1; d < base + 4; d++) {
      c.obus.push_back(p.code(frame(d), FrameSpec{false, d, true, false, 16}));
      c.recons.push_back(p.last_recon());
    }
  }
  c.obus.push_back(p.show_existing(hint));
  return c;
}

int same(const Coded& a, const Coded& b) {
  CHECK_EQ(a.obus.size(), b.obus.size());
  for (size_t i = 0; i < a.obus.size(); i++) CHECK(a.obus[i] == b.obus[i]);
  for (size_t i = 0; i < a.recons.size(); i++) CHECK_EQ(first_diff(a.recons[i], b.recons[i]), std::string());
  return 0;
}

int pipeline_sync_codes_alike() {
  for (Device dev : {Device::Cpu, Device::Auto}) {
    SeqConfig c{64, 48, 30, 1, 120};
    c.refs = 7;
    Pipeline a(c, dev), b(c, dev);
    a.code(scene_frame(64, 48), FrameSpec{true, 0});
    minigop(a, 0);
    b.sync_from(a);
    CHECK_EQ(same(minigop(a, 4), minigop(b, 4)), 0);
    // b goes its own way (other anchors, long enough to reuse the buffers that held a's frames),
    // then takes a's state again.
    for (int base = 8; base < 48; base += 4) minigop(b, base, -40);
    b.sync_from(a);
    CHECK_EQ(same(minigop(a, 8), minigop(b, 8)), 0);
    CHECK_EQ(same(minigop(a, 12), minigop(b, 12)), 0);
    // The same with the B frames coded at once (celv2enc's --parallel-b).
    for (int base = 16; base < 56; base += 4) minigop(b, base, -40, true);
    b.sync_from(a);
    for (int base = 16; base < 28; base += 4) CHECK_EQ(same(minigop(a, base, -16, true), minigop(b, base, -16, true)), 0);
  }
  return 0;
}

// The sequence qindex moved by set_qindex_offset codes as a Pipeline made at that qindex: frame by
// frame (code and code_group), the RD weight, and sync_from takes the offset along.
int pipeline_qindex_offset() {
  for (Device dev : {Device::Cpu, Device::Auto}) {
    SeqConfig c130{64, 48, 30, 1, 130}, c120{64, 48, 30, 1, 120};
    c130.refs = c120.refs = 7;
    Pipeline a(c130, dev), b(c120, dev), c(c120, dev);
    b.set_qindex_offset(10);
    CHECK_EQ(b.qindex_offset(), 10);
    CHECK(a.code(scene_frame(64, 48), FrameSpec{true, 0}) == b.code(scene_frame(64, 48), FrameSpec{true, 0}));
    CHECK(a.sse_per_bit() == b.sse_per_bit());
    CHECK_EQ(same(minigop(a, 0), minigop(b, 0)), 0);
    CHECK_EQ(same(minigop(a, 4, 0, true), minigop(b, 4, 0, true)), 0);
    c.sync_from(b);
    CHECK_EQ(c.qindex_offset(), 10);
    CHECK_EQ(same(minigop(b, 8), minigop(c, 8)), 0);
  }
  return 0;
}

// A chunk from its keyframe on a fresh Pipeline codes as on the pipeline that coded the frames
// before it (--workers): the keyframe at display 9 and the mini-GOP after it, frame by frame.
int pipeline_fresh_at_keyframe() {
  for (Device dev : {Device::Cpu, Device::Auto}) {
    SeqConfig c{64, 48, 30, 1, 120};
    c.refs = 7;
    Pipeline a(c, dev), b(c, dev);
    a.code(scene_frame(64, 48), FrameSpec{true, 0});
    minigop(a, 0);
    minigop(a, 4);
    const Frame420 key = scene_frame(64, 48, 18, 9);
    CHECK(a.code(key, FrameSpec{true, 9}) == b.code(key, FrameSpec{true, 9}));
    CHECK_EQ(same(minigop(a, 9), minigop(b, 9)), 0);
    CHECK_EQ(same(minigop(a, 13, -16, true), minigop(b, 13, -16, true)), 0);
  }
  return 0;
}

}  // namespace

int main() {
  int fails = 0;
  fails += RUN(pipeline_sync_codes_alike);
  fails += RUN(pipeline_qindex_offset);
  fails += RUN(pipeline_fresh_at_keyframe);
  if (fails) std::fprintf(stderr, "%d test(s) failed\n", fails);
  return fails ? 1 : 0;
}
