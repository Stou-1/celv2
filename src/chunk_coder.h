// One chunk of a video (a keyframe to the frame before the next keyframe) coded as celv2enc's frame
// loop codes it: keyframes, lone frames and mini-GOPs (with the GOP trial on a second Pipeline).
// Chunks start at keyframes, so a ChunkCoder per chunk codes the same bytes as one coder for the
// whole video (tests/test_pipeline_sync.cpp pipeline_fresh_at_keyframe).
#pragma once
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "gpu/engine.h"
#include "io/y4m.h"
#include "pipeline.h"

struct ChunkFrame {
  Frame420 f;
  int display = 0;
  bool key = false;
};

// The frames read ahead of the coders, at most `limit` (the reader waits for room).
class FrameBudget {
 public:
  explicit FrameBudget(size_t limit) : limit_(limit) {}
  bool take();      // the reader, before it reads a frame; false once cancelled
  void give(size_t n);  // a coder, when frames are coded
  void cancel();    // wakes the reader for good (an error elsewhere)

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  size_t limit_, held_ = 0;
  bool cancelled_ = false;
};

// One chunk's frames in display order, key flags set: the reader pushes them, the coder takes them.
class FrameQueue {
 public:
  explicit FrameQueue(FrameBudget& budget) : budget_(budget) {}
  void push(ChunkFrame f);
  void close();  // no more frames in this chunk
  // The i-th frame not yet popped, waiting for it; nullptr past the chunk's end. It stays valid
  // until popped (pushes do not move it).
  const ChunkFrame* at(size_t i);
  void pop(size_t n);
  size_t size();  // frames not yet popped
  // Popped frames stay (and keep their budget) until rewind() makes them the next again: the last
  // chunk, re-coded when the whole file is over --target-size.
  void keep();
  void rewind();

 private:
  FrameBudget& budget_;
  size_t off_ = 0;  // popped but kept
  bool keep_ = false;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<ChunkFrame> q_;
  bool closed_ = false;
};

// One output frame: its temporal unit (temporal delimiter first), display index, PSNR (Y, U, V;
// with LoopConfig::psnr) and reconstruction as y4m frame bytes (with LoopConfig::recon).
struct CodedFrame {
  std::vector<uint8_t> tu;
  int display = 0;
  double psnr[3] = {0, 0, 0};
  std::vector<uint8_t> recon;
};

// celv2enc's options the frame loop reads, and what main derives from them.
struct LoopConfig {
  int bframes = 3;
  bool parallel_b = true;
  bool psnr = false, recon = false;
  bool pyramid = true, deltaq = false;
  int tplq = 0, bq = 16, aq = -16, mq = 0, kq = 0;
  int key_boost = 48, anchor_boost = 24, level_step = 8, level_step16 = -1;
  double gop_still = 0;
  bool gop_trial = false;
};

class ChunkCoder {
 public:
  ChunkCoder(const SeqConfig& cfg, Device device, const LoopConfig& lc);
  // Codes the chunk in `in` (its first frame a keyframe) to its end. after_group gets each
  // keyframe's, lone frame's or mini-GOP's output frames, in display order, as they are coded.
  void code(FrameQueue& in, const std::function<void(std::vector<CodedFrame>&)>& after_group);
  // The sequence qindex offset of what is coded next (Pipeline::set_qindex_offset), main and trial alike.
  void set_qindex_offset(int d);
  const std::string& engine_description() const { return pipe_->engine_description(); }

 private:
  SeqConfig cfg_;
  Device device_;
  LoopConfig lc_;
  std::unique_ptr<Pipeline> pipe_, trial_;  // trial_: --gop-trial's other way of coding a mini-GOP
  bool gop_trial_;  // off for the rest of a chunk when the trial pipeline cannot be made or fails
};
