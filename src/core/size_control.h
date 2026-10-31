// --target-size: one qindex for the whole video from the --pass 1 statistics, then moved by at most
// 2 a mini-GOP as the coded bytes come in. A frame's payload at qindex q is taken as its pass-1
// payload x 2^(-(q - pass-1 qindex) / 24) x the ratio of the coded bytes so far to that estimate.
#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <vector>

#include "io/stats.h"

namespace celv2 {

class SizeControl {
 public:
  SizeControl(PassStats s, uint64_t budget, int workers = 1);  // budget: the whole IVF file's bytes
  // The finest qindex (1..255) whose estimate of the whole file is at most 98% of budget; 255 when none.
  static int start_qindex(const PassStats& s, uint64_t budget);
  int qindex() const { return q_; }  // the base qindex of the next frames
  // One IVF frame written (its payload), in file order; std::runtime_error past the statistics' frames.
  void coded(uint64_t payload);
  void update();  // after a mini-GOP (or a lone frame): the finest qindex within 2 of now that fits
  size_t frames() const { return done_; }
  bool coarsest() const { return done_ > 0 && finest_ >= 255; }  // every frame coded at qindex 255: no budget can be met above

  // Chunks (--workers), thread-safe. Chunk k sees only the chunks up to k - workers (the caller starts
  // it once they are done) and itself, so the qindexes do not depend on thread timing. The model is
  // pass-1 bytes x r x 2^(-(q - pass-1 qindex) / s), r and s fitted to the frames it sees.
  // The base qindex chunk k starts at: the first `workers` chunks the start qindex + 0, +1, -1, +2 ...
  // (frames at different qindexes early, for the slope), later ones the finest that fits.
  int chunk_start_qindex(size_t chunk, size_t first_frame, size_t frames, int at_least = 1);
  void forget(size_t chunk);  // drops what the chunk coded (it is coded again)
  void coded(size_t chunk, size_t frame, uint64_t payload, int qindex);  // any order
  int chunk_update(size_t chunk);  // the finest qindex within 2 of the chunk's that fits
  double slope() const;
  void set_workers(int workers) { workers_ = workers < 1 ? 1 : workers; }  // before any chunk call  // s over every coded frame: 12..48, 24 without 8 frames at each of 2 qindexes

 private:
  PassStats s_;
  uint64_t budget_;
  int q_;
  size_t done_ = 0;
  int finest_ = 256;  // the lowest qindex a frame was coded at
  struct Rec {
    size_t chunk, frame;
    uint64_t payload;
    int q;
  };
  struct Fit {
    double s, log2r;
  };
  Fit fit(const std::function<bool(const Rec&)>& seen) const;
  int finest_fitting(size_t chunk, int lo, int hi) const;  // with mu_ held
  int workers_;
  std::vector<Rec> recs_;
  std::map<size_t, int> chunk_q_;
  mutable std::mutex mu_;
  double used_ = 0, expected_ = 0;  // coded payload, and its pass-1 estimate at the qindex used
};

}  // namespace celv2
