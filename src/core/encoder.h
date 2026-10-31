// CPU encoding loop: follows the DecisionMap, predicts and reconstructs exactly like the AVM
// decoder, and writes the symbols through BlockWriter. Tiles are coded in parallel.
#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "bitstream/block_writer.h"
#include "bitstream/seq_header.h"
#include "gpu/decision_map.h"
#include "io/y4m.h"
#include "state/codec_state.h"

class Encoder {
 public:
  explicit Encoder(const SeqConfig& c);
  // One temporal unit (OBU sequence) with one shown reference frame: begin({key}), a temporal
  // delimiter and code(). KEY units carry the sequence header.
  std::vector<uint8_t> encode(const Frame420& src, const DecisionMap& dm, bool key,
                              const RowGate* gate = nullptr);
  // Sets up the next frame and returns its references' display order hints in AV2's ranking
  // (DecisionMap reference r is the r-th), so the decision can search the same frames.
  std::vector<int> begin(const FrameSpec& f);
  // Takes o's reference frames and counters (CodecState::sync_refs_from): the next frame begun
  // here codes as it would on o.
  void sync_from(const Encoder& o) { st_->sync_refs_from(*o.st_); }
  // Codes the frame begin() set up: its OBUs (a keyframe's preceded by the sequence header),
  // without a temporal delimiter. Each superblock row of dm is read once gate publishes it
  // (nullptr: every row is ready); a failed gate throws.
  std::vector<uint8_t> code(const Frame420& src, const DecisionMap& dm, const RowGate* gate = nullptr);
  // The show-existing-frame OBU that outputs the hidden frame with this display order hint.
  std::vector<uint8_t> show_existing(int display_order_hint);
  // Threads encode() codes tiles on: min(tiles, cores, cfg.threads > 0 ? cfg.threads : tiles).
  int tile_threads() const;
  int tile_rows() const;  // tile rows of the frame (AV2 may have adjusted SeqConfig's)
  const Frame420& last_recon() const { return st_->recon(); }
  // The display order hint of the frame begin() set up (as ref_display_hints report them).
  int display_order_hint() const { return static_cast<int>(st_->cm()->current_frame.display_order_hint); }
  // coef_scale() of a luma transform (for the CPU RD pass).
  double luma_scale(TX_SIZE tx);
  // The RD weight of a bit in squared sample error at a coded qindex: 0.025 squared quantizer steps
  // (as the CCSO search weighs its header bits).
  double sse_per_bit(int coded_qindex);

 private:
  friend class TileCoder;  // the per-tile block walk (encoder.cpp)
  // Forward-transform scale that matches AVM's inverse transform (measured once per tx size,
  // with the calling tile's xd; shared by all tiles).
  double coef_scale(int plane, TX_SIZE tx, MACROBLOCKD* xd);

  SeqConfig cfg_;
  std::unique_ptr<CodecState> st_;
  bool key_ = false;  // the frame begin() set up
  std::mutex scale_mu_;
  std::atomic<double> scale_[2][TX_SIZES_ALL] = {};  // 0 until measured; read without the lock
};
