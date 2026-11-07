// One encoding pipeline: a decision engine (GPU or CPU) + CPU encoder, frame by frame (used by
// celv2enc and tests).
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "core/encoder.h"
#include "gpu/decision_engine.h"

// Vulkan / Vulkan2: the Vulkan engine on the first / second discrete GPU. Gpu: Vulkan under the name
// saved settings and scripts use. Cuda: the CUDA engine, kept for one release to compare with.
enum class Device { Auto, Gpu, Cuda, Cpu, Vulkan, Vulkan2 };

// Auto: the Vulkan engine on the first discrete GPU, else CpuEngine. Gpu: that engine, or
// std::runtime_error ("no usable GPU: <reason>"). Cuda: GpuEngine, or std::runtime_error ("no usable
// CUDA GPU: <reason>") if probe() fails; only Cuda calls probe. Cpu: CpuEngine.
// description receives the one-line choice ("decisions on Vulkan / CUDA / CPU (...)").
std::unique_ptr<DecisionEngine> make_engine(Device device, int width, int height, int tile_threads,
                                            const std::function<ProbeResult()>& probe,
                                            std::string* description, int engines = 1);

// The decision devices this machine can use, as (--device value, name): the first two Vulkan GPUs
// ("vulkan", "vulkan2", "<name> (Vulkan)") in a build with Vulkan, then ("cpu", "CPU").
std::vector<std::pair<std::string, std::string>> list_devices();

class Pipeline {
 public:
  explicit Pipeline(const SeqConfig& c, Device device = Device::Auto);
  // A temporal unit with one shown reference frame: a temporal delimiter and code(src, {key}).
  std::vector<uint8_t> encode(const Frame420& src, bool key);
  // Codes src as spec and returns its OBUs (no temporal delimiter). KEY frames use the intra
  // decision; others search the reconstructions of the references Encoder::begin() reports (what
  // the decoder predicts from), in AV2's ranking.
  // next (optional): the frame coded after this one (FrameSpec::display next_display); what of its
  // decision needs no references is made while this one is coded.
  std::vector<uint8_t> code(const Frame420& src, const FrameSpec& spec, const Frame420* next = nullptr,
                            int next_display = -1);
  // Codes frames that depend on none of each other, as code() would one after the other: all but
  // the last must be non-reference frames that keep no slot (FrameSpec::keep_shown false). The
  // last is coded on this pipeline's state, the others at the same time on copies of it
  // (Encoder::sync_from); the engine decides them in turn. Returns each frame's OBUs; recons, when
  // given, gets each frame's reconstruction.
  std::vector<std::vector<uint8_t>> code_group(const std::vector<const Frame420*>& srcs,
                                               const std::vector<FrameSpec>& specs,
                                               std::vector<Frame420>* recons = nullptr);
  // The OBU that outputs the hidden frame with this display order hint.
  std::vector<uint8_t> show_existing(int display_order_hint) { return enc_.show_existing(display_order_hint); }
  // The display order hint of the last coded frame.
  int display_order_hint() const { return enc_.display_order_hint(); }
  const Frame420& last_recon() const { return enc_.last_recon(); }
  const std::string& engine_description() const { return description_; }
  // Takes o's state (the encoder's references and counters, the kept reconstructions): the next
  // frames coded here code as they would on o. Both must have the same SeqConfig.
  void sync_from(const Pipeline& o);
  // The RD weight of one bit in squared sample error at the sequence qindex (Encoder::sse_per_bit).
  double sse_per_bit() { return enc_.sse_per_bit(cfg_.coded_qindex(std::max(1, std::min(255, cfg_.qindex + qindex_offset_)))); }
  // Added to every frame's FrameSpec::qindex_delta from now on: codes as a Pipeline made at the
  // sequence qindex + d (--target-size moves it between mini-GOPs). sync_from takes it along.
  void set_qindex_offset(int d) { qindex_offset_ = d; }
  int qindex_offset() const { return qindex_offset_; }

 private:
  SeqConfig cfg_;
  int qindex_offset_ = 0;
  Encoder enc_;
  std::string description_;  // before engine_: make_engine writes it
  std::unique_ptr<DecisionEngine> engine_;
  // A reconstruction by display order hint, with a key unique in this pipeline (the engine's
  // DecisionEngine::launch ref_keys).
  struct Recon {
    int doh;
    long long key;
    Frame420 f;
  };
  // Reconstructions of the reference frames since the keyframe (the newest 8 + cfg.refs; the
  // slots hold at most the last cfg.refs).
  std::vector<Recon> recons_;
  // The last shown frame that is not a reference (it holds the slots above the turns, see
  // CodecState::begin_frame); display order hint -1: none since the keyframe.
  Recon last_shown_{-1, -1, Frame420{}};
  std::vector<std::unique_ptr<Encoder>> shadows_;  // code_group's copies of enc_'s state

  // The reconstructions (and keys) of the references Encoder::begin() reported.
  void references(const std::vector<int>& hints, std::vector<const Frame420*>& refs, std::vector<long long>& keys) const;
  // Keeps the reconstruction of the frame just coded as spec where later frames may predict from it.
  void keep(const FrameSpec& spec);
  // engine_->collect(dm, gate), with the CPU RD pass (core/rd_pass.h) on every superblock before
  // its row reaches gate when cfg_.rd_pass is on, each superblock at qindex plus its sb_dq offset.
  void collect(DecisionMap& dm, RowGate& gate, const Frame420& src, const std::vector<const Frame420*>& refs,
               int qindex, const std::vector<int>& sb_dq, const celv2::AqMap* map = nullptr, bool screen = false,
               const celv2::IntrabcIndex* ibc = nullptr);
};
