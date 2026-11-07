// The state the AVM decoder holds while decoding a frame (AV2_COMMON, MACROBLOCKD, mode-info grid,
// frame buffers, contexts), rebuilt with av2/common functions so that context derivation and
// prediction in celv2 run the same code the decoder runs.
#pragma once
#include <array>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "bitstream/seq_header.h"
#include "core/intrabc_index.h"
#include "core/perceptual_aq.h"
#include "io/y4m.h"

extern "C" {
#include "av2/common/av2_common_int.h"
#include "av2/common/blockd.h"
#include "av2/common/thread_common.h"
#include "avm_util/avm_thread.h"
}

// How one frame is coded. display: its display position (the order hint before wrapping); -1 is
// the frame after the last coded one. A frame that is not shown is output later by a
// show-existing-frame OBU (Encoder::show_existing); a non-reference frame refreshes no slot.
// delta_q_res: the superblock qindex offsets are coded in steps of this (a power of 2, up to 8).
constexpr int kDeltaQRes = 4;

struct FrameSpec {
  bool key = false;
  int display = -1;
  bool shown = true;
  bool reference = true;
  int qindex_delta = 0;  // added to the sequence qindex for this frame (clamped to 1..255)
  // A shown non-reference frame takes the slots above the turns (display order hints are derived
  // from shown frames there); false: it refreshes no slot, so frames coded after it do not depend
  // on it (the B frames of a mini-GOP but its last).
  bool keep_shown = true;
  // Per 64x64 superblock (raster, the decision map's grid): the qindex offset of its blocks, coded
  // as a delta qindex per coded superblock (the 64x64 offsets under it averaged). Empty: none.
  std::vector<int> sb_dq;
  // The frame's --paq importance map, which --psy-rd weighs its blocks by (null: none).
  std::shared_ptr<const celv2::AqMap> map;
  // --screen: the frame's screen content tools (palette, IntraBC) on; needs SeqConfig::screen.
  bool screen = false;
  // With screen: where the frame's source repeats itself, for IntraBC.
  std::shared_ptr<const celv2::IntrabcIndex> ibc;
  int refresh = -1;  // the slots it refreshes, in place of the turns (X_L0; -1: the turns)
};

// Every 64x64 offset of sb_dq (FrameSpec's grid for a width x height frame) replaced by its coded
// superblock's (coded_sb_px square): the group's mean rounded to the nearest multiple of kDeltaQRes,
// as begin_frame() codes it (without its clamp to the qindex range). Empty stays empty.
void snap_sb_dq(std::vector<int>& sb_dq, int width, int height, int coded_sb_px);

#ifdef CELV2_ORACLE
namespace celv2::oracle {
struct Frame;
}
void wienerns_shape(const WienernsFilterParameters* prm, const int16_t* t, int* s, int* sym);
#endif

class CodecState {
 public:
  explicit CodecState(const SeqConfig& c);
  ~CodecState();
  CodecState(const CodecState&) = delete;
  CodecState& operator=(const CodecState&) = delete;

  AV2_COMMON* cm() { return cm_; }
  // Each reference's distance for the global motion syntax, set before the frame is coded (0: the
  // decoder reads no model for it: restricted, or the same instant).
  int gm_ref_dist[INTER_REFS_PER_FRAME] = {};
  // Tile t's block state; tiles are coded independently, so each has its own (tile 0 by default).
  MACROBLOCKD* xd(int t = 0) { return tiles_[t].xd; }
  int num_tiles() const { return static_cast<int>(tiles_.size()); }
  const SeqConfig& config() const { return cfg_; }

  // Sets cm to what the decoder holds after parsing our frame header and the tile group header.
  // Reference frame j after a keyframe refreshes slot j % refs and every slot from refs up, so the
  // slots hold the last `refs` reference frames, which AV2 ranks by display distance.
  void begin_frame(const FrameSpec& f);
  void begin_frame(bool key) { begin_frame(FrameSpec{key}); }
  // After begin_frame: the references' display order hints, in AV2's ranking (reference 0 first).
  std::vector<int> ref_display_hints() const;
  // The slot holding the frame with this display order hint, or -1.
  int slot_of(int display_order_hint) const;
  // Per-tile init the decoder does before tile t's first superblock (decode_tiles()).
  void begin_tile(int t);
  // Runs the in-loop filters (CDEF's strength and the restoration units chosen against src),
  // updates the reference slots and copies the reconstruction into recon(). sse_per_bit: what a
  // bit costs the CCSO search and the restoration choice, in squared error. Then, once the tiles
  // are range coded (lr_flags() in hand), save_contexts().
  void end_frame(const Frame420& src, double sse_per_bit);
  // Per restoration unit, raster, the planes restored as the tiles were coded one after the other: 1
  // where restore() kept the filter, 0 where not (what the tiles' deferred flags code,
  // SymbolLog::replay()), kLrSkip for a plane restore() left without restoration (no flag coded).
  const std::vector<uint8_t>& lr_flags() const { return lr_flags_; }
  static constexpr int kWienerTaps = 18;  // wienerns_filter_uv's
  static constexpr uint8_t kLrSkip = 2;
  // A WIENER_NONSEP plane's frame filter syntax (read_wienerns_framefilters_hdr()), one bit a byte.
  const std::vector<uint8_t>& lr_filter_bits(int p) const { return lr_filter_bits_[p]; }
  // Saves the tiles' adapted CDFs (as the range coder left them) into the frame.
  void save_contexts();
  // Makes this state's reference slots (the frames, with the contexts they carry), default frame
  // context and frame counters those of o, a state of the same SeqConfig, copying the frames this
  // one does not hold yet: a frame coded here next codes as on o, so frames that depend on none of
  // each other can be coded on two states at once.
  void sync_refs_from(const CodecState& o);
  Frame420& recon() { return recon_; }
  // The qindex the blocks of the coded superblock at (mi_row, mi_col) use (FrameSpec::sb_dq).
  int sb_qindex(int mi_row, int mi_col) const;
  // Tile t's motion-compensation scratch buffers (the decoder's ThreadData::mc_buf).
  uint16_t** mc_buf(int t = 0) { return tiles_[t].mc_buf; }

  // A superblock-row worker of the parallel coder: its own block state, CDFs (xd->tile_ctx) and
  // motion-compensation scratch, taken from a pool that grows to the rows coded at once.
  struct RowWorker {
    MACROBLOCKD* xd = nullptr;
    FRAME_CONTEXT* ctx = nullptr;
    uint16_t* mc_buf[2] = {nullptr, nullptr};
  };
  RowWorker* acquire_row_worker();
  void release_row_worker(RowWorker* w);
  // w's block state becomes tile t's as begin_tile(t) left it, with w's own CDFs (unset) and scratch.
  void begin_row(RowWorker& w, int t);
  // Tile t's two CDF hand-over buffers: superblock row r leaves its CDFs after its second
  // superblock in handoff(t, r & 1) for row r + 1 to start from.
  FRAME_CONTEXT* handoff(int t, int k) { return tiles_[t].handoff[k]; }

 private:
  struct Tile : RowWorker {  // ctx: TileDataDec::tctx
    FRAME_CONTEXT* handoff[2] = {nullptr, nullptr};
  };
  static void alloc_worker(RowWorker& w);
  static void free_worker(RowWorker& w);
  RefCntBuffer* free_buffer();
  void copy_frame_buffer(RefCntBuffer* d, const RefCntBuffer* s);
  int derive_display_order_hint(int order_hint) const;
  // read_tile_info(): uniform spacing at the configured counts (at least the level minimum).
  void setup_tiles();
  // The decoder's in-loop deblocking, row-parallel on lf_workers_ (av2_loop_filter_frame_mt
  // gives the same pixels as the single-threaded filter).
  void deblock();
  // cfg_.deblock_search: sets the frame's deblocking offset to the candidate whose deblocked frame
  // is nearest src (Y + U + V squared error); the frame is left as it was.
  void pick_deblock(const Frame420& src);
  std::vector<uint8_t> deblock_keep_;  // pick_deblock(): the frame before deblocking
  // Picks one CDEF strength pair for the frame (cm->cdef_info, written by the frame header) and
  // filters the deblocked frame with it, on lf_workers_ when there are several.
  void cdef(const Frame420& src);
  // Picks each plane's CCSO offsets (cm->ccso_info, written by the frame header) from luma, the
  // deblocked luma (padded as extend_ccso_border()), and applies them as the decoder does.
  void apply_ccso(const std::vector<uint16_t>& luma);
  void ccso(const Frame420& src, std::vector<uint16_t>& luma, double sse_per_bit);
  // Loop restoration after CCSO: every luma unit filtered with PC_WIENER, then each kept where it
  // brings the frame nearer src by more than a bit's worth, else put back (lr_flags_).
  void restore(const Frame420& src, bool optimized, double sse_per_bit);
  void fit_wienerns(const Frame420& src, int p, double* a, double* v) const;
  int wienerns_syntax(int p, int match, int s, const int16_t* t, int16_t* ref, std::vector<uint8_t>* out);
  std::pair<int, int> wienerns_matches(int p);
#ifdef CELV2_ORACLE  // tools/oracle/filter_import.inc
 public:
  // A WIENER_NONSEP unit's syntax: per class, what it is coded against.
  struct LrPlan {
    bool exact[WIENERNS_MAX_CLASSES] = {};
    int ref[WIENERNS_MAX_CLASSES] = {}, size[WIENERNS_MAX_CLASSES] = {};
    int16_t taps[WIENERNS_MAX_CLASSES][WIENERNS_TAPS_MAX] = {};
  };
  const std::vector<LrPlan>& lr_plan(int p) const { return lr_plan_[p]; }
  void import_frame_filter_bits(int p);

 private:
  void import_filters(const celv2::oracle::Frame& o);
  void import_lr_types(const celv2::oracle::Frame& o);
  void import_lr_units(const celv2::oracle::Frame& o);
  void import_end_frame();
  std::vector<LrPlan> lr_plan_[MAX_MB_PLANE];
#endif
  std::vector<uint8_t> lr_flags_;
  std::vector<uint8_t> lr_filter_bits_[MAX_MB_PLANE];
  std::vector<uint16_t> lr_keep_[MAX_MB_PLANE];  // restore(): the planes before restoration
  AV2LrStruct lr_ctxt_ = {};
  AV2LrSync lr_sync_ = {};  // restore() on lf_workers_

  SeqConfig cfg_;
  AV2_COMMON* cm_ = nullptr;
  BufferPool* pool_ = nullptr;
  std::vector<Tile> tiles_;
  std::mutex rows_mu_;
  std::vector<std::unique_ptr<RowWorker>> rows_;  // all row workers
  std::vector<RowWorker*> free_rows_;
  int frame_number_ = 0;
  int since_key_ = 0;  // inter frames since the last keyframe
  int qindex_delta_ = 0;  // of the frame being coded
  int display_counter_ = 0;  // display order hint of the last coded frame
  bool tiles_warned_ = false;  // the tile-count clamp is reported once
  std::array<uint64_t, FRAME_BUFFERS> gen_{};  // per buffer: bumped when it takes a new frame
  // sync_refs_from(): the buffer of the other state each of ours holds a copy of, at its gen_.
  std::array<const RefCntBuffer*, FRAME_BUFFERS> mirror_of_{};
  std::array<uint64_t, FRAME_BUFFERS> mirror_gen_{};
  Frame420 recon_;
  std::vector<int> sb_q_;  // per coded superblock, raster: its qindex (empty: the frame's)
  int sb_q_cols_ = 0;
  std::vector<uint16_t> ccso_luma_;  // end_frame(): the deblocked luma CCSO reads, kept between frames
  std::vector<AVxWorker> lf_workers_;  // the last one runs on the calling thread
  AV2LfSync lf_sync_ = {};
  AV2CdefWorkerData* cdef_worker_ = nullptr;
  AV2CdefSync cdef_sync_ = {};
  int cdef_workers_ = 0;  // the CDEF buffers are allocated for this many workers (0: none)
};
