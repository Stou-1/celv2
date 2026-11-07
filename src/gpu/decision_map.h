// Boundary type between the GPU decision engine and the CPU encoder (no CUDA or AVM dependency).
#pragma once
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <vector>

#ifdef CELV2_ORACLE
struct OracleBlock;  // tools/oracle/oracle_format.h
namespace celv2::oracle {
struct Sb;
}
#endif

enum class Part : uint8_t { None, Horz, Vert, Split };
// The AV2 luma intra modes celv2 decides (angle delta 0); the first four keep their old values.
enum class IntraMode : uint8_t { DC, V, H, Smooth, SmoothV, SmoothH, Paeth, D45, D67, D113, D135, D157, D203 };
constexpr int kIntraModes = 13;

struct BlockDecision {
  uint8_t w4 = 0, h4 = 0;  // block size in 4-pixel units
  bool inter = false;
  IntraMode intra = IntraMode::DC;
  int16_t mv_row = 0, mv_col = 0;  // 1/8 pel
  uint8_t ref = 0;                  // inter: reference index, 0 = the previous frame, 1 = the one before...
  // Compound (two-reference average) when ref2 != kSingleRef: the second reference and its MV.
  static constexpr uint8_t kSingleRef = 255;
  uint8_t ref2 = kSingleRef;
  int16_t mv2_row = 0, mv2_col = 0;
#ifdef CELV2_ORACLE
  const OracleBlock* imp = nullptr;  // L2: avmenc's block, coded as it chose it
  bool piece = false;                // a 128 block's 64x64 after its first
#endif
};

// An engine's best choices for one candidate block of a 64x64 superblock, both kinds, with their
// costs (SATD * 256 plus lambda-weighted bits; INT64_MAX where there is none: no reference, or an
// intra block above 32x32).
struct BlockCandidates {
  int64_t inter_cost = INT64_MAX, intra_cost = INT64_MAX;
  BlockDecision inter;  // the best inter choice (w4 / h4 the block's)
  IntraMode intra = IntraMode::DC;
  // Each reference's best single-reference MV (1/8 pel, as BlockDecision's) and its cost / 256
  // (saturated; UINT32_MAX where the reference has none: every MV repeated, or no such reference).
  static constexpr int kRefs = 7;  // AV2's INTER_REFS_PER_FRAME
  uint32_t ref_cost[kRefs] = {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
  int16_t ref_mv_col[kRefs] = {}, ref_mv_row[kRefs] = {};
};

// The CPU encoder legalizes the tree: where AV2 forces or forbids a partition (frame edges),
// it substitutes the normative one and reuses the nearest block decision for missing leaves.
struct SbDecision {
  std::vector<Part> tree;             // preorder; SPLIT has 4 children, HORZ/VERT 2
  std::vector<BlockDecision> blocks;  // leaves in coding order
  // Every candidate block's choices (gpu/sb_blocks.h order, kSbBlocks), for the CPU RD pass; empty
  // where the decision does not come from an engine.
  std::vector<BlockCandidates> cands;
  // CELV2_WIDE W3: other trees and blocks of the RD pass, trial-coded against tree / blocks.
  struct Alt {
    std::vector<Part> tree;
    std::vector<BlockDecision> blocks;
  };
  std::vector<Alt> alts;
  // CELV2_WIDE W2: per block and reference the full search's MV (col, row; 1/8 pel), kNoMv for none.
  static constexpr int16_t kNoMv = INT16_MIN;
  std::vector<int16_t> wide_mv;
#ifdef CELV2_ORACLE
  const celv2::oracle::Sb* forced = nullptr;  // L1: the tree the RD pass keeps
#endif
};

struct DecisionMap {
  int sb_size = 0;
  int sb_cols = 0, sb_rows = 0;
  std::vector<SbDecision> sbs;  // raster order
};

// Every superblock NONE + DC intra.
DecisionMap uniform_decision(int width, int height, int sb_size);

// Hand-off from the GPU collector thread (producer) to tile threads (consumers): superblock
// rows become readable one at a time. A failure is recorded once and rethrown to every waiter.
// A row's contents are written before publish(row) and read only after wait_row(row) returns;
// the mutex orders the two.
class RowGate {
 public:
  RowGate() = default;
  RowGate(const RowGate&) = delete;
  RowGate& operator=(const RowGate&) = delete;

  void publish(int row);            // rows 0..row are ready (monotonic; smaller values ignored)
  void publish_one(int row);        // row is ready (rows may come in any order)
  void wait_row(int row) const;     // blocks until row is ready; rethrows the failure if any
  void fail(std::exception_ptr e);  // records e and wakes every waiter
  int ready_rows() const;           // number of rows published so far

 private:
  mutable std::mutex m_;
  mutable std::condition_variable cv_;
  int ready_ = 0;
  std::vector<char> one_;  // publish_one rows
  std::exception_ptr error_;
};
