// Stage 3's replay (docs/superpowers/specs/2026-11-04-av1-gap-breakdown-design.md), only in a CELV2_ORACLE
// build. X_ORACLE=<oracle_dump's file>, X_ORACLE_LEVEL=1..6 beside X_L0 (ChunkCoder's frame list):
//   1: avmenc's partitions, the blocks' modes and MVs CelV2's (RD pass and coder, as at L0);
//   2: its modes and MVs too, used as they are: no RD pass, no engine, no re-scoring in the coder;
//   3: its transform splits and types too;
//   4: its in-loop filter settings too (X_ORACLE_OWN_FILTERS=1: CelV2's own);
//   5: which transform blocks are all zero, skip_txfm with them, and what its prediction takes: delta q, IBP, CfL filter,
//      FSC, TMVP's step;
//   6: its levels, CelV2's quantizer bypassed (--bit-depth 8: the levels' depth).
//   X_ORACLE_Q below: from L3 to L4, levels by the coder's own bits.
//      X_ORACLE_128=1: its 128 blocks as they are, the map at 128: BAWP's template and the deblocking inside a 128 block
//      do not survive a split, so L6 reconstructs avmenc's frames only so.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "gpu/decision_engine.h"
#include "oracle_format.h"

namespace celv2::oracle {

// The decision of one 64x64 (DecisionMap's grid): a 128 block becomes its 64x64s, each with its mode and MV.
struct Sb {
  std::vector<Part> tree;
  std::vector<BlockDecision> blocks;
};

struct Frame {
  OracleFrame head;
  std::vector<OracleBlock> leaves;
  std::vector<Sb> sbs;  // raster
  std::vector<Sb> sbs128;  // the same at 128, nothing split
  std::vector<uint8_t> tx_part;  // per mi, at each max-size transform unit's top left
  std::vector<OracleTx> txs;
  std::vector<OracleCoef> coefs;
  std::vector<uint32_t> first_coef;  // per OracleTx
  std::vector<int32_t> tx_grid[3];   // per plane, per 4-sample cell: the OracleTx starting there, or -1
  OracleFilters filters;
  std::vector<int8_t> cdef_idx;  // per 64x64
  std::vector<uint8_t> ccso_flags[3];
  std::vector<uint8_t> lr_frame_filter[3], lr_unit_type[3], lr_unit_filter[3];
  std::vector<int32_t> lr_unit_at[3];  // a WIENER_NONSEP unit's place in lr_unit_filter
  // The transform block whose top left is (row, col) in plane's 4-sample cells, or nullptr.
  const OracleTx* tx_at(int plane, int row, int col) const {
    const int stride = head.mi_cols >> (plane ? 1 : 0);
    if (row >= head.mi_rows >> (plane ? 1 : 0) || col >= stride) return nullptr;
    const int32_t i = tx_grid[plane][static_cast<size_t>(row) * stride + col];
    return i < 0 ? nullptr : &txs[i];
  }
};

// What the replay could not express, by kind; printed at exit (count, then area in samples).
enum Kind {
  kSplit128x128, kSplit128x64, kSplit64x128,  // leaves coded as their 64x64s
  kRootVert,        // a 128 superblock's VERT: its 64x64s coded in raster order, not column by column
  kSameRefCompound, // a compound of one reference twice: the sequence has no such pairs, coded from that reference alone
  kNearLost,        // NEARMV or NEAR in a compound: no candidate holds the MV, coded with a difference
  kSkipLost,        // skip mode: no candidate holds the pair, coded as a plain compound
  kCompoundMode,    // any other compound mode no longer possible, coded as NEW_NEWMV
  kCflLost,         // CfL not allowed on the 64x64 of a 128 block: chroma DC
  kBawpLost,        // BAWP not allowed on the block
  kAmvdLost,        // adaptive MVD: the difference is not in its set, coded in full
  kPrecisionLost,   // the block's MV precision cannot reach the MV from this predictor: the frame's
  kMvMoved,         // the MV itself could not be coded
  kTxSplitLost,     // L3: a transform partition this block cannot code (a 128 block's 64x64)
  kFsc,             // L3: an intra block with forward skip coding: DCT
  kKinds
};
void count(Kind k, long long area);

// Q0 of docs/superpowers/specs/2026-11-06-real-cost-trellis-design.md, X_ORACLE_Q, letters: a TCQ's levels by real bits on the
// frame's starting CDFs, b avmenc's depth factors, c the CDFs of each superblock instead, d the plain path too.
// X_ORACLE_QCHECK=1: each block's rate against the symbols written.
enum { kRealTcq = 1, kRealDepth = 2, kRealRefresh = 4, kRealPlain = 8 };
int real_cost();
bool real_check();
int depth(int display);  // the frame's layer in avmenc's pyramid: 0 key, 1 anchor, 5 at most

int level();  // 0: off
bool filters();  // the stream's filter settings are in use
const Frame* frame(int display);  // nullptr when off; throws if the file has no such frame
int max_drl_bits();  // the stream's: the MV candidates past it differ
const OracleFrame& first_head();  // for what the sequence takes from the stream

// Wraps the decision engine: level 1 asks engine for the candidates, level 2 replaces it.
std::unique_ptr<DecisionEngine> import_engine(std::unique_ptr<DecisionEngine> engine);

}  // namespace celv2::oracle
