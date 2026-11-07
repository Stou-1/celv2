// A tile's symbols in coding order, logged while the decisions are made and range coded afterwards
// (docs/design/2026-10-30-parallel-coder.md). Logging adapts the CDFs exactly as avm_write_symbol()
// would, so the bit estimates read the same probabilities; replay() codes the log into a range coder
// with its own copy of the CDFs, which must start where the logged ones started (for the first
// superblock row of a tile; later rows' logs continue from the CDFs the rows before left).
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

extern "C" {
#include "av2/common/entropy.h"
#include "av2/common/entropymode.h"
#include "avm_dsp/binary_codes_writer.h"
#include "avm_dsp/bitwriter.h"
}

class SymbolLog {
 public:
  SymbolLog() = default;
  // fc: the CDFs every logged symbol's CDF lies in (the row's xd->tile_ctx); update: whether symbols
  // adapt them (avm_writer::allow_update_cdf).
  SymbolLog(FRAME_CONTEXT* fc, bool update) : base_(reinterpret_cast<avm_cdf_prob*>(fc)), update_(update) {}

  // avm_write_symbol(): logs, then adapts cdf.
  void symbol(int v, avm_cdf_prob* cdf, int n) {
    const ptrdiff_t off = cdf - base_;
    if (off < 0 || static_cast<size_t>(off) + n > kCdfs || n >= 64)
      throw std::logic_error("symbol CDF outside the tile's FRAME_CONTEXT");
    log_.push_back({static_cast<uint32_t>(off) << 9 | static_cast<uint32_t>(n) << 3 | kSymbol, v});
    if (count_) add_bits(cdf, v);
    if (update_) update_cdf(cdf, static_cast<int8_t>(v), n);
  }
  // A symbol decided after the tile is logged (a restoration unit's flag): replay() codes
  // deferred[index], or nothing where that is above n - 1 (the plane went without restoration). The
  // logged CDFs adapt as if it were 0 (they only steer bit estimates).
  void deferred(int index, avm_cdf_prob* cdf, int n) {
    const ptrdiff_t off = cdf - base_;
    if (off < 0 || static_cast<size_t>(off) + n > kCdfs || n >= 64)
      throw std::logic_error("symbol CDF outside the tile's FRAME_CONTEXT");
    log_.push_back({static_cast<uint32_t>(off) << 9 | static_cast<uint32_t>(n) << 3 | kDeferred, index});
    if (update_) update_cdf(cdf, 0, n);
  }
  void literal(int v, int bits) {
    log_.push_back({static_cast<uint32_t>(bits) << 3 | kLiteral, v});
    bits_ += bits;
  }
  void bit(int b) {
    log_.push_back({kBit, b});
    bits_ += 1;
  }
  void quniform(uint16_t n, uint16_t v) { log_.push_back({static_cast<uint32_t>(n) << 3 | kQuniform, v}); }
  // read_delta_qindex() for a superblock coded at qindex: the step from the qindex of the superblock
  // coded before it in raster order, which another row may decide, so replay() takes it. abs: the
  // step's size as the row sees it, adapting delta_q_cdf meanwhile.
  void delta_q(int qindex, int res, int abs) {
    log_.push_back({static_cast<uint32_t>(res) << 3 | kDeltaQ, qindex});
    if (update_) update_cdf(base_ + kDeltaQCdf, static_cast<int8_t>(abs < DELTA_Q_SMALL ? abs : DELTA_Q_SMALL),
                            DELTA_Q_PROBS + 1);
  }

  // Codes the log into w, adapting the CDFs at fc (same layout as the logged ones) as it goes;
  // base_qindex: the qindex delta_q() steps from, carried over to the next row's log. deferred:
  // the values of deferred() symbols, by their index.
  void replay(avm_writer* w, FRAME_CONTEXT* fc, int* base_qindex, const uint8_t* deferred = nullptr) const {
    avm_cdf_prob* const base = reinterpret_cast<avm_cdf_prob*>(fc);
    for (const Entry& e : log_) {
      switch (e.a & 7) {
        case kSymbol: avm_write_symbol(w, e.v, base + (e.a >> 9), (e.a >> 3) & 63); break;
        case kDeferred:
          if (!deferred) throw std::logic_error("a deferred symbol without its values");
          if (deferred[e.v] < ((e.a >> 3) & 63)) avm_write_symbol(w, deferred[e.v], base + (e.a >> 9), (e.a >> 3) & 63);
          break;
        case kLiteral: avm_write_literal(w, e.v, static_cast<int>(e.a >> 3)); break;
        case kBit: avm_write_bit(w, e.v); break;
        case kQuniform:
          avm_write_primitive_quniform(w, static_cast<uint16_t>(e.a >> 3), static_cast<uint16_t>(e.v));
          break;
        default: {  // kDeltaQ
          const int res = static_cast<int>(e.a >> 3), diff = e.v - *base_qindex;
          if (diff % res) throw std::logic_error("delta_qindex: superblock qindex off the delta_q_res grid");
          const int d = diff / res, abs = d < 0 ? -d : d;
          avm_write_symbol(w, abs < DELTA_Q_SMALL ? abs : DELTA_Q_SMALL, base + kDeltaQCdf, DELTA_Q_PROBS + 1);
          if (abs >= DELTA_Q_SMALL) {
            const int v = abs - DELTA_Q_SMALL_MINUS_2, rem_bits = get_msb(v);  // v >= 2, so rem_bits >= 1
            avm_write_literal(w, rem_bits - 1, 3);
            avm_write_literal(w, v - (1 << rem_bits), rem_bits);
          }
          if (abs) avm_write_bit(w, d < 0);
          *base_qindex = e.v;
        }
      }
    }
  }
  size_t size() const { return log_.size(); }
  // CELV2_WIDE W3: from count(true) the bits of what is logged add up in bits(); truncate() drops
  // everything logged after size() was n.
  void count(bool on) { count_ = on, bits_ = 0; }
  double bits() const { return bits_; }
  void truncate(size_t n) { log_.resize(n); }

 private:
  enum : uint32_t { kSymbol, kLiteral, kBit, kQuniform, kDeltaQ, kDeferred };
  static constexpr size_t kCdfs = sizeof(FRAME_CONTEXT) / sizeof(avm_cdf_prob);
  static constexpr size_t kDeltaQCdf = offsetof(FRAME_CONTEXT, delta_q_cdf) / sizeof(avm_cdf_prob);
  static_assert(kCdfs < (1u << 23), "CDF offsets take 23 bits");
  struct Entry {
    uint32_t a;  // symbol: CDF offset << 9 | symbols << 3 | kind; others: width, n or res << 3 | kind
    int32_t v;
  };
  void add_bits(const avm_cdf_prob* cdf, int v) {
    const int hi = v ? cdf[v - 1] : CDF_PROB_TOP, p = hi - cdf[v];
    bits_ -= std::log2((p > 1 ? p : 1) / static_cast<double>(CDF_PROB_TOP));
  }
  avm_cdf_prob* base_ = nullptr;
  bool update_ = false;
  bool count_ = false;
  double bits_ = 0;
  std::vector<Entry> log_;
};
