// CCSO filtering of one plane (ccso.c apply_ccso_filter()), celv2's own: tiles filtered across, no
// lossless segments, no BRU. The caller says which 32x32 blocks are filtered; the rows of blocks are
// independent (the luma is a separate copy), so they can run on threads of their own.
#pragma once

#include <cstdint>

namespace celv2 {

constexpr int kCcsoBlockLog2 = 5;  // CCSO_PROC_BLK_LOG2

struct CcsoPlane {
  const uint16_t* luma = nullptr;  // the deblocked luma at (0, 0), padded (extend_ccso_border())
  int luma_stride = 0;
  uint16_t* dst = nullptr;  // the plane after CDEF, filtered in place
  int dst_stride = 0;
  int w = 0, h = 0;    // the plane's (aligned) size, as the decoder's dst.width / height
  int sx = 0, sy = 0;  // its subsampling from luma
  int bit_depth = 8;
  bool bo_only = true;
  int thr = 0;               // quant_sz[scale_idx][quant_idx]
  int edge_clf = 0;          // 0: three classes per neighbour, 1: two
  int loc[2] = {};           // the two neighbours' offsets in the luma (derive_ccso_sample_pos())
  int max_band_log2 = 0;     // 0: one band
  const int8_t* offset = nullptr;  // (band << 4) + (class0 << 2) + class1
  const uint8_t* on = nullptr;     // per 32x32 block, row-major, on_stride a row: filter it or not
  int on_stride = 0;
};

// The 32x32 blocks of block row `row` (ccso.c av2_apply_ccso_filter_for_row()).
void ccso_apply_row(const CcsoPlane& p, int row);

// derive_ccso_sample_pos(): the two neighbours of ext_filter_support 0..6.
void ccso_neighbours(int ext_filter_support, int luma_stride, int loc[2]);

}  // namespace celv2
