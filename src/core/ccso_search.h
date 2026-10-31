// The encoder's choice of one plane's CCSO filter (ccso.c applies it): the classification (band of
// the deblocked luma sample, and with edge offsets the classes of its differences to two
// neighbours) and the table offset of each class, by squared error against the source plus bits.
#pragma once

#include <cstdint>

extern "C" {
#include "av2/common/av2_common_int.h"
}

struct CcsoChoice {
  bool bo_only = true;
  int scale_idx = 0, quant_idx = 0, ext_filter_support = 0, edge_clf = 0, max_band_log2 = 0;
  int8_t offset[CCSO_BAND_NUM * 16] = {};  // CcsoInfo::filter_offset layout: (band << 4) + (d0 << 2) + d1
  double gain = 0;  // squared error saved less the header bits' cost (0 for all-zero offsets)
};

// luma: the deblocked luma with CCSO_PADDING_SIZE edge samples on every side (extend_ccso_border),
// stride luma_stride; rec: the plane after CDEF, src: its source, w x h samples, subsampled from
// luma by ss_x / ss_y, all at bit_depth (8-bit sources take the first form). As ccso.c above 8 bits:
// the bands come from the luma brought down to max_band_log2 bits, the edge thresholds and offsets
// are in the depth's own units. sse_per_bit weighs header bits. Runs on up to `threads` threads.
CcsoChoice ccso_search(const uint16_t* luma, int luma_stride, const uint16_t* rec, int rec_stride, const uint8_t* src,
                       int src_stride, int w, int h, int ss_x, int ss_y, double sse_per_bit, int threads);
CcsoChoice ccso_search(const uint16_t* luma, int luma_stride, const uint16_t* rec, int rec_stride, const uint16_t* src,
                       int src_stride, int w, int h, int ss_x, int ss_y, double sse_per_bit, int threads, int bit_depth);
