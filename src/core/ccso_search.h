// The encoder's choice of one plane's CCSO filter (ccso.c applies it): the classification (band of
// the deblocked luma sample, and with edge offsets the classes of its differences to two
// neighbours) and the table offset of each class, by squared error against the source plus bits.
#pragma once

#include <immintrin.h>

#include <cstdint>
#include <vector>

extern "C" {
#include "av2/common/av2_common_int.h"
}

namespace celv2 {
struct CcsoPlane;
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

// The search's edge class of a luma difference d (-16..16: the thresholds of quant_sz's table below |d|), and the
// counting loop's 16-lane form of it, plus 16 (|d| < 2^15). Tests hold one against the other.
int ccso_bin_of(int d);
__m256i ccso_bins16(__m256i d);

// The flags per CCSO unit (docs/superpowers/specs/2026-11-10-ccso-per-unit-design.md).
// One class's squared-error change at each of the 8 offsets kOffset[i] x scale (setup_ccso()'s table), the
// filtered sample clipped to [0, 2^bd - 1] as the filter clips it.
struct CcsoDelta {
  int64_t d[8] = {};
  CcsoDelta& operator+=(const CcsoDelta& o) {
    for (int i = 0; i < 8; i++) d[i] += o.d[i];
    return *this;
  }
};
// Every unit's CcsoDelta in each class of p's classification (p.dst still unfiltered) at offsets of
// scale (scale_idx + 1), the classes compact: (band * nd + class0) * nd + class1 for nd edge classes (1
// band-only). Units are 1 << unit_log2_x by 1 << unit_log2_y samples, cols a row, over the w x h samples
// of src; stats[u * classes + k], unit u = row * cols + col. Returns classes (at most 72). Rows of units
// run on threads.
int ccso_unit_stats(const celv2::CcsoPlane& p, const uint8_t* src, int src_stride, int w, int h, int scale,
                    int unit_log2_x, int unit_log2_y, int cols, std::vector<CcsoDelta>* stats);
int ccso_unit_stats(const celv2::CcsoPlane& p, const uint16_t* src, int src_stride, int w, int h, int scale,
                    int unit_log2_x, int unit_log2_y, int cols, std::vector<CcsoDelta>* stats);
// Which groups of units take c's filter: stats[g * classes + k] summed per group. A group is on where
// the offsets lower its error; the offsets are then re-fitted over the on groups (twice). The least J of
// per-unit flags, every unit on (c's offsets or ones re-fitted over every unit) or the plane off (0),
// J counting header, offset and flag bits (an adaptive coder's, on the rows x cols units, unit_group[u]
// unit u's group): under 0, c holds the offsets and on[g] each group's flag; else the plane is better
// off (c and on unspecified).
double ccso_unit_flags(const CcsoDelta* stats, int groups, int classes, double sse_per_bit, CcsoChoice* c,
                       std::vector<uint8_t>* on, const int* unit_group, int rows, int cols);
