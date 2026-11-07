// Motion-compensated temporal filtering of a hidden reference (the alt-ref filtering of libaom /
// AVM's temporal_filter.c, by its method): each neighbour's motion-compensated block is averaged in
// with a weight per sample from its error against the frame to filter, so what lasts stays and
// noise that would not is averaged away; the frames that predict from it spend fewer bits on it.
#pragma once
#include <vector>

#include "io/y4m.h"

namespace celv2 {

// The noise level of a luma or chroma plane (AVM's av2_estimate_noise_from_single_plane() method:
// the mean |Laplacian| over smooth samples); -1 when too few samples are smooth.
double estimate_noise(const uint8_t* p, int stride, int w, int h);

// center averaged with others: per 16x16 luma block a full-pel motion search in each neighbour,
// then per sample weight exp(-min(e * d / n, 7)) against the center's 1, e the 5x5 window's mean
// squared error (luma; chroma adds its luma samples) blended 5:1 with the block's and divided by 20,
// n = 0.5 + ln(2 noise + 5) of the plane, d = max(1, motion / a tenth of the frame's shorter side);
// the result times strength (1: AVM's default strength at qindex 100 and up). Chroma follows the
// luma blocks' motion.
Frame420 temporal_filter(const Frame420& center, const std::vector<const Frame420*>& others, double strength = 1);

}  // namespace celv2
