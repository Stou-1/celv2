// Global motion estimation (encoder side, not normative): a rotation-zoom model of where each pixel
// of the current frame lies in a reference, from block matching on 4x-downscaled luma and a
// RANSAC fit.
#pragma once
#include <cstdint>

extern "C" {
#include "av2/common/av2_common_int.h"
}

// cur: the source luma (8-bit), ref: the reference's reconstructed luma (16-bit samples, of
// 8 + ref_shift bits), both w x h. On success wm holds a ROTZOOM model on AV2's global motion grid (av2_get_shear_params()
// valid); false when the motion is no better described than by a translation.
bool estimate_rotzoom(const uint8_t* cur, int cur_stride, const uint16_t* ref, int ref_stride, int ref_shift, int w, int h,
                      const struct scale_factors* sf, WarpedMotionParams* wm);
