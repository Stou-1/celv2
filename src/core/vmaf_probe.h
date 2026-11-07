// --target-vmaf (docs/superpowers/specs/2026-11-03-target-vmaf-design.md): the qindex search over a
// scene's probe VMAFs, and the VMAF of a probe measured with ffmpeg's libvmaf.
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "io/y4m.h"

namespace celv2 {

// The highest qindex (1..255) whose VMAF reaches target, from at most max_probes calls of vmaf(q):
// the first at start, the second where the usual slope of log(100 - VMAF) says (4 to 48 away), then
// where the line through the two nearest probes meets it; a probe within 0.15 above the target ends it,
// and one more than max_probes is taken while the best is over 1 above it.
// probes (optional) gets the number of calls. probed_only: the highest probed qindex at or above the
// target instead (the lowest probed when none is), so its probe can stand as the result.
int vmaf_search(double target, int start, const std::function<double(int q)>& vmaf, int max_probes = 4,
                int* probes = nullptr, bool probed_only = false);

// Whether `ffmpeg` runs and has the libvmaf filter.
bool ffmpeg_has_libvmaf(const std::string& ffmpeg);

// The mean VMAF (vmaf_v0.6.1) of recon (y4m frame payloads, as CodedFrame::recon) against src, one
// frame each, at src's size and depth: both written to the temporary folder as y4m, then ffmpeg.
// Throws std::runtime_error when ffmpeg fails or prints no score.
double ffmpeg_vmaf(const std::string& ffmpeg, const std::vector<const Frame420*>& src,
                   const std::vector<std::vector<uint8_t>>& recon, int fps_num, int fps_den);

}  // namespace celv2
