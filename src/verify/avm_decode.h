// In-process AVM decoding used to verify celv2's bitstreams (decoder only; never the AVM encoder).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "io/y4m.h"

// Decodes each temporal unit with AVM and returns, per output frame, the MD5 (lowercase hex) of the
// visible I420 area (Y, then U, then V; chroma size rounded up; above 8 bits, 16-bit little-endian
// samples). Throws std::runtime_error on any decode error or on output other than 8 or 10 bits.
std::vector<std::string> decode_frame_md5s(const std::vector<std::vector<uint8_t>>& temporal_units);

// Same hashing rule for 8-bit planes; w/h are the visible luma size.
std::string i420_md5(const uint8_t* y, int ys, const uint8_t* u, const uint8_t* v, int uvs, int w,
                     int h);

// i420_md5 of a Frame420's visible area (chroma rounded up); above 8 bits, of its full-depth
// samples as 16-bit little-endian bytes.
std::string frame_md5(const Frame420& f);
// Decoded output frames (visible area, padded like y4m input). Same errors as decode_frame_md5s.
std::vector<Frame420> decode_frames(const std::vector<std::vector<uint8_t>>& temporal_units);
// Empty if the visible areas match, else the first differing plane, 4x4 block and pixel.
std::string first_diff(const Frame420& a, const Frame420& b);
