// CDEF's strength search: one block's squared error at every strength, from the filter's parts, and the frame's set.
#pragma once
#include <cstdint>
#include <vector>

namespace celv2 {

// s = level * 4 + the secondary strength's index (0, 1, 2, 4), as CodecState::cdef() numbers them; s = 0 is unfiltered.
constexpr int kCdefStrengths = 64;

// err[s] += the squared error to the source of one 8x8 luma (luma) or 4x4 chroma block filtered as
// av2_cdef_filter_fb() filters it at strength s, for every s. in: the block's top left in CDEF's input buffer
// (CDEF_BSTRIDE, its borders around); dir and var: the block's direction and variance (av2_cdef_find_dir(); var only
// for luma); damping and coeff_shift as av2_cdef_filter_fb() takes them. Only the vw x vh pixels at the top left count
// (the visible picture); org (org16 if not null, else org8; stride os) at the block's top left.
void cdef_block_errors(const uint16_t* in, bool luma, int dir, int var, int damping, int coeff_shift,
                       const uint16_t* org16, const uint8_t* org8, int os, int vw, int vh, int64_t err[kCdefStrengths]);

// The frame's strength set, 1 to 8 (luma, chroma) pairs in use order (the most used first), and each group's index
// into it (docs/superpowers/specs/2026-11-10-cdef-per-unit-design.md): err[(g * 2 + p) * kCdefStrengths + s] is
// group g's squared error at plane type p (0 luma, 1 chroma) and strength s; J = error + sse_per_bit x the header's
// and the indices' bits. Empty when the frame does better unfiltered (index untouched).
struct CdefPair {
  int luma, chroma;
};
std::vector<CdefPair> cdef_choose_set(const int64_t* err, int groups, double sse_per_bit, std::vector<int8_t>* index);

}  // namespace celv2
