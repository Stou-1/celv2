// Tile-group OBU prefix for one frame: first_tile_group flag + uncompressed header + byte alignment
// (mirror of decodeframe.c av2_read_tilegroup_header / read_uncompressed_header).
#pragma once
#include <cstdint>
#include <vector>

#include "bitstream/seq_header.h"
#include "state/codec_state.h"

extern "C" {
#include "avm_dsp/bitwriter_buffer.h"
}

// Bytes of the size field before every tile but the last (tile_size_bytes).
constexpr int kTileSizeBytes = 4;

// Call after st.begin_frame(key); the header reflects the state CodecState set up.
std::vector<uint8_t> write_frame_header(const SeqConfig& c, CodecState& st, bool key);
// film_grain_params() of a shown frame (setup_film_grain()): apply_grain, then model 0 (the film grain
// model OBU before the frame holds it) and the seed.
void write_film_grain(avm_write_bit_buffer* wb, const avm_film_grain_t& g);
