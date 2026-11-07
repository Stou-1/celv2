// AV2 OBU framing: [uleb128 size including header][1-byte header][payload].
#pragma once
#include <cstdint>
#include <vector>

extern "C" {
#include "avm/avm_codec.h"  // OBU_TYPE values (OBU_SEQUENCE_HEADER, OBU_CLOSED_LOOP_KEY, ...)
#include "avm_dsp/grain_synthesis.h"
}

// Header without extension: obu_header_extension_flag=0, 5-bit type, 2-bit temporal layer 0.
std::vector<uint8_t> make_obu(int obu_type, const std::vector<uint8_t>& payload);

// OBU_CONTENT_INTERPRETATION (obu_ci.c), sent with keyframes: progressive scan and the colour
// description idc (1 BT.709 SDR, 2 BT.2100 PQ, 3 BT.2100 HLG) with the range; no chroma sample
// position, aspect ratio or timing.
std::vector<uint8_t> content_interpretation_obu(int colour_idc, bool full_range);
// OBU_METADATA_SHORT, prefix, for the current layer, not persisting past its frame: its type
// (OBU_METADATA_TYPE_*) and payload.
std::vector<uint8_t> metadata_obu(int metadata_type, const std::vector<uint8_t>& payload);
// OBU_FILM_GRAIN_MODEL (obu_fgm.c read_fgm_obu()) holding one 4:2:0 model as model `fgm_id` (0..7);
// the field widths as AVM's encoder picks them (bitstream_fgm.c write_fgm_obu()).
std::vector<uint8_t> film_grain_model_obu(const avm_film_grain_t& g, int fgm_id);
// Inserts `obus` into the OBU sequence `tu` before its first coded frame (a key, tile group or
// show-existing OBU), where the prefix OBUs of a frame unit go; throws if it has none.
void insert_before_first_frame(std::vector<uint8_t>& tu, const std::vector<uint8_t>& obus);
