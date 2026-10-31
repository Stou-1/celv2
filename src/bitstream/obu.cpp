#include "bitstream/obu.h"

extern "C" {
#include "avm/avm_image.h"
#include "avm_dsp/bitwriter_buffer.h"
}

std::vector<uint8_t> make_obu(int obu_type, const std::vector<uint8_t>& payload) {
  std::vector<uint8_t> out;
  uint64_t size = payload.size() + 1;  // the size field covers the 1-byte header
  do {
    uint8_t b = size & 0x7f;
    size >>= 7;
    if (size) b |= 0x80;
    out.push_back(b);
  } while (size);
  out.push_back(static_cast<uint8_t>((obu_type & 0x1f) << 2));
  out.insert(out.end(), payload.begin(), payload.end());
  return out;
}

namespace {

// The trailing one bit, then zeros to the byte boundary; the bytes written.
std::vector<uint8_t> finish(std::vector<uint8_t>& buf, avm_write_bit_buffer& wb) {
  avm_wb_write_bit(&wb, 1);
  while (!avm_wb_is_byte_aligned(&wb)) avm_wb_write_bit(&wb, 0);
  buf.resize(avm_wb_bytes_written(&wb));
  return buf;
}

}  // namespace

std::vector<uint8_t> content_interpretation_obu(int colour_idc, bool full_range) {
  std::vector<uint8_t> buf(16, 0);
  avm_write_bit_buffer wb = {buf.data(), 0};
  avm_wb_write_literal(&wb, AVM_SCAN_TYPE_PROGRESSIVE, 2);  // ci_scan_type_idc
  avm_wb_write_bit(&wb, 1);  // ci_color_description_present_flag
  avm_wb_write_bit(&wb, 0);  // ci_chroma_sample_position_present_flag
  avm_wb_write_bit(&wb, 0);  // ci_aspect_ratio_info_present_flag
  avm_wb_write_bit(&wb, 0);  // ci_timing_info_present_flag
  avm_wb_write_literal(&wb, 0, 2);  // ci_reserved_2bit
  avm_wb_write_rice_golomb(&wb, static_cast<uint32_t>(colour_idc), 2);  // av2_read_color_info()
  avm_wb_write_bit(&wb, full_range);
  avm_wb_write_bit(&wb, 0);  // ci_extension_present_flag
  return make_obu(OBU_CONTENT_INTERPRETATION, finish(buf, wb));
}

std::vector<uint8_t> metadata_obu(int metadata_type, const std::vector<uint8_t>& payload) {
  std::vector<uint8_t> buf;
  // metadata_is_suffix 0, muh_layer_idc current, muh_cancel_flag 0, muh_persistence_idc none
  buf.push_back(static_cast<uint8_t>((AVM_LAYER_CURRENT << 4) | AVM_NO_PERSISTENCE));
  for (uint64_t v = static_cast<uint64_t>(metadata_type);;) {  // uleb128 type
    const uint8_t byte = v & 0x7f;
    v >>= 7;
    buf.push_back(static_cast<uint8_t>(byte | (v ? 0x80 : 0)));
    if (!v) break;
  }
  buf.insert(buf.end(), payload.begin(), payload.end());
  buf.push_back(0x80);  // trailing bits
  return make_obu(OBU_METADATA_SHORT, buf);
}
