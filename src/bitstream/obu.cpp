#include "bitstream/obu.h"

#include <algorithm>
#include <stdexcept>

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

std::vector<uint8_t> film_grain_model_obu(const avm_film_grain_t& g, int fgm_id) {
  std::vector<uint8_t> buf(512, 0);
  avm_write_bit_buffer wb = {buf.data(), 0};
  auto bits_for = [](int v) {  // ceil(log2(v)) for v >= 1
    int b = 0;
    while ((1 << b) < v) b++;
    return b;
  };
  avm_wb_write_literal(&wb, 1 << fgm_id, 8);  // fgm_bit_map (MAX_FGM_NUM bits)
  avm_wb_write_uvlc(&wb, 0);  // fgm_chroma_idc: CHROMA_FORMAT_420
  avm_wb_write_bit(&wb, g.fgm_scale_from_channel0_flag);
  const int (*const points[3])[2] = {g.fgm_scaling_points_0, g.fgm_scaling_points_1, g.fgm_scaling_points_2};
  for (int c = 0; c < (g.fgm_scale_from_channel0_flag ? 1 : 3); c++) {
    avm_wb_write_literal(&wb, g.fgm_points[c], 4);
    if (!g.fgm_points[c]) continue;
    int max_incr = 0, max_scal = 0;
    for (int i = 0; i < g.fgm_points[c]; i++)
      max_incr = std::max(max_incr, points[c][i][0]), max_scal = std::max(max_scal, points[c][i][1]);
    const int bits_incr = std::max(1, bits_for(max_incr + 1)), bits_scal = std::max(5, bits_for(max_scal + 1));
    avm_wb_write_literal(&wb, bits_incr - 1, 3);
    avm_wb_write_literal(&wb, bits_scal - 5, 2);
    for (int i = 0; i < g.fgm_points[c]; i++) {
      avm_wb_write_literal(&wb, points[c][i][0] - (i ? points[c][i - 1][0] : 0), bits_incr);
      avm_wb_write_literal(&wb, points[c][i][1], bits_scal);
    }
  }
  avm_wb_write_literal(&wb, g.scaling_shift - 8, 2);
  avm_wb_write_literal(&wb, g.ar_coeff_lag, 2);
  const int num_luma = 2 * g.ar_coeff_lag * (g.ar_coeff_lag + 1), num_chroma = num_luma + (g.fgm_points[0] > 0);
  auto coeffs = [&](const int* a, int n) {  // offset by half their range, wide enough for the largest
    int top = 0, bottom = 0;
    for (int i = 0; i < n; i++) top = std::max(top, a[i]), bottom = std::min(bottom, a[i]);
    const int bits = std::max(5, bits_for(std::max(top + 1, -bottom)) + 1);
    avm_wb_write_literal(&wb, bits - 5, 2);
    for (int i = 0; i < n; i++) avm_wb_write_literal(&wb, a[i] + (1 << (bits - 1)), bits);
  };
  if (g.fgm_points[0]) coeffs(g.ar_coeffs_y, num_luma);
  if (g.fgm_points[1] || g.fgm_scale_from_channel0_flag) coeffs(g.ar_coeffs_cb, num_chroma);
  if (g.fgm_points[2] || g.fgm_scale_from_channel0_flag) coeffs(g.ar_coeffs_cr, num_chroma);
  avm_wb_write_literal(&wb, g.ar_coeff_shift - 6, 2);
  avm_wb_write_literal(&wb, g.grain_scale_shift, 2);
  if (g.fgm_points[1]) {
    avm_wb_write_literal(&wb, g.cb_mult, 8);
    avm_wb_write_literal(&wb, g.cb_luma_mult, 8);
    avm_wb_write_literal(&wb, g.cb_offset, 9);
  }
  if (g.fgm_points[2]) {
    avm_wb_write_literal(&wb, g.cr_mult, 8);
    avm_wb_write_literal(&wb, g.cr_luma_mult, 8);
    avm_wb_write_literal(&wb, g.cr_offset, 9);
  }
  avm_wb_write_bit(&wb, g.overlap_flag);
  avm_wb_write_bit(&wb, g.clip_to_restricted_range);
  if (g.clip_to_restricted_range) avm_wb_write_bit(&wb, g.mc_identity);
  avm_wb_write_bit(&wb, g.block_size);
  return make_obu(OBU_FILM_GRAIN_MODEL, finish(buf, wb));
}

void insert_before_first_frame(std::vector<uint8_t>& tu, const std::vector<uint8_t>& obus) {
  for (size_t p = 0; p < tu.size();) {
    uint64_t size = 0;
    size_t q = p;
    for (int shift = 0; q < tu.size(); shift += 7) {
      const uint8_t b = tu[q++];
      size |= static_cast<uint64_t>(b & 0x7f) << shift;
      if (!(b & 0x80)) break;
    }
    if (q >= tu.size()) break;
    const int type = (tu[q] >> 2) & 0x1f;
    if (type == OBU_CLOSED_LOOP_KEY || type == OBU_REGULAR_TILE_GROUP || type == OBU_REGULAR_SEF) {
      tu.insert(tu.begin() + static_cast<std::ptrdiff_t>(p), obus.begin(), obus.end());
      return;
    }
    p = q + size;
  }
  throw std::logic_error("insert_before_first_frame: no coded frame OBU");
}
