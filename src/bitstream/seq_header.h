// Sequence header for celv2's minimal AV2 tool set (spec section 2: almost every tool declared off).
#pragma once
#include <cstdint>
#include <vector>

extern "C" {
#include "av2/common/av2_common_int.h"
}

struct SeqConfig {
  int width, height;    // visible size
  int fps_num, fps_den;
  int qindex;           // 1..255 (0 would switch AV2 into lossless coding)
  // log2 of the tile columns/rows asked for (uniform spacing, clamped up to the level minimum),
  // and the CPU threads that code tiles in parallel (0 = one per tile).
  int tile_cols_log2 = 0, tile_rows_log2 = 0;
  int threads = 0;
  // Past frames kept as inter references (1..7). On anime clips 4 gave -8.6% bits at equal PSNR
  // against 1 for 8% of the 1080p speed; 7 gave another -1.9% for another 6%.
  int refs = 4;
  bool deblock = true;  // AVM's deblocking filter on all planes
  // Deblocking strength offset on every plane, in steps of 8 qindex (-16..15; + filters more).
  int deblock_delta = 0;
  // Per frame, the deblocking offset in {-4, -2, 0, 2, 4} of the lowest squared error (--preset max).
  bool deblock_search = false;
  bool cdef = true;  // CDEF after deblocking, one strength per frame chosen by squared error
  // Cross-component sample offset after CDEF: offsets by deblocked luma band and edge class, every
  // unit on (33 frames 1080p q100-190: anime B Y -2.04%, U -4.03%, V -2.47%; live action
  // Y -0.78%, U -2.05%, V -0.96% BD-rate).
  bool ccso = true;
  // Multiple intra reference lines, chosen as the coder searches the luma mode (33 frames 1080p
  // q100-190: anime B Y -0.31%, live action Y -0.05%, chroma -0.3 to -2.3% BD-rate).
  bool mrls = true;
  // Adaptive MVD: NEWMV differences from a short set of values, cheaply coded, chosen by the RD
  // cost (33 frames 1080p q100-190: anime B Y -0.11%, U -0.42%, V -0.42%; live action
  // Y -0.26%, U -0.53%, V -0.67% BD-rate).
  bool amvd = true;
  // Temporal MVs (enable_ref_frame_mvs), off: 33 frames 1080p q100-190, anime A Y -3.8% but live
  // action +0.6%, anime D +2.2%, the rest within 0.5%.
  bool tmv = false;
  // A coarser MV precision (1, 4, 8 pel) per NEWMV block, chosen by RD cost (enable_flex_mvres):
  // 33 frames 1080p q100-190, Y -0.1 to -1.5% on all five sources, the same speed.
  bool flex_mv = true;
  bool tcq = true;   // trellis coded quantization of luma
  bool compound = true;  // inter frames with 2+ references may average two (REFERENCE_MODE_SELECT)
  // The CPU RD pass re-picks the decision engine's partitions and modes from its candidates
  // (core/rd_pass.h; 120 frames 1080p q100-190: anime B -16.0 / -11.1%, anime C -14.1%,
  // anime D -13.4%, live action -10.0%, anime A -9.0% BD-rate, and another 0.6-2.1% with the
  // MVs of related blocks tried; anime A q130 10.2 -> 7.7 fps).
  bool rd_pass = true;
  // 8 or 10. qindex keeps the 8-bit scale: at 10 bits the frames code qindex + 48 (AV2 shares one
  // quantizer table, and 24 qindex double the step), so a qindex gives about the same quality.
  int bit_depth = 8;
  void validate() const;  // throws std::invalid_argument
  // The qindex coded for qindex q (8-bit scale).
  int coded_qindex(int q) const { return q + 24 * (bit_depth - 8); }
  int qindex_8bit(int coded) const { return coded - 24 * (bit_depth - 8); }
  // Sent with every keyframe when set: the colour description (content_interpretation_obu(); 0
  // sends none), and HDR static metadata (empty: none), the mastering display (24 bytes: R, G, B
  // and white point x / y in 0.16, max / min luminance in 24.8 / 18.14 cd/m2, big-endian) and the
  // content light level (MaxCLL, MaxFALL: 2 bytes each).
  int colour = 0;
  bool full_range = false;
  std::vector<uint8_t> mdcv, cll;
};

// The smallest AV2 level from 6.0 up whose maximum picture size, width, height and display rate
// fit the sequence, else SEQ_LEVEL_MAX (no level constraints). The level bounds the reference
// frames a decoder accepts (8 needs the picture within the level's maximum) and the tile limits.
AV2_LEVEL choose_seq_level(int width, int height, int fps_num, int fps_den);
// Fills `seq` with exactly the values the AVM decoder derives when it parses our header.
void fill_sequence_params(const SeqConfig& c, SequenceHeader* seq);
// Sequence header OBU payload (including trailing bits), serialized from fill_sequence_params().
std::vector<uint8_t> write_sequence_header(const SeqConfig& c);
