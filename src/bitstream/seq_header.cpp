// Portions of this file are derived from the AVM v1.0.0 decoder (av2/decoder),
// Copyright (c) 2021, Alliance for Open Media, under the BSD 3-Clause Clear License
// and the Alliance for Open Media Patent License 1.0: see third_party/avm/.

#include "bitstream/seq_header.h"

#include <cstring>
#include <stdexcept>

extern "C" {
#include "av2/common/level.h"
#include "av2/common/quant_common.h"
#include "avm_dsp/bitwriter_buffer.h"
}

void SeqConfig::validate() const {
  if (width <= 0 || height <= 0) throw std::invalid_argument("frame size must be positive");
  if (qindex < 1 || qindex > 255) throw std::invalid_argument("qindex must be 1..255");
  if (fps_num <= 0 || fps_den <= 0) throw std::invalid_argument("frame rate must be positive");
  if (tile_cols_log2 < 0 || tile_cols_log2 > 6 || tile_rows_log2 < 0 || tile_rows_log2 > 6)
    throw std::invalid_argument("tile columns/rows log2 must be 0..6");
  if (threads < 0) throw std::invalid_argument("threads must be >= 0");
  if (refs < 1 || refs > INTER_REFS_PER_FRAME) throw std::invalid_argument("refs must be 1..7");
  if (deblock_delta < -16 || deblock_delta > 15) throw std::invalid_argument("deblock delta must be -16..15");
  if (bit_depth != 8 && bit_depth != 10) throw std::invalid_argument("bit depth must be 8 or 10");
}

static int bits_for(int v) {  // number of bits needed to code v (v >= 1)
  int n = 1;
  while ((1 << n) <= v) n++;
  return n;
}

AV2_LEVEL choose_seq_level(int width, int height, int fps_num, int fps_den) {
  const int64_t pixels = static_cast<int64_t>(width) * height;
  const double rate = static_cast<double>(pixels) * fps_num / fps_den;  // luma samples per second
  for (int l = SEQ_LEVEL_6_0; l < SEQ_LEVELS; l++) {
    const AV2LevelSpec& d = av2_level_defs[l];
    if (pixels <= d.max_picture_size && width <= d.max_h_size && height <= d.max_v_size &&
        rate <= static_cast<double>(d.max_display_rate))
      return static_cast<AV2_LEVEL>(l);
  }
  return SEQ_LEVEL_MAX;
}

void fill_sequence_params(const SeqConfig& c, SequenceHeader* s) {
  c.validate();
  std::memset(s, 0, sizeof(*s));
  s->seq_header_id = 0;
  s->seq_profile_idc = MAIN_420_10_IP0;
  s->seq_max_level_idx = choose_seq_level(c.width, c.height, c.fps_num, c.fps_den);
  s->seq_tier = 0;
  s->subsampling_x = 1;
  s->subsampling_y = 1;
  s->monochrome = 0;
  s->bit_depth = c.bit_depth == 10 ? AVM_BITS_10 : AVM_BITS_8;
  s->monotonic_output_order_flag = 1;
  s->seq_max_mlayer_cnt = 1;
  s->num_bits_width = bits_for(c.width - 1 > 0 ? c.width - 1 : 1);
  s->num_bits_height = bits_for(c.height - 1 > 0 ? c.height - 1 : 1);
  s->max_frame_width = c.width;
  s->max_frame_height = c.height;
  // Values the decoder assigns when the display/decoder model info is absent.
  s->seq_max_initial_display_delay_minus_1 = BUFFER_POOL_MAX_SIZE - 1;
  s->decoder_model_info.num_units_in_decoding_tick = 1;
  s->seq_max_decoder_buffer_delay = 70000;
  s->seq_max_encoder_buffer_delay = 20000;
  // Partition group: 128x128 superblocks over the engines' 64x64 decisions (TileCoder::code_tile()):
  // -1.0% BD-rate on anime A 1080p, -0.2% / -0.4% on anime B / live action, 0.96x the speed.
  // No SDP/extended partitions, default aspect limit.
  s->sb_size = BLOCK_128X128;
  s->mib_size = mi_size_wide[s->sb_size];
  s->mib_size_log2 = mi_size_wide_log2[s->sb_size];
  s->max_pb_aspect_ratio_log2_m1 = 2;
  // Inter group: translation only, 8-frame DPB, everything else off.
  s->seq_enabled_motion_modes = (1 << SIMPLE_TRANSLATION);
  s->order_hint_info.order_hint_bits_minus_1 = 6;
  s->enable_bawp = 1;  // block-adaptive weighted prediction (BlockWriter::inter_block_mode_info())
  s->enable_drl_reorder = DRL_REORDER_DISABLED;
  s->ref_frames = 8;
  s->ref_frames_log2 = 3;
  // The MV bank and 5 MV-stack candidates instead of 2: -1.2% BD-rate on anime A 1080p, -1.35% /
  // -0.8% on the anime clips (6 candidates: -1.1% on anime A). Temporal MVs: SeqConfig::tmv.
  s->enable_refmvbank = 1;
  s->def_max_drl_bits = MIN_MAX_DRL_BITS + 3;
  s->def_max_bvp_drl_bits = MIN_MAX_IBC_DRL_BITS;
  s->enable_opfl_refine = AVM_OPFL_REFINE_NONE;
  // Screen content tools off; integer MV then defaults to "select".
  s->force_screen_content_tools = 0;
  s->force_integer_mv = 2;
  // Intra group: the edge filter and intra bi-prediction only change the prediction (no block
  // syntax), so AVM's predictor applies them in celv2's reconstruction too.
  s->enable_intra_edge_filter = 1;
  s->enable_adaptive_mvd = c.amvd;  // NEWMV only (BlockWriter::inter_block_mode_info())
  s->order_hint_info.enable_ref_frame_mvs = c.tmv;
  s->enable_flex_mvres = c.flex_mv;
  s->enable_mrls = c.mrls;  // intra reference lines 1-3 (BlockWriter::intra_modes() picks one)
  s->enable_ibp = 1;
  // Transform/quant/entropy group.
  s->enable_chroma_dctonly = 1;  // chroma always DCT_DCT (celv2 has only a DCT forward transform)
  s->reduced_tx_part_set = 1;
  s->equal_ac_dc_q = 1;
  // Filter group: CDEF (not on blocks without residual) or no in-loop filter but deblocking.
  s->enable_cdef = c.cdef;
  s->enable_cdef_on_skip_txfm = CDEF_ON_SKIP_TXFM_DISABLED;
  s->enable_ccso = c.ccso;  // after CDEF, offsets from the deblocked luma (CodecState::ccso())
  s->ccso_unit_matches_sb_size = 0;
  s->df_par_bits_minus2 = c.deblock_delta || c.deblock_search ? 3 : 0;  // 5-bit deblocking deltas when one is used
  s->enable_tcq = c.tcq ? TCQ_8ST : TCQ_DISABLE;  // trellis coded quantization of luma (every frame)
  s->enable_global_motion = 1;  // rotation / zoom of the whole frame (core/global_motion.h)
  // Forward skip coding (inter IDTX luma coded from its first level on, without TCQ; celv2 codes no
  // intra block in FSC mode) works but stays off: on anime A 1080p the bytes moved by 0.002% and on an
  // anime clip q90 it lost 0.06 dB at the same size (IDTX without TCQ).
  s->enable_fsc = 0;
  s->enable_idtx_intra = s->enable_fsc;  // implied by forward skip coding
}

static void write_sequence_tools(const SequenceHeader* s, avm_write_bit_buffer* wb) {
  // Partition group (decodeframe.c read_sequence_partition_group_tool_flags).
  avm_wb_write_bit(wb, s->sb_size == BLOCK_256X256);
  if (s->sb_size != BLOCK_256X256) avm_wb_write_bit(wb, s->sb_size == BLOCK_128X128);
  avm_wb_write_bit(wb, s->enable_sdp);
  avm_wb_write_bit(wb, s->enable_ext_partitions);
  avm_wb_write_bit(wb, s->max_pb_aspect_ratio_log2_m1 < 2);
  if (s->max_pb_aspect_ratio_log2_m1 < 2) avm_wb_write_bit(wb, s->max_pb_aspect_ratio_log2_m1);
  // Segment group.
  avm_wb_write_bit(wb, s->enable_ext_seg);
  avm_wb_write_bit(wb, s->seq_seg_info_present_flag);
  // Intra group.
  avm_wb_write_bit(wb, s->enable_intra_dip);
  avm_wb_write_bit(wb, s->enable_intra_edge_filter);
  avm_wb_write_bit(wb, s->enable_mrls);
  avm_wb_write_bit(wb, s->enable_cfl_intra);
  avm_wb_write_literal(wb, s->cfl_ds_filter_index, 2);
  avm_wb_write_bit(wb, s->enable_mhccp);
  avm_wb_write_bit(wb, s->enable_ibp);
  // Inter group (not a single-picture header).
  for (int mm = INTERINTRA; mm < MOTION_MODES; mm++)
    avm_wb_write_bit(wb, (s->seq_enabled_motion_modes >> mm) & 1);
  avm_wb_write_bit(wb, s->enable_masked_compound);
  avm_wb_write_bit(wb, s->order_hint_info.enable_ref_frame_mvs);
  if (s->order_hint_info.enable_ref_frame_mvs) avm_wb_write_bit(wb, s->order_hint_info.reduced_ref_frame_mvs_mode);
  avm_wb_write_literal(wb, s->order_hint_info.order_hint_bits_minus_1, 4);
  avm_wb_write_bit(wb, s->enable_refmvbank);
  avm_wb_write_bit(wb, s->enable_drl_reorder == DRL_REORDER_DISABLED);
  if (s->enable_drl_reorder != DRL_REORDER_DISABLED)
    avm_wb_write_bit(wb, s->enable_drl_reorder == DRL_REORDER_CONSTRAINT);
  avm_wb_write_bit(wb, s->enable_explicit_ref_frame_map);
  avm_wb_write_bit(wb, s->ref_frames != 8);
  avm_wb_write_literal(wb, s->number_of_bits_for_lt_frame_id, 3);
  avm_wb_write_primitive_quniform(wb, MAX_MAX_DRL_BITS - MIN_MAX_DRL_BITS + 1,
                                  s->def_max_drl_bits - MIN_MAX_DRL_BITS);
  avm_wb_write_bit(wb, s->allow_frame_max_drl_bits);
  avm_wb_write_primitive_quniform(wb, MAX_MAX_IBC_DRL_BITS - MIN_MAX_IBC_DRL_BITS + 1,
                                  s->def_max_bvp_drl_bits - MIN_MAX_IBC_DRL_BITS);
  avm_wb_write_bit(wb, s->allow_frame_max_bvp_drl_bits);
  avm_wb_write_literal(wb, s->num_same_ref_compound, 2);
  avm_wb_write_bit(wb, s->enable_tip != 0);
  avm_wb_write_bit(wb, s->enable_mv_traj);
  avm_wb_write_bit(wb, s->enable_bawp);
  avm_wb_write_bit(wb, s->enable_cwp);
  avm_wb_write_bit(wb, s->enable_imp_msk_bld);
  avm_wb_write_bit(wb, s->enable_lf_sub_pu);
  avm_wb_write_literal(wb, s->enable_opfl_refine, 2);
  avm_wb_write_bit(wb, s->enable_refinemv);
  avm_wb_write_bit(wb, s->enable_bru > 0);
  avm_wb_write_bit(wb, s->enable_adaptive_mvd);
  avm_wb_write_bit(wb, s->enable_mvd_sign_derive);
  avm_wb_write_bit(wb, s->enable_flex_mvres);
  avm_wb_write_bit(wb, s->enable_global_motion);
  avm_wb_write_bit(wb, s->enable_short_refresh_frame_flags);
  // Screen content group: tools forced off.
  avm_wb_write_bit(wb, 0);
  avm_wb_write_bit(wb, s->force_screen_content_tools);
  // Transform/quant/entropy group.
  avm_wb_write_bit(wb, s->enable_fsc);
  if (!s->enable_fsc) avm_wb_write_bit(wb, s->enable_idtx_intra);
  avm_wb_write_bit(wb, s->enable_ist);
  avm_wb_write_bit(wb, s->enable_inter_ist);
  avm_wb_write_bit(wb, s->enable_chroma_dctonly);
  avm_wb_write_bit(wb, s->enable_inter_ddt);
  avm_wb_write_bit(wb, s->reduced_tx_part_set);
  avm_wb_write_bit(wb, s->enable_cctx);
  avm_wb_write_bit(wb, s->enable_tcq != 0);
  if (s->enable_tcq) avm_wb_write_bit(wb, s->enable_tcq - 1);  // single_picture_header_flag is 0
  if (s->enable_tcq == TCQ_DISABLE || s->enable_tcq >= TCQ_8ST_FR)
    avm_wb_write_bit(wb, s->enable_parity_hiding);
  avm_wb_write_bit(wb, s->enable_avg_cdf);
  avm_wb_write_bit(wb, s->separate_uv_delta_q);
  avm_wb_write_bit(wb, s->equal_ac_dc_q);
  avm_wb_write_unsigned_literal(wb, s->base_uv_ac_delta_q - DELTA_DCQUANT_MIN, DELTA_DCQUANT_BITS);
  avm_wb_write_bit(wb, s->uv_ac_delta_q_enabled);
  // Filter group.
  avm_wb_write_bit(wb, s->disable_loopfilters_across_tiles);
  avm_wb_write_bit(wb, s->enable_cdef);
  avm_wb_write_bit(wb, s->enable_gdf);
  avm_wb_write_bit(wb, s->enable_restoration);
  avm_wb_write_bit(wb, s->enable_ccso);
  if (s->enable_ccso) avm_wb_write_bit(wb, s->ccso_unit_matches_sb_size);
  avm_wb_write_bit(wb, s->enable_cdef_on_skip_txfm == CDEF_ON_SKIP_TXFM_ALWAYS_ON);
  avm_wb_write_bit(wb, s->enable_cdef_on_skip_txfm == CDEF_ON_SKIP_TXFM_DISABLED);
  avm_wb_write_literal(wb, s->df_par_bits_minus2, 2);
  // Tile config.
  avm_wb_write_bit(wb, s->seq_tile_info_present_flag);
}

std::vector<uint8_t> write_sequence_header(const SeqConfig& c) {
  SequenceHeader s;
  fill_sequence_params(c, &s);
  std::vector<uint8_t> buf(256, 0);
  avm_write_bit_buffer wb = {buf.data(), 0};
  avm_wb_write_uvlc(&wb, s.seq_header_id);
  avm_wb_write_literal(&wb, s.seq_profile_idc, PROFILE_BITS);
  avm_wb_write_bit(&wb, s.single_picture_header_flag);
  avm_wb_write_literal(&wb, s.seq_max_level_idx, LEVEL_BITS);
  avm_wb_write_bit(&wb, s.seq_tier);  // level >= 4.0 and not single-picture
  avm_wb_write_uvlc(&wb, CHROMA_FORMAT_420);
  avm_wb_write_uvlc(&wb, s.bit_depth == AVM_BITS_10 ? 0 : 1);  // bit-depth LUT index: 10->0, 8->1, 12->2
  avm_wb_write_literal(&wb, s.seq_lcr_id, 3);
  avm_wb_write_bit(&wb, s.still_picture);
  avm_wb_write_literal(&wb, s.max_tlayer_id, TLAYER_BITS);
  avm_wb_write_literal(&wb, s.max_mlayer_id, MLAYER_BITS);
  avm_wb_write_bit(&wb, s.monotonic_output_order_flag);
  avm_wb_write_literal(&wb, s.num_bits_width - 1, 4);
  avm_wb_write_literal(&wb, s.num_bits_height - 1, 4);
  avm_wb_write_literal(&wb, s.max_frame_width - 1, s.num_bits_width);
  avm_wb_write_literal(&wb, s.max_frame_height - 1, s.num_bits_height);
  avm_wb_write_bit(&wb, s.conf.conf_win_enabled_flag);
  avm_wb_write_bit(&wb, s.seq_max_display_model_info_present_flag);
  avm_wb_write_bit(&wb, s.decoder_model_info_present_flag);
  write_sequence_tools(&s, &wb);
  avm_wb_write_bit(&wb, s.film_grain_params_present);
  avm_wb_write_bit(&wb, s.seq_extension_present_flag);
  // Trailing bits.
  if (avm_wb_is_byte_aligned(&wb))
    avm_wb_write_literal(&wb, 0x80, 8);
  else
    avm_wb_write_bit(&wb, 1);
  buf.resize(avm_wb_bytes_written(&wb));
  return buf;
}
