#include <stdexcept>
#include <vector>

#include "bitstream/obu.h"
#include "bitstream/seq_header.h"
#include "check.h"

extern "C" {
#include "avm/avm_decoder.h"
#include "avm/avmdx.h"
}

static int peek(int w, int h) {
  SeqConfig c{w, h, 30, 1, 120};
  const auto obu = make_obu(OBU_SEQUENCE_HEADER, write_sequence_header(c));
  avm_codec_stream_info_t si{};
  CHECK_EQ(int(avm_codec_peek_stream_info(avm_codec_av2_dx(), obu.data(), obu.size(), &si)),
           int(AVM_CODEC_OK));
  CHECK_EQ(int(si.w), w);
  CHECK_EQ(int(si.h), h);
  return 0;
}

static int seqheader_peek_matches() { return peek(1920, 1080); }
static int seqheader_odd_size() { return peek(201, 121); }

static int seqconfig_rejects_qindex0() {
  for (int q : {0, 256}) {
    SeqConfig c{64, 64, 30, 1, q};
    bool threw = false;
    try { c.validate(); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
  }
  return 0;
}

// The smallest level from 6.0 up whose picture size, dimensions and display rate fit, else
// "no level constraints"; up to 8K at 30 fps this stays the old fixed 6.0.
static int seq_level_follows_frame_size() {
  CHECK_EQ(int(choose_seq_level(1920, 1080, 30, 1)), int(SEQ_LEVEL_6_0));
  CHECK_EQ(int(choose_seq_level(7680, 4320, 30, 1)), int(SEQ_LEVEL_6_0));
  CHECK_EQ(int(choose_seq_level(7680, 4320, 60, 1)), int(SEQ_LEVEL_6_1));
  CHECK_EQ(int(choose_seq_level(15360, 8640, 30, 1)), int(SEQ_LEVEL_7_0));
  CHECK_EQ(int(choose_seq_level(30720, 17280, 30, 1)), int(SEQ_LEVEL_8_0));
  CHECK_EQ(int(choose_seq_level(40000, 20000, 30, 1)), int(SEQ_LEVEL_MAX));
  SequenceHeader s;
  fill_sequence_params(SeqConfig{15360, 8640, 30, 1, 120}, &s);
  CHECK_EQ(int(s.seq_max_level_idx), int(SEQ_LEVEL_7_0));
  return 0;
}

int main() {
  if (int r = seq_level_follows_frame_size()) return r;
  if (int r = seqheader_peek_matches()) return r;
  if (int r = seqheader_odd_size()) return r;
  if (int r = seqconfig_rejects_qindex0()) return r;
  return 0;
}
