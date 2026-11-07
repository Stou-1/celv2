#include "verify/avm_decode.h"

#include "io/y4m.h"

#include <cstdio>
#include <stdexcept>

extern "C" {
#include "avm/avm_decoder.h"
#include "avm/avmdx.h"
#include "common/md5_utils.h"
}

namespace {

std::string hex(const unsigned char d[16]) {
  char s[33];
  for (int i = 0; i < 16; i++) std::snprintf(s + 2 * i, 3, "%02x", d[i]);
  return std::string(s, 32);
}

// Hashes rows of a plane; `wide` means 16-bit samples holding 8-bit values.
void md5_plane(MD5Context* ctx, const uint8_t* p, int stride, int w, int h, bool wide) {
  std::vector<uint8_t> row(w);
  for (int r = 0; r < h; r++) {
    const uint8_t* src = p + static_cast<ptrdiff_t>(r) * stride;
    if (wide) {
      const uint16_t* s16 = reinterpret_cast<const uint16_t*>(src);
      for (int c = 0; c < w; c++) row[c] = static_cast<uint8_t>(s16[c]);
      MD5Update(ctx, row.data(), w);
    } else {
      MD5Update(ctx, src, w);
    }
  }
}

template <typename T>
void copy_plane(T* dst, int ds, const uint8_t* src, int ss, int w, int h, bool wide) {
  for (int r = 0; r < h; r++) {
    const uint8_t* s = src + static_cast<ptrdiff_t>(r) * ss;
    for (int c = 0; c < w; c++)
      dst[r * ds + c] = static_cast<T>(wide ? reinterpret_cast<const uint16_t*>(s)[c] : s[c]);
  }
}

}  // namespace

std::string i420_md5(const uint8_t* y, int ys, const uint8_t* u, const uint8_t* v, int uvs, int w,
                     int h) {
  MD5Context ctx;
  MD5Init(&ctx);
  const int cw = (w + 1) / 2, ch = (h + 1) / 2;
  md5_plane(&ctx, y, ys, w, h, false);
  md5_plane(&ctx, u, uvs, cw, ch, false);
  md5_plane(&ctx, v, uvs, cw, ch, false);
  unsigned char d[16];
  MD5Final(d, &ctx);
  return hex(d);
}

std::vector<Frame420> decode_frames(const std::vector<std::vector<uint8_t>>& temporal_units, bool film_grain) {
  avm_codec_ctx_t codec;
  if (avm_codec_dec_init(&codec, avm_codec_av2_dx(), nullptr, 0) != AVM_CODEC_OK)
    throw std::runtime_error("avm_codec_dec_init failed");
  if (avm_codec_control(&codec, AV2D_SET_SKIP_FILM_GRAIN, film_grain ? 0 : 1) != AVM_CODEC_OK) {
    avm_codec_destroy(&codec);
    throw std::runtime_error("AV2D_SET_SKIP_FILM_GRAIN failed");
  }
  std::vector<Frame420> out;
  try {
    for (const auto& tu : temporal_units) {
      if (avm_codec_decode(&codec, tu.data(), tu.size(), nullptr) != AVM_CODEC_OK) {
        const char* detail = avm_codec_error_detail(&codec);
        throw std::runtime_error(std::string("AVM decode failed: ") + avm_codec_error(&codec) +
                                 (detail ? std::string(" / ") + detail : std::string()));
      }
      avm_codec_iter_t it = nullptr;
      while (avm_image_t* img = avm_codec_get_frame(&codec, &it)) {
        if (img->bit_depth != 8 && img->bit_depth != 10) throw std::runtime_error("AVM output is not 8- or 10-bit");
        const bool wide = (img->fmt & AVM_IMG_FMT_HIGHBITDEPTH) != 0;
        Frame420 f = make_frame(static_cast<int>(img->d_w), static_cast<int>(img->d_h), static_cast<int>(img->bit_depth));
        const int cw = (f.width + 1) / 2, ch = (f.height + 1) / 2;
        auto copy = [&](auto& y, auto& u, auto& v) {
          copy_plane(y.data(), f.stride_y(), img->planes[0], img->stride[0], f.width, f.height, wide);
          copy_plane(u.data(), f.stride_uv(), img->planes[1], img->stride[1], cw, ch, wide);
          copy_plane(v.data(), f.stride_uv(), img->planes[2], img->stride[2], cw, ch, wide);
        };
        if (f.bit_depth > 8) copy(f.hy, f.hu, f.hv);
        else copy(f.y, f.u, f.v);
        pad_frame(f);
        round_to_8bit(f);
        out.push_back(std::move(f));
      }
    }
  } catch (...) {
    avm_codec_destroy(&codec);
    throw;
  }
  avm_codec_destroy(&codec);
  return out;
}

std::vector<std::string> decode_frame_md5s(const std::vector<std::vector<uint8_t>>& temporal_units) {
  std::vector<std::string> out;
  for (const auto& f : decode_frames(temporal_units)) out.push_back(frame_md5(f));
  return out;
}

std::string first_diff(const Frame420& a, const Frame420& b) {
  if (a.width != b.width || a.height != b.height) return "frame sizes differ";
  const char* names[3] = {"Y", "U", "V"};
  if (a.bit_depth != b.bit_depth) return "bit depths differ";
  auto plane_diff = [&](int p, const auto& pa, const auto& pb) -> std::string {
    const int stride = p ? a.stride_uv() : a.stride_y();
    const int w = p ? (a.width + 1) / 2 : a.width, h = p ? (a.height + 1) / 2 : a.height;
    for (int r = 0; r < h; r++)
      for (int c = 0; c < w; c++)
        if (pa[r * stride + c] != pb[r * stride + c]) {
          char msg[160];
          std::snprintf(msg, sizeof(msg), "first diff: plane %s 4x4 block (row %d, col %d), pixel "
                        "(%d,%d): %d vs %d", names[p], r / 4, c / 4, r, c, pa[r * stride + c],
                        pb[r * stride + c]);
          return msg;
        }
    return "";
  };
  for (int p = 0; p < 3; p++) {
    const std::string d = a.bit_depth > 8 ? plane_diff(p, p == 0 ? a.hy : p == 1 ? a.hu : a.hv, p == 0 ? b.hy : p == 1 ? b.hu : b.hv)
                                          : plane_diff(p, p == 0 ? a.y : p == 1 ? a.u : a.v, p == 0 ? b.y : p == 1 ? b.u : b.v);
    if (!d.empty()) return d;
  }
  return "";
}

std::string frame_md5(const Frame420& f) {
  if (f.bit_depth > 8) {
    const std::vector<uint8_t> bytes = write_y4m_frame_bytes(f);  // 16-bit little-endian samples
    MD5Context ctx;
    MD5Init(&ctx);
    MD5Update(&ctx, bytes.data(), static_cast<unsigned>(bytes.size()));
    unsigned char d[16];
    MD5Final(d, &ctx);
    return hex(d);
  }
  return i420_md5(f.y.data(), f.stride_y(), f.u.data(), f.v.data(), f.stride_uv(), f.width,
                  f.height);
}
