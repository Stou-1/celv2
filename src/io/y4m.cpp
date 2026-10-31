#include "io/y4m.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

Frame420 make_frame(int width, int height, int bit_depth) {
  if (width <= 0 || height <= 0) throw std::runtime_error("frame size must be positive");
  if (bit_depth != 8 && bit_depth != 10) throw std::runtime_error("bit depth must be 8 or 10");
  Frame420 f;
  f.bit_depth = bit_depth;
  f.width = width;
  f.height = height;
  f.pw = (width + 7) & ~7;
  f.ph = (height + 7) & ~7;
  f.y.assign(static_cast<size_t>(f.pw) * f.ph, 0);
  f.u.assign(static_cast<size_t>(f.pw / 2) * (f.ph / 2), 0);
  f.v.assign(f.u.size(), 0);
  if (bit_depth > 8) f.hy.assign(f.y.size(), 0), f.hu.assign(f.u.size(), 0), f.hv.assign(f.v.size(), 0);
  return f;
}

void round_to_8bit(Frame420& f) {
  if (f.bit_depth == 8) return;
  const int shift = f.bit_depth - 8;
  for (auto [h, p] : {std::pair{&f.hy, &f.y}, std::pair{&f.hu, &f.u}, std::pair{&f.hv, &f.v}})
    for (size_t i = 0; i < h->size(); i++)
      (*p)[i] = static_cast<uint8_t>(std::min(255, ((*h)[i] + (1 << (shift - 1))) >> shift));
}

template <typename T>
static void pad_plane(T* p, int stride, int w, int h, int pw, int ph) {
  for (int r = 0; r < h; r++)
    for (int c = w; c < pw; c++) p[r * stride + c] = p[r * stride + w - 1];
  for (int r = h; r < ph; r++)
    for (int c = 0; c < pw; c++) p[r * stride + c] = p[(h - 1) * stride + c];
}

void pad_frame(Frame420& f) {
  pad_plane(f.y.data(), f.stride_y(), f.width, f.height, f.pw, f.ph);
  const int cw = (f.width + 1) / 2, ch = (f.height + 1) / 2;
  pad_plane(f.u.data(), f.stride_uv(), cw, ch, f.pw / 2, f.ph / 2);
  pad_plane(f.v.data(), f.stride_uv(), cw, ch, f.pw / 2, f.ph / 2);
  if (f.bit_depth > 8) {
    pad_plane(f.hy.data(), f.stride_y(), f.width, f.height, f.pw, f.ph);
    pad_plane(f.hu.data(), f.stride_uv(), cw, ch, f.pw / 2, f.ph / 2);
    pad_plane(f.hv.data(), f.stride_uv(), cw, ch, f.pw / 2, f.ph / 2);
  }
}

Y4mReader::Y4mReader(const std::string& path) {
  if (path == "-") {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);  // text mode would rewrite CR/LF bytes in the frames
#endif
    f_ = stdin;
  } else {
    f_ = std::fopen(path.c_str(), "rb");
    if (!f_) throw std::runtime_error("cannot open " + path);
    owned_ = true;
  }
  try {  // the destructor does not run for a constructor that throws
    std::string header;
    if (!read_line(header)) throw std::runtime_error("empty y4m input: " + path);
    std::istringstream ss(header);
    std::string tok;
    ss >> tok;
    if (tok != "YUV4MPEG2") throw std::runtime_error("not a y4m file: " + path);
    std::string color = "420";
    while (ss >> tok) {
      switch (tok[0]) {
        case 'W': w_ = std::stoi(tok.substr(1)); break;
        case 'H': h_ = std::stoi(tok.substr(1)); break;
        case 'F': {
          const auto colon = tok.find(':');
          fps_num_ = std::stoi(tok.substr(1, colon - 1));
          fps_den_ = std::stoi(tok.substr(colon + 1));
          break;
        }
        case 'C': color = tok.substr(1); break;
        default: break;
      }
    }
    if (color == "420p10")
      bit_depth_ = 10;
    else if (color != "420" && color != "420jpeg" && color != "420mpeg2" && color != "420paldv")
      throw std::runtime_error("unsupported y4m colorspace C" + color + " (only 4:2:0 at 8 or 10 bits)");
    if (w_ <= 0 || h_ <= 0) throw std::runtime_error("y4m header has no size");
    out_depth_ = bit_depth_;
  } catch (...) {
    if (owned_) std::fclose(f_);
    throw;
  }
}

void Y4mReader::set_bit_depth(int d) {
  if (d != 8 && d != 10) throw std::invalid_argument("bit depth must be 8 or 10");
  out_depth_ = d;
}

Y4mReader::~Y4mReader() {
  if (owned_) std::fclose(f_);
}

bool Y4mReader::read_line(std::string& line) {
  line.clear();
  int c = std::fgetc(f_);
  if (c == EOF) return false;
  for (; c != EOF && c != '\n'; c = std::fgetc(f_)) line.push_back(static_cast<char>(c));
  return true;
}

bool Y4mReader::read(Frame420& out) {
  std::string line;
  if (!read_line(line)) return false;
  if (line.rfind("FRAME", 0) != 0) throw std::runtime_error("bad y4m frame marker");
  if (out.width != w_ || out.height != h_ || out.bit_depth != bit_depth_)
    out = make_frame(w_, h_, bit_depth_);  // pad_frame refills padding (the file's depth first)
  const int cw = (w_ + 1) / 2, ch = (h_ + 1) / 2;
  auto read_plane = [&](auto* p, int stride, int w, int h) {
    const size_t n = sizeof(*p);  // 2: little-endian samples, as Windows stores uint16_t
    for (int r = 0; r < h; r++)
      if (std::fread(p + static_cast<size_t>(r) * stride, n, w, f_) != static_cast<size_t>(w))
        throw std::runtime_error("truncated y4m frame");
  };
  if (bit_depth_ > 8) {
    read_plane(out.hy.data(), out.stride_y(), w_, h_);
    read_plane(out.hu.data(), out.stride_uv(), cw, ch);
    read_plane(out.hv.data(), out.stride_uv(), cw, ch);
    for (auto* h : {&out.hy, &out.hu, &out.hv})
      for (uint16_t& s : *h)
        if (s >> bit_depth_) throw std::runtime_error("y4m sample above the bit depth");
  } else {
    read_plane(out.y.data(), out.stride_y(), w_, h_);
    read_plane(out.u.data(), out.stride_uv(), cw, ch);
    read_plane(out.v.data(), out.stride_uv(), cw, ch);
  }
  pad_frame(out);
  round_to_8bit(out);
  if (out_depth_ == bit_depth_) return true;
  // To the depth asked for: 8 -> 10 exactly (the 8-bit planes stay as they are), 10 -> 8 rounded.
  if (out_depth_ > 8) {
    Frame420 wide = make_frame(w_, h_, out_depth_);
    for (auto [p8, p16] : {std::pair{&out.y, &wide.hy}, std::pair{&out.u, &wide.hu}, std::pair{&out.v, &wide.hv}})
      for (size_t i = 0; i < p8->size(); i++) (*p16)[i] = static_cast<uint16_t>((*p8)[i] << (out_depth_ - 8));
    wide.y.swap(out.y), wide.u.swap(out.u), wide.v.swap(out.v);
    out = std::move(wide);
  } else {
    out.bit_depth = 8;
    out.hy.clear(), out.hu.clear(), out.hv.clear();
  }
  return true;
}

std::vector<uint8_t> write_y4m_frame_bytes(const Frame420& f) {
  std::vector<uint8_t> b;
  const int cw = (f.width + 1) / 2, ch = (f.height + 1) / 2;
  auto put = [&](const auto* p, int stride, int w, int h) {
    for (int r = 0; r < h; r++) {
      const uint8_t* const s = reinterpret_cast<const uint8_t*>(p + r * stride);
      b.insert(b.end(), s, s + w * sizeof(*p));
    }
  };
  if (f.bit_depth > 8) {
    put(f.hy.data(), f.stride_y(), f.width, f.height);
    put(f.hu.data(), f.stride_uv(), cw, ch);
    put(f.hv.data(), f.stride_uv(), cw, ch);
    return b;
  }
  put(f.y.data(), f.stride_y(), f.width, f.height);
  put(f.u.data(), f.stride_uv(), cw, ch);
  put(f.v.data(), f.stride_uv(), cw, ch);
  return b;
}
