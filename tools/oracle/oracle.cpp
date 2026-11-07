#include "oracle.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

namespace celv2::oracle {
namespace {

const char* const kNames[kKinds] = {"128x128 split into 64x64", "128x64 split into 64x64", "64x128 split into 64x64",
                                    "VERT superblock in raster order", "same-reference compound", "NEAR with no candidate",
                                    "skip mode with no candidate", "other compound mode", "CfL not allowed",
                                    "BAWP not allowed", "AMVD set", "MV precision", "MV moved", "transform split not codable",
                                    "forward skip coding"};
std::atomic<long long> counts[kKinds], areas[kKinds];

void print() {
  for (int k = 0; k < kKinds; k++)
    std::fprintf(stderr, "oracle: %s: %lld blocks, %lld samples\n", kNames[k], counts[k].load(), areas[k].load());
}

BlockDecision decision(const OracleBlock& b, int w4, int h4, bool piece) {
  BlockDecision d;
  d.w4 = static_cast<uint8_t>(w4), d.h4 = static_cast<uint8_t>(h4);
  d.inter = b.inter != 0;
  if (b.inter) {
    d.ref = static_cast<uint8_t>(b.ref[0]), d.mv_row = b.mv[0][0], d.mv_col = b.mv[0][1];
    if (b.ref[1] >= 0) d.ref2 = static_cast<uint8_t>(b.ref[1]), d.mv2_row = b.mv[1][0], d.mv2_col = b.mv[1][1];
  }
  d.imp = &b, d.piece = piece;
  return d;
}

// The superblock's tree and leaves laid onto Frame::sbs. r, c, h, w in mi units.
struct Reader {
  Frame& f;
  const uint8_t* tree;
  const OracleBlock* leaf;
  int mi_rows, mi_cols, sb_cols;

  Sb& sb(int r, int c) { return f.sbs[static_cast<size_t>(r / 16) * sb_cols + c / 16]; }
  long long area(int r, int c, int h, int w) const {
    return 16LL * std::min(h, mi_rows - r) * std::min(w, mi_cols - c);
  }

  // A node inside one 64x64. Past the frame: a stand-in leaf the coder never reaches.
  void small(Sb& s, int r, int c, int h, int w) {
    const uint8_t p = *tree++;
    if (p == 255) {
      s.tree.push_back(Part::None), s.blocks.push_back(s.blocks.back());
    } else if (p == 0) {
      s.tree.push_back(Part::None), s.blocks.push_back(decision(*leaf++, w, h, false));
    } else if (p == 1) {
      s.tree.push_back(Part::Horz);
      small(s, r, c, h / 2, w), small(s, r + h / 2, c, h / 2, w);
    } else if (p == 2) {
      s.tree.push_back(Part::Vert);
      small(s, r, c, h, w / 2), small(s, r, c + w / 2, h, w / 2);
    } else {
      s.tree.push_back(Part::Split);
      for (int i = 0; i < 4; i++) small(s, r + (i >> 1) * h / 2, c + (i & 1) * w / 2, h / 2, w / 2);
    }
  }

  // A node of 128 on some side, or a 64x64: root says the superblock's own partition.
  void node(int r, int c, int h, int w, bool under_vert) {
    if (h <= 16 && w <= 16) {
      if (r >= mi_rows || c >= mi_cols) return void(tree++);
      if (under_vert) count(kRootVert, area(r, c, h, w));
      return small(sb(r, c), r, c, h, w);
    }
    const uint8_t p = *tree++;
    if (p == 255) return;
    if (p == 0) {
      const OracleBlock& b = *leaf++;
      count(h == 32 && w == 32 ? kSplit128x128 : w == 32 ? kSplit128x64 : kSplit64x128, area(r, c, h, w));
      bool piece = false;
      for (int y = r; y < r + h && y < mi_rows; y += 16)
        for (int x = c; x < c + w && x < mi_cols; x += 16) {
          sb(y, x).tree = {Part::None};
          sb(y, x).blocks = {decision(b, 16, 16, piece)};
          piece = true;
        }
    } else if (p == 1) {
      node(r, c, h / 2, w, under_vert), node(r + h / 2, c, h / 2, w, under_vert);
    } else if (p == 2) {
      node(r, c, h, w / 2, h == 32 && w == 32), node(r, c + w / 2, h, w / 2, h == 32 && w == 32);
    } else {
      for (int i = 0; i < 4; i++) node(r + (i >> 1) * h / 2, c + (i & 1) * w / 2, h / 2, w / 2, false);
    }
  }
};

struct File {
  int level = 0, max_drl_bits = 0;
  bool filters = false;
  std::map<int, Frame> frames;  // by display; the first coded with a display number wins
  std::map<int, int> depths;     // by display: one more than the deeper of the frames coded before on each side
};

File load() {
  File out;
  const char* const path = std::getenv("X_ORACLE");
  if (!path) return out;
  const char* const lv = std::getenv("X_ORACLE_LEVEL");
  out.level = lv ? std::atoi(lv) : 0;
  if (out.level < 1 || out.level > 6) throw std::runtime_error("X_ORACLE_LEVEL: 1 to 6");
  out.filters = out.level >= 4 && !std::getenv("X_ORACLE_OWN_FILTERS");
  std::ifstream in(path, std::ios::binary);
  char magic[8];
  int32_t size[2];
  in.read(magic, 8).read(reinterpret_cast<char*>(size), 8);
  if (!in || std::memcmp(magic, "CELV2OR1", 8)) throw std::runtime_error("X_ORACLE: not an oracle_dump file");
  const int mi_cols = (size[0] + 3) / 4 + ((size[0] + 3) / 4 & 1), mi_rows = (size[1] + 3) / 4 + ((size[1] + 3) / 4 & 1);
  const int sb_cols = (size[0] + 63) / 64, sb_rows = (size[1] + 63) / 64;
  OracleFrame h;
  while (in.read(reinterpret_cast<char*>(&h), sizeof(h))) {
    std::vector<uint8_t> tree(h.n_tree);
    std::vector<OracleBlock> leaves(h.n_blocks);
    std::vector<uint8_t> tx_part(static_cast<size_t>(h.mi_rows) * h.mi_cols);
    std::vector<OracleTx> txs(h.n_tx);
    std::vector<OracleCoef> coefs(h.n_coef);
    in.read(reinterpret_cast<char*>(tree.data()), tree.size());
    in.read(reinterpret_cast<char*>(leaves.data()), leaves.size() * sizeof(OracleBlock));
    in.read(reinterpret_cast<char*>(tx_part.data()), tx_part.size());
    in.read(reinterpret_cast<char*>(txs.data()), txs.size() * sizeof(OracleTx));
    in.read(reinterpret_cast<char*>(coefs.data()), coefs.size() * sizeof(OracleCoef));
    OracleFilters fl;
    in.read(reinterpret_cast<char*>(&fl), sizeof(fl));
    std::vector<int8_t> cdef_idx(fl.cdef_units);
    in.read(reinterpret_cast<char*>(cdef_idx.data()), cdef_idx.size());
    std::vector<uint8_t> ccso_flags[3], lr_ff[3], lr_type[3], lr_filter[3];
    std::vector<int32_t> lr_at[3];
    for (int p = 0; p < 3; p++) {
      ccso_flags[p].resize(fl.ccso_units[p]);
      in.read(reinterpret_cast<char*>(ccso_flags[p].data()), ccso_flags[p].size());
    }
    for (int p = 0; p < 3; p++) {
      if (!fl.lr_units[p]) continue;
      lr_ff[p].resize(fl.wiener_bytes);
      lr_type[p].resize(fl.lr_units[p]);
      in.read(reinterpret_cast<char*>(lr_ff[p].data()), lr_ff[p].size());
      in.read(reinterpret_cast<char*>(lr_type[p].data()), lr_type[p].size());
      int n = 0;
      lr_at[p].assign(fl.lr_units[p], -1);
      for (int i = 0; i < fl.lr_units[p]; i++)
        if (lr_type[p][i] == 2) lr_at[p][i] = n++;  // RESTORE_WIENER_NONSEP
      lr_filter[p].resize(static_cast<size_t>(n) * fl.wiener_bytes);
      in.read(reinterpret_cast<char*>(lr_filter[p].data()), lr_filter[p].size());
    }
    if (!in) throw std::runtime_error("X_ORACLE: truncated file");
    out.max_drl_bits = std::max(out.max_drl_bits, h.max_drl_bits);
    if (out.frames.count(h.display)) continue;
    {
      const auto right = out.depths.lower_bound(h.display);
      int d = 0;
      if (!h.key) d = right == out.depths.end() ? 1 : std::min(5, std::max(right->second, std::prev(right)->second) + 1);
      out.depths[h.display] = d;
    }
    Frame& f = out.frames[h.display];
    f.head = h, f.leaves = std::move(leaves);
    f.tx_part = std::move(tx_part), f.txs = std::move(txs), f.coefs = std::move(coefs);
    f.filters = fl, f.cdef_idx = std::move(cdef_idx);
    for (int p = 0; p < 3; p++) {
      f.ccso_flags[p] = std::move(ccso_flags[p]), f.lr_frame_filter[p] = std::move(lr_ff[p]);
      f.lr_unit_type[p] = std::move(lr_type[p]), f.lr_unit_filter[p] = std::move(lr_filter[p]), f.lr_unit_at[p] = std::move(lr_at[p]);
    }
    for (int p = 0; p < 3; p++) f.tx_grid[p].assign(static_cast<size_t>(h.mi_rows >> (p ? 1 : 0)) * (h.mi_cols >> (p ? 1 : 0)), -1);
    f.first_coef.resize(f.txs.size());
    for (size_t i = 0, c = 0; i < f.txs.size(); c += f.txs[i++].n_coef) {
      const OracleTx& t = f.txs[i];
      f.first_coef[i] = static_cast<uint32_t>(c);
      f.tx_grid[t.plane][static_cast<size_t>(t.row) * (h.mi_cols >> (t.plane ? 1 : 0)) + t.col] = static_cast<int32_t>(i);
    }
    f.sbs.resize(static_cast<size_t>(sb_cols) * sb_rows);
    Reader rd{f, tree.data(), f.leaves.data(), mi_rows, mi_cols, sb_cols};
    for (int r = 0; r < mi_rows; r += 32)
      for (int c = 0; c < mi_cols; c += 32) rd.node(r, c, 32, 32, false);
    if (rd.tree != tree.data() + tree.size() || rd.leaf != f.leaves.data() + f.leaves.size())
      throw std::runtime_error("X_ORACLE: a tree does not fit its frame");
    rd.tree = tree.data(), rd.leaf = f.leaves.data();
    for (int r = 0; r < mi_rows; r += 32)
      for (int c = 0; c < mi_cols; c += 32) {
        f.sbs128.emplace_back();
        rd.small(f.sbs128.back(), r, c, 32, 32);
      }
    for (const Sb& s : f.sbs)
      if (s.tree.empty()) throw std::runtime_error("X_ORACLE: a 64x64 without a decision");
  }
  std::atexit(print);
  return out;
}

const File& file() {
  static const File f = load();
  return f;
}

class ImportEngine : public DecisionEngine {
 public:
  explicit ImportEngine(std::unique_ptr<DecisionEngine> inner) : inner_(std::move(inner)) {}
  void launch(const Frame420& src, const std::vector<const Frame420*>& refs, int qindex, DecisionMap& dm,
              const std::vector<long long>& ref_keys, long long src_key) override {
    cur_ = frame(static_cast<int>(src_key));
    if (level() == 1) return inner_->launch(src, refs, qindex, dm, ref_keys, src_key);
    const bool whole = std::getenv("X_ORACLE_128") != nullptr;
    const std::vector<Sb>& sbs = whole ? cur_->sbs128 : cur_->sbs;
    dm.sb_size = whole ? 128 : 64;
    dm.sb_cols = (src.width + dm.sb_size - 1) / dm.sb_size, dm.sb_rows = (src.height + dm.sb_size - 1) / dm.sb_size;
    dm.sbs.resize(sbs.size());
    for (size_t i = 0; i < dm.sbs.size(); i++) {
      dm.sbs[i] = SbDecision{};
      dm.sbs[i].tree = sbs[i].tree, dm.sbs[i].blocks = sbs[i].blocks;
    }
  }
  void prepare(const Frame420& next, long long key) override {
    if (level() == 1) inner_->prepare(next, key);
  }
  void collect(DecisionMap& dm, RowGate& gate) override {
    if (level() == 1) return inner_->collect(dm, gate);
    gate.publish(dm.sb_rows - 1);
  }
  void set_row_streams(int n) override { inner_->set_row_streams(n); }
  const Sb* forced(size_t sb) const override { return level() == 1 ? &cur_->sbs[sb] : nullptr; }

 private:
  std::unique_ptr<DecisionEngine> inner_;
  const Frame* cur_ = nullptr;
};

}  // namespace

void count(Kind k, long long area) { counts[k]++, areas[k] += area; }

int real_cost() {
  static const int m = [] {
    int r = 0;
    if (const char* s = std::getenv("X_ORACLE_Q"))
      for (; *s; s++) r |= *s == 'a' ? kRealTcq : *s == 'b' ? kRealDepth : *s == 'c' ? kRealRefresh : *s == 'd' ? kRealPlain : 0;
    return r;
  }();
  return m;
}

bool real_check() {
  static const bool on = std::getenv("X_ORACLE_QCHECK") != nullptr;
  return on;
}

int depth(int display) {
  const File& f = file();
  const auto it = f.depths.find(display);
  return it == f.depths.end() ? 5 : it->second;
}

int level() { return file().level; }

bool filters() { return file().filters; }

int max_drl_bits() { return file().max_drl_bits; }

const OracleFrame& first_head() { return file().frames.begin()->second.head; }

const Frame* frame(int display) {
  const File& f = file();
  if (!f.level) return nullptr;
  const auto it = f.frames.find(display);
  if (it == f.frames.end()) throw std::runtime_error("X_ORACLE: no frame at display " + std::to_string(display));
  return &it->second;
}

std::unique_ptr<DecisionEngine> import_engine(std::unique_ptr<DecisionEngine> engine) {
  return std::make_unique<ImportEngine>(std::move(engine));
}

}  // namespace celv2::oracle
