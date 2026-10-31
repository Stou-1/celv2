#include "vk/vk_engine.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "gpu/inter_common.h"
#include "gpu/kernels_ref.h"
#include "gpu/sb_out.h"
#include "vk/vk_kernels.h"

namespace {

constexpr int kIntraCosts = celv2::kSbBlocks * celv2::kNumModes;  // gpu/kernels.h
constexpr int kInterSlots = 4400;
// Workgroups a superblock of the cost kernel; any number gives the same costs.
constexpr int kShares = 4;

}  // namespace

// GpuEngine::Impl's structure on Vulkan: the work is recorded in batches on one queue (a launch's
// motion search, then each superblock row with the download of its decisions, so collect() takes
// rows as they finish), references are kept under their keys, and prepare() uploads the next
// source and its intra costs behind the current launch.
struct VulkanEngine::Impl {
  int width, height, sb_size, sb_cols, sb_rows;
  celv2::vk::Context ctx;
  celv2::vk::Kernels k{ctx};
  // A luma pyramid (level 0, a reference's with its quarter-pel phases), with its upload buffer.
  struct Pyramid {
    celv2::vk::Buffer lv[celv2::kMeLevels];
    celv2::vk::DevPlane plane[celv2::kMeLevels];
    celv2::vk::HostBuffer stage;
  };
  // A reference's pyramid kept across launches under its key: a reconstruction is the reference
  // of several frames.
  struct RefSlot {
    long long key = -1;   // -1: none (or the caller gave no keys)
    long long used = -1;  // the launch that last used it
    Pyramid p;
  };
  std::vector<std::unique_ptr<RefSlot>> slots;
  long long launches = 0;
  // A source's pyramid and intra costs. Two: while one frame is decided, prepare() makes the next one's.
  struct Src {
    long long key = -1;  // prepare()d for the launch with this src_key (-1: none)
    Pyramid p;
    celv2::vk::Buffer intra;
  };
  Src srcs[2];
  int cur_set = 0;  // the Src of the last launch
  celv2::vk::Buffer mvx[celv2::kMaxRefs][celv2::kMeLevels], mvy[celv2::kMaxRefs][celv2::kMeLevels];
  // The cost kernel's output for one superblock row (rows run one after the other on the queue, so
  // the next row's costs wait for this row's finalize): 1/sb_rows of the frame's (8 GB at 16K).
  celv2::vk::Buffer costs, out;
  celv2::vk::HostBuffer host_out;
  int row_streams = 1;
  std::vector<int> order;          // the superblock rows in the order launched
  std::vector<uint64_t> row_done;  // row_done[r]: the batch that brings row r's SbOut to host_out

  Impl(int w, int h, int sb, int device)
      : width(w), height(h), sb_size(sb), sb_cols((w + sb - 1) / sb), sb_rows((h + sb - 1) / sb), ctx(device),
        row_done(sb_rows, 0) {}
  ~Impl() {
    try {
      ctx.finish();
    } catch (...) {  // a lost device: nothing left to wait for
    }
    auto free_pyramid = [&](Pyramid& p) {
      for (celv2::vk::Buffer& b : p.lv) ctx.free(b);
      ctx.free(p.stage);
    };
    for (Src& s : srcs) free_pyramid(s.p), ctx.free(s.intra);
    for (auto& s : slots) free_pyramid(s->p);
    for (int r = 0; r < celv2::kMaxRefs; r++)
      for (int l = 0; l < celv2::kMeLevels; l++) ctx.free(mvx[r][l]), ctx.free(mvy[r][l]);
    ctx.free(costs), ctx.free(out), ctx.free(host_out);
  }

  // b holds at least n bytes (reallocated when smaller).
  void ensure(celv2::vk::Buffer& b, size_t n) {
    if (b.size >= n) return;
    ctx.free(b);
    b = ctx.buffer(n);
  }
  void ensure(celv2::vk::HostBuffer& b, size_t n) {
    if (b.size >= n) return;
    ctx.free(b);
    b = ctx.host_buffer(n);
  }

  // GpuEngine::Impl::upload, recorded in the open batch: the padded luma plane and its pyramid; a
  // reference's level 0 also gets its quarter-pel phases.
  void upload(const Frame420& f, Pyramid& p, bool phases) {
    const size_t n = f.y.size();
    ensure(p.lv[0], phases ? n * celv2::kSubpelPhases : n);
    ensure(p.stage, n);
    std::memcpy(p.stage.map, f.y.data(), n);
    ctx.copy(p.lv[0], 0, p.stage, 0, n);
    p.plane[0] = {&p.lv[0], f.stride_y(), f.pw, f.ph};
    if (phases) {
      k.subpel(p.plane[0], p.lv[0], n);
      p.plane[0].phase = n;
    }
    for (int l = 1; l < celv2::kMeLevels; l++) {
      const int w = p.plane[l - 1].w / 2, h = p.plane[l - 1].h / 2;
      ensure(p.lv[l], static_cast<size_t>(w) * h);
      k.downsample(p.plane[l - 1], p.lv[l]);
      p.plane[l] = {&p.lv[l], w, w, h};
    }
  }
};

VulkanEngine::VulkanEngine(int width, int height, int sb_size, int device) {
  if (width <= 0 || height <= 0) throw std::runtime_error("VulkanEngine: frame size must be positive");
  if (sb_size != 64) throw std::runtime_error("VulkanEngine: only 64x64 superblocks are supported");
  impl_ = std::make_unique<Impl>(width, height, sb_size, device);
}

VulkanEngine::~VulkanEngine() = default;

const std::string& VulkanEngine::device_name() const { return impl_->ctx.name(); }

void VulkanEngine::set_row_streams(int n) { impl_->row_streams = n; }

void VulkanEngine::prepare(const Frame420& next, long long key) {
  Impl& m = *impl_;
  if (key < 0 || next.width != m.width || next.height != m.height) return;
  // The Src the last launch does not use: its previous work (two launches back) was collected.
  Impl::Src& s = m.srcs[1 - m.cur_set];
  s.key = -1;
  m.ctx.begin_batch();
  try {
    m.upload(next, s.p, false);
    m.ensure(s.intra, static_cast<size_t>(m.sb_cols) * m.sb_rows * kIntraCosts * sizeof(int64_t));
    m.k.intra(s.p.plane[0], m.sb_cols, 0, m.sb_rows, kShares, s.intra);
  } catch (...) {
    m.ctx.finish();
    throw;
  }
  m.ctx.submit();
  s.key = key;
}

void VulkanEngine::launch(const Frame420& src, const std::vector<const Frame420*>& refs, int qindex, DecisionMap& dm,
                          const std::vector<long long>& ref_keys, long long src_key) {
  Impl& m = *impl_;
  if (src.width != m.width || src.height != m.height)
    throw std::runtime_error("VulkanEngine: source size does not match the engine");
  for (const Frame420* r : refs)
    if (r->width != m.width || r->height != m.height)
      throw std::runtime_error("VulkanEngine: reference size does not match the engine");
  if (refs.size() > static_cast<size_t>(celv2::kMaxRefs)) throw std::runtime_error("VulkanEngine: too many references");
  if (!ref_keys.empty() && ref_keys.size() != refs.size())
    throw std::invalid_argument("VulkanEngine: one key per reference");
  const int nrefs = static_cast<int>(refs.size());
  const size_t nsb = static_cast<size_t>(m.sb_cols) * m.sb_rows;
  try {  // a failure leaves no batch open or running
    m.ctx.begin_batch();
    // The Src prepare()d for this frame, else the other one, uploaded now (its intra costs per row).
    int use = -1;
    for (int i = 0; i < 2; i++)
      if (src_key >= 0 && m.srcs[i].key == src_key) use = i;
    const bool prepared = use >= 0;
    if (!prepared) {
      use = 1 - m.cur_set;
      m.upload(src, m.srcs[use].p, false);
      m.ensure(m.srcs[use].intra, nsb * kIntraCosts * sizeof(int64_t));
    }
    m.srcs[use].key = -1;  // used: a later prepare() may overwrite it
    m.cur_set = use;
    Impl::Src& cs = m.srcs[use];
    celv2::vk::Refs vr;
    vr.n = nrefs;
    const long long now = ++m.launches;
    for (int ri = 0; ri < nrefs; ri++) {
      // The slot holding this key, else one that no reference of this launch needs, else a new one.
      const long long key = ref_keys.empty() ? -1 : ref_keys[ri];
      Impl::RefSlot* slot = nullptr;
      for (auto& s : m.slots)
        if (key >= 0 && s->key == key) slot = s.get();
      if (!slot) {
        for (auto& s : m.slots)
          if (s->used != now && std::find(ref_keys.begin(), ref_keys.end(), s->key) == ref_keys.end()) {
            slot = s.get();
            break;
          }
        if (!slot) m.slots.push_back(std::make_unique<Impl::RefSlot>()), slot = m.slots.back().get();
        slot->key = -1;
        m.upload(*refs[ri], slot->p, true);
        slot->key = key;
      }
      slot->used = now;
      int cols = 0, rows = 0;
      for (int l = celv2::kMeLevels - 1; l >= 0; l--) {  // GpuEngine::launch's motion search
        const int pcols = cols, prows = rows;
        cols = (cs.p.plane[l].w + 7) / 8, rows = (cs.p.plane[l].h + 7) / 8;
        const size_t n = static_cast<size_t>(cols) * rows * sizeof(int);
        m.ensure(m.mvx[ri][l], n), m.ensure(m.mvy[ri][l], n);
        const bool top = l == celv2::kMeLevels - 1;
        m.k.me_level(cs.p.plane[l], slot->p.plane[l], top ? nullptr : &m.mvx[ri][l + 1],
                     top ? nullptr : &m.mvy[ri][l + 1], top ? 0 : pcols, top ? 0 : prows, m.mvx[ri][l], m.mvy[ri][l],
                     cols, rows, l == 0);
      }
      vr.plane[ri] = slot->p.plane[0];
      vr.mvx[ri] = &m.mvx[ri][0], vr.mvy[ri] = &m.mvy[ri][0];
      vr.mv_cols = cols, vr.mv_rows = rows;
    }
    const size_t sb_costs = static_cast<size_t>(std::max(1, nrefs)) * kInterSlots * sizeof(int64_t);
    m.ensure(m.costs, m.sb_cols * sb_costs);
    m.ensure(m.out, nsb * sizeof(celv2::SbOut));
    m.ensure(m.host_out, nsb * sizeof(celv2::SbOut));
    m.ctx.submit();
    const int64_t lam = lambda_q8(qindex);
    // Row streams: the first row of each run of rows, then the second, ... (set_row_streams).
    const int streams = std::max(1, std::min(m.row_streams, m.sb_rows));
    const int run = (m.sb_rows + streams - 1) / streams;
    m.order.clear();
    for (int i = 0; i < run; i++)
      for (int s = 0; s < streams; s++)
        if (s * run + i < m.sb_rows) m.order.push_back(s * run + i);
    for (int r : m.order) {
      m.ctx.begin_batch();
      if (!prepared) m.k.intra(cs.p.plane[0], m.sb_cols, r, 1, kShares, cs.intra);
      m.k.costs(cs.p.plane[0], vr, m.sb_cols, r, 1, kShares, lam, m.costs);
      m.k.finalize(cs.p.plane[0], vr, m.sb_cols, r, 1, lam, cs.intra, m.costs, m.out);
      const size_t first = static_cast<size_t>(r) * m.sb_cols * sizeof(celv2::SbOut);
      m.ctx.copy(m.host_out, first, m.out, first, m.sb_cols * sizeof(celv2::SbOut));
      m.row_done[r] = m.ctx.submit();
    }
  } catch (...) {
    m.ctx.finish();
    throw;
  }
  dm.sb_size = m.sb_size;
  dm.sb_cols = m.sb_cols;
  dm.sb_rows = m.sb_rows;
  dm.sbs.assign(nsb, SbDecision{});
}

void VulkanEngine::collect(DecisionMap& dm, RowGate& gate) {
  Impl& m = *impl_;
  try {
    const celv2::SbOut* const out = static_cast<const celv2::SbOut*>(m.host_out.map);
    for (int r : m.order) {
      m.ctx.wait(m.row_done[r]);
      const size_t first = static_cast<size_t>(r) * m.sb_cols;
      for (size_t i = first; i < first + m.sb_cols; i++) celv2::sb_out_to_decision(out[i], dm.sbs[i]);
      gate.publish_one(r);
    }
  } catch (...) {
    gate.fail(std::current_exception());
    throw;
  }
}
