#include "gpu/engine.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

#include "gpu/inter_common.h"
#include "gpu/kernels.h"
#include "gpu/kernels_ref.h"

namespace {

void cuda_ok(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <typename T>
struct DevBuf {
  T* p = nullptr;
  size_t n = 0;
  void resize(size_t count) {
    if (count <= n) return;
    if (p) cudaFree(p);
    p = nullptr, n = 0;  // a failed allocation below leaves an empty buffer, not a stale size
    cuda_ok(cudaMalloc(&p, count * sizeof(T)), "cudaMalloc");
    n = count;
  }
  ~DevBuf() {
    if (p) cudaFree(p);
  }
};

// Page-locked host staging: copies from it are truly asynchronous, so a frame's uploads, kernels
// and row downloads queue up on the stream instead of stalling at every pageable copy.
template <typename T>
struct PinnedBuf {
  T* p = nullptr;
  size_t n = 0;
  void resize(size_t count) {
    if (count <= n) return;
    if (p) cudaFreeHost(p);
    p = nullptr, n = 0;
    cuda_ok(cudaMallocHost(&p, count * sizeof(T)), "cudaMallocHost");
    n = count;
  }
  ~PinnedBuf() {
    if (p) cudaFreeHost(p);
  }
};

}  // namespace

struct GpuEngine::Impl {
  int width, height, sb_size, sb_cols, sb_rows;
  int k = 1;  // blocks_per_sb(sb_cols)
  cudaStream_t stream = nullptr;  // every upload, kernel and download of this engine
  std::vector<cudaEvent_t> ev;    // ev[r]: superblock row r's SbOut is in host_out
  int row_streams = 1;            // set_row_streams
  std::vector<int> order;         // the superblock rows in the order launched
  // A reference's luma pyramid (level 0 with its quarter-pel phases), kept across launches under
  // its key: a reconstruction is the reference of several frames.
  struct RefSlot {
    long long key = -1;  // -1: none (or the caller gave no keys)
    long long used = -1;  // the launch that last used it
    DevBuf<uint8_t> lv[celv2::kMeLevels];
    celv2::DevPlane plane[celv2::kMeLevels];
    PinnedBuf<uint8_t> stage;
  };
  std::vector<std::unique_ptr<RefSlot>> slots;
  long long launches = 0;
  // A source frame's luma pyramid and intra costs (launch_intra). Two: while one frame is decided,
  // prepare() can make the next one's.
  struct Src {
    long long key = -1;  // prepare()d for the launch with this src_key (-1: none)
    DevBuf<uint8_t> lv[celv2::kMeLevels];
    celv2::DevPlane plane[celv2::kMeLevels];
    PinnedBuf<uint8_t> stage;
    DevBuf<int64_t> intra;
  };
  Src srcs[2];
  int cur_set = 0;  // the Src of the last launch
  int64_t last_lam = 0;
  // Each reference's pyramid is in slots; the planes of the launch and their motion fields.
  celv2::DevPlane cur_plane[celv2::kMeLevels], ref_plane[celv2::kMaxRefs][celv2::kMeLevels];
  DevBuf<int> mvx[celv2::kMaxRefs][celv2::kMeLevels], mvy[celv2::kMaxRefs][celv2::kMeLevels];
  int mv_cols[celv2::kMeLevels], mv_rows[celv2::kMeLevels];
  DevBuf<int64_t> costs;  // cost_kernel -> finalize_kernel, costs_per_sb(refs) per superblock
  DevBuf<celv2::SbOut> out;
  PinnedBuf<celv2::SbOut> host_out;

  ~Impl() {
    if (stream) cudaStreamSynchronize(stream);  // the buffers freed after this body may be in use
    for (cudaEvent_t e : ev)
      if (e) cudaEventDestroy(e);
    if (stream) cudaStreamDestroy(stream);
  }

  // Uploads the padded luma plane and builds its pyramid (queued on stream); a reference's level 0
  // also gets its quarter-pel phases.
  void upload(const Frame420& f, PinnedBuf<uint8_t>& stage, DevBuf<uint8_t>* lv,
              celv2::DevPlane* planes, int levels, bool phases = false) {
    const size_t n = f.y.size();
    lv[0].resize(phases ? n * celv2::kSubpelPhases : n);
    stage.resize(n);
    std::memcpy(stage.p, f.y.data(), n);
    cuda_ok(cudaMemcpyAsync(lv[0].p, stage.p, n, cudaMemcpyHostToDevice, stream), "upload luma");
    planes[0] = celv2::DevPlane{lv[0].p, f.stride_y(), f.pw, f.ph};
    if (phases) {
      celv2::launch_subpel(planes[0], lv[0].p, n, stream);
      planes[0].phase = n;
    }
    for (int l = 1; l < levels; l++) {
      const int w = planes[l - 1].w / 2, h = planes[l - 1].h / 2;
      lv[l].resize(static_cast<size_t>(w) * h);
      celv2::launch_downsample(planes[l - 1], lv[l].p, stream);
      planes[l] = celv2::DevPlane{lv[l].p, w, w, h};
    }
  }
};

GpuEngine::GpuEngine(int width, int height, int sb_size) {
  if (width <= 0 || height <= 0) throw std::runtime_error("GpuEngine: frame size must be positive");
  if (sb_size != 64) throw std::runtime_error("GpuEngine: only 64x64 superblocks are supported");
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
    throw std::runtime_error("GpuEngine: no CUDA device available");
  impl_ = std::make_unique<Impl>();
  Impl& m = *impl_;
  m.width = width;
  m.height = height;
  m.sb_size = sb_size;
  m.sb_cols = (width + sb_size - 1) / sb_size;
  m.sb_rows = (height + sb_size - 1) / sb_size;
  // Before any stream exists: the process-wide one-time init runs on the default stream and
  // synchronizes the device, so it can never overlap an engine's queued work.
  m.k = celv2::blocks_per_sb(m.sb_cols);
  cuda_ok(cudaStreamCreate(&m.stream), "cudaStreamCreate");
  m.ev.assign(m.sb_rows, nullptr);
  for (cudaEvent_t& e : m.ev)
    cuda_ok(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "cudaEventCreate");
}

GpuEngine::~GpuEngine() = default;

ProbeResult probe_gpu() {
  ProbeResult r;
  int devices = 0;
  cudaError_t e = cudaGetDeviceCount(&devices);
  if (e != cudaSuccess) {
    cudaGetLastError();  // clear the sticky error
    r.reason = cudaGetErrorString(e);
    return r;
  }
  if (devices < 1) {
    r.reason = "no CUDA device";
    return r;
  }
  cudaDeviceProp prop{};
  e = cudaGetDeviceProperties(&prop, 0);
  if (e != cudaSuccess) {
    cudaGetLastError();
    r.reason = cudaGetErrorString(e);
    return r;
  }
  r.name = prop.name;
  r.cc_major = prop.major;
  r.cc_minor = prop.minor;
  if (prop.major * 10 + prop.minor < 75) {
    r.reason = "compute " + std::to_string(prop.major) + "." + std::to_string(prop.minor) +
               " is older than Turing (7.5)";
    return r;
  }
  try {
    celv2::blocks_per_sb(1);
  } catch (const std::exception& ex) {
    r.reason = ex.what();
    return r;
  }
  r.ok = true;
  return r;
}

DecisionMap GpuEngine::decide_intra(const Frame420& src, int qindex) {
  DecisionMap dm;
  RowGate gate;
  launch(src, nullptr, qindex, dm);
  collect(dm, gate);
  return dm;
}

DecisionMap GpuEngine::decide_inter(const Frame420& src, const Frame420& prev_src, int qindex) {
  return decide_inter(src, std::vector<const Frame420*>{&prev_src}, qindex);
}

DecisionMap GpuEngine::decide_inter(const Frame420& src, const std::vector<const Frame420*>& refs, int qindex) {
  DecisionMap dm;
  RowGate gate;
  launch(src, refs, qindex, dm);
  collect(dm, gate);
  return dm;
}

void GpuEngine::prepare(const Frame420& next, long long key) {
  Impl& m = *impl_;
  if (key < 0 || next.width != m.width || next.height != m.height) return;
  // The Src the last launch does not use: its previous work (two launches back) was collected.
  Impl::Src& s = m.srcs[1 - m.cur_set];
  try {
    m.upload(next, s.stage, s.lv, s.plane, celv2::kMeLevels);
    s.intra.resize(static_cast<size_t>(m.sb_cols) * m.sb_rows * celv2::kIntraCosts);
    celv2::launch_intra(s.plane[0], m.sb_cols, 0, m.sb_rows, m.k, s.intra.p, m.stream);
    s.key = key;
  } catch (...) {
    s.key = -1;
    throw;
  }
}

void GpuEngine::launch(const Frame420& src, const std::vector<const Frame420*>& refs, int qindex,
                       DecisionMap& dm, const std::vector<long long>& ref_keys, long long src_key) {
  Impl& m = *impl_;
  if (src.width != m.width || src.height != m.height)
    throw std::runtime_error("GpuEngine: source size does not match the engine");
  for (const Frame420* r : refs)
    if (r->width != m.width || r->height != m.height)
      throw std::runtime_error("GpuEngine: reference size does not match the engine");
  if (refs.size() > static_cast<size_t>(celv2::kMaxRefs)) throw std::runtime_error("GpuEngine: too many references");
  const int nrefs = static_cast<int>(refs.size());
  try {  // a failure must not leave uploads, rows or downloads queued on the stream
    // The Src prepare()d for this frame, else the other one, uploaded now (its intra costs per row).
    int use = -1;
    for (int i = 0; i < 2; i++)
      if (src_key >= 0 && m.srcs[i].key == src_key) use = i;
    const bool prepared = use >= 0;
    if (!prepared) {
      use = 1 - m.cur_set;
      m.upload(src, m.srcs[use].stage, m.srcs[use].lv, m.srcs[use].plane, celv2::kMeLevels);
    }
    m.srcs[use].key = -1;  // used: a later prepare() may overwrite it
    m.cur_set = use;
    Impl::Src& cs = m.srcs[use];
    for (int l = 0; l < celv2::kMeLevels; l++) m.cur_plane[l] = cs.plane[l];
    celv2::DevRefs dr;
    dr.n = nrefs;
    if (!ref_keys.empty() && ref_keys.size() != refs.size())
      throw std::invalid_argument("GpuEngine: one key per reference");
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
        m.upload(*refs[ri], slot->stage, slot->lv, slot->plane, celv2::kMeLevels, true);
        slot->key = key;
      }
      slot->used = now;
      for (int l = 0; l < celv2::kMeLevels; l++) m.ref_plane[ri][l] = slot->plane[l];
      for (int l = celv2::kMeLevels - 1; l >= 0; l--) {
        m.mv_cols[l] = (m.cur_plane[l].w + 7) / 8;
        m.mv_rows[l] = (m.cur_plane[l].h + 7) / 8;
        const size_t n = static_cast<size_t>(m.mv_cols[l]) * m.mv_rows[l];
        m.mvx[ri][l].resize(n);
        m.mvy[ri][l].resize(n);
        const bool top = l == celv2::kMeLevels - 1;
        celv2::launch_me_level(m.cur_plane[l], m.ref_plane[ri][l], top ? nullptr : m.mvx[ri][l + 1].p,
                              top ? nullptr : m.mvy[ri][l + 1].p, top ? 0 : m.mv_cols[l + 1],
                              top ? 0 : m.mv_rows[l + 1], m.mvx[ri][l].p, m.mvy[ri][l].p, m.mv_cols[l],
                              m.mv_rows[l], l == 0, m.stream);
      }
      dr.plane[ri] = m.ref_plane[ri][0];
      dr.mvx[ri] = m.mvx[ri][0].p;
      dr.mvy[ri] = m.mvy[ri][0].p;
      dr.mv_cols = m.mv_cols[0];
      dr.mv_rows = m.mv_rows[0];
    }
    const size_t nsb = static_cast<size_t>(m.sb_cols) * m.sb_rows;
    m.costs.resize(nsb * celv2::costs_per_sb(nrefs));
    cs.intra.resize(nsb * celv2::kIntraCosts);
    m.out.resize(nsb);
    m.host_out.resize(nsb);
    const int64_t lam = lambda_q8(qindex);
    m.last_lam = lam;
    // Row streams: the first row of each run of rows, then the second, ... (set_row_streams).
    const int streams = std::max(1, std::min(m.row_streams, m.sb_rows));
    const int run = (m.sb_rows + streams - 1) / streams;
    m.order.clear();
    for (int i = 0; i < run; i++)
      for (int s = 0; s < streams; s++)
        if (s * run + i < m.sb_rows) m.order.push_back(s * run + i);
    for (int r : m.order) {
      const size_t first = static_cast<size_t>(r) * m.sb_cols;
      if (!prepared) celv2::launch_intra(m.cur_plane[0], m.sb_cols, r, 1, m.k, cs.intra.p, m.stream);
      celv2::launch_costs(m.cur_plane[0], dr, m.sb_cols, r, 1, m.k, lam, m.costs.p, m.stream);
      celv2::launch_finalize(m.cur_plane[0], dr, m.sb_cols, r, 1, lam, cs.intra.p, m.costs.p, m.out.p, m.stream);
      cuda_ok(cudaMemcpyAsync(m.host_out.p + first, m.out.p + first,
                              m.sb_cols * sizeof(celv2::SbOut), cudaMemcpyDeviceToHost, m.stream),
              "download decisions");
      cuda_ok(cudaEventRecord(m.ev[r], m.stream), "cudaEventRecord");
    }
    dm.sb_size = m.sb_size;
    dm.sb_cols = m.sb_cols;
    dm.sb_rows = m.sb_rows;
    dm.sbs.assign(nsb, SbDecision{});
  } catch (...) {
    cudaStreamSynchronize(m.stream);
    throw;
  }
}

void GpuEngine::set_row_streams(int n) { impl_->row_streams = n; }

void GpuEngine::collect(DecisionMap& dm, RowGate& gate) {
  Impl& m = *impl_;
  try {
    for (int r : m.order) {
      cuda_ok(cudaEventSynchronize(m.ev[r]), "decision");
      const size_t first = static_cast<size_t>(r) * m.sb_cols;
      for (size_t i = first; i < first + m.sb_cols; i++) celv2::sb_out_to_decision(m.host_out.p[i], dm.sbs[i]);
      gate.publish_one(r);
    }
  } catch (...) {
    cudaStreamSynchronize(m.stream);  // leave no queued work behind, even on failure
    gate.fail(std::current_exception());
    throw;
  }
}

std::vector<int64_t> GpuEngine::intra_costs(int sb) const {
  const Impl& m = *impl_;
  std::vector<int64_t> out(celv2::kIntraCosts);
  cuda_ok(cudaMemcpy(out.data(), m.srcs[m.cur_set].intra.p + static_cast<size_t>(sb) * celv2::kIntraCosts,
                     out.size() * sizeof(int64_t), cudaMemcpyDeviceToHost),
          "download costs");
  for (size_t i = 0; i < out.size(); i++) out[i] += m.last_lam * celv2::mode_bits(static_cast<int>(i % celv2::kNumModes));
  return out;
}

std::vector<int> gpu_satd_batch(const uint8_t* a, const uint8_t* b, int n, int w4, int h4) {
  DevBuf<uint8_t> da, db;
  DevBuf<int> dout;
  da.resize(static_cast<size_t>(n) * 4096);
  db.resize(static_cast<size_t>(n) * 4096);
  dout.resize(n);
  cuda_ok(cudaMemcpy(da.p, a, static_cast<size_t>(n) * 4096, cudaMemcpyHostToDevice), "upload a");
  cuda_ok(cudaMemcpy(db.p, b, static_cast<size_t>(n) * 4096, cudaMemcpyHostToDevice), "upload b");
  celv2::launch_satd_batch(da.p, db.p, n, w4, h4, dout.p);
  std::vector<int> out(n);
  cuda_ok(cudaMemcpy(out.data(), dout.p, n * sizeof(int), cudaMemcpyDeviceToHost), "download satd");
  return out;
}
