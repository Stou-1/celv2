#include "pipeline.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>
#include <future>
#include <utility>
#include <vector>
#include <stdexcept>
#include <thread>

#include "bitstream/obu.h"
#include "core/rd_pass.h"
#include "core/threads.h"
#include "gpu/cpu_engine.h"
#include "gpu/engine.h"
#ifdef CELV2_HAVE_VULKAN
#include "vk/vk_context.h"
#include "vk/vk_engine.h"
#endif

std::vector<std::pair<std::string, std::string>> list_devices(const std::function<ProbeResult()>& probe) {
  std::vector<std::pair<std::string, std::string>> out;
  const ProbeResult p = probe();
  if (p.ok) out.emplace_back("gpu", p.name + " (CUDA)");
#ifdef CELV2_HAVE_VULKAN
  const std::vector<std::string> vk = celv2::vk::Context::devices();
  for (size_t i = 0; i < vk.size() && i < 2; i++) out.emplace_back(i ? "vulkan2" : "vulkan", vk[i] + " (Vulkan)");
#endif
  out.emplace_back("cpu", "CPU");
  return out;
}

std::unique_ptr<DecisionEngine> make_engine(Device device, int width, int height, int tile_threads,
                                            const std::function<ProbeResult()>& probe,
                                            std::string* description) {
  constexpr int kSbSize = 64;
  if (device == Device::Vulkan || device == Device::Vulkan2) {
#ifdef CELV2_HAVE_VULKAN
    auto e = std::make_unique<VulkanEngine>(width, height, kSbSize, device == Device::Vulkan2 ? 1 : 0);
    *description = "decisions on Vulkan (" + e->device_name() + ")";
    return e;
#else
    throw std::runtime_error("this celv2enc was built without Vulkan");
#endif
  }
  if (device == Device::Cpu) {
    *description = "decisions on CPU (requested with --device cpu)";
  } else {
    const ProbeResult p = probe();
    if (p.ok) {
      *description = "decisions on GPU (" + p.name + ", compute " + std::to_string(p.cc_major) +
                     "." + std::to_string(p.cc_minor) + ")";
      if (device == Device::Gpu) return std::make_unique<GpuEngine>(width, height, kSbSize);
      // Auto is still at startup: a GPU that passes the probe but fails to start (e.g. out of
      // memory on a busy GPU) falls back to the CPU; an explicit --device gpu reports the error.
      try {
        return std::make_unique<GpuEngine>(width, height, kSbSize);
      } catch (const std::exception& e) {
        *description = std::string("decisions on CPU (GPU engine failed to start: ") + e.what() + ")";
      }
    } else {
      if (device == Device::Gpu) throw std::runtime_error("no usable GPU: " + p.reason);
      *description = "decisions on CPU (" + p.reason + ")";
    }
#ifdef CELV2_HAVE_VULKAN
    // Auto without CUDA: the first Vulkan GPU (the same decisions; anime A 1080p q130 coded at 3.1 fps
    // with the Arc Pro B50 against 0.09 fps with the CPU engine), else the CPU.
    try {
      auto e = std::make_unique<VulkanEngine>(width, height, kSbSize, 0);
      *description = "decisions on Vulkan (" + e->device_name() + ")";
      return e;
    } catch (const std::exception&) {
    }
#endif
  }
  // The decision workers and the tile threads together fit in the cores.
  const int cores = static_cast<int>(std::thread::hardware_concurrency());
  return std::make_unique<CpuEngine>(width, height, kSbSize, decision_threads(cores, tile_threads));
}

Pipeline::Pipeline(const SeqConfig& c, Device device)
    : cfg_(c),
      enc_(c),
      engine_(make_engine(device, c.width, c.height, enc_.tile_threads(), probe_gpu, &description_)) {
  // Every tile row starts coding as soon as its first superblock row is decided (16K with 4 tile
  // rows: the lower 12 of 16 tiles waited for the rows above them).
  engine_->set_row_streams(enc_.tile_rows());
}

namespace {

// A copy made on several threads (one took 0.04 s for a 16K frame). Above 8 bits, full_depth also
// copies the full-depth planes (the decisions need only the 8-bit ones).
Frame420 copy_frame(const Frame420& s, bool full_depth) {
  Frame420 d;
  d.width = s.width, d.height = s.height, d.pw = s.pw, d.ph = s.ph;
  d.bit_depth = full_depth ? s.bit_depth : 8;
  d.y.resize(s.y.size()), d.u.resize(s.u.size()), d.v.resize(s.v.size());
  if (d.bit_depth > 8) d.hy.resize(s.hy.size()), d.hu.resize(s.hu.size()), d.hv.resize(s.hv.size());
  const int n = std::max(1, std::min(16, static_cast<int>(s.y.size() >> 22)));  // 4 MB a thread at least
  auto part = [&](int k) {
    auto copy = [&](const auto& src, auto& dst) {
      const size_t a = src.size() * k / n, b = src.size() * (k + 1) / n;
      std::copy(src.begin() + a, src.begin() + b, dst.begin() + a);
    };
    copy(s.y, d.y), copy(s.u, d.u), copy(s.v, d.v);
    if (d.bit_depth > 8) copy(s.hy, d.hy), copy(s.hu, d.hu), copy(s.hv, d.hv);
  };
  celv2::parallel(n, part);
  return d;
}

}  // namespace

std::vector<uint8_t> Pipeline::encode(const Frame420& src, bool key) {
  std::vector<uint8_t> tu = make_obu(OBU_TEMPORAL_DELIMITER, {});
  const std::vector<uint8_t> obus = code(src, FrameSpec{key});
  tu.insert(tu.end(), obus.begin(), obus.end());
  return tu;
}

std::vector<uint8_t> Pipeline::code(const Frame420& src, const FrameSpec& given, const Frame420* next,
                                    int next_display) {
  FrameSpec spec = given;
  spec.qindex_delta += qindex_offset_;
  // The inter decision searches the reconstructions (what the decoder predicts from), so its cost
  // includes the references' coding error. launch() copies them.
  DecisionMap dm;
  RowGate gate;
  std::vector<const Frame420*> refs;
  std::vector<long long> keys;
  references(enc_.begin(spec), refs, keys);
  engine_->launch(src, refs, std::max(1, std::min(255, cfg_.qindex + spec.qindex_delta)), dm, keys, spec.display);
  if (next && next_display >= 0) engine_->prepare(*next, next_display);
  // Tile threads code each superblock row as soon as the collector publishes it.
  std::future<void> collector;
  try {
    collector = std::async(std::launch::async, [&] {
      collect(dm, gate, src, refs, std::max(1, std::min(255, cfg_.qindex + spec.qindex_delta)), spec.sb_dq);
    });
  } catch (...) {
    // No collector thread: drain the launched work here (collect leaves none pending on failure).
    try {
      RowGate unused;
      engine_->collect(dm, unused);
    } catch (...) {
    }
    throw;
  }
  std::vector<uint8_t> tu;
  try {
    tu = enc_.code(src, dm, &gate);
  } catch (...) {
    collector.wait();  // collect() uses dm and gate, and the next launch needs it finished
    throw;
  }
  collector.get();  // a decision error surfaces here
  keep(spec);
  return tu;
}

void Pipeline::collect(DecisionMap& dm, RowGate& gate, const Frame420& src, const std::vector<const Frame420*>& refs,
                       int qindex, const std::vector<int>& sb_dq) {
  if (!cfg_.rd_pass) return engine_->collect(dm, gate);
  RdPassParams p;
  p.qindex = qindex;
  p.tcq = cfg_.tcq;
  for (int t = 0; t < TX_SIZES_ALL; t++) {  // the pass's transforms: 4x8 to 64, at most 2:1
    const int w = tx_size_wide[t], h = tx_size_high[t];
    if (w <= 2 * h && h <= 2 * w) p.scale[t] = enc_.luma_scale(static_cast<TX_SIZE>(t));
  }
  // Workers take superblocks in raster order as the engine's rows arrive (on `inner`, in any
  // order); a row goes on to `gate` once all of its superblocks are done.
  RowGate inner;
  const size_t cols = static_cast<size_t>(dm.sb_cols), n = cols * dm.sb_rows;
  if (!sb_dq.empty() && sb_dq.size() != n) throw std::invalid_argument("collect: one sb_dq offset per decision superblock");
  std::atomic<size_t> next{0};
  std::unique_ptr<std::atomic<int>[]> done(new std::atomic<int>[dm.sb_rows]());
  std::atomic<bool> stop{false};
  auto work = [&] {
    try {
      for (size_t i; !stop && (i = next++) < n;) {
        const int r = static_cast<int>(i / cols);
        inner.wait_row(r);
        if (sb_dq.empty()) {
          rd_pass_sb(src, refs, static_cast<int>(i % cols) * dm.sb_size, r * dm.sb_size, p, dm.sbs[i]);
        } else {  // the superblock's own qindex (FrameSpec::sb_dq)
          RdPassParams ps = p;
          ps.qindex = std::max(1, std::min(255, qindex + sb_dq[i]));
          rd_pass_sb(src, refs, static_cast<int>(i % cols) * dm.sb_size, r * dm.sb_size, ps, dm.sbs[i]);
        }
        if (++done[r] == static_cast<int>(cols)) gate.publish_one(r);
      }
    } catch (...) {
      stop = true;
      gate.fail(std::current_exception());
    }
  };
  std::vector<std::thread> pool;
  // Both hardware threads of a core are left to each tile thread: 1080p (2 tiles) with 44 workers
  // instead of 48 ran 4-6% faster, the tiles no longer sharing cores with the pass (the same bytes).
  const int hw = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  const int nt = std::max({1, hw / 2, hw - 2 * enc_.tile_threads()});  // one at least, or no row is ever published
  std::exception_ptr err;
  try {
    for (int t = 0; t < nt; t++) pool.emplace_back(work);
    engine_->collect(dm, inner);  // fails `inner` on error, which ends the workers
  } catch (...) {
    err = std::current_exception();
    stop = true;
    inner.fail(err);
  }
  for (std::thread& t : pool) t.join();
  if (err) {
    gate.fail(err);
    std::rethrow_exception(err);
  }
}

// Reference keys, unique in the process: pipelines that take each other's state (sync_from) share
// them, and an engine must never see one key for two pictures.
static std::atomic<long long> next_key{0};

void Pipeline::references(const std::vector<int>& hints, std::vector<const Frame420*>& refs,
                          std::vector<long long>& keys) const {
  for (int doh : hints) {
    const Recon* r = nullptr;
    if (last_shown_.doh == doh) r = &last_shown_;
    for (const Recon& c : recons_)
      if (c.doh == doh) r = &c;
    if (!r) throw std::logic_error("Pipeline: a reference's reconstruction is missing");
    refs.push_back(&r->f);
    keys.push_back(r->key);
  }
}

void Pipeline::sync_from(const Pipeline& o) {
  enc_.sync_from(o.enc_);
  qindex_offset_ = o.qindex_offset_;
  recons_ = o.recons_;
  last_shown_ = o.last_shown_;
}

void Pipeline::keep(const FrameSpec& spec) {
  if (spec.key) recons_.clear(), last_shown_.doh = -1;
  if (spec.key || spec.reference) {
    recons_.push_back(Recon{enc_.display_order_hint(), next_key++, copy_frame(enc_.last_recon(), false)});
    if (static_cast<int>(recons_.size()) > cfg_.refs + 8) recons_.erase(recons_.begin());
  } else if (spec.shown && spec.keep_shown) {
    last_shown_ = Recon{enc_.display_order_hint(), next_key++, copy_frame(enc_.last_recon(), false)};
  }
}

std::vector<std::vector<uint8_t>> Pipeline::code_group(const std::vector<const Frame420*>& srcs,
                                                       const std::vector<FrameSpec>& given,
                                                       std::vector<Frame420>* recons) {
  std::vector<FrameSpec> specs = given;
  for (FrameSpec& s : specs) s.qindex_delta += qindex_offset_;
  const size_t n = srcs.size();
  if (n == 0 || specs.size() != n) throw std::invalid_argument("code_group: one spec per frame");
  for (size_t i = 0; i + 1 < n; i++)
    if (specs[i].key || specs[i].reference || specs[i].keep_shown)
      throw std::invalid_argument("code_group: a frame other frames may depend on is not last");
  while (shadows_.size() + 1 < n) shadows_.push_back(std::make_unique<Encoder>(cfg_));
  std::vector<Encoder*> encs;
  for (size_t i = 0; i + 1 < n; i++) shadows_[i]->sync_from(enc_), encs.push_back(shadows_[i].get());
  encs.push_back(&enc_);
  std::vector<std::vector<const Frame420*>> refs(n);
  std::vector<std::vector<long long>> keys(n);
  for (size_t i = 0; i < n; i++) references(encs[i]->begin(specs[i]), refs[i], keys[i]);
  // The engine decides one frame at a time; each frame's tiles are coded while the next is decided.
  std::vector<DecisionMap> dms(n);
  std::vector<std::unique_ptr<RowGate>> gates;
  for (size_t i = 0; i < n; i++) gates.push_back(std::make_unique<RowGate>());
  std::vector<std::future<std::vector<uint8_t>>> coders;
  std::exception_ptr err;
  for (size_t i = 0; i < n && !err; i++) {
    try {
      engine_->launch(*srcs[i], refs[i], std::max(1, std::min(255, cfg_.qindex + specs[i].qindex_delta)), dms[i],
                      keys[i], specs[i].display);
      if (i + 1 < n) engine_->prepare(*srcs[i + 1], specs[i + 1].display);
      auto collector = std::async(std::launch::async, [&, i] {
        collect(dms[i], *gates[i], *srcs[i], refs[i], std::max(1, std::min(255, cfg_.qindex + specs[i].qindex_delta)),
                specs[i].sb_dq);
      });
      coders.push_back(std::async(std::launch::async, [&, i] { return encs[i]->code(*srcs[i], dms[i], gates[i].get()); }));
      collector.get();  // a decision error surfaces here (and fails the gate the coder waits on)
    } catch (...) {
      err = std::current_exception();
    }
  }
  std::vector<std::vector<uint8_t>> out;
  for (auto& c : coders) {  // every coder finishes before the locals it uses go
    try {
      out.push_back(c.get());
    } catch (...) {
      if (!err) err = std::current_exception();
    }
  }
  if (err) std::rethrow_exception(err);
  if (recons) {
    recons->clear();
    for (Encoder* e : encs) recons->push_back(copy_frame(e->last_recon(), true));
  }
  keep(specs.back());
  return out;
}
