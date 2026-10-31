#include "chunk_coder.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <thread>
#include <utility>

#include "bitstream/obu.h"
#include "core/scene_cut.h"
#include "core/tpl.h"

namespace {

std::atomic<bool> trial_warned{false};  // a failed GOP trial is reported once, not for every chunk

// Squared error of plane o against s (the same layout) over w x h.
template <class P>
double plane_sse(const P& s, const P& o, int stride, int w, int h) {
  double se = 0;
  for (int r = 0; r < h; r++)
    for (int c = 0; c < w; c++) {
      const int d = s[static_cast<size_t>(r) * stride + c] - o[static_cast<size_t>(r) * stride + c];
      se += static_cast<double>(d) * d;
    }
  return se;
}

// Y + U + V squared error of shown against source at the coded depth (the full-depth planes above 8 bits).
double frame_sse(const Frame420& source, const Frame420& shown) {
  const int cw = (source.width + 1) / 2, ch = (source.height + 1) / 2;
  if (source.bit_depth > 8)
    return plane_sse(source.hy, shown.hy, source.stride_y(), source.width, source.height) +
           plane_sse(source.hu, shown.hu, source.stride_uv(), cw, ch) + plane_sse(source.hv, shown.hv, source.stride_uv(), cw, ch);
  return plane_sse(source.y, shown.y, source.stride_y(), source.width, source.height) +
         plane_sse(source.u, shown.u, source.stride_uv(), cw, ch) + plane_sse(source.v, shown.v, source.stride_uv(), cw, ch);
}

// PSNR of one plane at the coded depth, peak (1 << depth) - 1 (99 when identical).
template <class P>
double plane_psnr(const P& s, const P& o, int stride, int w, int h, int depth) {
  const double peak = (1 << depth) - 1;
  const double mse = plane_sse(s, o, stride, w, h) / (static_cast<double>(w) * h);
  return mse > 0 ? std::min(99.0, 10 * std::log10(peak * peak / mse)) : 99.0;
}

}  // namespace

bool FrameBudget::take() {
  std::unique_lock<std::mutex> l(mu_);
  cv_.wait(l, [&] { return held_ < limit_ || cancelled_; });
  if (cancelled_) return false;
  held_++;
  return true;
}

void FrameBudget::give(size_t n) {
  {
    std::lock_guard<std::mutex> l(mu_);
    held_ -= std::min(held_, n);
  }
  cv_.notify_all();
}

void FrameBudget::cancel() {
  {
    std::lock_guard<std::mutex> l(mu_);
    cancelled_ = true;
  }
  cv_.notify_all();
}

void FrameQueue::push(ChunkFrame f) {
  {
    std::lock_guard<std::mutex> l(mu_);
    q_.push_back(std::move(f));
  }
  cv_.notify_all();
}

void FrameQueue::close() {
  {
    std::lock_guard<std::mutex> l(mu_);
    closed_ = true;
  }
  cv_.notify_all();
}

const ChunkFrame* FrameQueue::at(size_t i) {
  std::unique_lock<std::mutex> l(mu_);
  cv_.wait(l, [&] { return off_ + i < q_.size() || closed_; });
  return off_ + i < q_.size() ? &q_[off_ + i] : nullptr;
}

size_t FrameQueue::size() {
  std::lock_guard<std::mutex> l(mu_);
  return q_.size() - off_;
}

void FrameQueue::keep() {
  std::lock_guard<std::mutex> l(mu_);
  keep_ = true;
}

void FrameQueue::rewind() {
  std::lock_guard<std::mutex> l(mu_);
  off_ = 0;
}

void FrameQueue::pop(size_t n) {
  {
    std::lock_guard<std::mutex> l(mu_);
    n = std::min(n, q_.size() - off_);
    if (keep_) {
      off_ += n;
      return;
    }
    q_.erase(q_.begin(), q_.begin() + static_cast<std::ptrdiff_t>(n));
  }
  budget_.give(n);
}

ChunkCoder::ChunkCoder(const SeqConfig& cfg, Device device, const LoopConfig& lc)
    : cfg_(cfg), device_(device), lc_(lc), pipe_(std::make_unique<Pipeline>(cfg, device)), gop_trial_(lc.gop_trial) {}

void ChunkCoder::set_qindex_offset(int d) {
  pipe_->set_qindex_offset(d);
  if (trial_) trial_->set_qindex_offset(d);
}

void ChunkCoder::code(FrameQueue& in, const std::function<void(std::vector<CodedFrame>&)>& after_group) {
  const SeqConfig& cfg = cfg_;
  const LoopConfig& a = lc_;
  const int bframes = a.bframes;
  gop_trial_ = lc_.gop_trial;  // per chunk: a trial lost to a failure is not lost for the chunks this worker takes next
  const bool keep_recons = a.psnr || a.recon || gop_trial_;
  auto q = [&](size_t i) -> const ChunkFrame& { return *in.at(i); };  // only for frames known to be there
  Frame420 last_ref;  // the source of the last reference coded before the next mini-GOP (TPL's first frame)
  std::vector<CodedFrame> group;
  auto output = [&](const std::vector<uint8_t>& obus, int display, const Frame420& shown, const Frame420& source) {
    CodedFrame c;
    c.tu = make_obu(OBU_TEMPORAL_DELIMITER, {});
    c.tu.insert(c.tu.end(), obus.begin(), obus.end());
    c.display = display;
    if (a.psnr) {
      const int cw = (source.width + 1) / 2, ch = (source.height + 1) / 2, d = source.bit_depth;
      if (d > 8) {
        c.psnr[0] = plane_psnr(source.hy, shown.hy, source.stride_y(), source.width, source.height, d);
        c.psnr[1] = plane_psnr(source.hu, shown.hu, source.stride_uv(), cw, ch, d);
        c.psnr[2] = plane_psnr(source.hv, shown.hv, source.stride_uv(), cw, ch, d);
      } else {
        c.psnr[0] = plane_psnr(source.y, shown.y, source.stride_y(), source.width, source.height, d);
        c.psnr[1] = plane_psnr(source.u, shown.u, source.stride_uv(), cw, ch, d);
        c.psnr[2] = plane_psnr(source.v, shown.v, source.stride_uv(), cw, ch, d);
      }
    }
    if (a.recon) c.recon = write_y4m_frame_bytes(shown);
    group.push_back(std::move(c));
  };
  auto end_group = [&](size_t coded) {
    after_group(group);
    group.clear();
    in.pop(coded);
  };

  // With --pyramid the middle of each gap between references holding 3 or more frames is coded
  // after the anchor, hidden and kept as a reference, coarse to fine: (frame, level 1...) of a
  // mini-GOP of m frames.
  auto mids_of = [&](size_t m) {
    std::vector<std::pair<size_t, int>> mids;
    std::function<void(int, int, int)> split = [&](int lo, int hi, int level) {
      // with 1 or 2 slots the middle frame is gone when shown; the anchor and the hidden frames
      // must leave a slot for the last shown one
      if (hi - lo - 1 < 3 || cfg.refs < 3 || static_cast<int>(mids.size()) + 2 >= cfg.refs) return;
      const int mid = (lo + hi) / 2;
      mids.emplace_back(static_cast<size_t>(mid), level);
      split(lo, mid, level + 1), split(mid, hi, level + 1);
    };
    if (a.pyramid) split(-1, static_cast<int>(m) - 1, 1);
    return mids;
  };
  const int max_coded_q = cfg.bit_depth == 8 ? 255 : 303;  // MAXQ_8_BITS / MAXQ_10_BITS
  auto coded_q = [&](int delta) { return cfg.coded_qindex(std::max(1, std::min(255, cfg.qindex + delta))); };
  // TPL of the mini-GOP q(start .. start + m) coded after the reference frame `ref`, in its coding
  // order: ref, the anchor (its last frame), the mids, then the rest, each predicting from the
  // nearest references on both sides.
  auto tpl_stats = [&](const Frame420& ref, size_t start, size_t m, const std::vector<std::pair<size_t, int>>& mids) {
    std::vector<int> refs_at = {-1, static_cast<int>(m) - 1};  // positions coded as references so far
    std::vector<celv2::TplFrame> fr;
    std::vector<int> index(m + 1, -1);  // position + 1 -> TPL frame
    auto add = [&](int pos) {
      celv2::TplFrame t;
      t.src = pos < 0 ? &ref : &q(start + pos).f;
      int lo = -2, hi = static_cast<int>(m);
      for (int r : refs_at)
        if (index[r + 1] >= 0) {
          if (r < pos) lo = std::max(lo, r);
          if (r > pos) hi = std::min(hi, r);
        }
      if (lo > -2) t.refs.push_back(index[lo + 1]);
      if (hi < static_cast<int>(m)) t.refs.push_back(index[hi + 1]);
      index[pos + 1] = static_cast<int>(fr.size());
      fr.push_back(t);
    };
    add(-1), add(static_cast<int>(m) - 1);
    for (const auto& md : mids) refs_at.push_back(static_cast<int>(md.first)), add(static_cast<int>(md.first));
    for (int pos = 0; pos + 1 < static_cast<int>(m); pos++)
      if (index[pos + 1] < 0) add(pos);
    return celv2::tpl_model(fr, static_cast<int>(std::thread::hardware_concurrency()));
  };
  // Superblock qindex offsets (FrameSpec::sb_dq) of ref, the anchor and each mid from their
  // tpl_stats() (bases: their qindex deltas, in that order).
  auto tpl_offsets = [&](const std::vector<celv2::TplStats>& st, const std::vector<int>& deltas) {
    std::vector<std::vector<int>> out;
    for (size_t k = 0; k < deltas.size(); k++)
      out.push_back(celv2::tpl_sb_offsets(st[k], cfg.width, cfg.height, coded_q(deltas[k]), max_coded_q, 9 * kDeltaQRes - 1));
    return out;
  };
  // --tplq: the qindex delta whose step is the B frames' (--bq) times r0^0.7 of these stats (30
  // frames, 2: 0.5 / 0.7 / 1.0 gave -1.2 / -2.3 / +0.2% on average).
  auto tpl_delta = [&](const celv2::TplStats& s) {
    const int coded = celv2::qindex_for_step_ratio(coded_q(a.bq), std::pow(celv2::tpl_r0(s), 0.7), max_coded_q);
    return std::clamp(coded - cfg.coded_qindex(cfg.qindex), 1 - cfg.qindex, 255 - cfg.qindex);
  };
  // A frame shown by a mini-GOP: its unit's OBUs (the first one's led by the hidden frames coded
  // before it), its display index, its reconstruction (empty unless kept) and its source.
  struct Shown {
    std::vector<uint8_t> obus;
    int display;
    Frame420 recon;
    const Frame420* src;
  };
  // Codes q(start, start + m) on p as one mini-GOP (m >= 2, no keyframe among them) after the
  // reference whose source is ref (TPL's first frame) and returns its shown frames in display
  // order; the caller outputs them and sets last_ref. Two may run at once on two pipelines.
  auto code_minigop = [&](Pipeline& p, size_t start, size_t m, const Frame420& ref) {
    std::vector<Shown> out;
    // The last one first, hidden; the others from both sides (not references); then show it.
    const ChunkFrame& anchor = q(start + m - 1);
    // The mids (mids_of()), each level 8 qindex above the one before (8-frame mini-GOPs, 30 frames
    // live action / anime: 4 / 8 / 12 gave -3.7 / -0.8%, -4.4 / -0.9%, -4.5 / -0.3%).
    const int kLevelStep = m > 8 && a.level_step16 >= 0 ? a.level_step16 : a.level_step;
    const std::vector<std::pair<size_t, int>> mids = mids_of(m);
    const size_t first = mids.empty() ? 0 : mids[0].first;  // the frame coded next
    // The more of the picture stays still through the mini-GOP, the more of the anchor later frames
    // reuse: its qindex goes down by up to 24 as the still share goes from 1/2 to all (120 frames
    // 1080p q100-190: anime C -6.1%, anime A -4.3%, live action -2.4%, anime D -2.3%,
    // anime B -2.2 / -0.4%; the fast Xiph clips, under 1/2, the same bytes; up to 16 / 32:
    // -0.5 to -5.6% / +0.2 to -6.9%; without the 1/2 floor, 12 lost up to 1.3% on busy shots).
    const double kStillBoost = a.anchor_boost;
    int anchor_q = a.aq - static_cast<int>(std::lround(kStillBoost * std::max(0.0, still_share(q(start).f, anchor.f) - 0.5) * 2));
    std::vector<celv2::TplStats> st;
    if ((a.deltaq || a.tplq == 2) && ref.width == cfg.width && ref.height == cfg.height) st = tpl_stats(ref, start, m, mids);
    if (a.tplq == 2 && !st.empty()) anchor_q = tpl_delta(st[1]);
    std::vector<int> deltas = {0, anchor_q};
    int top = 0;  // the mids' deepest level
    for (const auto& md : mids) top = std::max(top, md.second);
    for (size_t k = 0; k < mids.size(); k++) {
      const int level = mids[k].second;
      if (a.tplq == 2 && !st.empty())  // from the anchor's qindex (level 0) to the B frames' (top + 1)
        deltas.push_back(anchor_q + static_cast<int>(std::lround((a.bq - anchor_q) * level / static_cast<double>(top + 1))));
      else
        deltas.push_back(a.mq + (level - 1) * kLevelStep);
    }
    std::vector<std::vector<int>> dq;
    if (a.deltaq && !st.empty()) dq = tpl_offsets(st, deltas);
    FrameSpec as{false, anchor.display, false, true, anchor_q};
    if (!dq.empty()) as.sb_dq = dq[1];
    std::vector<uint8_t> hidden = p.code(anchor.f, as, &q(start + first).f, q(start + first).display);
    const int anchor_hint = p.display_order_hint();
    const Frame420 anchor_recon = keep_recons ? p.last_recon() : Frame420{};
    // With --pyramid the middle B frame of a run of 3 or more is coded next, hidden and kept as a
    // reference (qindex offset mq); it is shown in its turn below. (A middle frame in each half too
    // lost on anime A 1080p, and the four hidden references in a row broke the stream at --refs 4.)
    std::vector<int> mid_hint(m, -1);
    std::vector<Frame420> mid_recon(m);
    for (size_t k = 0; k < mids.size(); k++) {
      const size_t mid = mids[k].first;
      const bool more = k + 1 < mids.size();
      FrameSpec ms{false, q(start + mid).display, false, true, deltas[2 + k]};
      if (!dq.empty()) ms.sb_dq = dq[2 + k];
      const auto obus = p.code(q(start + mid).f, ms, more ? &q(start + mids[k + 1].first).f : nullptr, more ? q(start + mids[k + 1].first).display : -1);
      hidden.insert(hidden.end(), obus.begin(), obus.end());
      mid_hint[mid] = p.display_order_hint();
      if (keep_recons) mid_recon[mid] = p.last_recon();
    }
    // The other B frames, from both sides (not references). In parallel only the last keeps a slot,
    // so they depend on none of each other and their tiles are coded at once (the same bytes as one
    // after the other); else each takes it in turn.
    std::vector<size_t> leaves;
    for (size_t i = 0; i + 1 < m; i++)
      if (mid_hint[i] < 0) leaves.push_back(i);
    std::vector<const Frame420*> srcs;
    std::vector<FrameSpec> specs;
    for (size_t k = 0; k < leaves.size(); k++)
      srcs.push_back(&q(start + leaves[k]).f),
          specs.push_back(FrameSpec{false, q(start + leaves[k]).display, true, false, a.bq, !a.parallel_b || k + 1 == leaves.size()});
    auto shown = [&](std::vector<uint8_t> obus, size_t i, Frame420 r) {
      if (out.empty()) obus.insert(obus.begin(), hidden.begin(), hidden.end());
      out.push_back(Shown{std::move(obus), q(start + i).display, std::move(r), &q(start + i).f});
    };
    auto show_mid = [&](size_t i) { shown(p.show_existing(mid_hint[i]), i, mid_recon[i]); };
    if (a.parallel_b && srcs.size() > 1) {
      std::vector<Frame420> rs;
      const auto obus = p.code_group(srcs, specs, keep_recons ? &rs : nullptr);
      for (size_t i = 0, k = 0; i + 1 < m; i++)
        if (mid_hint[i] >= 0)
          show_mid(i);
        else
          shown(obus[k], i, rs.empty() ? Frame420{} : rs[k]), k++;
    } else {
      for (size_t i = 0, k = 0; i + 1 < m; i++) {
        if (mid_hint[i] >= 0) {
          show_mid(i);
          continue;
        }
        const bool more = k + 1 < leaves.size();
        auto obus = p.code(q(start + i).f, specs[k], more ? &q(start + leaves[k + 1]).f : nullptr,
                           more ? q(start + leaves[k + 1]).display : -1);
        shown(std::move(obus), i, keep_recons ? p.last_recon() : Frame420{});
        k++;
      }
    }
    shown(p.show_existing(anchor_hint), m - 1, anchor_recon);
    return out;
  };

  while (in.at(0)) {
    if (q(0).key) {
      // A keyframe's qindex goes down as an anchor's (below), by up to 48 with the still share to the
      // frame up to 8 ahead (120 frames 1080p q100-190 after the anchors' boost: anime B
      // -2.0 / -2.2%, anime D -2.2%, anime A -0.5%, live action -0.2%, anime C -0.1%, the fast Xiph
      // clips the same bytes; q40-100 live action / anime -1.2 / -0.1%; 32 / 64: -0.1 to -2.2% /
      // +0.2 to -2.7%). Never past the next keyframe: the chunk ends before it.
      const double kKeyStillBoost = a.key_boost;
      size_t ahead = 0;
      // Up to bframes + 1 ahead, as far as the frame loop ever held (and as far as the read-ahead reaches).
      while (ahead < 8 && ahead < static_cast<size_t>(bframes) + 1 && in.at(ahead + 1) && !q(ahead + 1).key) ahead++;
      int key_q = a.kq - (ahead ? static_cast<int>(std::lround(kKeyStillBoost * std::max(0.0, still_share(q(0).f, q(ahead).f) - 0.5) * 2)) : 0);
      std::vector<celv2::TplStats> st;
      if ((a.deltaq || a.tplq) && ahead) {  // the mini-GOP after it, as it will be coded
        const size_t m = std::min<size_t>(ahead, static_cast<size_t>(bframes) + 1);
        st = tpl_stats(q(0).f, 1, m, mids_of(m));
        if (a.tplq) key_q = tpl_delta(st[0]) + a.kq;
      }
      FrameSpec ks{true, q(0).display, true, true, key_q};
      if (a.deltaq && !st.empty()) ks.sb_dq = tpl_offsets(st, {key_q})[0];
      output(pipe_->code(q(0).f, ks), q(0).display, pipe_->last_recon(), q(0).f);
      last_ref = q(0).f;
      end_group(1);
      continue;
    }
    // A mini-GOP: the frames up to the next keyframe, at most bframes + 1.
    size_t m = 0;
    while (m < static_cast<size_t>(bframes) + 1 && in.at(m) && !q(m).key) m++;
    // --gop-still: over 8 frames only where the picture stays still to the last one (120 frames
    // 1080p, 16 frames everywhere / where the still share is 0.7 or more: kaguya -9.9 / -9.9%, live
    // action -2.6 / -1.7%, AIR +10.8 / 0%, AM +6.8 / +1.4%, Dragon Maid +5.1 / +1.9%, the fast Xiph
    // clips +5.4 to +8.8 / 0%).
    if (m > 8 && a.gop_still > 0 && still_share(q(0).f, q(m - 1).f) < a.gop_still) m = 8;
    if (m <= 1) {
      output(pipe_->code(q(0).f, FrameSpec{false, q(0).display}), q(0).display, pipe_->last_recon(), q(0).f);
      last_ref = q(0).f;
      end_group(1);
      continue;
    }
    if (m > 8 && gop_trial_ && !trial_) {
      try {
        trial_ = std::make_unique<Pipeline>(cfg, device_);
      } catch (const std::exception& e) {
        if (!trial_warned.exchange(true)) std::fprintf(stderr, "celv2enc: --gop-trial off for the chunk (said once): %s\n", e.what());
        gop_trial_ = false;
      }
    }
    std::vector<Shown> shown;
    if (m > 8 && gop_trial_) {
      // --gop-trial: the m frames as one mini-GOP (on a copy of the state, on its own thread)
      // against 8 and the rest, at the same time; the lower squared error + sse_per_bit x bits is
      // kept (the trial's state taken with it).
      trial_->sync_from(*pipe_);
      std::vector<Shown> one;
      std::exception_ptr trial_err;
      std::thread trial_thread([&] {
        try {
          if (std::getenv("CELV2_TEST_TRIAL_FAIL")) throw std::runtime_error("CELV2_TEST_TRIAL_FAIL is set");  // tests/test_cli.cpp
          one = code_minigop(*trial_, 0, m, last_ref);
        } catch (...) {
          trial_err = std::current_exception();
        }
      });
      try {
        shown = code_minigop(*pipe_, 0, 8, last_ref);
        if (m - 8 >= 2) {
          for (Shown& s : code_minigop(*pipe_, 8, m - 8, q(7).f)) shown.push_back(std::move(s));
        } else {  // one frame left: as the loop codes a lone frame
          auto obus = pipe_->code(q(8).f, FrameSpec{false, q(8).display});
          shown.push_back(Shown{std::move(obus), q(8).display, pipe_->last_recon(), &q(8).f});
        }
      } catch (...) {
        trial_thread.join();
        throw;
      }
      trial_thread.join();
      // GPU memory is taken once coding starts, so a trial that could be built can still run out:
      // only the trial is lost, the main pipeline's coding stands.
      auto cost = [&](const std::vector<Shown>& v) {
        double d = 0, bits = 0;
        for (const Shown& s : v) d += frame_sse(*s.src, s.recon), bits += 8.0 * s.obus.size();
        return d + pipe_->sse_per_bit() * bits;
      };
      if (trial_err) {
        try {
          std::rethrow_exception(trial_err);
        } catch (const std::exception& e) {
          if (!trial_warned.exchange(true)) std::fprintf(stderr, "celv2enc: --gop-trial off for the chunk (said once): %s\n", e.what());
        }
        trial_.reset();
        gop_trial_ = false;
      } else if (cost(one) < cost(shown)) {
        std::swap(pipe_, trial_), shown = std::move(one);
      }
    } else {
      shown = code_minigop(*pipe_, 0, m, last_ref);
    }
    for (Shown& s : shown) output(s.obus, s.display, s.recon, *s.src);
    last_ref = q(m - 1).f;
    end_group(m);
  }
}
