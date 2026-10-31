#include "core/size_control.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

namespace celv2 {
namespace {

// Of the budget, so a late misestimate does not go over (CELV2_TEST_AIM, percent: tests/test_cli.cpp).
double aim() {
  static const double a = std::getenv("CELV2_TEST_AIM") ? std::atof(std::getenv("CELV2_TEST_AIM")) / 100 : 0.98;
  return a;
}
constexpr double kIvfFile = 32, kIvfFrame = 12;

// Frame i's pass-1 payload at qindex q.
double payload_at(const PassStats& s, size_t i, int q) { return static_cast<double>(s.bytes[i]) * std::pow(2.0, -(q - s.qindex) / 24.0); }

}  // namespace

int SizeControl::start_qindex(const PassStats& s, uint64_t budget) {
  for (int q = 1; q <= 255; q++) {
    double file = kIvfFile + kIvfFrame * s.bytes.size();
    for (size_t i = 0; i < s.bytes.size(); i++) file += payload_at(s, i, q);
    if (file <= aim() * budget) return q;
  }
  return 255;
}

SizeControl::SizeControl(PassStats s, uint64_t budget, int workers)
    : s_(std::move(s)), budget_(budget), q_(start_qindex(s_, budget)), workers_(std::max(1, workers)) {}

void SizeControl::coded(uint64_t payload) {
  if (done_ >= s_.bytes.size())
    throw std::runtime_error("the input has more frames than --stats (" + std::to_string(s_.bytes.size()) + ")");
  finest_ = std::min(finest_, q_);
  used_ += static_cast<double>(payload);
  expected_ += payload_at(s_, done_, q_);
  done_++;
}

void SizeControl::update() {
  const double ratio = expected_ > 0 ? used_ / expected_ : 1;
  auto fits = [&](int q) {
    double file = kIvfFile + kIvfFrame * s_.bytes.size() + used_;
    for (size_t i = done_; i < s_.bytes.size(); i++) file += ratio * payload_at(s_, i, q);
    return file <= aim() * budget_;
  };
  for (int q = std::max(1, q_ - 2); q <= std::min(255, q_ + 2); q++)
    if (fits(q)) return void(q_ = q);
  q_ = std::min(255, q_ + 2);
}

SizeControl::Fit SizeControl::fit(const std::function<bool(const Rec&)>& seen) const {
  // y = log2(coded / pass 1) against x = q - pass-1 qindex, weighted by the pass-1 bytes.
  double w = 0, wx = 0, wy = 0, wxx = 0, wxy = 0;
  std::map<int, size_t> per_q;
  double used = 0, expected24 = 0;
  for (const Rec& r : recs_) {
    if (!seen(r)) continue;
    const double p1 = static_cast<double>(s_.bytes[r.frame]);
    used += static_cast<double>(r.payload), expected24 += payload_at(s_, r.frame, r.q);
    if (p1 <= 0 || r.payload == 0) continue;
    const double x = r.q - s_.qindex, y = std::log2(static_cast<double>(r.payload) / p1);
    w += p1, wx += p1 * x, wy += p1 * y, wxx += p1 * x * x, wxy += p1 * x * y;
    per_q[r.q]++;
  }
  size_t enough = 0;
  for (const auto& kv : per_q) enough += kv.second >= 8;
  const double var = w > 0 ? wxx / w - (wx / w) * (wx / w) : 0;
  if (enough < 2 || var <= 0)  // the 24-qindex model, its ratio as in update()
    return Fit{24, expected24 > 0 && used > 0 ? std::log2(used / expected24) : 0};
  const double b = (wxy / w - (wx / w) * (wy / w)) / var;
  const double s = b < 0 ? std::clamp(-1 / b, 12.0, 48.0) : 48;
  return Fit{s, (wy + wx / s) / w};
}

double SizeControl::slope() const {
  std::lock_guard<std::mutex> l(mu_);
  return fit([](const Rec&) { return true; }).s;
}

int SizeControl::finest_fitting(size_t chunk, int lo, int hi) const {
  auto seen = [&](const Rec& r) { return r.chunk == chunk || r.chunk + workers_ <= chunk; };
  const Fit f = fit(seen);
  std::vector<bool> coded(s_.bytes.size());
  double file = kIvfFile + kIvfFrame * s_.bytes.size(), rest = 0;
  for (const Rec& r : recs_)
    if (seen(r)) coded[r.frame] = true, file += static_cast<double>(r.payload);
  for (size_t i = 0; i < s_.bytes.size(); i++)
    if (!coded[i]) rest += static_cast<double>(s_.bytes[i]);
  for (int q = std::max(1, lo); q <= std::min(255, hi); q++)
    if (file + rest * std::pow(2.0, f.log2r - (q - s_.qindex) / f.s) <= aim() * budget_) return q;
  return std::min(255, hi);
}

int SizeControl::chunk_start_qindex(size_t chunk, size_t, size_t, int at_least) {
  std::lock_guard<std::mutex> l(mu_);
  int q;
  if (chunk < static_cast<size_t>(workers_)) {
    const int k = static_cast<int>(chunk), d = (k + 1) / 2 * (k % 2 ? 1 : -1);
    q = std::clamp(start_qindex(s_, budget_) + d, 1, 255);
  } else {
    q = finest_fitting(chunk, 1, 255);
  }
  return chunk_q_[chunk] = std::clamp(std::max(q, at_least), 1, 255);
}

void SizeControl::forget(size_t chunk) {
  std::lock_guard<std::mutex> l(mu_);
  recs_.erase(std::remove_if(recs_.begin(), recs_.end(), [&](const Rec& r) { return r.chunk == chunk; }), recs_.end());
  done_ = recs_.size();
  finest_ = 256;
  for (const Rec& r : recs_) finest_ = std::min(finest_, r.q);
}

void SizeControl::coded(size_t chunk, size_t frame, uint64_t payload, int qindex) {
  std::lock_guard<std::mutex> l(mu_);
  if (frame >= s_.bytes.size())
    throw std::runtime_error("the input has more frames than --stats (" + std::to_string(s_.bytes.size()) + ")");
  recs_.push_back(Rec{chunk, frame, payload, qindex});
  finest_ = std::min(finest_, qindex);
  done_++;
}

int SizeControl::chunk_update(size_t chunk) {
  std::lock_guard<std::mutex> l(mu_);
  int& q = chunk_q_[chunk];
  return q = finest_fitting(chunk, q - 2, q + 2);
}

}  // namespace celv2
