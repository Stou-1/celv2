#include "core/psy.h"

#include <algorithm>
#include <cmath>

namespace celv2 {

namespace {

// t(E)'s ends, in log2(variance + 1) of 8-bit samples (variance 15 and 255).
constexpr double kFlatLog = 4, kBusyLog = 8;

}  // namespace

double psy_term(const int32_t* S, const int32_t* c, const int32_t* dq, int n, double step) {
  double src = 0, rec = 0;
  for (int k = 1; k < n; k++) {
    src += std::abs(static_cast<double>(S[k]));
    rec += std::abs(static_cast<double>(S[k]) - (c[k] - dq[k]));
  }
  return std::abs(src - rec) / step;
}

double psy_weight(const AqMap& m, int x, int y, int w, int h, PsyRule rule) {
  const int c0 = std::clamp(x / 16, 0, m.cols - 1), c1 = std::clamp((x + w - 1) / 16, 0, m.cols - 1);
  const int r0 = std::clamp(y / 16, 0, m.rows - 1), r1 = std::clamp((y + h - 1) / 16, 0, m.rows - 1);
  double sum = 0;
  int n = 0;
  for (int r = r0; r <= r1; r++)
    for (int c = c0; c <= c1; c++) {
      const size_t i = static_cast<size_t>(r) * m.cols + c;
      const double t = std::clamp((m.log_energy[i] - kFlatLog) / (kBusyLog - kFlatLog), 0.0, 1.0);
      sum += t * (rule == PsyRule::Line ? m.coherence[i] : 1 - m.coherence[i]);
      n++;
    }
  return n ? sum / n : 0;
}

}  // namespace celv2
