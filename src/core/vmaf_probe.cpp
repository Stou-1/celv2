#include "core/vmaf_probe.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <regex>
#include <stdexcept>

#include <windows.h>

namespace celv2 {

namespace {

// VMAF as log(100 - VMAF): close to a line in qindex over the range a scene's probes span.
double vlog(double v) { return std::log(std::max(100 - v, 0.01)); }
// Its usual slope per qindex (100 - VMAF about doubles every 40), for the second probe.
constexpr double kSlope = 0.017;
// A probe this little above the target ends the search (0.3 left chika 1080p 5% over one qindex with
// the same VMAF: 0.3 VMAF was 2 qindex there).
constexpr double kClose = 0.15;
constexpr double kFar = 1.0;

// The qindex where the line through (a, va) and (b, vb) reaches target (a < b), or `fallback` when the
// probes do not fall with qindex.
double crossing(int a, double va, int b, double vb, double target, double fallback) {
  const double m = (vlog(vb) - vlog(va)) / (b - a);
  return m > 1e-6 ? a + (vlog(target) - vlog(va)) / m : fallback;
}

std::wstring wide(const std::string& s) {
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
  std::wstring w(n > 0 ? n - 1 : 0, L'\0');
  if (n > 1) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
  return w;
}

// Runs the command line (UTF-8) with its stdout and stderr into out; the exit code (-1 if it cannot start).
int run(const std::string& cmd, std::string& out) {
  out.clear();
  FILE* p = _wpopen(wide("\"" + cmd + " 2>&1\"").c_str(), L"rb");  // cmd.exe drops the outer quotes
  if (!p) return -1;
  char buf[4096];
  for (size_t n; (n = std::fread(buf, 1, sizeof buf, p)) > 0;) out.append(buf, n);
  return _pclose(p);
}

std::string quote(const std::string& s) { return "\"" + s + "\""; }

}  // namespace

int vmaf_search(double target, int start, const std::function<double(int)>& vmaf, int max_probes, int* probes, bool probed_only) {
  std::map<int, double> v;  // qindex -> probe VMAF
  int q = std::clamp(start, 1, 255), estimate = q;
  // The highest qindex probed at or above target, and the lowest above it probed below (-1: none).
  auto bracket = [&](int& lo, int& hi) {
    lo = hi = -1;
    for (const auto& [k, s] : v)
      if (s >= target) lo = k;
    for (const auto& [k, s] : v)
      if (s < target && k > lo) {
        hi = k;
        break;
      }
  };
  // One probe more where the best so far is over kFar above the target (a still Flash scene: 98.3 at
  // qindex 168 after 4, the target between 168 and 190).
  auto far_above = [&] {
    double best = -1;
    for (const auto& [k, s] : v)
      if (s >= target) best = s;
    return best - target > kFar;
  };
  for (int n = 0; (n < max_probes || (n == max_probes && far_above())) && !v.count(q); n++) {
    v[q] = vmaf(q);
    if (v[q] >= target && v[q] - target < kClose) {
      estimate = q;
      break;
    }
    int lo, hi;
    bracket(lo, hi);
    if (lo >= 0 && hi >= 0) {  // between them, where the line through them meets the target
      estimate = std::clamp(static_cast<int>(std::floor(crossing(lo, v[lo], hi, v[hi], target, lo))), lo, hi - 1);
      if (hi - lo <= 1) break;
      q = std::clamp(estimate, lo + 1, hi - 1);
    } else if (lo >= 0) {  // every probe at or above: further up
      estimate = lo;
      if (lo == 255) break;
      const auto top = std::prev(v.end());
      // one probe: by the usual slope (4 to 48 up); two that do not fall with qindex: the top
      const double c = v.size() > 1 ? crossing(std::prev(top)->first, std::prev(top)->second, lo, top->second, target, 255)
                                    : std::clamp(lo + (vlog(target) - vlog(top->second)) / kSlope, lo + 4.0, lo + 48.0);
      q = std::clamp(static_cast<int>(std::floor(c)), lo + 1, 255);
    } else {  // every probe below: further down
      const int low = v.begin()->first;
      if (low == 1) {
        estimate = 1;
        break;
      }
      const auto second = std::next(v.begin());
      const double c = v.size() > 1 ? crossing(low, v.begin()->second, second->first, second->second, target, 1)
                                    : std::clamp(low + (vlog(target) - vlog(v.begin()->second)) / kSlope, low - 48.0, low - 4.0);
      q = std::clamp(static_cast<int>(std::floor(c)), 1, low - 1);
      estimate = q;
    }
  }
  if (probes) *probes = static_cast<int>(v.size());
  // v.begin() is the lowest qindex probed, the best VMAF.
  if ((probed_only || estimate <= 1) && v.begin()->second < target)
    std::fprintf(stderr, "celv2enc: target VMAF %.2f not reached: %.2f at qindex %d, the best probed\n", target, v.begin()->second, v.begin()->first);
  if (probed_only) {
    int best = -1;
    for (const auto& [k, s] : v)
      if (s >= target) best = k;
    return best >= 0 ? best : v.begin()->first;
  }
  return std::clamp(estimate, 1, 255);
}

bool ffmpeg_has_libvmaf(const std::string& ffmpeg) {
  std::string out;
  return run(quote(ffmpeg) + " -nostdin -hide_banner -filters", out) == 0 && out.find(" libvmaf ") != std::string::npos;
}

double ffmpeg_vmaf(const std::string& ffmpeg, const std::vector<const Frame420*>& src,
                   const std::vector<std::vector<uint8_t>>& recon, int fps_num, int fps_den) {
  if (src.empty() || src.size() != recon.size()) throw std::invalid_argument("ffmpeg_vmaf: one recon frame per source frame");
  static std::atomic<int> serial{0};
  const auto dir = std::filesystem::temp_directory_path();
  const std::string tag = "celv2_vmaf_" + std::to_string(GetCurrentProcessId()) + "_" + std::to_string(serial++);
  const auto ref_path = dir / (tag + "_ref.y4m"), dis_path = dir / (tag + "_dis.y4m");
  const Frame420& f0 = *src[0];
  const std::string header = "YUV4MPEG2 W" + std::to_string(f0.width) + " H" + std::to_string(f0.height) + " F" +
                             std::to_string(fps_num) + ":" + std::to_string(fps_den) + " Ip A1:1 " +
                             (f0.bit_depth > 8 ? "C420p10" : "C420jpeg") + "\n";
  struct Cleanup {
    std::filesystem::path a, b;
    ~Cleanup() {
      std::error_code e;
      std::filesystem::remove(a, e), std::filesystem::remove(b, e);
    }
  } cleanup{ref_path, dis_path};
  auto write = [&](const std::filesystem::path& path, const std::function<std::vector<uint8_t>(size_t)>& payload) {
    FILE* f = _wfopen(path.wstring().c_str(), L"wb");
    if (!f) throw std::runtime_error("cannot write " + path.u8string());
    bool ok = std::fwrite(header.data(), 1, header.size(), f) == header.size();
    for (size_t i = 0; ok && i < src.size(); i++) {
      const std::vector<uint8_t> b = payload(i);
      ok = std::fwrite("FRAME\n", 1, 6, f) == 6 && std::fwrite(b.data(), 1, b.size(), f) == b.size();
    }
    if (std::fclose(f) != 0 || !ok) throw std::runtime_error("writing " + path.u8string() + " failed (disk full?)");
  };
  write(ref_path, [&](size_t i) { return write_y4m_frame_bytes(*src[i]); });
  write(dis_path, [&](size_t i) { return recon[i]; });
  std::string out;
  const std::string cmd = quote(ffmpeg) + " -nostdin -hide_banner -nostats -i " + quote(dis_path.u8string()) + " -i " +
                          quote(ref_path.u8string()) + " -lavfi \"[0:v][1:v]libvmaf=n_threads=8\" -f null -";
  const int rc = run(cmd, out);
  std::smatch m;
  if (rc != 0 || !std::regex_search(out, m, std::regex(R"(VMAF score: ([0-9.]+))")))
    throw std::runtime_error("ffmpeg's VMAF failed (exit " + std::to_string(rc) + "): " +
                             out.substr(out.size() > 400 ? out.size() - 400 : 0));
  return std::stod(m[1].str());
}

}  // namespace celv2
