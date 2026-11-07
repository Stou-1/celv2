// engine_bench <in.y4m> <frames> [qindex] [refs] [cpu]: the decision engines alone (no coding), ms per
// frame; cpu adds the CPU engine (minutes a frame at 16K). Each frame is decided against the refs (2 by default) before it, nearest first (source
// frames as references), with the next frame prepared ahead, as Pipeline::code does.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu/cpu_engine.h"
#include "io/y4m.h"
#include "vk/vk_context.h"
#include "vk/vk_engine.h"

int main(int argc, char** argv) {
  if (argc < 3) return std::fprintf(stderr, "usage: engine_bench <in.y4m> <frames> [qindex] [refs] [cpu]\n"), 2;
  try {
    Y4mReader in(argv[1]);
    const int n = std::atoi(argv[2]), q = argc > 3 ? std::atoi(argv[3]) : 130, nr = argc > 4 ? std::atoi(argv[4]) : 2;
    if (nr < 1 || nr > 7) throw std::runtime_error("refs must be 1..7");
    std::vector<Frame420> f;
    for (Frame420 x; static_cast<int>(f.size()) < n && in.read(x);) f.push_back(x);
    std::printf("%dx%d, %zu frames, qindex %d\n", in.width(), in.height(), f.size(), q);
    // Frames 0 .. refs - 1 are references only and frame refs warms up: the mean is over the rest.
    if (static_cast<int>(f.size()) < nr + 2) throw std::runtime_error("needs refs + 2 frames at least");
    std::vector<std::pair<std::string, std::unique_ptr<DecisionEngine>>> engines;
    auto add = [&](const std::string& name, auto make) {  // an engine that cannot start is skipped
      try {
        engines.emplace_back(name, make());
      } catch (const std::exception& e) {
        std::printf("%-40s skipped: %s\n", name.c_str(), e.what());
      }
    };
    if (argc > 5) add("CPU", [&] { return std::make_unique<CpuEngine>(in.width(), in.height(), 64, 0); });
    const std::vector<std::string> vk = celv2::vk::Context::devices();
    for (int d = 0; d < static_cast<int>(vk.size()); d++)
      add("Vulkan " + vk[d], [&] { return std::make_unique<VulkanEngine>(in.width(), in.height(), 64, d); });
    for (auto& [name, e] : engines) {
      double total = 0;
      for (size_t i = nr; i < f.size(); i++) {
        const auto t0 = std::chrono::steady_clock::now();
        DecisionMap dm;
        RowGate gate;
        std::vector<const Frame420*> refs;
        std::vector<long long> keys;
        for (int k = 1; k <= nr; k++) refs.push_back(&f[i - k]), keys.push_back(static_cast<long long>(i - k));
        e->launch(f[i], refs, q, dm, keys, static_cast<long long>(i));
        if (i + 1 < f.size()) e->prepare(f[i + 1], static_cast<long long>(i + 1));
        e->collect(dm, gate);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (i > static_cast<size_t>(nr)) total += ms;  // the first frame warms up (allocations, pipeline caches)
      }
      std::printf("%-40s %8.1f ms/frame\n", name.c_str(), total / std::max<size_t>(1, f.size() - nr - 1));
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "engine_bench: %s\n", e.what());
    return 1;
  }
}
