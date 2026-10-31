// engine_bench <in.y4m> <frames> [qindex]: the decision engines alone (no coding), ms per frame.
// Each frame is decided against the two before it (source frames as references, as an inter frame
// with two references), with the next frame prepared ahead, as Pipeline::code does.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu/engine.h"
#include "io/y4m.h"
#include "vk/vk_context.h"
#include "vk/vk_engine.h"

int main(int argc, char** argv) {
  if (argc < 3) return std::fprintf(stderr, "usage: engine_bench <in.y4m> <frames> [qindex]\n"), 2;
  try {
    Y4mReader in(argv[1]);
    const int n = std::atoi(argv[2]), q = argc > 3 ? std::atoi(argv[3]) : 130;
    std::vector<Frame420> f;
    for (Frame420 x; static_cast<int>(f.size()) < n && in.read(x);) f.push_back(x);
    std::printf("%dx%d, %zu frames, qindex %d\n", in.width(), in.height(), f.size(), q);
    // Frames 0 and 1 are references only and frame 2 warms up: the mean is over the rest.
    if (f.size() < 4) throw std::runtime_error("needs 4 frames at least");
    std::vector<std::pair<std::string, std::unique_ptr<DecisionEngine>>> engines;
    auto add = [&](const std::string& name, auto make) {  // an engine that cannot start is skipped
      try {
        engines.emplace_back(name, make());
      } catch (const std::exception& e) {
        std::printf("%-40s skipped: %s\n", name.c_str(), e.what());
      }
    };
    add("CUDA", [&] { return std::make_unique<GpuEngine>(in.width(), in.height(), 64); });
    const std::vector<std::string> vk = celv2::vk::Context::devices();
    for (int d = 0; d < static_cast<int>(vk.size()); d++)
      add("Vulkan " + vk[d], [&] { return std::make_unique<VulkanEngine>(in.width(), in.height(), 64, d); });
    for (auto& [name, e] : engines) {
      double total = 0;
      for (size_t i = 2; i < f.size(); i++) {
        const auto t0 = std::chrono::steady_clock::now();
        DecisionMap dm;
        RowGate gate;
        e->launch(f[i], {&f[i - 1], &f[i - 2]}, q, dm, {static_cast<long long>(i - 1), static_cast<long long>(i - 2)},
                  static_cast<long long>(i));
        if (i + 1 < f.size()) e->prepare(f[i + 1], static_cast<long long>(i + 1));
        e->collect(dm, gate);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (i > 2) total += ms;  // the first frame warms up (allocations, pipeline caches)
      }
      std::printf("%-40s %8.1f ms/frame\n", name.c_str(), total / std::max<size_t>(1, f.size() - 3));
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "engine_bench: %s\n", e.what());
    return 1;
  }
}
