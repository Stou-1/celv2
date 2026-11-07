// The decision engine on Vulkan (plan step 5): GpuEngine's work with the vk::Kernels, so the same
// decisions on any Vulkan GPU (the RTX 5090, the Arc Pro B50), queued as GpuEngine queues them.
#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include "gpu/decision_engine.h"

// How a frame of this many pixels holds a reference's quarter-pel phases: '1' all 16 planes, 'B' the 4 half-pel
// ones, 'A' none, each interpolated when read (the same decisions in all); 0: not even that fits. engines: how many
// share free_vram, each allocating only at its first frame.
char vk_phase_layout(int64_t pixels, uint64_t free_vram, int engines = 1);

class VulkanEngine : public DecisionEngine {
 public:
  // device: the index into celv2::vk::Context::devices(). engines: vk_phase_layout()'s. Throws std::runtime_error.
  VulkanEngine(int width, int height, int sb_size, int device = 0, int engines = 1);
  ~VulkanEngine() override;
  VulkanEngine(const VulkanEngine&) = delete;
  VulkanEngine& operator=(const VulkanEngine&) = delete;

  using DecisionEngine::launch;
  void launch(const Frame420& src, const std::vector<const Frame420*>& refs, int qindex, DecisionMap& dm,
              const std::vector<long long>& ref_keys = {}, long long src_key = -1) override;
  void prepare(const Frame420& next, long long key) override;
  void collect(DecisionMap& dm, RowGate& gate) override;
  void set_row_streams(int n) override;
  const std::string& device_name() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
