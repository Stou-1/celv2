// The decision engine on Vulkan (plan step 5): GpuEngine's work with the vk::Kernels, so the same
// decisions on any Vulkan GPU (the RTX 5090, the Arc Pro B50), queued as GpuEngine queues them.
#pragma once
#include <memory>
#include <vector>

#include "gpu/decision_engine.h"

class VulkanEngine : public DecisionEngine {
 public:
  // device: the index into celv2::vk::Context::devices(). Throws std::runtime_error.
  VulkanEngine(int width, int height, int sb_size, int device = 0);
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
