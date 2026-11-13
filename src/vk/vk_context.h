// A Vulkan compute device for the decision kernels (plan step 5): device-local buffers that the
// kernels reach by device address (as the CUDA kernels take pointers), compute pipelines from
// embedded SPIR-V with push constants only, dispatches run at once or recorded in batches. The
// loader (vulkan-1.dll) is opened at run time, so celv2 links and runs without Vulkan installed.
#pragma once
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace celv2::vk {

struct Fns;
struct Device;

// A buffer in device memory; `addr` is what a kernel's push constants point at.
struct Buffer {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  size_t size = 0;
  uint64_t addr = 0;
};

// A buffer in host memory that batches copy from or into, mapped at `map`.
struct HostBuffer {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  size_t size = 0;
  void* map = nullptr;
};

// One compute shader taking `push_bytes` of push constants (its parameters, buffers by address).
struct Kernel {
  VkPipeline pipe = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  int push_bytes = 0;
  const char* name = "kernel";  // CELV2_VK_TIMES' label
};

class Context {
 public:
  // The discrete GPUs' names, in the loader's order (empty without a Vulkan loader or GPU).
  static std::vector<std::string> devices();
  // Opens devices()[index]; throws std::runtime_error on failure.
  explicit Context(int index = 0);
  ~Context();
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  // The queue family a Context takes, from each family's flags: `forced` if it has compute
  // (CELV2_VK_FAMILY, for the runs), else one with compute and no graphics, else the first with
  // compute. flags.size() when none fits.
  static uint32_t compute_family(const std::vector<VkQueueFlags>& flags, int forced);

  const std::string& name() const { return name_; }
  VkQueueFlags queue_flags() const { return queue_flags_; }
  VkDevice device() const { return dev_; }
  VkQueue queue() const { return queue_; }
  // What the largest device-local heap has left (VK_EXT_memory_budget), or its size without that.
  uint64_t free_vram() const;

  Buffer buffer(size_t size);
  void free(Buffer& b);
  HostBuffer host_buffer(size_t size);  // cached where the driver has it (fast to read back)
  void free(HostBuffer& b);
  // Synchronous copies (not inside a batch).
  void upload(const Buffer& dst, const void* src, size_t size);
  void download(void* dst, const Buffer& src, size_t size);

  // A batch: from begin_batch() to submit(), dispatch(), copy() and update() are recorded, each
  // after everything before it (on the queue, so across batches too), and run on submit(), which
  // returns the value wait() takes to know the batch done. Outside a batch dispatch() and update()
  // run at once and wait. One thread records and submits; any thread may wait().
  void begin_batch();
  uint64_t submit();
  void wait(uint64_t value);
  // Submits the open batch (if any) and waits for every batch.
  void finish();
  void copy(const Buffer& dst, size_t dst_off, const HostBuffer& src, size_t src_off, size_t size);
  void copy(const HostBuffer& dst, size_t dst_off, const Buffer& src, size_t src_off, size_t size);
  // dst = data (size a multiple of 4, at most 65536 bytes), in order with the batch's work.
  void update(const Buffer& dst, const void* data, size_t size);

  template <size_t N>
  Kernel kernel(const uint32_t (&spv)[N], int push_bytes, int subgroup = 0) {
    return kernel_words(spv, N, push_bytes, subgroup);
  }
  // subgroup > 0: every subgroup is exactly that wide and full (the CUDA kernels' warps of 32).
  // (Not an overload of kernel(): kernel(spv, push, subgroup) would pick it, words = push.)
  Kernel kernel_words(const uint32_t* spv, size_t words, int push_bytes, int subgroup);
  // Runs k on gx x gy x gz workgroups and waits for it. In a batch: records it, then a barrier()
  // unless then_barrier is false (dispatches on separate buffers, ordered by one barrier() after them).
  void dispatch(const Kernel& k, const void* push, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1,
                bool then_barrier = true);
  void barrier();  // in a batch: everything recorded so far before everything after

  // CELV2_VK_TIMES=1 (the speed-first spec's step 0): GPU time stamps around each dispatch and copy in a batch,
  // summed per GPU and kernel name over the process, and printed at its end. A barrier goes before each, so the
  // timed work runs one dispatch at a time (no overlap: totals above an untimed run's) and each interval is that
  // dispatch's own time.
  struct KernelTime {
    std::string device, kernel;
    uint64_t calls = 0;
    double ms = 0;
  };
  static std::vector<KernelTime> kernel_times();

 private:
  void copy(VkBuffer dst, VkBuffer src, size_t size);
  void staging(size_t size);
  void begin();
  void submit_wait();
  void to_host(VkCommandBuffer cb);
  // Waits for this Context's submitted batches (not the open one, not other Contexts'). Before a free:
  // vkDeviceWaitIdle would wait for every Context on the GPU and needs all its queues held.
  void idle();

  // The batches' command buffers, each free again once the value it signals is reached.
  struct Batch {
    VkCommandBuffer cb = VK_NULL_HANDLE;
    uint64_t value = 0;
    VkQueryPool qp = VK_NULL_HANDLE;  // CELV2_VK_TIMES: two stamps per mark
    std::vector<const char*> marks;
  };
  static constexpr uint32_t kMarks = 256;  // stamped dispatches and copies a batch, at most
  void stamp_begin();
  void stamp_end(const char* name);
  void harvest(Batch& b);  // b's stamps into the totals (b done)
  bool times_ = false;
  double tick_ms_ = 0;
  uint64_t tick_mask_ = 0;
  static constexpr size_t kMaxBatches = 256;
  std::vector<Batch> batches_;
  Batch* open_ = nullptr;
  VkSemaphore timeline_ = VK_NULL_HANDLE;
  uint64_t submitted_ = 0;

  // The GPU's device, shared by every Context on it (V2 of the Vulkan wait spec); the rest of the
  // handles below are its, copied. queue_mu_ guards queue_, which other Contexts may share.
  std::shared_ptr<Device> d_;
  Fns* f_ = nullptr;
  std::mutex* queue_mu_ = nullptr;
  std::string name_;
  VkInstance inst_ = VK_NULL_HANDLE;
  VkPhysicalDevice pd_ = VK_NULL_HANDLE;
  VkDevice dev_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  uint32_t family_ = 0;
  VkQueueFlags queue_flags_ = 0;
  VkPhysicalDeviceMemoryProperties mem_props_{};
  VkCommandPool pool_ = VK_NULL_HANDLE;
  VkCommandBuffer cb_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  std::vector<Kernel> kernels_;
  VkBuffer stage_ = VK_NULL_HANDLE;  // host-visible, for upload / download
  VkDeviceMemory stage_mem_ = VK_NULL_HANDLE;
  void* stage_map_ = nullptr;
  size_t stage_size_ = 0;
};

}  // namespace celv2::vk
