#include "vk/vk_context.h"

#include <windows.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace celv2::vk {

#define CELV2_VK_INSTANCE_FNS(X)                                                                          \
  X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties)                    \
  X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceMemoryProperties) X(vkCreateDevice)   \
  X(vkGetDeviceProcAddr) X(vkGetPhysicalDeviceProperties2) X(vkGetPhysicalDeviceMemoryProperties2)
#define CELV2_VK_DEVICE_FNS(X)                                                                            \
  X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkDestroyBuffer)                            \
  X(vkGetBufferMemoryRequirements) X(vkAllocateMemory) X(vkFreeMemory) X(vkBindBufferMemory)             \
  X(vkMapMemory) X(vkCreateShaderModule) X(vkDestroyShaderModule) X(vkCreatePipelineLayout)              \
  X(vkDestroyPipelineLayout) X(vkCreateComputePipelines) X(vkDestroyPipeline) X(vkGetBufferDeviceAddress) \
  X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer)     \
  X(vkEndCommandBuffer) X(vkCmdBindPipeline) X(vkCmdPushConstants) X(vkCmdDispatch)                      \
  X(vkCmdCopyBuffer) X(vkCreateFence) X(vkDestroyFence) X(vkResetFences) X(vkWaitForFences)              \
  X(vkQueueSubmit) X(vkDeviceWaitIdle) X(vkCreateSemaphore) X(vkDestroySemaphore) X(vkWaitSemaphores)     \
  X(vkGetSemaphoreCounterValue) X(vkCmdPipelineBarrier) X(vkCmdUpdateBuffer)

struct Fns {
  HMODULE dll = nullptr;
  PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
  PFN_vkCreateInstance vkCreateInstance = nullptr;
#define CELV2_VK_DECL(name) PFN_##name name = nullptr;
  CELV2_VK_INSTANCE_FNS(CELV2_VK_DECL)
  CELV2_VK_DEVICE_FNS(CELV2_VK_DECL)
#undef CELV2_VK_DECL
  ~Fns() {
    if (dll) FreeLibrary(dll);
  }
};

namespace {

void ok(VkResult r, const char* what) {
  if (r != VK_SUCCESS) throw std::runtime_error(std::string("Vulkan: ") + what + " failed (" + std::to_string(r) + ")");
}

// The loader and an instance, with the instance functions; nullptr without a loader.
std::unique_ptr<Fns> open_instance(VkInstance* inst) {
  auto f = std::make_unique<Fns>();
  f->dll = LoadLibraryA("vulkan-1.dll");
  if (!f->dll) return nullptr;
  f->vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(f->dll, "vkGetInstanceProcAddr"));
  if (!f->vkGetInstanceProcAddr) return nullptr;
  f->vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(f->vkGetInstanceProcAddr(nullptr, "vkCreateInstance"));
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "celv2";
  app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  if (!f->vkCreateInstance || f->vkCreateInstance(&ici, nullptr, inst) != VK_SUCCESS) return nullptr;
#define CELV2_VK_LOAD(name) f->name = reinterpret_cast<PFN_##name>(f->vkGetInstanceProcAddr(*inst, #name));
  CELV2_VK_INSTANCE_FNS(CELV2_VK_LOAD)
#undef CELV2_VK_LOAD
  return f;
}

// CELV2_NO_VULKAN: as on a machine without a GPU, for the tests.
std::vector<VkPhysicalDevice> discrete(const Fns& f, VkInstance inst) {
  if (std::getenv("CELV2_NO_VULKAN")) return {};
  uint32_t n = 0;
  f.vkEnumeratePhysicalDevices(inst, &n, nullptr);
  std::vector<VkPhysicalDevice> all(n), out;
  f.vkEnumeratePhysicalDevices(inst, &n, all.data());
  for (VkPhysicalDevice pd : all) {
    VkPhysicalDeviceProperties p;
    f.vkGetPhysicalDeviceProperties(pd, &p);
    if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) out.push_back(pd);
  }
  return out;
}

}  // namespace

std::vector<std::string> Context::devices() {
  VkInstance inst = VK_NULL_HANDLE;
  const std::unique_ptr<Fns> f = open_instance(&inst);
  std::vector<std::string> names;
  if (!f) return names;
  for (VkPhysicalDevice pd : discrete(*f, inst)) {
    VkPhysicalDeviceProperties p;
    f->vkGetPhysicalDeviceProperties(pd, &p);
    names.push_back(p.deviceName);
  }
  f->vkDestroyInstance(inst, nullptr);
  return names;
}

uint64_t Context::free_vram() const {
  VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
  VkPhysicalDeviceMemoryProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &budget};
  f_->vkGetPhysicalDeviceMemoryProperties2(pd_, &props);
  uint64_t best = 0, size = 0;
  for (uint32_t i = 0; i < props.memoryProperties.memoryHeapCount; i++) {
    const VkMemoryHeap& h = props.memoryProperties.memoryHeaps[i];
    if (!(h.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) || h.size <= size) continue;
    size = h.size;
    best = budget.heapBudget[i] ? budget.heapBudget[i] - std::min(budget.heapBudget[i], budget.heapUsage[i]) : h.size;
  }
  return best;
}

Context::Context(int index) {
  f_ = open_instance(&inst_);
  if (!f_) throw std::runtime_error("Vulkan: no loader (vulkan-1.dll) or no instance");
  const std::vector<VkPhysicalDevice> pds = discrete(*f_, inst_);
  if (index < 0 || index >= static_cast<int>(pds.size())) {
    f_->vkDestroyInstance(inst_, nullptr);
    throw std::runtime_error("Vulkan: no discrete GPU " + std::to_string(index));
  }
  pd_ = pds[index];
  VkPhysicalDeviceProperties props;
  f_->vkGetPhysicalDeviceProperties(pd_, &props);
  name_ = props.deviceName;
  f_->vkGetPhysicalDeviceMemoryProperties(pd_, &mem_props_);

  uint32_t nq = 0;
  f_->vkGetPhysicalDeviceQueueFamilyProperties(pd_, &nq, nullptr);
  std::vector<VkQueueFamilyProperties> qf(nq);
  f_->vkGetPhysicalDeviceQueueFamilyProperties(pd_, &nq, qf.data());
  while (family_ < nq && !(qf[family_].queueFlags & VK_QUEUE_COMPUTE_BIT)) family_++;
  const float prio = 1;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = family_, qci.queueCount = 1, qci.pQueuePriorities = &prio;
  // The kernels read 8-bit planes and keep 16 / 64-bit costs (both GPUs have these).
  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  f12.storageBuffer8BitAccess = VK_TRUE, f12.shaderInt8 = VK_TRUE, f12.bufferDeviceAddress = VK_TRUE;
  f12.timelineSemaphore = VK_TRUE;  // batches (both GPUs have it)
  // Warps of 32 for the cost kernel (the B50 would pick 16 or 32).
  VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  f13.subgroupSizeControl = VK_TRUE, f13.computeFullSubgroups = VK_TRUE;
  f12.pNext = &f13;
  VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  f11.pNext = &f12, f11.storageBuffer16BitAccess = VK_TRUE;
  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  f2.pNext = &f11, f2.features.shaderInt16 = VK_TRUE, f2.features.shaderInt64 = VK_TRUE;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.pNext = &f2, dci.queueCreateInfoCount = 1, dci.pQueueCreateInfos = &qci;
  if (f_->vkCreateDevice(pd_, &dci, nullptr, &dev_) != VK_SUCCESS) {
    f_->vkDestroyInstance(inst_, nullptr);
    throw std::runtime_error("Vulkan: cannot create a device on " + name_);
  }
#define CELV2_VK_LOAD(name) f_->name = reinterpret_cast<PFN_##name>(f_->vkGetDeviceProcAddr(dev_, #name));
  CELV2_VK_DEVICE_FNS(CELV2_VK_LOAD)
#undef CELV2_VK_LOAD
  f_->vkGetDeviceQueue(dev_, family_, 0, &queue_);

  VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, cpci.queueFamilyIndex = family_;
  ok(f_->vkCreateCommandPool(dev_, &cpci, nullptr, &pool_), "vkCreateCommandPool");
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = pool_, cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, cbai.commandBufferCount = 1;
  ok(f_->vkAllocateCommandBuffers(dev_, &cbai, &cb_), "vkAllocateCommandBuffers");
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  ok(f_->vkCreateFence(dev_, &fci, nullptr, &fence_), "vkCreateFence");
  VkSemaphoreTypeCreateInfo sti{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  sti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE, sti.initialValue = 0;
  VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  sci.pNext = &sti;
  ok(f_->vkCreateSemaphore(dev_, &sci, nullptr, &timeline_), "vkCreateSemaphore");
  batches_.reserve(kMaxBatches);  // open_ points into it
}

Context::~Context() {
  if (!dev_) return;
  f_->vkDeviceWaitIdle(dev_);
  for (Kernel& k : kernels_) {
    f_->vkDestroyPipeline(dev_, k.pipe, nullptr);
    f_->vkDestroyPipelineLayout(dev_, k.layout, nullptr);
  }
  if (stage_) f_->vkDestroyBuffer(dev_, stage_, nullptr), f_->vkFreeMemory(dev_, stage_mem_, nullptr);
  f_->vkDestroySemaphore(dev_, timeline_, nullptr);
  f_->vkDestroyFence(dev_, fence_, nullptr);
  f_->vkDestroyCommandPool(dev_, pool_, nullptr);
  f_->vkDestroyDevice(dev_, nullptr);
  f_->vkDestroyInstance(inst_, nullptr);
}

namespace {

uint32_t memory_type(const VkPhysicalDeviceMemoryProperties& mp, uint32_t bits, VkMemoryPropertyFlags want) {
  for (uint32_t t = 0; t < mp.memoryTypeCount; t++)
    if ((bits >> t & 1) && (mp.memoryTypes[t].propertyFlags & want) == want) return t;
  throw std::runtime_error("Vulkan: no suitable memory type");
}

}  // namespace

Buffer Context::buffer(size_t size) {
  Buffer b;
  b.size = size;
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = std::max<size_t>(size, 4);
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
              VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  ok(f_->vkCreateBuffer(dev_, &bci, nullptr, &b.buf), "vkCreateBuffer");
  VkMemoryRequirements mr;
  f_->vkGetBufferMemoryRequirements(dev_, b.buf, &mr);
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = mr.size;
  mai.memoryTypeIndex = memory_type(mem_props_, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VkMemoryAllocateFlagsInfo maf{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  maf.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  mai.pNext = &maf;
  if (f_->vkAllocateMemory(dev_, &mai, nullptr, &b.mem) != VK_SUCCESS) {
    f_->vkDestroyBuffer(dev_, b.buf, nullptr);
    throw std::runtime_error("Vulkan: out of device memory (" + std::to_string(size) + " bytes)");
  }
  ok(f_->vkBindBufferMemory(dev_, b.buf, b.mem, 0), "vkBindBufferMemory");
  VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  ai.buffer = b.buf;
  b.addr = f_->vkGetBufferDeviceAddress(dev_, &ai);
  return b;
}

void Context::free(Buffer& b) {
  if (b.buf) f_->vkDeviceWaitIdle(dev_), f_->vkDestroyBuffer(dev_, b.buf, nullptr), f_->vkFreeMemory(dev_, b.mem, nullptr);
  b = Buffer{};
}

HostBuffer Context::host_buffer(size_t size) {
  HostBuffer h;
  h.size = size;
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = std::max<size_t>(size, 4), bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  ok(f_->vkCreateBuffer(dev_, &bci, nullptr, &h.buf), "vkCreateBuffer");
  VkMemoryRequirements mr;
  f_->vkGetBufferMemoryRequirements(dev_, h.buf, &mr);
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = mr.size;
  const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  try {  // cached memory reads back fast; not every driver has it coherent
    mai.memoryTypeIndex = memory_type(mem_props_, mr.memoryTypeBits, host | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
  } catch (const std::runtime_error&) {
    mai.memoryTypeIndex = memory_type(mem_props_, mr.memoryTypeBits, host);
  }
  if (f_->vkAllocateMemory(dev_, &mai, nullptr, &h.mem) != VK_SUCCESS) {
    f_->vkDestroyBuffer(dev_, h.buf, nullptr);
    throw std::runtime_error("Vulkan: out of host memory (" + std::to_string(size) + " bytes)");
  }
  ok(f_->vkBindBufferMemory(dev_, h.buf, h.mem, 0), "vkBindBufferMemory");
  ok(f_->vkMapMemory(dev_, h.mem, 0, VK_WHOLE_SIZE, 0, &h.map), "vkMapMemory");
  return h;
}

void Context::free(HostBuffer& h) {
  if (h.buf) f_->vkDeviceWaitIdle(dev_), f_->vkDestroyBuffer(dev_, h.buf, nullptr), f_->vkFreeMemory(dev_, h.mem, nullptr);
  h = HostBuffer{};
}

void Context::staging(size_t size) {
  if (size <= stage_size_) return;
  HostBuffer old{stage_, stage_mem_, stage_size_, stage_map_};
  free(old);
  stage_ = VK_NULL_HANDLE, stage_size_ = 0;
  const HostBuffer h = host_buffer(size);
  stage_ = h.buf, stage_mem_ = h.mem, stage_map_ = h.map, stage_size_ = size;
}

void Context::begin() {
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  ok(f_->vkBeginCommandBuffer(cb_, &bi), "vkBeginCommandBuffer");
}

void Context::submit_wait() {
  ok(f_->vkEndCommandBuffer(cb_), "vkEndCommandBuffer");
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1, si.pCommandBuffers = &cb_;
  ok(f_->vkQueueSubmit(queue_, 1, &si, fence_), "vkQueueSubmit");
  ok(f_->vkWaitForFences(dev_, 1, &fence_, VK_TRUE, UINT64_MAX), "vkWaitForFences");
  ok(f_->vkResetFences(dev_, 1, &fence_), "vkResetFences");
}

void Context::copy(VkBuffer dst, VkBuffer src, size_t size) {
  begin();
  const VkBufferCopy region{0, 0, size};
  f_->vkCmdCopyBuffer(cb_, src, dst, 1, &region);
  submit_wait();
}

void Context::upload(const Buffer& dst, const void* src, size_t size) {
  if (size > dst.size) throw std::invalid_argument("Vulkan upload: larger than the buffer");
  if (!size) return;
  staging(size);
  std::memcpy(stage_map_, src, size);
  copy(dst.buf, stage_, size);
}

void Context::download(void* dst, const Buffer& src, size_t size) {
  if (size > src.size) throw std::invalid_argument("Vulkan download: larger than the buffer");
  if (!size) return;
  staging(size);
  copy(stage_, src.buf, size);
  std::memcpy(dst, stage_map_, size);
}

Kernel Context::kernel_words(const uint32_t* spv, size_t words, int push_bytes, int subgroup) {
  Kernel k;
  k.push_bytes = push_bytes;
  const VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(push_bytes)};
  VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pli.pushConstantRangeCount = push_bytes ? 1 : 0, pli.pPushConstantRanges = &pcr;
  ok(f_->vkCreatePipelineLayout(dev_, &pli, nullptr, &k.layout), "vkCreatePipelineLayout");
  VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smi.codeSize = words * 4, smi.pCode = spv;
  VkShaderModule sm;
  ok(f_->vkCreateShaderModule(dev_, &smi, nullptr, &sm), "vkCreateShaderModule");
  VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main"};
  VkPipelineShaderStageRequiredSubgroupSizeCreateInfo sg{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
  if (subgroup > 0) {
    VkPhysicalDeviceVulkan13Properties p13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &p13;
    f_->vkGetPhysicalDeviceProperties2(pd_, &p2);
    if (static_cast<uint32_t>(subgroup) < p13.minSubgroupSize || static_cast<uint32_t>(subgroup) > p13.maxSubgroupSize) {
      f_->vkDestroyShaderModule(dev_, sm, nullptr);
      f_->vkDestroyPipelineLayout(dev_, k.layout, nullptr);
      throw std::runtime_error("Vulkan: " + name_ + " cannot run subgroups of " + std::to_string(subgroup));
    }
    sg.requiredSubgroupSize = static_cast<uint32_t>(subgroup);
    cpi.stage.pNext = &sg;
    cpi.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
  }
  cpi.layout = k.layout;
  const VkResult r = f_->vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &cpi, nullptr, &k.pipe);
  f_->vkDestroyShaderModule(dev_, sm, nullptr);
  ok(r, "vkCreateComputePipelines");
  kernels_.push_back(k);
  return k;
}

void Context::dispatch(const Kernel& k, const void* push, uint32_t gx, uint32_t gy, uint32_t gz) {
  if (!open_) begin();
  const VkCommandBuffer cb = open_ ? open_->cb : cb_;
  f_->vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipe);
  if (k.push_bytes) f_->vkCmdPushConstants(cb, k.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, k.push_bytes, push);
  f_->vkCmdDispatch(cb, gx, gy, gz);
  if (open_) barrier();
  else submit_wait();
}

void Context::barrier() {
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  f_->vkCmdPipelineBarrier(open_->cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb,
                           0, nullptr, 0, nullptr);
}

void Context::begin_batch() {
  if (open_) throw std::logic_error("Vulkan: a batch is already open");
  uint64_t done = 0;
  ok(f_->vkGetSemaphoreCounterValue(dev_, timeline_, &done), "vkGetSemaphoreCounterValue");
  for (Batch& b : batches_)
    if (b.value <= done) {
      open_ = &b;
      break;
    }
  if (!open_) {
    if (batches_.size() == kMaxBatches) {  // all in flight: wait for the oldest
      wait(done + 1);
      return begin_batch();
    }
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = pool_, cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, cbai.commandBufferCount = 1;
    Batch b;
    ok(f_->vkAllocateCommandBuffers(dev_, &cbai, &b.cb), "vkAllocateCommandBuffers");
    batches_.push_back(b);
    open_ = &batches_.back();
  }
  open_->value = UINT64_MAX;  // not free until submitted and done
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  ok(f_->vkBeginCommandBuffer(open_->cb, &bi), "vkBeginCommandBuffer");
}

uint64_t Context::submit() {
  if (!open_) throw std::logic_error("Vulkan: no batch to submit");
  Batch& b = *open_;
  open_ = nullptr;
  ok(f_->vkEndCommandBuffer(b.cb), "vkEndCommandBuffer");
  const uint64_t value = submitted_ + 1;
  VkTimelineSemaphoreSubmitInfo tsi{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  tsi.signalSemaphoreValueCount = 1, tsi.pSignalSemaphoreValues = &value;
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.pNext = &tsi, si.commandBufferCount = 1, si.pCommandBuffers = &b.cb;
  si.signalSemaphoreCount = 1, si.pSignalSemaphores = &timeline_;
  ok(f_->vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
  b.value = submitted_ = value;
  return value;
}

void Context::wait(uint64_t value) {
  VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  wi.semaphoreCount = 1, wi.pSemaphores = &timeline_, wi.pValues = &value;
  ok(f_->vkWaitSemaphores(dev_, &wi, UINT64_MAX), "vkWaitSemaphores");
}

void Context::finish() {
  if (open_) submit();
  if (submitted_) wait(submitted_);
}

void Context::copy(const Buffer& dst, size_t dst_off, const HostBuffer& src, size_t src_off, size_t size) {
  if (!open_) throw std::logic_error("Vulkan: copy outside a batch");
  if (dst_off + size > dst.size || src_off + size > src.size) throw std::invalid_argument("Vulkan copy: out of range");
  const VkBufferCopy region{src_off, dst_off, size};
  f_->vkCmdCopyBuffer(open_->cb, src.buf, dst.buf, 1, &region);
  barrier();
}

void Context::copy(const HostBuffer& dst, size_t dst_off, const Buffer& src, size_t src_off, size_t size) {
  if (!open_) throw std::logic_error("Vulkan: copy outside a batch");
  if (dst_off + size > dst.size || src_off + size > src.size) throw std::invalid_argument("Vulkan copy: out of range");
  const VkBufferCopy region{src_off, dst_off, size};
  f_->vkCmdCopyBuffer(open_->cb, src.buf, dst.buf, 1, &region);
  barrier();
}

void Context::update(const Buffer& dst, const void* data, size_t size) {
  if (size % 4 || size > 65536 || size > dst.size) throw std::invalid_argument("Vulkan update: bad size");
  if (!open_) return upload(dst, data, size);
  f_->vkCmdUpdateBuffer(open_->cb, dst.buf, 0, size, data);
  barrier();
}

}  // namespace celv2::vk
