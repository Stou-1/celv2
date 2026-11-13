#include "vk/vk_context.h"

#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
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
  X(vkGetSemaphoreCounterValue) X(vkCmdPipelineBarrier) X(vkCmdUpdateBuffer) X(vkCreateQueryPool)            \
  X(vkDestroyQueryPool) X(vkCmdResetQueryPool) X(vkCmdWriteTimestamp) X(vkGetQueryPoolResults)                              X(vkGetPipelineExecutablePropertiesKHR) X(vkGetPipelineExecutableStatisticsKHR)

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

uint32_t Context::compute_family(const std::vector<VkQueueFlags>& flags, int forced) {
  const uint32_t n = static_cast<uint32_t>(flags.size());
  if (forced >= 0) return forced < static_cast<int>(n) && (flags[forced] & VK_QUEUE_COMPUTE_BIT) ? forced : n;
  // A compute-only family first (V1 of 2026-11-07-vulkan-wait-design.md); it may still copy.
  for (uint32_t i = 0; i < n; i++)
    if ((flags[i] & VK_QUEUE_COMPUTE_BIT) && !(flags[i] & VK_QUEUE_GRAPHICS_BIT)) return i;
  for (uint32_t i = 0; i < n; i++)
    if (flags[i] & VK_QUEUE_COMPUTE_BIT) return i;
  return n;
}

// One GPU's instance, device and queues, shared by the Contexts on it (V2 of the Vulkan wait spec):
// separate devices are separate contexts to the OS, which take turns on the GPU.
struct Device {
  std::unique_ptr<Fns> f;
  VkInstance inst = VK_NULL_HANDLE;
  VkPhysicalDevice pd = VK_NULL_HANDLE;
  VkDevice dev = VK_NULL_HANDLE;
  std::string name;
  VkPhysicalDeviceMemoryProperties mem_props{};
  uint32_t family = 0;
  VkQueueFlags flags = 0;
  std::vector<VkQueue> queues;
  std::vector<std::mutex> queue_mu;  // vkQueueSubmit needs its queue held
  uint32_t next = 0;                 // the queue the next Context takes, under g_devices_mu
  bool stats = false;                // CELV2_VK_STATS: each pipeline's driver statistics to stderr

  explicit Device(int index);
  ~Device() {
    if (dev) f->vkDeviceWaitIdle(dev), f->vkDestroyDevice(dev, nullptr);
    if (inst) f->vkDestroyInstance(inst, nullptr);
  }
};

Device::Device(int index) {
  f = open_instance(&inst);
  if (!f) throw std::runtime_error("Vulkan: no loader (vulkan-1.dll) or no instance");
  const std::vector<VkPhysicalDevice> pds = discrete(*f, inst);
  if (index < 0 || index >= static_cast<int>(pds.size())) {
    f->vkDestroyInstance(inst, nullptr), inst = VK_NULL_HANDLE;
    throw std::runtime_error("Vulkan: no discrete GPU " + std::to_string(index));
  }
  pd = pds[index];
  VkPhysicalDeviceProperties props;
  f->vkGetPhysicalDeviceProperties(pd, &props);
  name = props.deviceName;
  f->vkGetPhysicalDeviceMemoryProperties(pd, &mem_props);

  uint32_t nq = 0;
  f->vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
  std::vector<VkQueueFamilyProperties> qf(nq);
  f->vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf.data());
  std::vector<VkQueueFlags> all;
  for (const VkQueueFamilyProperties& q : qf) all.push_back(q.queueFlags);
  const char* forced = std::getenv("CELV2_VK_FAMILY");
  family = Context::compute_family(all, forced && *forced ? std::atoi(forced) : -1);
  if (family >= nq) {
    f->vkDestroyInstance(inst, nullptr), inst = VK_NULL_HANDLE;
    throw std::runtime_error("Vulkan: no compute queue on " + name);
  }
  flags = all[family];
  const std::vector<float> prio(qf[family].queueCount, 1.0f);
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = family, qci.queueCount = qf[family].queueCount, qci.pQueuePriorities = prio.data();
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
  VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR fx{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
  const char* ext = VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME;
  stats = std::getenv("CELV2_VK_STATS") != nullptr;
  if (stats) fx.pipelineExecutableInfo = VK_TRUE, f13.pNext = &fx, dci.enabledExtensionCount = 1, dci.ppEnabledExtensionNames = &ext;
  if (f->vkCreateDevice(pd, &dci, nullptr, &dev) != VK_SUCCESS) {
    dev = VK_NULL_HANDLE, f->vkDestroyInstance(inst, nullptr), inst = VK_NULL_HANDLE;
    throw std::runtime_error("Vulkan: cannot create a device on " + name);
  }
#define CELV2_VK_LOAD(name) f->name = reinterpret_cast<PFN_##name>(f->vkGetDeviceProcAddr(dev, #name));
  CELV2_VK_DEVICE_FNS(CELV2_VK_LOAD)
#undef CELV2_VK_LOAD
  queues.resize(qf[family].queueCount);
  for (uint32_t q = 0; q < queues.size(); q++) f->vkGetDeviceQueue(dev, family, q, &queues[q]);
  queue_mu = std::vector<std::mutex>(queues.size());
}

namespace {

// Each GPU's Device while a Context holds it; the last one out closes it, a later one opens it again.
std::mutex g_devices_mu;
std::map<int, std::weak_ptr<Device>> g_devices;
std::map<int, int> g_opened;  // CELV2_VK_QUEUES=devices: the Contexts opened on each GPU

}  // namespace

Context::Context(int index) {
  {
    const std::lock_guard<std::mutex> lock(g_devices_mu);
    // Every Context on the first queue: on the Arc Pro B50 (driver 32.0.101.8976) kernels with scratch
    // memory (cost 18 KB, finalize, me) give wrong results or lose the device while two queues of one
    // VkDevice run them at once, as if the queues shared one scratch area; separate VkDevices do not
    // (probes_on_queues, 2026-11-08). One queue is also the faster there. CELV2_VK_QUEUES=many: the k-th takes
    // queue k mod count (the runs); a number n: every one on queue n mod count; devices: the k-th on a
    // VkDevice of its own (k mod 4), its first queue, as V1.
    const char* many = std::getenv("CELV2_VK_QUEUES");
    const bool spread = many && std::string(many) == "many", own = many && std::string(many) == "devices";
    const int key = own ? index + 1000 * (1 + g_opened[index]++ % 4) : index;
    d_ = g_devices[key].lock();
    if (!d_) d_ = std::make_shared<Device>(index), g_devices[key] = d_;
    const uint32_t nq = static_cast<uint32_t>(d_->queues.size());
    const uint32_t q = spread ? d_->next++ % nq : many && *many && !own ? static_cast<uint32_t>(std::atoi(many)) % nq : 0;
    queue_ = d_->queues[q], queue_mu_ = &d_->queue_mu[q];
  }
  f_ = d_->f.get(), inst_ = d_->inst, pd_ = d_->pd, dev_ = d_->dev, name_ = d_->name;
  mem_props_ = d_->mem_props, family_ = d_->family, queue_flags_ = d_->flags;

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
  if (const char* t = std::getenv("CELV2_VK_TIMES"); t && *t && *t != '0') {
    VkPhysicalDeviceProperties props;
    f_->vkGetPhysicalDeviceProperties(pd_, &props);
    uint32_t nq = 0;
    f_->vkGetPhysicalDeviceQueueFamilyProperties(pd_, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    f_->vkGetPhysicalDeviceQueueFamilyProperties(pd_, &nq, qf.data());
    const uint32_t bits = family_ < nq ? qf[family_].timestampValidBits : 0;
    times_ = bits > 0;  // a family without time stamps: nothing timed
    tick_ms_ = props.limits.timestampPeriod * 1e-6;
    tick_mask_ = bits >= 64 ? ~0ull : (1ull << bits) - 1;
  }
}

Context::~Context() {
  if (!dev_) return;
  idle();
  for (Kernel& k : kernels_) {
    f_->vkDestroyPipeline(dev_, k.pipe, nullptr);
    f_->vkDestroyPipelineLayout(dev_, k.layout, nullptr);
  }
  if (stage_) f_->vkDestroyBuffer(dev_, stage_, nullptr), f_->vkFreeMemory(dev_, stage_mem_, nullptr);
  for (Batch& b : batches_)
    if (b.qp) f_->vkDestroyQueryPool(dev_, b.qp, nullptr);
  f_->vkDestroySemaphore(dev_, timeline_, nullptr);
  f_->vkDestroyFence(dev_, fence_, nullptr);
  f_->vkDestroyCommandPool(dev_, pool_, nullptr);
  // d_ goes after this: the last Context on the GPU closes its device.
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
  if (b.buf) idle(), f_->vkDestroyBuffer(dev_, b.buf, nullptr), f_->vkFreeMemory(dev_, b.mem, nullptr);
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
  if (h.buf) idle(), f_->vkDestroyBuffer(dev_, h.buf, nullptr), f_->vkFreeMemory(dev_, h.mem, nullptr);
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

// The device's writes made visible to host reads after the wait (a fence or semaphore alone does not).
void Context::to_host(VkCommandBuffer cb) {
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT, mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  f_->vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0,
                           nullptr);
}

void Context::submit_wait() {
  to_host(cb_);
  ok(f_->vkEndCommandBuffer(cb_), "vkEndCommandBuffer");
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1, si.pCommandBuffers = &cb_;
  {
    const std::lock_guard<std::mutex> lock(*queue_mu_);
    ok(f_->vkQueueSubmit(queue_, 1, &si, fence_), "vkQueueSubmit");
  }
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
  if (d_->stats) cpi.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
  const VkResult r = f_->vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &cpi, nullptr, &k.pipe);
  f_->vkDestroyShaderModule(dev_, sm, nullptr);
  ok(r, "vkCreateComputePipelines");
  if (d_->stats) {  // the investigation of the B50's queues (2026-11-08): scratch and spills per kernel
    VkPipelineInfoKHR pi{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
    pi.pipeline = k.pipe;
    uint32_t ne = 0;
    f_->vkGetPipelineExecutablePropertiesKHR(dev_, &pi, &ne, nullptr);
    std::vector<VkPipelineExecutablePropertiesKHR> ep(ne, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
    f_->vkGetPipelineExecutablePropertiesKHR(dev_, &pi, &ne, ep.data());
    for (uint32_t e = 0; e < ne; e++) {
      VkPipelineExecutableInfoKHR ei{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
      ei.pipeline = k.pipe, ei.executableIndex = e;
      uint32_t ns = 0;
      f_->vkGetPipelineExecutableStatisticsKHR(dev_, &ei, &ns, nullptr);
      std::vector<VkPipelineExecutableStatisticKHR> st(ns, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
      f_->vkGetPipelineExecutableStatisticsKHR(dev_, &ei, &ns, st.data());
      std::string line = "stats " + std::to_string(words) + " words " + ep[e].name + " (subgroup " +
                         std::to_string(ep[e].subgroupSize) + "):";
      for (const VkPipelineExecutableStatisticKHR& x : st) {
        line += std::string(" ") + x.name + "=";
        line += x.format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR ? std::to_string(x.value.f64)
                : x.format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR ? std::to_string(x.value.i64)
                : x.format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR ? std::to_string(x.value.b32)
                                                                                 : std::to_string(x.value.u64);
      }
      std::fprintf(stderr, "%s\n", line.c_str());
    }
  }
  kernels_.push_back(k);
  return k;
}

namespace {
std::mutex g_times_mu;
std::map<std::pair<std::string, std::string>, Context::KernelTime> g_times;
// Prints the totals at the process's end, when CELV2_VK_TIMES gathered any.
struct TimesPrinter {
  ~TimesPrinter() {
    for (const Context::KernelTime& t : Context::kernel_times())
      std::fprintf(stderr, "celv2enc: vk-times: %s: %-12s %8llu calls %10.1f ms %8.1f us each\n", t.device.c_str(),
                   t.kernel.c_str(), static_cast<unsigned long long>(t.calls), t.ms, t.calls ? 1000 * t.ms / t.calls : 0.0);
  }
} g_times_printer;
}  // namespace

std::vector<Context::KernelTime> Context::kernel_times() {
  const std::lock_guard<std::mutex> lock(g_times_mu);
  std::vector<KernelTime> v;
  for (const auto& e : g_times) v.push_back(e.second);
  return v;
}

// The barrier first: without it a GPU that overlaps work (the RTX 5090) starts the dispatch before the work ahead
// ends, and the stamp, written when that work ends, comes late (cost inter read 0.7 us). Timed batches therefore
// run their work one after another.
void Context::stamp_begin() {
  if (!times_ || !open_ || open_->marks.size() >= kMarks) return;
  barrier();
  f_->vkCmdWriteTimestamp(open_->cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, open_->qp,
                          static_cast<uint32_t>(2 * open_->marks.size()));
}

void Context::stamp_end(const char* name) {
  if (!times_ || !open_ || open_->marks.size() >= kMarks) return;
  barrier();  // likewise: the RTX 5090 wrote this one before the dispatch ended (0.7 us a row's costs)
  f_->vkCmdWriteTimestamp(open_->cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, open_->qp,
                          static_cast<uint32_t>(2 * open_->marks.size() + 1));
  open_->marks.push_back(name);
}

void Context::harvest(Batch& b) {
  if (b.marks.empty()) return;
  std::vector<uint64_t> t(2 * b.marks.size());
  const VkResult r = f_->vkGetQueryPoolResults(dev_, b.qp, 0, static_cast<uint32_t>(t.size()), t.size() * 8, t.data(), 8,
                                              VK_QUERY_RESULT_64_BIT);
  if (r == VK_SUCCESS) {
    const std::lock_guard<std::mutex> lock(g_times_mu);
    for (size_t i = 0; i < b.marks.size(); i++) {
      KernelTime& k = g_times[{name_, b.marks[i]}];
      k.device = name_, k.kernel = b.marks[i];
      k.calls++, k.ms += static_cast<double>((t[2 * i + 1] - t[2 * i]) & tick_mask_) * tick_ms_;
    }
  }
  b.marks.clear();
}

void Context::dispatch(const Kernel& k, const void* push, uint32_t gx, uint32_t gy, uint32_t gz, bool then_barrier) {
  if (!open_) begin();
  const VkCommandBuffer cb = open_ ? open_->cb : cb_;
  stamp_begin();
  f_->vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipe);
  if (k.push_bytes) f_->vkCmdPushConstants(cb, k.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, k.push_bytes, push);
  f_->vkCmdDispatch(cb, gx, gy, gz);
  stamp_end(k.name);
  if (!open_) submit_wait();
  else if (then_barrier) barrier();
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
      harvest(b);
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
  if (times_) {
    if (!open_->qp) {
      VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
      qi.queryType = VK_QUERY_TYPE_TIMESTAMP, qi.queryCount = 2 * kMarks;
      ok(f_->vkCreateQueryPool(dev_, &qi, nullptr, &open_->qp), "vkCreateQueryPool");
    }
    f_->vkCmdResetQueryPool(open_->cb, open_->qp, 0, 2 * kMarks);
  }
}

uint64_t Context::submit() {
  if (!open_) throw std::logic_error("Vulkan: no batch to submit");
  Batch& b = *open_;
  open_ = nullptr;
  to_host(b.cb);
  ok(f_->vkEndCommandBuffer(b.cb), "vkEndCommandBuffer");
  const uint64_t value = submitted_ + 1;
  VkTimelineSemaphoreSubmitInfo tsi{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  tsi.signalSemaphoreValueCount = 1, tsi.pSignalSemaphoreValues = &value;
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.pNext = &tsi, si.commandBufferCount = 1, si.pCommandBuffers = &b.cb;
  si.signalSemaphoreCount = 1, si.pSignalSemaphores = &timeline_;
  {
    const std::lock_guard<std::mutex> lock(*queue_mu_);
    ok(f_->vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
  }
  b.value = submitted_ = value;
  return value;
}

void Context::wait(uint64_t value) {
  VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  wi.semaphoreCount = 1, wi.pSemaphores = &timeline_, wi.pValues = &value;
  ok(f_->vkWaitSemaphores(dev_, &wi, UINT64_MAX), "vkWaitSemaphores");
}

void Context::idle() {
  try {
    if (submitted_) wait(submitted_);
    for (Batch& b : batches_)
      if (b.value <= submitted_) harvest(b);
  } catch (const std::runtime_error&) {  // a lost device: nothing left to wait for (and destructors call this)
  }
}

void Context::finish() {
  if (open_) submit();
  if (submitted_) wait(submitted_);
  for (Batch& b : batches_) harvest(b);
}

void Context::copy(const Buffer& dst, size_t dst_off, const HostBuffer& src, size_t src_off, size_t size) {
  if (!open_) throw std::logic_error("Vulkan: copy outside a batch");
  if (dst_off + size > dst.size || src_off + size > src.size) throw std::invalid_argument("Vulkan copy: out of range");
  const VkBufferCopy region{src_off, dst_off, size};
  stamp_begin();
  f_->vkCmdCopyBuffer(open_->cb, src.buf, dst.buf, 1, &region);
  stamp_end("upload");
  barrier();
}

void Context::copy(const HostBuffer& dst, size_t dst_off, const Buffer& src, size_t src_off, size_t size) {
  if (!open_) throw std::logic_error("Vulkan: copy outside a batch");
  if (dst_off + size > dst.size || src_off + size > src.size) throw std::invalid_argument("Vulkan copy: out of range");
  const VkBufferCopy region{src_off, dst_off, size};
  stamp_begin();
  f_->vkCmdCopyBuffer(open_->cb, src.buf, dst.buf, 1, &region);
  stamp_end("download");
  barrier();
}

void Context::update(const Buffer& dst, const void* data, size_t size) {
  if (size % 4 || size > 65536 || size > dst.size) throw std::invalid_argument("Vulkan update: bad size");
  if (!open_) return upload(dst, data, size);
  f_->vkCmdUpdateBuffer(open_->cb, dst.buf, 0, size, data);
  barrier();
}

}  // namespace celv2::vk
