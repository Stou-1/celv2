# Vulkan port (plan step 5) — groundwork (2026-10-29)

The CUDA decision engine rewritten on Vulkan compute, for the RTX 5090 and the Arc Pro B50.
Nothing here is built yet.

## What there is to port

About 840 lines of CUDA (`gpu/kernels.cu`, `gpu/engine.cu`) plus the shared integer primitives in
`gpu/intra_common.h` and `gpu/inter_common.h` (`CELV2_HD`, compiled for both host and device).

| Kernel | Job | Port notes |
|---|---|---|
| `downsample_kernel` | 2x2 average pyramid | Trivial. |
| `me_kernel` | Full search per pyramid level | Plain loops. |
| `subpel_kernel` | Quarter-pel planes (8-tap) | Trivial. |
| `cost_kernel` | Intra / inter costs per block | Shared `int16_t` array; one `__shfl_down_sync` reduction over `lanes` (must match the CUDA sums). |
| `finalize_kernel` | Partition choice per superblock | Shared `int64_t` costs: needs `shaderInt64` (both GPUs have it). |
| `inter_tasks_kernel` | One-thread setup | Fold into the host or a push constant. |
| `satd_kernel` | Batched SATD for the CPU | Trivial. |

The host side uses only streams, async copies, pinned memory and events: these map to one compute
queue, staging buffers, and timeline semaphores (both GPUs support them).

`gpu/kernels_ref.cpp` is the CPU reference of every kernel, and `test_gpu_intra` / `test_gpu_me`
check the CUDA results against it bit for bit. The Vulkan kernels get the same tests, so the port
is correct when those tests pass on both GPUs; nothing else in the encoder changes
(`DecisionEngine` is the interface, `VulkanEngine` sits next to `GpuEngine`).

## What the two GPUs offer (vulkaninfo, today's drivers)

| | RTX 5090 | Arc Pro B50 |
|---|---|---|
| Vulkan | 1.4.351 | 1.4.356 |
| Subgroup size | 32 | 16-32 (default 32) |
| Shared memory | 48 KB | 48 KB |
| int8 / int16 / int64, 8/16-bit storage | yes | yes |
| Timeline semaphores | yes | yes |

The B50 can run subgroups of 16: either request 32 (`VK_EXT_subgroup_size_control`, in range) or
reduce through shared memory. The reduction must match the CUDA order, so it adds integers only
(exact in any order).

## Choices

- **Shader language: Slang**, compiled to SPIR-V at build time. It has generics and `inline`
  functions close to the C++ in `*_common.h`, so the primitives port nearly line for line. The fallback
  is GLSL with `glslc`. Both ship with the Vulkan SDK.
- **No Vulkan wrapper library**: plain `vulkan-1` (loader present in System32) with a small
  helper for buffers and pipelines. Volk is not needed for one queue and a few pipelines.
- CUDA stays as the default until the Vulkan engine matches it bit for bit and is no slower on the
  5090; then `--device vulkan`, and auto-pick when no CUDA device exists (the B50).

## Toolchain on this machine

- The LunarG SDK installer (1.4.363.0) needs administrator rights, so for now there are portable
  copies in `%LOCALAPPDATA%\bin`: `slang-2026.19` (`bin\slangc.exe`) and
  `Vulkan-Headers-1.4.363.0\include`. There is no `vulkan-1.lib`: the loader in System32 is opened
  with `LoadLibrary` and `vkGetInstanceProcAddr` (`VK_NO_PROTOTYPES`). There are no validation layers
  until the SDK is installed; then the build switches to `find_package(Vulkan)`.
- Spike (2026-10-29): `downsample_kernel` in Slang, built with `slangc -target spirv`, ran on both
  GPUs and matched the CPU on every sample (1920x1080, 1918x1078 and 15360x8640; 8-bit storage
  buffers, push constants). 16K: 9.6 ms on the 5090 and 43 ms on the B50, from host-visible memory,
  so the speed is not representative yet.

## Order

1. Done (branch feat/vulkan): `cmake/vulkan.cmake` finds the headers and slangc (SDK or the
   portable copies) and embeds each `src/vk/shaders/*.slang` as SPIR-V words; `vk::Context`
   (device, device-local buffers, staging, pipelines, synchronous dispatch) and `vk::Kernels`;
   `test_vk_kernels` checks `downsample` against `downsample_px` on both GPUs (1080p, odd sizes, 16K).
2. Done: subpel, me (top level, lower level, level 0 refined) and satd, also matched bit for bit on
   both GPUs. `src/vk/shaders/common.slang` is the shared primitives line for line; the kernels take
   planes by device address (buffer device address, push constants only), as the CUDA ones take
   pointers, so there are no descriptor sets.
3. Done: cost and finalize. On the same inputs as the CUDA kernels (320x200, 3 references, 160
   blocks of which 103 inter and 24 compound; and intra only) their intra costs, inter costs and
   SbOut decisions are byte-identical to CUDA's on both GPUs. The cost kernel's warp reduction is a
   subgroup shuffle in pipelines that require full subgroups of 32.
   `VulkanEngine` (`--device vulkan` / `vulkan2`: the first / second discrete GPU) sits behind
   `DecisionEngine`: 10 frames of anime A 1080p at q130 code to the same bytes as with the CUDA
   engine, on the RTX 5090 and on the Arc Pro B50. It still runs each frame synchronously and
   re-uploads every reference.
4. Done (see below): asynchronous submission (row by row, as the CUDA engine), the references
   cached by key and the next frame prepared ahead; the 5090 against CUDA and the B50 at 1080p and
   4K, and 16K.

## Speed, first version (2026-10-29, idle CPU)

`engine_bench` (the decision engines alone, two references, the next frame prepared), ms a frame:

| | 1080p | 4K | 16K |
|---|---|---|---|
| CUDA (RTX 5090) | 10.6 | 39.3 | 640 |
| Vulkan, RTX 5090 | 18.2 | 65.5 | 1002 |
| Vulkan, Arc Pro B50 | 123 | 432 | 6810 |

Whole encodes at q130 (same bytes on all three): anime A 1080p 5.53 / 5.12 / 2.55 fps, 4K 2.48 /
2.10 / 0.86 fps. On the 5090 the gap is the synchronous first version (every step waited for,
references re-uploaded, decisions downloaded whole). The B50 is 7x slower than the 5090 on the same
code: smaller, and it has no native 64-bit integers, which the cost and finalize kernels use for
every sum; 32-bit sums where they cannot overflow are the first thing to try there.


## Speed, queued version (2026-10-29 evening, idle CPU)

`VulkanEngine` now queues its work as `GpuEngine` does: `vk::Context` records batches (each step
behind a full memory barrier, so the order is the CUDA stream's) and submits each with a timeline
semaphore value. A launch is one batch for the uploads and the motion search, then one per
superblock row ending with that row's download, so `collect()` publishes rows as they finish.
References are kept by key (`RefSlot`, as CUDA's), and `prepare()` uploads the next source and its
intra costs ahead. The bytes are unchanged on all three engines.

`engine_bench`, ms a frame (the first version in brackets):

| | 1080p | 4K |
|---|---|---|
| CUDA (RTX 5090) | 10.4 | 41.2 |
| Vulkan, RTX 5090 | 13.7 (18.2) | 37.6 (65.5) |
| Vulkan, Arc Pro B50 | 126 (123) | 397 (432) |

Whole encodes, anime A 1080p q130 (30 frames, main as of the speed work): CUDA 8.09 fps, Vulkan on
the 5090 7.82 (7.05), on the B50 3.06 (2.58); 4K (16 frames) on the B50 1.28 fps. The CPU engine
codes the same 1080p at 0.09 fps, so `--device auto` without a CUDA device now takes the first
Vulkan GPU, then the CPU.

The B50 stays about 9x behind the 5090 on the same kernels, near their compute ratio. The cost
kernel's sums were 32-bit already (only the stores and addresses are 64-bit), so 64-bit emulation is
not what holds it back; any further gain there needs less work per candidate, not the port.

16K (8 frames, q130): CUDA 0.336 fps, Vulkan on the 5090 0.338, on the B50 0.122, the same bytes.
The B50 first failed there: its 16 GB did not hold the cost kernel's output for the whole frame
(7 references x 4400 candidates x 8 bytes a superblock, 8 GB at 16K) on top of the references'
quarter-pel planes (2.1 GB each). The kernels now index the costs from their first row on and the
engine keeps one row's worth (59 MB), since rows run one after the other. (Moving the base address
back by the row instead gave wrong costs at 16K: the address left the buffer by gigabytes.) A
failed launch now ends with an error instead of std::terminate from the engine's destructor.

## Validation layers (2026-10-30)

With the LunarG SDK 1.4.363 installed (its slangc 2026.17 is older than the portable 2026.19 the
build uses, so the build keeps that), the engine runs clean under the Khronos validation layer with
synchronization validation: `test_vk_kernels`, and 10 frames 1080p and 9 frames 4K through
`celv2enc --device vulkan` (RTX 5090) and `vulkan2` (Arc Pro B50) report no errors or warnings, and
the streams are the same with and without the layer and on both GPUs. To check again:

```
$env:VK_INSTANCE_LAYERS = 'VK_LAYER_KHRONOS_validation'; $env:VK_LAYER_VALIDATE_SYNC = '1'
.\build\celv2enc.exe -i clip.y4m -o out.ivf --frames 10 --device vulkan
```

Messages go to stderr as `Validation Error` / `Validation Warning` lines. Run it again after
narrowing the barriers (speed plan): synchronization validation is what catches a missing one.
