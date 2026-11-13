# Vulkan's waiting (design)

2026-11-07. From the speed ledger's "Why fps fell" (2b17000). Encoder only, no format change, the same bytes on every row.
Vendor-neutral: one code path, no driver or GPU names in it.

## Why

| Default settings, quiet machine | S1 fps, cores busy | S2 AIR fps, cores busy |
|---|---|---|
| CUDA | 10.92, 32.5 | 10.22, 34.0 |
| Vulkan | 10.19, 29.9 | 9.44, 31.7 |

- Vulkan costs 6.7 and 7.6%. The user waived 5% on 2026-11-05, on the under-load figures (3.3, 4.8%). The retire-CUDA spec's rule is: past 5%, tuned first, or the user decides.
- CPU time a frame is the same, so this is waiting. On 3 workers it is 20 and 24 ms a frame per worker (275 to 294 ms, 294 to 318 ms).
- One engine alone does not hold that much: `engine_bench` on 2026-10-29 had 1080p at 10.4 ms on CUDA, 13.7 on Vulkan, and 4K faster on Vulkan (2 references, an older build).
- Tile threads wait for the first row, and the first row waits for every reference's motion search, on both engines. A delay on the GPU therefore lands on idle cores, which fits cores busy falling while CPU time does not rise.

## What differs

From reading `src/vk/` and `src/gpu/engine.cu`, and `vulkaninfo` on both GPUs. Not measured.

| | CUDA | Vulkan |
|---|---|---|
| Engines on one GPU | one process context, a stream each | a VkInstance, VkDevice and queue each: 3 at 1080p, twice that with the GOP trial |
| Queue | compute | the first family with compute: the graphics family on both GPUs |
| Motion search | 7 references x 5 levels, one chain on the stream | the same chain, with an all-commands barrier after each of the 35 dispatches |
| Waiting for a row | `cudaEventSynchronize`, no blocking flag: spins or yields | `vkWaitSemaphores`: sleeps. By reading, this is why the ledger's decision row fell from 38 to 11 ms |
| Work handed over a frame at 1080p | launches on the stream (their cost not read) | 19 `vkQueueSubmit`: uploads and search, 17 rows, `prepare()` |

**The first suspect is the first row.** Separate devices are separate contexts to the OS, and contexts share a GPU engine by turns; CUDA's streams in one context can run together. A row's kernels fill little of a 5090, so three workers' work would fit side by side. In turns, each worker's first row waits for the others' slices. This is a reading: neither side of it has been measured.

| Family | RTX 5090 | Arc Pro B50 |
|---|---|---|
| Graphics and compute | 0: 16 queues | 0: 1 queue |
| Compute only | 2: 8 queues | 1: 4 queues (no transfer bit; compute queues may copy all the same) |

## Runs

Every run: S1 and S2 AIR, ledger's settings, RTX 5090, old and new alternating, 3 rounds, medians. The default's spread is 1-2% in fps, so a row is kept at +1% or better on both, and no worse than -1% on S2 Dragon Maid, S3 and the B50 on S1. The rows stack in the order kept.

**V0, where the wait is. No change to the output.**
- `CELV2_PROFILE` gains two wall times a frame, mean over frames: `launch()`, and the end of `launch()` to the first row published. Both engines.
- S1 and S2 AIR, `--workers auto` and `--workers 1`, CUDA and Vulkan, 2 rounds alternating. `engine_bench` at 7 references on the current build, both engines.

Reading:
- Gap at `--workers 1` under 2%: it is between engines. V1, V2, V3, then V4 if needed.
- Gap stays: it is in each engine. V1, V3, V4, then V2.
- `launch()` 3 ms or more past CUDA's: rows in groups (row 0 alone, then 2 to 4 a batch) join the list after V3.

**V1, a compute queue.** `Context()` takes a family with compute and no graphics, else the first with compute. `CELV2_VK_FAMILY` picks a family, for the runs.

**V2, one device per GPU, a queue per engine.**
- A process-wide device per GPU index, shared by `shared_ptr`, made and dropped under a mutex: VkInstance, VkDevice, `Fns`, the family's queues. The last engine out closes it, and a later engine opens it again (the target size passes).
- Each `Context` keeps its own command pool, batches, timeline, fence, staging buffer and kernels. Its interface does not change. No buffer is shared between engines.
- The k-th `Context` takes queue k mod count. Contexts on one queue submit under that queue's mutex: this happens at 6 engines on the B50. Each context's batches stay in order, and its barriers also order the other's work: correct, slower.
- **`vkDeviceWaitIdle` goes from `free()` and `~Context()`.** It waits for every engine, and it needs all queues held. It becomes `wait(submitted_)`, the engine's own work. Not `finish()`: `ensure()` frees inside an open batch, and today's wait does not submit that batch either.
- `CELV2_VK_QUEUES=1` forces every context onto one queue, for the tests.

Queues of one device may not run together on every GPU. Nothing here promises it. If they do not, V2 shows nothing there and the row closes on that GPU's numbers.

**V3, the motion search by level.**
- New references' uploads first, as now. Then level by level from the top: every reference's `me_level()`, one barrier. 35 barriers become 5, and each level is 7 references wide.
- `Context::dispatch()` gets a form with no barrier, and `barrier()` becomes public. Each reference has its own `mvx`, `mvy`, so nothing changes in them.
- One stream cannot do this. This is the row where Vulkan may pass CUDA.

**V4, spin, then sleep.** `wait()` polls `vkGetSemaphoreCounterValue` for up to 200 us, then calls `vkWaitSemaphores`. Bound: +1% CPU a frame on S1 and S2 AIR. Only if the gap is still over 2%.

## Reading

- **Gap under 2%:** done. A "Saved" row in the ledger, and the same table in `docs/reports/2026-11-05-vulkan-only.md`.
- **Between 2 and 5%:** kept rows stand, reported to the user. The waiver covers it.
- **Over 5% after every row:** to the user with the table, for a new waiver or more work.
- A gain past CUDA is kept: it goes towards the new defaults' 3.8 and 5.1%.

`--device cuda` stays until this is read: it is what these rows are measured against.

## Speed and memory

Every row saves or holds time. V4 alone spends CPU, under its bound. V2 drops two devices' fixed memory, so check it after the 16K VRAM measurement, not during it.

## Tests

- **Bytes:** every row, on both GPUs: the same bytes as before, at `--workers 1`, `2` and `auto`, with `--device cpu` agreeing where the tests already compare it.
- **V1:** `test_vk_kernels` on the compute family of both GPUs, copies included (the B50's has no transfer bit).
- **V2:**
  - Three contexts on three threads give the results one context gives, with their own queues and with `CELV2_VK_QUEUES=1`.
  - `free()` in one context while another's batch runs: returns without waiting for it.
  - All engines closed, then a new one opened: it works.
- **V3:** the MVs equal the old order's on a frame with 7 references, and with a reference new to the slots.
- **V4:** a value reached before the call, during the poll, and after it.

## Not in this

The kernels' own speed, more workers, the row order, and retiring CUDA.
