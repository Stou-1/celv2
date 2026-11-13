# Vulkan as the default GPU (2026-11-05): CUDA's step 1 and 2 done

Spec: docs/superpowers/specs/2026-11-04-retire-cuda-design.md.

- Vulkan on the RTX 5090 gives CUDA's bytes on every run, 0.7-4.8% slower in whole-encode fps. The user waived the 5%
  gate, seeing this (2026-11-05).
- `--device auto` and `gpu` take Vulkan (97fbf2b). `--device cuda` stays until the release that follows.
- Checking the Arc Pro B50 found a bug that is not the B50's: the bytes depended on which worker coded which chunk: MHCCP
  read stale luma past the frame's edge. Fixed, with a test.
- Step 3 done: no test needs CUDA but `cli_device_cuda_matches_gpu`, which skips without it. Step 4 (removing CUDA) waits
  on a release.

## Speed

`--workers auto --refs 7 --qindex 130`, the speed ledger's runs, celv2_h at 1ced58a, one run each (S2-AIR three), under
load (eight avmenc on 48 threads), so the fps are below the ledger's baseline. Decision is `CELV2_PROFILE`'s CPU-side stage,
not the GPU's time.

| Run | CUDA fps | Vulkan fps | Difference | B50 fps | Decision s, CUDA / Vulkan / B50 |
|---|---|---|---|---|---|
| a16k (9 frames) | 0.136 | 0.135 | -0.7% | 0.044 | 9.95 / 7.22 / 13.36 |
| S1 | 9.51 | 9.19 | -3.3% | 1.40 | 20.38 / 5.64 / 8.02 |
| S2 AIR | 9.34 | 8.88 | -4.8% | 1.45 | 19.12 / 5.63 / 8.01 |
| S2 Dragon Maid | 7.69 | 7.49 | -2.6% | 1.28 | 21.59 / 5.44 / 9.48 |
| S3 | 2.08 | 2.06 | -1.3% | 0.43 | 12.12 / 4.07 / 4.75 |
| S4 | 2.14 | 2.12 | -1.1% | 0.79 | 31.95 / 10.20 / 14.16 |

S2 AIR is the mean of its three runs (Vulkan -5.0, -5.1, -4.5%). The gap is largest at 1080p, where the GPU's share is
largest, and smallest at 16K, where the CPU limits the encode. The B50 is 3-7x slower than the RTX 5090.

Not measured here: the engines alone (engine_bench), a quiet machine, and Vulkan's peak VRAM at 16K on both cards.

## Tests with Vulkan as the GPU

A throwaway build took `gpu`, `auto` and no `--device` to Vulkan. 33 tests: all pass but two, both understood:
- `cli_device_gpu_without_gpu_fails`: it hid CUDA, not Vulkan. It now uses `CELV2_NO_VULKAN`.
- `gui_tests`: fails under `-j 4` and passes alone, as before this work.

**Not covered, against the spec's "every test":** test_gpu_me, test_gpu_intra, test_keyframe_gpu and test_ippp call
`GpuEngine::decide_inter` / `decide_intra` / `intra_costs` directly, which VulkanEngine does not have. They stayed on CUDA.
Moved in step 3 through DecisionEngine::decide_intra / decide_inter. Their three CUDA-only cases went (the SATD and
probe hooks; the intra costs now in test_vk_kernels, against the CPU rule). test_vk_kernels no longer checks inter costs
and finalize kernel by kernel: the whole decisions against the CPU engine cover them, less directly.

## The bytes that depended on the worker

The B50 differed from the RTX 5090 on one temporal unit of S1 (unit 523, 294 bytes, same length) and on S4, the same every
run. Found in order:
1. The engines agree: a hash of every superblock's tree, blocks and candidates matched on all 306,000 superblocks of S1.
2. On the RTX 5090 alone, `--workers 1` gives other bytes than 2 or 4. So the output depended on which ChunkCoder took
   which chunk, and the B50's slowness only changed that assignment.
3. Clip of source frames 170-260 shows it: the keyframe that starts the second chunk differs with the same decisions.
4. Clearing the new frame buffer removes it, to 0 or to 0xFFFF alike. The visible luma is the part that matters, not the
   chroma, not the borders.
5. `--mhccp 0` removes it too. Off and still differing: CDEF, CCSO, restoration, CfL, `--cfl-rd`, the transform
   partition, `--rd-intra`, the RD pass, MRLS, deblock search, TCQ.

**Found (the design answers' check, docs/reports/2026-11-05-design-answers.md):**
- AVM's decode equals `--recon` on the clip and on S1, before and after be63bcf, 1 and 2 workers. No mismatch.
- Random luma in new frames gives the same bytes as 0: unwritten frame luma is not what is read.
- The two streams differ first at a block past the frame's bottom (16x32 at y = 1056), and only in its MHCCP prediction.
- `mhccp_implicit_fetch_neighbor_luma()` stops at the frame's edge, but the prediction reads the transform block's whole
  size from `cfl.mhccp_ref_buf_q3`: past the edge, what earlier blocks left. Those rows are not shown, so the decoder agrees;
  their residual is coded, so the bytes depended on the blocks before.
- Why no constant showed it: a value unlike the picture makes MHCCP lose those blocks, whatever the value. Only a plausible one
  lets it win: 2900, 3040 and 3200 there gave three different streams.

**Fix:** the edge repeated into that area before the prediction (`TileCoder::mhccp_fill_outside()`). The frame clear of be63bcf
is gone. `cli_mhccp_past_the_edge` (synthetic, `CELV2_TEST_MHCCP_STALE`) fails without the fill and passes with it.

The same bytes on:

| Run | Device, workers | Bytes |
|---|---|---|
| S1 | RTX 5090 Vulkan auto, B50 auto, RTX 5090 1 and 2 workers | 4,370,064 |
| S4 | RTX 5090 Vulkan auto, B50 auto | 4,361,490 |

Against before any of this: S1 -913 bytes (-0.02%), S4 -19,651 (-0.45%), since those rows now predict well.

## Waiting, V0 (2026-11-08)

The vulkan-wait spec's V0: the wall-time profile lines of 5b850b8, on a clean Release build (c623dea plus engine_bench's
CUDA). RTX 5090, `--refs 7 --qindex 130 --tune animation`, 2 rounds alternating, medians. Same size on every run.

| Run | fps CUDA / Vulkan | Vulkan against CUDA | launch() ms | launch() end to row 0, ms |
|---|---|---|---|---|
| S1, auto | 10.67 / 9.88 | -7.4% | 1.88 / 1.18 | 34.6 / 49.6 |
| S1, 1 worker | 6.38 / 6.22 | -2.5% | 0.84 / 0.79 | 25.3 / 24.1 |
| S2 AIR, auto | 9.94 / 9.44 | -5.0% | 1.85 / 1.30 | 29.5 / 38.0 |
| S2 AIR, 1 worker | 5.58 / 5.80 | +3.9% | 0.88 / 0.79 | 20.1 / 21.3 |

`engine_bench`, S1, 30 frames, one engine alone, ms a frame: 2 references CUDA 15.7, Vulkan 5090 32.9, B50 180.1;
7 references CUDA 39.4, Vulkan 5090 55.5, B50 472.1.

Read by the spec:
- **At one worker the first row is as early on Vulkan** (24.1 against 25.3, 21.3 against 20.1 ms), and the fps gap is
  -2.5 and +3.9%, inside the rounds' spread. At `--workers auto` Vulkan's first row comes 15 and 8.5 ms later. The gap is
  between engines: V1, V2, V3, then V4.
- `launch()` is shorter on Vulkan: no row groups.
- One engine alone is slower on Vulkan over the whole frame (+16 ms at both 2 and 7 references), but at one worker the
  CPU hides it. V3 is the row for that.

## V1, a compute queue (2026-11-08)

Same exe, `CELV2_VK_FAMILY=0` (the graphics family) against the compute-only family, `--workers auto`, 3 rounds alternating:

| Run | graphics fps | compute fps | change | first row, ms |
|---|---|---|---|---|
| S1 | 9.92 | 9.91 | -0.1% | 42.7 / 47.6 |
| S2 AIR | 9.40 | 9.47 | +0.7% | 37.4 / 40.8 |

The same bytes. Not +1% on both, so V1 is not kept on its own. **The code stays, against the spec's rule:** it costs
nothing measurable, and V2 takes queues from the family: the B50's graphics family has 1 queue, its compute family 4
(the 5090: 16 and 8). V2 is run on both families; if the compute family adds nothing there, the default goes back to
the first family with compute.

## V2, one device per GPU (2026-11-08)

V1's exe against V2's, and V2 on the graphics family (`CELV2_VK_FAMILY=0`), `--workers auto`, 3 rounds alternating:

| Run | V1 fps | V2 fps | V2, graphics family | first row, ms (V1 / V2) |
|---|---|---|---|---|
| S1 | 9.66 | 10.32 (+6.8%) | 10.01 (+3.6%) | 42.8 / 28.6 |
| S2 AIR | 9.39 | 9.54 (+1.6%) | 9.32 (-0.8%) | 37.3 / 26.1 |

The same bytes. **Kept.** The first row now comes earlier than on CUDA (34.6 and 29.5 ms in V0). The compute family is
worth 3 and 2 points on top, so V1's default stays. Against V0's CUDA medians V2 is -3.3% (S1) and -4.0% (S2 AIR), on
runs of different hours: V3 next, then one CUDA run beside the result. S2 Dragon Maid, S3 and the B50 are checked once
for V2 and V3 together.

## V3, the motion search by level, and one queue per GPU (2026-11-08)

**The B50 and V2's queues.** With each Context on its own queue of the compute family, the Arc Pro B50
returned bad motion vectors and celv2enc crashed in `rd_pass predict()` once several chunks ran (600 frames,
3 workers); its display blanked at each crash, with no TDR in the event log. On one queue it runs clean.
Every Context now takes the GPU's first queue; `CELV2_VK_QUEUES=many` keeps V2's spread for runs. Deviation
from the spec, which gave each Context its own queue: the B50 cannot run that. `three_engines_match_cpu`
crashed on the B50 before the fix and passes after. Why the B50 fails is investigated apart
(branch investigate/b50-queues).

V2's exe against V3 on one queue, V3 on V2's queues, and CUDA, `--workers auto`, 3 rounds alternating:

| Run | V2 | V3, one queue | V3, many queues | CUDA |
|---|---|---|---|---|
| S1 | 10.45 | 10.47 (+0.2%) | 10.50 (+0.5%) | 10.51 (+0.6%) |
| S2 AIR | 9.56 | 9.70 (+1.5%) | 9.78 (+2.3%) | 10.00 (+4.6%) |

The same bytes everywhere. V3 alone (many queues against V2) gives +0.5 and +2.3%; the first run gave
+1.7 and -1.0%. Neither run reached +1% on both, so V3 was first dropped (bfc1ccb). One queue costs 0.3 and
0.8% on the 5090, within the spread, and is kept for the B50.

V1's exe against V3 on one queue: S2 Dragon Maid 7.95 -> 8.37 fps (+5.2%), S3 2.16 -> 2.19 (+1.4%), the B50 on
S1 1.47 -> 2.34 (+59.4%, the same bytes, no crash).

**V3 kept after all, against the spec's rule.** The V4 run showed the B50 at 1.71 fps without V3. In one run,
3 rounds alternating, on the B50 on S1: without V3 1.72 fps, with V3 2.34 (+36.2%), the same bytes. On the 5090 V3
never cost more than 1% in any of its runs. The rule measures the 5090 only and checks the B50 for losses; a
gain this size on the second GPU outweighs a 5090 gain under +1%. Of the B50's +59% against V1, V2 with one queue
gives +16.5% and V3 the rest.

Gap to CUDA after V1, V2 and one queue: 0.4% on S1, 3.1% on S2 AIR. Over 2% on AIR: V4 next.

## V4, spin, then sleep (2026-11-08)

`wait()` polled `vkGetSemaphoreCounterValue` for up to 200 us, then called `vkWaitSemaphores`. Against V2 with one
queue (without V3: V3 was out when V4 was built), 3 rounds alternating:

| Run | V2, one queue | V4 | CUDA |
|---|---|---|---|
| S1 | 10.22 | 10.30 (+0.8%) | 10.83 (+6.0%) |
| S2 AIR | 9.65 | 9.41 (-2.5%) | 9.94 (+3.0%) |

S2 Dragon Maid -0.7%, S3 +0.2%, the B50 0% (1.71 both). The same bytes. CPU a frame (cores / fps) within V4's +1%
bound: 2.97 -> 2.96 on S1, 3.35 -> 3.37 on AIR. **V4 not kept** (reverted; the spin_then_block test goes with it).

## Reading of the wait spec (2026-11-08)

Kept: V1, V2 with one queue per GPU, V3. The gap to CUDA with them is 0.4% on S1 and 3.1% on S2 AIR (the V3 run,
CUDA beside it). CUDA's own speed moves between runs of different hours (S1: 10.51 and 10.83 fps), so a gap is
read only beside a CUDA run of the same hour. **Between 2 and 5% after every row:** the kept rows stand and the
gap goes to the user, as the spec says; the 2026-11-05 waiver (5%) covers it.

**The user accepted the gap on 2026-11-08** and asked for the B50 branch to be merged and CUDA retired.

## The B50 branch merged, CUDA removed (2026-11-08): step 4

**Merge (e392769).** investigate/b50-queues came in clean: `idle()` no longer throws after a lost device, and a
host barrier ends every batch (`Context::to_host`: the Vulkan memory model needs it before the host reads what
the GPU wrote; a timeline wait alone does not make it visible). test_cli, test_vk_kernels and test_gpu_me pass.

**Step 4 of the retire-CUDA spec.** `src/gpu/engine.cu`, `kernels.cu`, `kernels.h` and `engine.h` (GpuEngine,
probe_gpu) are gone; `--device cuda` with them (an error naming the values left). `make_engine` loses its probe,
`ProbeResult` goes, `intra_common.h` loses its `__CUDACC__` branch, CMake the CUDA language and architectures,
engine_bench its CUDA row, test_cli its `--device cuda` case, the release notice its CUDA Runtime section.
Comments that named the removed files say so. The bytes are unchanged (the same MD5s as before on every run below).
All 33 tests pass (ctest, build_dev). A clean configure and Release build in a new directory, with `CUDA_PATH`
unset and every CUDA directory off the PATH (no nvcc), succeeds, and its `--list-devices` shows the RTX 5090 and
the B50 (Vulkan) and the CPU.

Speed, `--workers auto`, rounds alternating, the same bytes on every run:

| Run | v3q (84617c4) | merge (e392769) | merge, no CUDA | no CUDA, barrier on row batches only |
|---|---|---|---|---|
| S1, 5090, 3 rounds | 10.64 | 10.23 (-3.8%) | | |
| S2 AIR, 5090, 3 rounds | 9.72 | 9.71 (-0.1%) | | |
| S1, B50, 3 rounds | 2.35 | 2.32 (-1.5%) | | |
| S1, B50, 3 rounds, again | 2.35 | | 2.35 (0.0%) | 2.35 (+0.1%) |
| S1, 5090, 5 rounds | 10.62 | | 10.70 (+0.7%) | 10.52 (-0.9%) |

The first run's -3.8% was one round's swing (rounds 10.17 to 10.65); five rounds put the merged, CUDA-free build
at +0.7%. The last column (the host barrier only in the batches whose results the host reads, from the transfer
stage) saves nothing: not kept, the barrier stays on every batch.

## Files

Streams, speed_vk.txt and the throwaway patches (dm_hash_patch.py, mh_patch.py, w_clip.py):
`celv2-bench\vk_diff`.
