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

## Files

Streams, speed_vk.txt and the throwaway patches (dm_hash_patch.py, mh_patch.py, w_clip.py):
`celv2-bench\vk_diff`.
