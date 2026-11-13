# Speed first: stage A's next steps, then the Vulkan kernels (design)

2026-11-08. The user decided that every candidate in docs/reports/2026-11-08-improvement-candidates.md will be
built, and that speed comes first. This spec orders the speed candidates (C20, C18, C19, C23, C24) under the
speed-budget spec (2026-11-04-speed-budget-design.md), whose rules still hold: stage A changes no bytes; a speed-up
that changes bytes is a coding change with its own BD-rate; stage C runs one shader version and must not be slower
on either GPU.

Why first: the size candidates that come next cost CPU (C1 real-cost trellis +0.5 to +3%, C2 per-unit filters +1
to +2.5%, C5 key-frame W3 +8 to +25% on cut-heavy anime). The speed rule asks each to be offset. This builds the
offsets before they are spent, and the ledger shows the balance.

## Where the time goes (aa3631c, ms a frame, `--paq 0`; docs/design/speed-budget.md)

| Stage | S1 | S2 AIR | 吉村 |
|---|---|---|---|
| transforms | 944 | 1230 | 1293 |
| quantization | 793 | 689 | 664 |
| RD pass, its own work | 607 | 781 | 1414 |

`CELV2_PROFILE` books transforms and quantization to their stage whatever the caller, so the RD pass's inclusive
share is not known. The first step fixes that.

## Step 0: measure, no byte change (about 1 day)

1. **Caller split in `CELV2_PROFILE`**: transforms and quantization booked as "under the RD pass" or "under the
   coder" (`src/core/profile.h`). Printed beside today's lines, off by default as now.
2. **`tools/sampler`** (symbols build, `--workers 1`), 吉村 and AM, 33 frames, q130: self time of `tx_cost`,
   `tcq_prep`, `tcq_levels` / `tcq_cost`, `fwd_dct`, `most_compact`, `Pass::inter_cost`, `Pass::intra_cost`.
3. **Repeat counters** (behind `CELV2_PROFILE`), AM, DM, 吉村, 33 frames, q130:
   - RD pass: `tx_cost` calls by size, and the share repeating an earlier (x, y, tx, ref, mv, ref2, mv2) in the same
     superblock; prediction samples made more than once;
   - coder: `rd_cost_` calls per NEWMV block that repeat an MV already costed in the same block.
4. **SATD miss log** (no byte change): on intra frames' 64x64 / 64x32 / 32x64 blocks, how often the mode the RD pass
   picks falls outside the top 3 and top 4 by `block_mode_satd()` (`src/gpu/intra_common.h:149`). The 7
   intra-gap frames and S1's key frames.
5. **Vulkan timestamps**: `VkQueryPool` around each dispatch in `engine_bench` (prepare, upload, me, subpel, cost,
   finalize). S1, 30 frames, 2 and 7 references, RTX 5090 and Arc Pro B50, with `CELV2_VK_STATS`.

The results go into the ledger and set the order of steps 2-4 (largest share first).

**Done 2026-11-08** (docs/reports/2026-11-08-speed-step0.md). It changed the order:
- step 2 goes first, keyed by the residual;
- step 4 is not built (its SATD miss rate is 23-48%);
- step 5 starts with `me.slang`.

## Step 1: exact residual and intra prediction (C18 (c)), after step 2

- `Pass::intra_cost` (`src/core/rd_pass.cpp:239-256`) calls `intra_pred()` once per sample. Build each block's
  prediction once from its edge arrays, as the coder does, then subtract.
- The residual for blocks inside the padded plane skips the per-sample clamp (AVX2; the clamped path stays for
  blocks at the edge).
- Must reproduce `intra_pred()` and `dc_value()` exactly for every mode, size and edge. `intra_common.h` is shared
  with the shaders: the shaders keep calling the per-sample form.
- Test: a unit test comparing the block form with the per-sample form on every mode and size, with frame edges.

## Step 2: compute each identical trial once (C20), first

**Revised by step 0.** No `inter_cost` trial repeats a block and its MVs; identical residuals repeat in 38% of S1's
transformed samples (73% of Dragon Maid's). The memo is in `tx_cost()`, keyed by size and residual, per superblock
and thread:
- an all-zero residual is answered without the transform;
- another residual is looked up by its hash and checked against the stored residual (no false hits);
- bypassed under `--psy-rd` / `--psy-trellis`.

The cost depends only on the residual, the size and the frame's parameters, so the bytes are the same. The coder
half is dropped: the coder's transforms and quantization are 5-9% of the CPU. The bullets below are the first
draft, kept for the record.

- **RD pass, per superblock:**
  - memoize the luma `Cost` of a 32x32 transform, keyed by (x, y, ref, mv, ref2, mv2);
  - cache predictions per (ref, mv) over the 64x64, filled per 8x8 cell when first read;
  - move the per-square `std::vector`s into per-thread scratch (about 100 KB a thread).
- The memo is off whenever the cost reads more than the key: `--psy-rd`, `--psy-trellis` (block-wide weights),
  W4 chroma, any `CELV2_WIDE` switch.
- **Coder:** the step descent's (`block_writer.cpp:2046`) and AMVD's (`:2101`) trials go through the existing
  `rd` helper (`:2011-2016`), keyed by MV; today they call `rd_cost_` directly. The key covers mode, BAWP, inter-intra and warp state. A memoized value is stored with the bound it
  was computed under and reused only when that bound is at least as loose as the current one.
- `tried` order and ties stay as they are.

## Step 3: the RD pass's other hot loops, exact (C18 (a), (b), (d), (e))

Built in the order step 0's sampler shows. Each part is kept only if S1's CPU a frame drops by more than the noise
(1%) with the same bytes.

- **(a) `tx_cost`**: reuse `tcq_prep`'s per-position error, keep the add chain in its order, skip the zero terms,
  and sum level bits only below eob. Same bytes. The int64 form (exact sums, divided once) changes bytes at ties:
  not here; it would be a coding change of its own.
- **(b) `tcq_prep`**: find eob with integer thresholds; divide only the positions the Viterbi reads; skip the
  two `std::fill` calls when eob is 0 on the no-cost path.
- **(d) TCQ Viterbi vector build**: build each position's candidates four positions at a time, laid out so each
  state vector is one load. Same pairs, same order, same ties. A microbenchmark on blocks dumped from AM and 吉村
  comes first; built only if it shows a gain there.
- **(e) Transforms**: the 64-point DCT computes only outputs 0-31 (about 15% of its operations, not half);
  `most_compact`'s passes with the line index innermost; straight-line AVX2 4- and 8-point kernels. Same
  per-element operation order, no FMA, so every `CELV2_SIMD` level gives the same values.

## Step 4: SATD shortlist for 64-size intra (C19 (a)), a coding change

- In the RD pass's 64-size intra loop (`src/core/rd_pass.cpp:239-291`), rank the 13 modes by `block_mode_satd()`
  and score only the best K with `intra_cost`. K from step 0's miss log (3 or 4).
- The coder re-picks every intra block's mode itself (`block_writer.cpp:554-556`), so only the choice between a
  64-size block and smaller ones is exposed. `kIntra64Ratio` (1.1, `rd_pass.cpp:41`) is retuned with the shortlist.
- Same in the `--rd-intra` loop (on only under `--preset max`).
- Ceiling: `--intra64 0`'s measured cost, 2.8 / 5.4 / 1.1% CPU (S1, S2 AIR, 吉村); expected about half of that.
- **Not this:** cutting the coder's MRL trials to the best 3 modes (it undoes 613f72c; avmenc loses 13.8% on Flash
  without MRL).
- **Kept if** BD-rate is within ±0.1% on the eight sources (below) and S1 / S2 AIR gain.

## Step 5: Vulkan engine kernels (C23, under stage C)

Built only for the kernels step 0's timestamps show as most of the B50's time. One shader version; kept only if
neither GPU is slower (engine time and whole-encode fps, S1 and a16k).

- `kShares` (fixed 4) as an environment sweep, 4 / 8 / 12 / 16. On the RTX 5090 Vulkan's time per reference
  already equals CUDA's (4.52 against 4.74 ms); its gap is a fixed part of about 17 ms a frame, so this is not
  expected to close it.
- `cost.slang`: each distinct (MV, 4x4) Hadamard sum computed once per superblock and reference; the superblock's
  source loaded into groupshared memory once.
- `me.slang`: the 8x8 block in registers, the reference window in groupshared memory.
- No dynamically indexed private arrays (`kSubpelTaps`, `kAngles`, finalize's `rc` / `rmx` / `rmy`): removes the
  scratch memory, and lets Intel run SIMD16.
- Aim: the B50's engine time per reference, and 16K (9.68 s a frame on the B50).

## Step 6: workers (C24), measuring only

- `--workers` 3, 4 and 5 on a source with 12 or more chunks of mixed length (S1's 6 and S2 AIR's 9 are
  multiples of 3 and would not show it), two alternating rounds, quiet machine, RTX 5090 and B50. Same md5 in all.
- The rule for `auto` changes only if 4 or 5 is faster on both GPUs.
- Making `auto` the CLI's default (today 1; the GUI already passes auto): same bytes, about +60% on multi-scene
  input from a plain CLI run. It changes memory use for scripted users, so it goes to the user as a proposal with
  the numbers.

## Not in this

- C17 (the RD pass's scoring on the GPU): its own spec, after this one. The user decided on 2026-11-08 that it
  comes before 10-bit / HDR, and that the RD pass's scoring is rewritten in fixed point (stage B's rule kept):
  CPU and GPU give the same bytes. The rewrite changes the bytes once, checked by BD-rate. Its widths are chosen
  for 10-bit from the start (residuals 4x, squares 16x), and the real-cost trellis's product version (Q1) is
  written in the same fixed point, not ported twice. Both GPUs report `shaderFloat64`; the RTX 5090 does not
  preserve fp64 denormals. Its gate is step 0's caller split: the RD pass's inclusive share of S1's CPU a frame
  at least 40%.
- C21 (two mids at once at `--workers 1`), C22 (`--target-size` by chunks), G3.3 (coarse motion search ahead):
  after this spec, by the ledger.
- C19 (b) (SATD screen on pool MVs and pairs): a logging run only, later. Each pool MV still needs its motion
  compensation, so the saving is smaller than the trial count suggests.

## Measure

- **Every step:**
  - S1-S4 and 吉村: CPU a frame and fps, `--workers auto --refs 7 --qindex 130`, old and new alternating, medians;
  - the ledger's "Saved" row (docs/design/speed-budget.md).
- **Steps 1-3 and 5 (same bytes):**
  - byte-identical output on S1-S4 and 吉村;
  - all `CELV2_SIMD` levels;
  - 8 and 10 bits;
  - `--device cpu`, Vulkan on the RTX 5090, and the B50.
- **Step 4 (bytes change):**
  - BD-rate by global-MSE PSNR-Y on the eight sources (AIR, Dragon Maid, AM demo, chika, Angel Beats, 吉村,
    crowd_run, park_joy), 120 frames, `--refs 7`, q100-190;
  - XPSNR beside it.
- **Speed rule:** none of these may slow the encoder. If a step does slow it on some run, it is not kept unless
  another step in this spec offsets it, and the report shows both. If no offset is found, the user decides.
- **Expected, not measured, not additive:** CPU a frame about -10 to -20% on S1 and S2 AIR for steps 1-4;
  B50 engine time -20 to -45% for step 5. The size work after this spends from that balance.

## Tests

- Step 1: the block form of intra prediction equals the per-sample form (all modes, sizes, edges).
- Step 2: the memo's on and off give the same bytes, including under `--psy-rd`, where it must turn itself off.
- Step 3: each kernel equals the scalar code on dumped blocks, at every `CELV2_SIMD` level.
- Step 5: `test_vk_kernels` and the engine-match tests unchanged.
- Every existing test passes.
