# The RD pass's scoring on the GPU, in fixed point (design)

**Step 1 done in part (2026-11-09): docs/reports/2026-11-09-c17-step1.md.** The quantizers and every cost of the
RD pass are integers (src/core/rd_int.h: J units, 1/512 bits, the TCQ Viterbi and the plain path, --rd-zero, int64
sums saturating at 2^62); BD-rate -0.03% mean, worst +0.04%. The transform stays fwd_dct() (float): AVM's integer
DCT lost 0.1-0.4% on anime even at the coder's scale per frequency, so the GPU port needs a more precise integer DCT
first (tried-not-kept). Step 2 is on hold (the user, 2026-11-09): about 0 fps at 48 threads, over the 60% GPU gate.

**Step 0b done (2026-11-09): docs/reports/2026-11-09-c17-step0b.md.** The CPU's integer path is faster than today's,
not slower: transform 0.60-0.71x (AVM's own int32 butterflies), TCQ 0.76x. The naive GPU kernel takes 16 / 42 ms a
frame on the 5090 (S1 / DM), which projects to 66-69% busy: over the 60% gate until the Viterbi is packed. At 48
threads fps does not follow the RD pass's CPU (a cap near 9.6 fps on S1); at 16 threads it does (+11.9% fps for -15%
CPU). Step 1 should use AVM's own int32 butterflies and shifts, not the 16-bit split below.

**Status: draft, not approved for building (2026-11-09).** A review found eight ways it would change bytes and
fourteen other problems; they are listed under "Review" at the end and must be folded in before step 1. The
review's main point: the gain is not established. Before the byte-changing step 1, a cheap prototype (step 0b)
decides:
- a throwaway GPU kernel scoring dumped S1 and Dragon Maid transforms, timed on the RTX 5090;
- an integer DCT (madd16) against today's `fwd_dct()` on dumped residuals.

The gates are the projected RD kernel time a frame, the projected GPU busy share with three workers under 60%, and
the CPU integer path no slower. Two measurements since the draft weigh against it:
- C24: more workers lost fps, because the engines' one shared queue already limits fps (3 workers 10.03 fps, 4
  9.80, 5 8.64 on a 1200-frame mix).
- CPU savings have given little fps at `--workers auto` (S1: -13.8% CPU for +2.1% fps).

2026-11-09. C17 of docs/reports/2026-11-08-improvement-candidates.md, given its own spec by the speed-first spec
(2026-11-08-speed-first-design.md, "Not in this"). Encoder only, no format change. Stage B of the speed budget
(2026-11-04-speed-budget-design.md): Vulkan, one shader version, a CPU reference with identical results, slow GPUs
on the CPU. The bytes change once, at step 1. From then on the CPU and the GPU give the same bytes.

## Decisions

The user's (2026-11-08):
- The RD pass's scoring is rewritten in fixed point; stage B's integer-or-fixed-point rule is kept. The CPU still
  makes every choice.
- The rewrite changes the bytes once, judged by BD-rate: global-MSE PSNR-Y, the eight sources, 120 frames,
  `--refs 7`, q100-190.
- Widths for 10 bits from the start: residuals 4x, squares 16x.
- C17 comes before 10-bit / HDR.
- The real-cost trellis's product version (Q1, 2026-11-06-real-cost-trellis-design.md) is written in the same
  fixed point.

This spec's:
- **Transform:** AVM's own forward DCT matrices in integers, AVM's total shift split so the first pass fits 16 bits.
- **Units:** one integer cost unit per superblock, 1/(1024 dqv²) of a squared step; bits in 1/512 bit, AVM's cost
  unit. The quantizers search as today, in these integers, with today's tie rules.
- **GPU:** every trial `Pass::block()` scores with a transform (each `inter_cost()` and `intra_cost()` call), per
  superblock row, in the row's batch after `finalize`. Their costs come down beside the row's `SbOut`; the CPU
  runs `block()` and `square()` on them as now.
- **CPU:** the same scoring, with the memo, wherever the GPU does not score. `--screen`'s palette and IntraBC
  trials have no transform: they stay in `block()` on the CPU in every case.
- **Where:** a share of the superblock rows goes to the GPU. At start none or all, by a timing (stage B's
  slow-GPU rule); other shares by environment, for the runs.
- **Order:** the ceiling (no code), the CPU rewrite with its BD-rate, the GPU port bit-exact against it,
  placement. Each step has its stop.

## Why

Measured on S1 (`CELV2_RD_TIMES`, 2026-11-09, after e1782ee), ms a frame:

| Part | ms |
|---|---|
| `inter_cost()` (its `predict()` 196, residuals 31) | 1148 |
| `intra_cost()` | 419 |
| `tx_cost()`, inside both: memo lookup 84, uncached 1004 | 1205 |
| ... uncached: `fwd_dct()` (float Lee) 369, `quantize()` (double TCQ; 50 M of S1's 193 M calls reach the Viterbi) 461, sums 125, the 64-point loss 18 | |

- The RD pass with its transforms and quantization is 67-71% of the CPU a frame on S1, S2 AIR and 吉村 (step 0's
  caller split, docs/reports/2026-11-08-speed-step0.md, and after step 3). C17's gate (40%) passes.
- Which trials run depends on no cost. `square()` scores every block once and `block()` every candidate
  (src/core/rd_pass.cpp:478-606, 610-680); the candidates come from the engine. `residual_cost()`'s early stop
  (rd_pass.cpp:342) changes which parts are computed, not the result: costs only grow as parts are added. So a
  row's trials can all be scored at once, anywhere, and chosen afterwards.
- Integers match by nature. Floats would need every rounding pinned: Vulkan lets division be inexact and fuse into
  FMA unless each operation is marked, and the RTX 5090 flushes fp64 denormals.
- GPU time today: the engine's kernels take about 57 ms a frame on the RTX 5090 and 615 ms on the Arc Pro B50
  (1080p, 7 references, time stamps with a barrier on each side, so above an untimed run's; `engine_bench`
  untimed: 21 ms of kernels in 70.1 ms a frame on the 5090). Three workers share one queue. The B50 is the limit
  already.

## Fixed point

### The transform

- `tx_cost()` (rd_pass.cpp:220-269) transforms with AVM's forward DCT2 matrices `tx_kernel_dct2_size{4..64}[FWD_TXFM]`
  (av2/common/txb_common.c:404-820): columns, then rows, as `fwd_transform()` (src/core/transform.cpp:225-236), in
  the even-odd butterflies of `fwd_txfm_dct2_size*_c` (av2/encoder/hybrid_fwd_txfm.c:167, 241). A 64-point pass
  computes outputs 0-31 only, as AVM's skipped lines (hybrid_fwd_txfm.c:985-998).
- Shifts: the first pass by s1 = log2(h) + bd - 9, the second by s2 = fwd_tx_shift[t][0] + fwd_tx_shift[t][1] - s1
  (av2/common/common_data.h:821), then AVM's sqrt(2) step on 2:1 sizes (`round_shift(x * 5793, 12)`). bd is the RD
  pass's sample depth (8 today: it reads 8-bit planes, src/pipeline.cpp:88-89).
  - The total is AVM's, so the coefficients are on AVM's scale. Only the first pass's rounding differs: s1 is AVM's
    or up to 2 bits more at 8 bits.
  - The bound: every computed row of every matrix has sum |m| <= 64N. Summed from the tables 2026-11-09: at most
    256, 512, 1024, 2048, 4096 for 4-64 points, equal for the DC row and row N/2.
  - So the first pass stays within ±32736 at 10 bits (±32640 at 8). Both passes are 16-bit multiply-adds
    (`_mm256_madd_epi16` on the CPU), and the second's sums stay under 2^27.
- The gain is 2^(3 - txs) for every size the pass uses (txs = `av2_get_tx_scale()`, av2/common/idct.c:761-765). So:
  - `p.scale[]` and its calibration leave the RD pass (pipeline.cpp:204-210);
  - the 64-point loss (`dropped_cost()`, src/core/quant.cpp:525-537) becomes exact:
    max(0, (sum r² << 2 (3 - txs)) - sum c²).
- The coder keeps `fwd_dct()` (src/core/encoder.cpp:1064): this spec is the RD pass's scoring.
- Rejected:
  - a fixed-point Lee: its factors 1/(2 cos) reach 40.7 at 64 points, and its rounding errors grow with them;
  - AVM's own shifts: their first pass needs 17-19 bits at 32 and 64 points, so the second pass would need a
    32-bit multiply per tap.

### Units

A superblock has one qindex (FrameSpec::sb_dq).
- dqv = `av2_ac_quant_QTX(qindex, 0, 0, kQDepth)`: the step with 3 fractional bits (av2/common/quant_common.c:131-160,
  avm_dsp/avm_dsp_common.h:55).
- A transform's step is dqv / 2^(3 + txs) in coefficient units.
- CelV2 sets no DC delta, so the DC step is the AC step: both are `get_q(qindex)`. An assert holds it.

| Name | Definition | Bound at 10 bits |
|---|---|---|
| residual r | int16 | ±1023 |
| coefficient c | int32 | < 2^17 |
| `X2` | `abs(c) << (4 + txs)`: 2 dqv times the coefficient in steps | < 2^24 |
| `rec2` | `(2l - q1) * dqv`, 0 for l = 0: level l's ideal reconstruction (q1 = 1 under TCQ's Q1) | |
| J unit | 1/(1024 dqv²) of a squared step, the same in every transform size | |
| search term | `((X2 - rec2)^2 << 8) + lam * bits(l)` | < 2^57 |
| a transform's distortion | sum of `(c - dq)^2 << (16 + 2 txs)`, dq the decoder's dequantized level | < 2^57 |
| bits | 1/512 bit (`AV2_PROB_COST_SHIFT`, av2/common/cost.h:26), int32 | < 2^25 a transform |
| `lam` of a weight w (steps² a bit) | `llround(2 * w * dqv^2)`, made on the CPU | < 2^35 |
| a trial's J | its distortion + `lam_rd` x its bits, side bits included | < 2^58 |

- `lam_rd` is for `bit_weight` (0.025; `X_RDW`, limited to 0.2 by these bounds), `lam_tcq` for `kTcqBitWeight`
  (0.05) and `lam_plain` for `kBitWeight` (0.2) (src/core/quant.h:33, 37).
- A squared step is 1024 dqv² units in every transform, so a superblock's J values compare as today's
  `d + bit_weight * bits` did (rd_pass.cpp:59-63).
- Viterbi path costs stay below 2^60. The sums in `block()` and `square()` saturate at kInf = 2^62 (today 1e300,
  rd_pass.cpp:35): a block without a choice absorbs what is added to it, as 1e300 did.
- `kIntra64Ratio`'s x1.1 becomes J + J / 10 (rd_pass.cpp:497).
- The pair rule `ref_cost[r] > ref_cost[first] * kPairRatio` becomes `5 * ref_cost[r] > 6 * ref_cost[first]`
  (rd_pass.cpp:555). For 32-bit costs the double product is exactly x·6/5 wherever that is an integer, so this
  changes no byte, and the shader can follow it.

### Tables, made once on the CPU and uploaded

- Level bits: `llround(512 * level_bits_of(l))` (quant.cpp:46-52) for l < 2^19, above every max level (0xffffff / 40).
  0.6, 2.6, 3.6, 4.4 bits become 307, 1331, 1843, 2253.
- End of block: `llround(512 * (2 + 1.5 log2 e))`, e = 1..1024 (quant.cpp:72-78); 512 for none.
- Side bits x512: `mode_bits`, `kPartBits`, `kIntraInInterBits`, `inter_bits` (src/gpu/intra_common.h:32-33,
  src/gpu/inter_common.h:50-52), `kCompoundBits` + ref + ref2 and `block_bits` (rd_pass.cpp:42; `kCompoundBits` moves
  to the shared header). Palette and IntraBC bits (rd_pass.cpp:463-474, 597): `llround(512 *` today's formula`)`,
  CPU only.
- Per qindex 1..255: dqv, `lam_rd`, `lam_tcq`, `lam_plain`, the end-of-block threshold, the max levels.
- `get_scan(t, DCT_DCT)` for the 12 sizes the pass uses (4x8 to 64x64, at most 2:1), and the DCT matrices.

No shader computes a log2 or a weight.

### The quantizers

**TCQ** (`tcq_prep`, `tcq_cands`, `tcq_fresh`, `tcq_levels`: quant.cpp:86-289): the same steps in the same order, in
these integers.
- Candidates: floor(e) = `X2 / (2 dqv)` and floor(e + 0.5) = `(X2 + dqv) / (2 dqv)`, exact integer quotients. The
  CPU may take them by a double division and floor, which is exact for these bounds; the shader divides integers.
- Each candidate's search term is as above with `lam_tcq`. The max levels and the decoder's dequantization are as
  today (quant.cpp:163, 272-287).
- End of block: one past the last scan index where level 1's term under Q0 is at most level 0's, i.e.
  `X2 >= dqv + ceil(lam_tcq * (bits(1) - bits(0)) / (1024 dqv))`. That is one threshold per qindex, with no
  division per coefficient, and it also gates `--rd-zero`'s fresh starts. It is today's 0.55 steps, with `>=`.
- The eight states, their pairs, ties, `--rd-zero`'s starts and its all-zero end are as quant.cpp:181-185, 135-151
  and 249-253 say: the first least in the order compared, a continuing path winning a tie. Integer costs tie more
  often; the rules decide them as before.

**The plain path** (`--tcq 0`, `quantize()`: quant.cpp:460-520) is transliterated the same way with `lam_plain`.
The trailing-1 trim's 0.6 and 1.0 bits become bits(0) and 512, and `zero_wins()` follows likewise.

**The pass's sums** (rd_pass.cpp:239-262):
- distortion: `(c - dq)^2 << (16 + 2 txs)` over every position (dq is 0 past the end), plus the 64-point loss in
  the same units;
- bits: the end of block's, plus the level bits of every position before the end.

These are exact integer sums, so their order does not matter.

**Two forms, one result.**
- The CPU may backtrack the chosen path as today, then sum.
- The reference form and the shader carry each state's distortion and bits beside its search cost, taken over with
  the path it continues. The winning state's sums are then the pass's.
- Same comparisons, same path, same sums; a test holds the two equal.

### --psy-* (CPU only)

- `--psy-trellis`'s term enters the search as an integer weight, rounded once per block.
- `--psy-rd`'s term (src/core/psy.h:20) stays double and is rounded into J units.
- Both default to 0 and lost their own runs (docs/design/tried-not-kept.md). They are not BD-checked here.

## The CPU path

- `src/gpu/rd_common.h` is the scalar reference (`CELV2_HD`, as intra_common.h and inter_common.h), integers only:
  the transform, both quantizers in the carried-sums form, and a trial's sums. The shader is its port, line for line.
- AVX2 forms in transform.cpp and quant.cpp equal it at every `CELV2_SIMD` level:
  - the transform by 16-bit multiply-adds;
  - the Viterbi on int64 lanes, its levels and previous states packed as today.
- The memo stays (rd_pass.cpp:106-218):
  - a Cost is int64 distortion and int32 bits, so it holds no weight;
  - `scale` leaves its key (rd_pass.cpp:119, 203, 213), since the transform's gain is fixed.
- After step 1's A/B (the old build is the A), the RD pass's double scoring goes. `quantize()` and `quantize_cost()`
  stay double for the coder.
- Expected, not measured: about today's speed.
  - The multiply-adds are about the float Lee's count.
  - The quantizer loses its two division loops over every position: `tcq_prep`'s (quant.cpp:91-97) and the
    sums' (rd_pass.cpp:246-251). Only positions the Viterbi reaches take a quotient.

## The GPU path

**What it scores:** every trial `block()` would score with a transform.
- Intra: the engine's mode. All 13 modes on 64-size blocks of intra frames (`--intra64`, rd_pass.cpp:484-497) and
  under `--rd-intra`.
- Inter: the block's own choice, then the pool's (at most 10, in `square()`'s order: rd_pass.cpp:617-623), then the
  pairs with the cheapest reference (at most 6: rd_pass.cpp:551-565).
- A candidate equal by `same` (rd_pass.cpp:521-524) to an earlier one of the block is not scored: `block()` never
  reads it.
- Not on the GPU: palette and IntraBC (no transform; `block()` on the CPU), and `CELV2_WIDE`'s W1, W2, W4, W6.

**Slots.** Each block has fixed slots, so the CPU finds a trial's J without a list.
- Intra frames: 13, by mode.
- Inter frames: 1 for the engine's intra mode (13 by mode under `--rd-intra`), then 18 inter: the block's own, the
  pool's by index, the pairs by reference index.
- 19 slots of 8 bytes a block on the default inter frame: 65 KB a superblock, 33 MB a 1080p frame, 2.1 GB at 16K,
  per engine, in host memory. On the GPU, one row.
- A slot the shader did not score holds -1. Reading one is a bug: it throws.

**In the row's batch.** vk_engine.cpp:260-268 records intra, costs, finalize and the row's SbOut copied down. After
finalize, the RD kernel:
- reads the row's SbOut from `out` on the GPU, with no round trip, and the source and references' level-0 planes
  already there;
- writes the row's slots, copied down beside the SbOut under the same `row_done` value.

Then:
- `collect()` (vk_engine.cpp:279-293) gives each SbDecision a pointer to its slots, valid until the next launch.
- `rd_pass_sb()` reads them where it would call `inter_cost()` or `intra_cost()`.
- `DecisionEngine::launch()` takes the frame's RdPassParams, built in `Pipeline::code()` before the launch
  (today in `collect()`: pipeline.cpp:184-210).
- Per frame the GPU gets a byte per superblock (its qindex) and the qindex table.

**Predictions.**
- Inter: the shader computes `predict()`'s values (rd_pass.cpp:273-330), subpel_px at the moved position with
  each tap clamped.
  - The stored quarter-pel phases hold subpel_px at the position clamped first (src/vk/shaders/common.slang:20-36).
  - That is the same only where the position is inside the padded plane, so the shader reads the phases only there
    and filters from the whole-pel plane elsewhere.
- Intra: `intra_pred()` and `dc_value()` from the source's edges, already in common.slang.

**One shader for both GPUs** (stage C's rules):
- integers only (int16, int32, int64: enabled, src/vk/vk_context.cpp:179-190), no floats;
- no subgroup operations and no required subgroup size (`cost.slang` needs 32: src/vk/vk_kernels.cpp:129-130);
- no dynamically indexed private arrays: the eight states unrolled, tables in groupshared or device memory. On the
  B50, scratch memory breaks a second queue and slows the kernel (tried-not-kept, "Several queues on the Arc Pro
  B50");
- no 64-bit multiply in the Viterbi's inner loop (Intel emulates it): `lam` times a level's bits comes from a table
  for the superblock's qindex;
- a zero residual takes the zero cost without its transform, the memo's first answer (rd_pass.cpp:106-109). No
  other memo on the GPU;
- the mapping of trials to workgroups is free: each trial's J depends only on its inputs. Start from a workgroup per
  (superblock, block): its residual and transforms in groupshared memory, an invocation per transform for the
  quantizer. The plan picks the mapping by time stamps on both GPUs.

## Where it runs

- **Always the CPU:**
  - `--device cpu`;
  - frames with `--psy-rd` or `--psy-trellis` above 0;
  - any `CELV2_WIDE` switch;
  - the oracle's builds;
  - when the GPU's buffers do not fit: a failed allocation keeps the CPU, and the encode never fails (stage B's
    VRAM rule).
- **By rows.** A share s in eighths: row r goes to the GPU when floor((r + 1) s / 8) > floor(r s / 8). The pass's
  threads score the other rows with the memo, as now (pipeline.cpp:219-249). Every share gives the same bytes.
- **auto** (stage B's slow-GPU rule), once per process and GPU:
  - the first inter frame runs at s = 4 and times both sides: the RD dispatches by time stamps, and the CPU's rows
    by their wall time from the engine's row to its last superblock;
  - then s = 8 when the GPU's time a row is under half the CPU's, else 0;
  - the factor is checked by step 3's runs.
- `CELV2_RD_SHARE=0..8` forces a share (runs, tests).

## Order of work

**Step 0, the ceiling, no code.** S1 and S2 AIR on the RTX 5090, ledger's settings, 2 rounds alternating:
- the default against `--rd-pass 0`: fps and CPU a frame;
- the default run's GPU busy share and `CELV2_VK_TIMES`.

Go when `--rd-pass 0` is at least 15% faster on both and the GPU is busy under 60% (stage B's third condition). The
candidates report expects +10 to +35%, and the GPU spends time `--rd-pass 0` does not. Short of either: to the
user with the numbers. C23 (the engine's kernels) would make room first.

**Step 1, the CPU rewrite:** rd_common.h, the AVX2 forms, the pass in J units. Kept if:
- BD-rate: mean +0.10% or better, no source above +0.30%;
- CPU a frame within +1% of today's on S1, S2 AIR, 吉村, S3 and S4;
- the same md5 at every `CELV2_SIMD` level, `--workers 1`, `2` and `auto`, and with `--device cpu` against Vulkan.

If it falls short:
- on BD-rate, the first suspect is the transform's first-pass shift: an A/B with AVM's own shifts, at 8 bits with
  a 32-bit second pass, as a diagnosis only. Then `lam`'s rounding.
- on speed, the speed rule applies (below).

**Step 2, the GPU port, bit-exact.** `rd.slang` and its tables, the row's dispatch and copy, the slots in
`rd_pass_sb()`, `CELV2_RD_SHARE`. No auto yet. Kept when every test below holds on both GPUs. The RD kernel's
time a frame (`CELV2_VK_TIMES`, S1, both GPUs) goes into the report.

**Step 3, placement.** Forced shares 0, 2, 4, 6, 8 on both GPUs: S1, S2 AIR, 吉村, S3, S4 and a16k, with
`CELV2_VK_TIMES` beside.
- Kept if auto gives S1 and S2 AIR at least +5% fps on the RTX 5090, and no run is more than 1% slower than share 0
  on either GPU.
- If a middle share beats both 0 and 8 by over 3% on some run: step 4.
- If neither GPU gains 5%, the GPU code goes, with a row in tried-not-kept (revisit when C23 frees the engine's
  time, or on a faster GPU). Step 1 stays as Q1's base.

**Step 4, only if step 3 asks:** auto picks a share in eighths from the same timing.

## Measure

- **Speed, every step:** the ledger's settings (`--workers auto --refs 7 --qindex 130`), old and new alternating,
  3 rounds, medians (S3, S4, a16k: 1); CPU a frame and fps. The results go into docs/design/speed-budget.md.
- **BD-rate, step 1:** global-MSE PSNR-Y on AIR, Dragon Maid, AM demo, chika, Angel Beats, 吉村, crowd_run and
  park_joy, 120 frames, `--refs 7`, q100-190; XPSNR beside it.
- **Same bytes, steps 2-4:**
  - md5 equal at every share on S1-S4 and 吉村, `--workers 1`, `2` and `auto`;
  - 8- and 10-bit input;
  - both GPUs;
  - `--device cpu` equal to share 0.
- **Speed rule:** step 1 may not slow the CPU path, because the B50 and `--device cpu` run it and the GPU offsets
  nothing there. If it does, an offset on the CPU path comes first; if there is none, the user decides, with the
  BD-rate beside. Steps 2-4 may not slow any run: auto keeps the CPU where the GPU loses.
- **Expected, not measured:**
  - step 1 about neutral in CPU a frame;
  - step 3 on the RTX 5090 at `--workers auto`: fps +10 to +35% (the candidates report, centre +20%), capped near
    1000 / (engine + RD kernel ms) a frame. The engine is about 57 ms timed, 21 ms untimed in `engine_bench`; the
    RD kernel is not known (estimate 20-40 ms a frame on the 5090, ten times that on the B50);
  - 0 on the B50 and at `--workers 1`.

## Tests

- **Transform:**
  - rd_common.h's equals a plain matrix product with the same shifts, at every size;
  - worst blocks (±1023 flat, checkerboard and ramps; ±255 at 8 bits) keep the first pass in int16 and the second
    in int32;
  - the coefficients are within ±2 of `fwd_dct()`'s on dumped residuals (the gain);
  - the AVX2 form equals the reference on dumped residuals at every `CELV2_SIMD` level.
- **Quantizers:** the carried sums equal backtracking and summing on dumped blocks from S1, AM and 吉村 (a million
  at least):
  - TCQ with and without `--rd-zero`, and the plain path;
  - every size, 8 and 10 bits;
  - synthetic coefficients on every threshold and tie.
- **The pass:** test_rd_pass's four cases hold (`p.scale` goes); the memo on and off give the same bytes.
- **The shader** (test_vk_kernels):
  - every slot equals the CPU's on dumped superblocks: S1 inter frames, a key frame, `--rd-intra`, `--tcq 0`,
    `--rd-zero 0`;
  - at 1080p with a partial last row and column, and with MVs reaching past the padded plane;
  - on both GPUs;
  - the B50's pipeline statistics show no scratch.
- **Encodes:** the md5 list above. `--psy-rd`, `--psy-trellis`, `--screen 1` and a `CELV2_WIDE` switch run, and
  keep the CPU.
- Every existing test passes.

## Not in this

- The coder's transform and quantizers: `fwd_dct()`, `quantize()` and `quantize_cost()` stay double.
  - Q1 replaces the coder's last quantization in these units: X2, the J unit, Q9 bits (AVM's cost tables are Q9
    already), the same ties, in the backtracking form, since it writes levels.
  - The Q1 spec gets a line pointing here.
- 10-bit samples in the RD pass: that is the 10-bit / HDR step. The formats hold them; bd enters only s1 and the
  dequantization's clamp. 12 bits is not covered, because `lam` times bits can pass 2^62.
- On the GPU: the choice (`block()` and `square()`), psy, the W switches, and a memo.
- The engine's kernels (C23).
- After this, each by its tried-not-kept row's revisit condition, which this meets:
  - the pass's thread count (pipeline.cpp:254), from "Fewer RD-pass threads";
  - warp trials, from "the parallel RD pass makes many trials cheap".
- AVX-512 (tried-not-kept: the DCT built for AVX-512 was slower).

## Review (2026-11-09, to fold in)

Verdict: build with fixes.

Would change bytes:
- Stale slots. The GPU slot buffer is one row, reused for every row (the same pattern as `costs`, vk_engine.cpp:55-57, 260-268). 'A slot the shader did not score holds -1' is only true if every slot is overwritten each row. If the CPU's `same`/pool/pair skipping (rd_pass.cpp:521-565) ever differs from the shader's, the CPU silently reads the previous row's J instead of throwing. Shares then give different bytes, and nothing catches it.
- The Viterbi's infinity is not specified. Today a missing candidate costs +inf (quant.cpp:186, 199-209): inf+x stays inf, and inf<=inf is true. That matters in `take` for a pair's second next state, which uses _CMP_LE_OQ (quant.cpp:229). Under --rd-zero all states start at inf (quant.cpp:188). An integer sentinel without a clamp grows as sentinel+x, so comparisons among unreachable states come out differently from today. Over up to 1024 positions it can also overflow int64. The spec's only saturation rule is for block()/square() at 2^62. The CPU AVX2 form and the shader must clamp the same way, and the spec does not say how.
- Exactness item (2) misstates the tie rule as 'even level first in a pair'. The code (quant.cpp:183-185, 224-229) keeps the even level in the pair's FIRST next state and the ODD one in the second (`other <= even` under the `second` mask). A shader written from (2) would differ on ties, and ties are much more frequent in integers.
- The pair rule as `5*ref_cost[r] > 6*ref_cost[first]` (rd_pass.cpp:555) is exact only in 64-bit arithmetic. ref_cost is uint32 (decision_map.h:48, values up to UINT32_MAX-1), so 32-bit shader arithmetic overflows. The double-equivalence argument in (7) itself checks out.
- Sizes are missing. The spec lists '12 sizes (4x8 to 64x64)', but after step 1 deletes the double scoring, the integer CPU path must also cover TX_4X4: W4's chroma of 8x8 blocks (rd_pass.cpp:414-428, 450-458; cw = w/2) and the 4x4 that pipeline.cpp:204-210 already scales. The oracle's 4:1/8:1 leaves need it too (rd_pass.cpp:686-724, pipeline.cpp:206-207). Neither the tables, the scans nor test (a) include these, so those paths either break or quietly keep the float. Also, s1 = log2(h)+bd-9 is 3 bits above AVM for TX_16X64 ({2,13}) and 1 bit below for TX_32X8 ({3,10}) (common_data.h:821-840). The bounds still hold, but 'AVM's or up to 2 bits more' is wrong there.
- The transform is not 'AVM's scale with only the first pass's rounding different', and the 64-point loss is not 'exact'. I summed AVM's DCT2 tables (txb_common.c:404ff). sum|m| <= 64N holds, max entry 90. But row energies deviate from 64^2 N by -0.95% to +0.30% per 1-D pass (4-pt row 1: 2*(83^2+35^2) = 16228 against 16384), down to about -1.9% in 2-D, with off-diagonal leakage up to 0.37%. So max(0, G^2*sum r^2 - sum c^2) carries a spurious 0.1-1.9% of the kept energy. Today's dropped_cost (quant.cpp:525-537) uses the float orthonormal DCT (transform.cpp:144-191), which is accurate to rounding. That DCT is also the coder's own transform for DCT_DCT (encoder.cpp:1064, 481, 601). So step 1 moves the RD estimate away from what the coder quantizes, by more than the shift and lam effects that risk 3 lists. Test (a)'s 'within +-2 of fwd_dct()' will fail on real residuals: about 0.5% amplitude on a 1000-level coefficient is about 5.
- --psy-rd still transforms the source with fwd_dct(src, ..., p.scale[t]) and compares it with the residual's coefficients (rd_pass.cpp:263-266). Once p.scale is gone and coeff comes from the integer transform, S must come from the same integer transform. The spec says nothing about this, so the term would mix two scales. This is CPU-only and default off, but it is a result change the spec does not describe.
- The RD kernel reads the GPU's source plane, which may be the one prepare() uploaded from `next` (vk_engine.cpp:154-171, 189-199). The CPU pass reads the `src` handed to code(). Any frame coded from a different picture than the one prepared (key-filter codes `filtered`, chunk_coder.cpp:508) would make share 0 and share 8 differ. Today only the engine's candidates would differ, and identically for CPU and GPU scoring. Low risk, but there is no guard and no test.

Other problems:
- Step 1's 'about today's speed' is unsupported (my estimate, not measured). Second-pass inputs reach +-32736, so the first butterfly level (+-65472) overflows int16. With _mm256_madd_epi16 the second pass therefore has to be a full matrix product: a 32-point row set costs about 512 madd plus about 480 add per 8 rows, against the float Lee's about 289 ops (transform.cpp:38-83), roughly 3.4x. The first pass, which can keep int16 butterflies, is about 1.4x. Overall the transform likely costs 2-2.5x today's 368 ms a frame on S1 (speed-budget.md:64), about +15-25% CPU a frame before the quantizer's division savings (two vdivpd loops, quant.cpp:91-97 and rd_pass.cpp:246-251). The +1% gate likely fails. The text calls the transform 'both passes 16-bit multiply-adds' and 'the even-odd butterflies' at once; for the second pass these contradict each other.
- The RD-kernel estimate does not follow from its own formula: 1567 ms / (40-80) x 1.3 = 25-51 ms, not 20-40. The memo factor of 1.3 is optimistic. Step 0 measured 38.4% of S1's and 72.6% of Dragon Maid's transformed samples as repeats (speed-step0.md), which means 1.62x and 3.65x more transform work without the memo, less only the share of zero residuals, which was never counted.
- fps responds weakly to CPU savings at --workers auto. The memo cut 13.8% of CPU for +2.1% fps; fwd_dct cut 6.5% for +2.6%; S1 ran 28.8 of 48 threads busy (speed-budget.md 'Saved'). --rd-pass 0 is not a clean ceiling either: it also changes the decisions and therefore the coder's work.
- Step 0's GPU gate checks today's busy share (<60%). The RD kernel would add an estimated 25-51 ms a frame on top of 21-61 ms of engine kernels, on one queue shared by three workers. On those estimates the queue reaches roughly 70-100% busy at +20% fps. The gate should use the projected busy share, not today's.
- Putting the RD dispatch into the row's batch, under the same row_done, makes CPU-scored rows and the other two workers' engine rows wait behind it. Every dispatch and copy ends in an ALL_COMMANDS barrier (vk_context.cpp:532-550, 639-647). This works against stage B's rule that 'the GPU work overlaps the CPU's, so nothing waits on it' (2026-11-04-speed-budget-design.md, Rules).
- The auto rule compares one engine's timestamped kernel time a row with CPU wall time a row. It ignores that three engines share the queue, and the timed run's barriers perturb what it measures. Bytes are unaffected, but the rule can pick s=8 and make the 5090 the bottleneck.
- The no-scratch rule conflicts with the reuse of common.slang. Its intra_pred() and subpel_px() index kAngles and kSubpelTaps dynamically (common.slang:58-62, 212, 226). The speed-first spec's step 5 names exactly these as B50 scratch. Either step 5 becomes a prerequisite or the RD kernel needs its own forms; otherwise the zero-scratch test fails. Eight states with three carried int64-or-int32 values also risk register spills, which are scratch too on Intel.
- 'No 64-bit multiply in the Viterbi's inner loop' is not reachable as written. Each candidate's ((X2-rec2)^2 << 8) needs a product of up to 2^56; only lam x bits is tabled. Tabling lam x bits(l) for every l < 2^19 per qindex would be about 1 GB. The table has to cover small levels only, with a fallback multiply.
- '--screen 1 ... keep the CPU' in the tests contradicts 'Where it runs', which sends --screen frames' transform trials to the GPU. --screen auto is the default (aa3631c in the ledger). Mixed GPU-J and CPU palette/IBC-J comparisons (rd_pass.cpp:578-604) then have no share-invariance test.
- The zero-residual 'memo's first answer (rd_pass.cpp:106-109)' does not exist: tx_cost() hashes every residual (rd_pass.cpp:188-218). It was only proposed in speed-step0.md. Adding it is still exact (a pure function), but the citation is wrong.
- The 57 ms (5090) and 615 ms (B50) engine times are not in the cited report, which gives 61 and 674 ms at refs 7 (speed-step0.md).
- Two of the 18 inter slots are always empty. Every block is in its own pool (rd_pass.cpp:619, skipped at 526), and `first` never pairs with itself (555). 17 slots would do.
- The md5 matrix is about 300 full encodes (5 shares x 5 sources x 3 worker settings x 2 depths x 2 GPUs), including S4 and the B50 at forced shares: many hours. Share invariance can mostly be shown on 33-frame clips plus slot dumps.
- Checked and correct: sum|m| <= 64N with maximum entry 90; the int16 first pass (+-32736 at 10 bits) and the second pass under 2^27; DC step equal to AC step (quant_common.c:131-160, get_q); the phase planes against predict() at positions outside the plane (common.slang:20-36 against rd_pass.cpp:283-307); the early stop (rd_pass.cpp:342); the dedup; and the kPairRatio double equivalence. The cited line numbers check out.

Fixes asked for:
- Add a step 0b before step 1. Prototype the integer transform plus TCQ as a throwaway GPU kernel on dumped S1 and Dragon Maid transforms and time it on the 5090. Separately, microbenchmark a madd16 integer DCT against fwd_dct() on dumped residuals. Gate step 1 on the projected RD-kernel time a frame, the projected GPU busy share with three workers under 60%, and the CPU integer path being no slower once the quantizer's savings are counted. Step 1 is the expensive, byte-changing step, and it is only worth doing if the GPU gain exists.
- Specify the second pass as a full-matrix madd16 product (or int32 butterflies) and budget its cost. Name a CPU-side offset before step 1 rather than after it fails.
- Remove the 64-point bias. Either weight each kept coefficient's square by an integer per-frequency table (Q15 of 64^2 N / ||row||^2 for u and v), or compute the dropped outputs' energy directly. Drop the word 'exact'. Replace test (a)'s +-2 with an exact match against a plain integer matrix product, and report the drift from fwd_dct() as a statistic.
- Define the Viterbi sentinel: kInfV = 2^61, missing candidates cost kInfV, and every state's cost is clamped with min(cost, kInfV) after every position, the same on CPU and GPU. Equal sentinels then tie the way inf did (LE true).
- Before each row's RD dispatch, fill the row's slot range with -1 (vkCmdFillBuffer), or have the shader write every slot. Add a test that corrupts one slot and expects the throw.
- Do the pair rule in 64-bit, in rd_common.h and in the shader.
- Replace exactness (2) with quant.cpp:183-185's wording (lower state first; even level wins in the pair's first next state, odd in the second), and add synthetic tie cases for both next states.
- Cover TX_4X4 and the oracle's 4:1/8:1 sizes in rd_common.h, the scan tables and test (a), or state explicitly that W4 and the oracle keep a double path.
- Under --psy-rd, take S from the same integer transform as coeff.
- Make C23 step 5's removal of kAngles and kSubpelTaps a prerequisite, or give the RD kernel switch or groupshared forms. Check spills as well as private arrays in the B50 pipeline statistics.
- Table lam x bits only for small levels (for example l < 64) in groupshared memory per superblock qindex, with a 64-bit fallback above that. Say so in the spec.
- Give the slots their own timeline value, or record the RD dispatches after the engine's rows, so CPU-scored rows and the other workers are not delayed.
- Fix the --screen test line, and add share-invariance md5 on a screen source (Flash, --screen 1).
- Correct the estimates: 25-51 ms on the 5090 by the spec's own formula; memo factor 1.6x on S1 and 3.7x on Dragon Maid by samples, less zeros. Cite 61/674 ms or name where 57/615 come from. Correct the zero-residual citation.
- Assert in launch() that the prepared source's key and dimensions match the src that code() passes to collect(), and include a --key-filter run in the share md5 tests.
- Run the full share x workers x depth md5 matrix on 33-frame clips, and full length only for share 0 against 8 on each GPU.

The reviewer's gain assessment (estimates): Measured inputs, from the docs and the brief: the RD pass inclusive is 66.8/71.2/70.9% of CPU a frame (S1/AIR/吉村, after C20 and C18c). The 5090's engine kernels take 61 ms a frame timed at refs 7, 21 ms untimed in engine_bench; the B50's take 674 ms. CPU savings have turned into little fps on S1: -13.8% CPU gave +2.1% fps, -6.5% gave +2.6%, and only 28.8 of 48 threads were busy.  My estimates, not measured: - Step 1 is likely not neutral. Because the second pass needs a full-matrix madd16 product, the integer DCT should cost about 2-2.5x the float Lee, roughly +15-25% CPU a frame on S1 before the quantizer saves maybe 3-7%. So step 1 probably fails its +1% gate unless an offset is found. The B50 and --device cpu would carry that slowdown with nothing to offset it. - The RD kernel is 25-51 ms a frame on the 5090 by the spec's own formula, and more on Dragon Maid-like flat content without the memo (up to about 3.7x the transform work). - With three workers on one queue, the 5090 likely becomes the limit near +10-20% fps. The +10 to +35% range is plausible only at its low end, and only if step 0's --rd-pass 0 ceiling shows the CPU is the limit; that ceiling also mixes in the changed decisions' coder work. - B50 and --workers 1: 0. 16K: a gain is possible but smaller than stated, since the RD kernel scales with pixels (about 64x 1080p, so 1.6-3.2 s a frame against 0.97 s of engine time), and it needs 2.1 GB of host-visible memory per engine.  Overall the design can be made byte-exact with the fixes above. Its gains are not established, and a cheap GPU and CPU prototype (step 0b) should decide before the byte-changing step 1.
