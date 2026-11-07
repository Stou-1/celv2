# Where the AV1 gap comes from: compare, widen CelV2's search, then replay avmenc's decisions (design)

2026-11-04. A measuring design: its output is a ranked list of what to fix, each item then getting its
own spec. Nothing here ships: the switches and the tool are for measurement only.

**Updated 2026-11-05 from stage 1's streams.** New: stage 1b, the mini-GOP's length and the pyramid's
qindex, which stage 2's switches do not touch. It needs no new code and runs beside stage 2.

**Updated 2026-11-06 from stage 1b and W3.** The length is ruled out (best +5.91% on the moving four); W3
gives -2.93% at 4.76x. New: W7, a switch the list left out (zeroing residual by RD), first in the order of
the switches still to build, and G5, the no-tune default's mini-GOP.

**Updated 2026-11-06 from W7.** W7a + b gives -0.80%: a small default candidate, not the cause of the gap. New: W7's
result, one more run (W7f), bytes by true distance in place of `lv.py`'s levels, and L0 first in stage 3.

**Updated 2026-11-06 from W7f and the run without W1 and W4.** Zeroing is finished, W7a + b has its own spec
(2026-11-06-rd-zeroing-design.md), stage 3 is open at L0.

**Updated 2026-11-06 from L0.** The gap is coefficient bytes at the same step. L1-L4 wait for F0, avm0c without its
temporal filter: the replay needs those streams either way.

**Updated 2026-11-06 from F0.** The filter is not the cause on Dragon Maid and Angel Beats: the tool is built on F0's
streams, L1 and L2 first. 128 blocks stay split in the replay.

**Updated 2026-11-06 from L1 and L2.** Decisions are a quarter to a third of the gap, side bytes are closed, coefficients at
the same modes are left. L3 and L4 go ahead, L5 and L6 are added to part the quantizer from the coder. AM's filter is its
own 10%, outside the ladder.

**Updated 2026-11-06 from L6.** The gap at the same modes is the levels in nonzero blocks: not the coder, not zeroing. The
fix has its spec, 2026-11-06-real-cost-trellis-design.md.

## Why

The goal is to beat AV1 everywhere. On 2026-11-02 (docs/reports/2026-11-02-av1-gap.md) CelV2 was +30
to +84% larger than AV1's slowest settings. The ME reach fix narrowed that, but it is still large.
That report concludes the gap is in search and decisions, not tools: every AVM tool CelV2 lacks was
within +-1% on avmenc. Which decisions cost how much is not known.

**From earlier measurements:**
- avmenc with CelV2's tools, cpu-used 0, against CelV2 (33 frames, d656bb6): AM +32.7%, Dragon Maid
  +60.1%, 吉村 -11.8% (CelV2 smaller).
- With CelV2 against SVT-AV1 preset 0 (120 frames: AM +29.6%, Dragon Maid +64.1%, 吉村 +58.2%), that
  puts avmenc cpu-used 0 near SVT on anime (AM about -2%, Dragon Maid about +3%) and far behind on
  吉村 (about +79%).
- This is an estimate only: different frame counts, CelV2 versions and avmenc tool sets.
- If it holds, imitating avmenc explains the anime gap but only reaches AV1 there; live action needs
  CelV2's own search widened.

**Already widened:**
- the RD pass's every intra mode (`--rd-intra`, -0.18%, in `--preset max`);
- the ME reach (3 to 5 pyramid levels: Dragon Maid -10.0%, Angel Beats -5.1%);
- the GOP trial (mixed stretches -2.0 to -2.4%).

**Planned, never done:** `--preset max` stages 2 (decisions by the real coding cost) and 3 (wider
candidates).

## Stage 1: is avmenc a target?

- **Sources:** the eight 1080p sources (`g_air`, `g_md`, `g_am30`, `g_ck`, `g_ab` (720p), `g_yo`,
  `g_cr`, `g_pa` in
  `scratchpad\me\tmp\`,
  a temporary folder: copy them somewhere stable first), the first 33 frames.
- **Settings:** qindex 100-190 / crf 20-56, BD-rate by global-MSE PSNR-Y.
- **Encoders on the same 33 frames:**
  - avmenc cpu-used 0, every AVM tool;
  - avmenc cpu-used 0, CelV2's tools (gap/avmabl.py's base);
  - SVT-AV1 preset 0 and libaom cpu-used 0 (av1bench.py's settings);
  - CelV2, default and `--preset max` (anime `--tune animation`).
- **Time:** avmenc cpu-used 0 takes about 2,500 s for 33 frames on one thread; run each source and
  point as its own process.
- **Gives:** where avmenc is a target (at or ahead of AV1) and where it is not.

## Stage 1b: the mini-GOP's length and the pyramid's qindex

**Why.** Read from gap33's streams, no encoder run (`celv2-bench\gapstreams`: gacct, a
per-frame reader on AVM's inspection build, and its tables). 33 frames, so the key frame weighs more than
in a real video.

- **Where:** the 32 frames after the key frame, in coefficients. At the same PSNR over those frames,
  avmenc cpu-used 0 with CelV2's tools against CelV2: coefficient bytes -36 to -82%, everything else
  -40 to +33%. avmenc's larger key frame flatters this.
- **Search depth, the first suspect (stage 2):** avmenc's cpu-used 6 and 0 code nearly the same qindex per
  frame, and 0 is about -12 to -34% on the whole clip. cpu-used 6 against CelV2 is -0.4 to -16.8% on Dragon
  Maid, AM, Angel Beats, crowd_run and park_joy (piecewise-linear BD-rate; the first range is a ratio
  of two of them, so an approximation).
- **Structure, the second suspect (here):** every other encoder runs a longer pyramid.

  | Encoder | Top of the pyramid every | Anchors in 33 frames | Anchors' share of the bytes (q130, q160) |
  |---|---|---|---|
  | CelV2 | 8 frames, nearly always | 4 (Angel Beats 6, Dragon Maid 3) | 15-58% |
  | avmenc cpu-used 0 | about 16 | 2 | 5-32% |
  | libaom cpu-used 0 (crf 44) | 16-32 | 1-2 (Angel Beats 3) | not counted |
  | SVT-AV1 preset 0 (crf 44) | 32 | 1 (AM 2) | not counted |

  Each CelV2 anchor is predicted from the past only. On crowd_run with the key frames level (35.9 and 35.5
  dB, about 210 KB), CelV2's first anchor takes 255 KB, more than the key frame, and avmenc's 105 KB. SVT-AV1
  does this at 40 s an encode against CelV2's 8 s, so deep search is not all of it.
- **avmenc's large key premium is not needed:** libaom's and SVT-AV1's PSNR per frame is about as flat as
  CelV2's, and they still win.

**Against it.** All measured before the 5-level motion search (2026-11-02), so at +-52 pixels of reach:
- 2026-10-28, 120 frames: 16 frames, live action -13.5% against 8 frames' -9.3%, anime A +11.3% against -3.3%.
- 2026-10-30, 120 frames, 16 frames everywhere: kaguya -9.9%, live action -2.6%, AIR +10.8%, AM +6.8%, Dragon
  Maid +5.1%, the fast Xiph clips +5.4 to +8.8% (chunk_coder.cpp, docs/design/2026-10-30-tunes.md).
- 2026-10-30, mixed AM / Angel Beats stretches: `--bframes 7` -4.44 / -1.86% against the tune's 16-frame choice.

The "16-frame mini-GOP" row in docs/design/tried-not-kept.md waits on per-frame qindex from reference use.
That has held since `--tplq 3` (2026-11-02), and no re-measurement is on record. Nor is any run over 16
frames. Under `--tune animation` a 16-frame mini-GOP's mids are also nearly flat: `--level-step-16 2` gives
0, +2, +4, then +16 at the B frames, where avmenc rises from its anchor by 15, 7, 4, 3.

**The runs: existing options, no new code.** Added to each source's usual settings (anime `--tune
animation`, `--refs 7`). `--bframes 15` is needed since live action has no tune.

| Run | Options | What it tries |
|---|---|---|
| G0 | none | today |
| G1 | `--bframes 15 --gop-still 0 --level-step-16` 2, 4, 8 | 16 frames everywhere, mids as they are |
| G2 | `--bframes 15 --gop-still 0 --mq 2 --level-step-16 6` | mids near avmenc's (it codes them 14, 7, 3 under the B frames: +2, +9, +13 at `--bq 16`), anchor from TPL |
| G3 | G2 with `--tplq 0 --aq -13` | avmenc's anchor too (29 under the B frames) |

- **Sources:** the eight, 120 frames, q100-190. These decide; 33 frames only for comparison with stage 1.
- **Measures:** BD-rate by global-MSE PSNR-Y at 10 bits against G0, and time. At q130 and q160 gacct's
  anchor share and coefficient bytes, beside G0's.
- **Time:** about an hour on one process (five settings' worth of runs, about 30 s an encode), less in
  parallel. Load does not matter for the BD-rates, only for the time column.

**Reading it, on the mean of crowd_run, park_joy, AM and Angel Beats, the moving sources the earlier runs
lost on, at the best of G1-G3:**
- **-3% or better:** the length is a cause. Then G4, 32 frames: `--bframes` past 15 and a sixth level, a
  small measurement build. The result goes to the design session, which writes how to choose the length:
  `--gop-still` looks only at the first and last frames.
- **Within +-1%, or worse:** not the length at these offsets. The gap rests on stage 2.
- **Between:** to the design session, with gacct's tables.

Either way the "16-frame mini-GOP" row is updated, and the result takes its place in the ranked list.
Stage 2 does not wait for this. Stage 3's L0 would answer the same question, at the cost of the tool.

**Speed:** measurement only. If it becomes a feature, its spec carries the bound. 16-frame mini-GOPs were
faster on 2026-10-28 (9.1 against 8.2 fps on anime A).

## Stage 2: CelV2's own search widened (measurement switches)

`CELV2_WIDE=<list>` (environment, not a CLI option, not in the GUI) turns on any of these. Off, the
bytes are unchanged.

| Switch | What it widens |
|---|---|
| W1 partitions | the engine keeps more candidates per block, and the RD pass compares every partition shape |
| W2 MVs | per reference, the best of an integer full search (+-64) and the neighbours' MVs added as candidates |
| W3 real cost | `--preset max` stage 2: the RD pass's best 3 candidates per superblock trial-coded by the real coder, chosen by CDF bits and 10-bit error with chroma |
| W4 chroma | chroma error and bits in the RD pass's decisions |
| W5 transforms | every transform type and split compared by RD, with every intra mode (`--rd-intra`) |
| W6 compound | every pair of references tried |
| W7 zeroing | a transform block's levels weighed against none, the end of block chosen, the weights by level (below) |

- Each switch on its own, then all together, on the eight sources (33 frames), with the time each
  costs.
- On live action (吉村), where avmenc is no target, this is the main tool.
- A switch that gains much becomes a candidate for `--preset max` or the default, each by its own
  spec.
- **Order of those still to build:** W7, W6, W2, then W1, W4, W5. W7 is the smallest and has the most
  direct evidence. W6 next: compound (without skip mode) is 28-31% of the area of avmenc's lower three levels
  and 11-17% of CelV2's (crowd_run q160).

### W7: zeroing residual by RD

**Why.** gap33's streams, frames at the same pyramid level and about the same PSNR (`lv.py`, 33 frames, both at
q130; no coefficient: the area of inter blocks with none; a few percent off, see "W7's result"):

| Source, level | PSNR CelV2 / avm0c | qindex | Bytes | Coefficient bytes | No coefficient |
|---|---|---|---|---|---|
| crowd_run, 8 frames at distance 2 | 35.0 / 35.1 | 138 / 127 | 531 / 346 kB | 381 / 161 kB | 51 / 76% |
| crowd_run, 15 leaves | 34.5 / 34.7 | 146 / 130 | 433 / 296 kB | 243 / 95 kB | 71 / 88% |
| park_joy, distance 2 | 36.0 / 35.9 | 138 / 127 | 528 / 363 kB | 449 / 248 kB | 58 / 73% |
| AM, distance 2 | 42.3 / 42.3 | 138 / 127 | 48 / 33 kB | 32 / 15 kB | 40 / 63% |
| AM, leaves | 42.1 / 42.1 | 146 / 130 | 31 / 24 kB | 17 / 10 kB | 57 / 73% |

avmenc's quantizer is finer by 11-16, and it leaves 15-25 points more area without a coefficient, for about
half the coefficient bytes. In CelV2 the symbols of small levels (`read_coeffs_reverse_2d`) are 42-48% of the
bits on crowd_run and AM at q130, in avmenc 22-28%. **Not proven:** avmenc's references are better, its key
frame about 2.5 dB above CelV2's on crowd_run, and that alone makes residual less needed. W3 points the same way
(its choice is best with the bits at x2).

**What the code does.**
- `quantize_tcq()`: `tcq_prep()` sets the end of block from one coefficient beating level 0, and the
  trellis must then keep a level there (`TcqCands`). So the end of block is never chosen, and a transform
  block with one coefficient over about 0.55 steps cannot become all zero.
- The plain path (chroma, 1D luma, and `--tcq 0`) drops trailing +-1 levels one at a time, and has no whole-block check.
- The coder has no zero candidate either: its transform choices in encoder.cpp compare coded types only, and
  `skip_txfm` is always 0.
- avmenc: each transform block against all zero (tx_search.c:2942), the block against skip (3253, 3526), and
  the lambda up to x1.41 down the pyramid (`rd_layer_depth_factor`, rd.c:654). CelV2's bit weights are one
  constant each for every frame (`kBitWeight` 0.2, `kTcqBitWeight` 0.05), tuned without any of this.

**The switch, in steps, each measured alone.** All in the level choice, so `quantize()` and `quantize_cost()`
agree and the RD pass sees it. On luma and chroma.

| Step | What | Scan |
|---|---|---|
| W7a | After the levels: J of the levels (the cost as now, plus the all-zero flag and the end of block, rough bits) against J of none (`dropped()` over the block, plus the flag). Less wins | the bits of this comparison x1, x2, x4 |
| W7b | The end of block chosen: the trellis also run from up to 3 earlier ends, least J wins. A rerun for the measurement; if it is kept, a path may start at any position | x1 and W7a's best |
| W7c | The bit weights by level, reference frames to leaves: avmenc's factors (1, 1, 1.125, 1.25, 1.25, 1.41 by depth), then double their rise | W7a and b's best |
| W7d | Only if a-c gain 1% or more: a block whose planes are all zero codes `skip_txfm` 1 (writer, contexts, deblocking), its own spec | |

**After them, with the best of W7a-c:**
- `--bq` 8 and 0 (with `--mq`, `--aq` as they are): avmenc's leaf sits at its base qindex, CelV2's at +16. The
  tried-not-kept row (`--bq 8 --mq -8 --aq -8`, +3.8 to +7.1%) had no stronger zeroing.
- G1 step 4 again on the moving four: the "16-frame mini-GOP" row waits on a stronger B frame coder.

**Reading it.**
- Measures: BD-rate as the other switches, time, and at q130 / q160 gacct's no-coefficient area and coefficient
  bytes by true distance (below) beside avm0c's, to see whether the gap in the table closes.
- -1% or better on the mean of the eight: to the ranked list, a candidate for the default, since it adds one
  sum per transform block and takes inverse transforms away (expected; the time column says). Its own spec then
  runs the 120-frame sources and SSIMULACRA2 (`--psy-trellis` exists because dropping small levels can show:
  the user views before a default changes).
- Within +-0.3%: the table's gap is the references, not the zeroing. Row in tried-not-kept.

**W7's result** (33 frames, q100-190, against off; docs/reports/2026-11-04-av1-gap-breakdown.md). Between the two
readings, so it came back to design.

| Run | Mean of the eight | Note |
|---|---|---|
| W7a x1, x2, x4 | -0.37, +2.14, +7.99% | |
| W7b | -0.24% | |
| W7c x1, x2 | +0.11, +0.32% | with W7a: -0.00, +0.42% |
| **W7a + b** | **-0.80%** | Angel Beats -3.44, AM -1.45, chika -1.13, 吉村 +0.21, crowd_run +0.00; 120 frames, the moving four: -1.01% |
| W7a + b, `--bq` 8, 0 | -0.38, +3.02% | chika -3.41, -3.55; AM +2.52, +11.18 |
| W7a + b, G1 step 4 | +4.14% against G0 | the moving four, 120 frames |

**Levels by true distance.** `lv.py` buckets by display order. After a cut the pyramid is shifted, so Angel Beats'
"L1" held its key frame at frame 3 (190 kB), the anchor at 11 (90 kB) and the mids: its "406 against 89 kB" is not a
leaf gap. `truelv2.py` and `truelv3.py` (celv2-bench\gapstreams) take a frame's distance from the coding order. Reports
use them from now on. The rows of the table above move by a few percent; its reading stands.

| q130 | Key + anchors, off / avm0c (kB) | Below the anchors, off / W7a + b / avm0c (kB) | No coefficient there, same order |
|---|---|---|---|
| crowd_run | 1104 / 1177 | 1540 / 1407 / 892 | 59 / 69 / 83% |
| park_joy | 857 / 1081 | 1490 / 1367 / 953 | 61 / 68 / 79% |
| AM | 287 / 249 | 143 / 123 / 79 | 48 / 53 / 70% |
| Angel Beats | 313 / 234 | 241 / 231 / 130 | 35 / 35 / 59% |
| Dragon Maid | 58 / 75 | 161 / 157 / 89 | 99 / 99 / 99% |
| chika | 126 / 144 | 43 / 41 / 22 | 95 / 96 / 99% |
| 吉村 | 231 / 306 | 63 / 59 / 36 | 91 / 93 / 97% |
| AIR | 43 / 107 | 7.3 / 7.1 / 3.7 | 98 / 98 / 100% |

- **Where the gap is:** below the anchors, CelV2 spends 1.6-2.0 times avm0c's bytes at q130 and 1.8-2.3 at q160 (AIR's
  2 kB aside) on every source, with a coarser quantizer (leaves q146 against q130). Key frames and anchors together are
  within about a third either way, AIR aside. Same base qindex, not the same quality: at q130 whole clips are within
  0.4 dB, Angel Beats 0.54 dB in CelV2's favour; at q160 crowd_run is 0.64 dB behind.
- **Zeroing is not the cause.** On crowd_run W7 moves the area from 59 to 69% and takes 8.6% of those bytes, for a
  BD-rate of +0.00%: what it drops was worth about what it cost. Two fifths of the area gap, a fifth of the byte gap, no
  gain. x2 and x4 lose nearly everywhere. So most of the residual is needed, given CelV2's prediction.
- **Decided:**
  - W7a + b is a default candidate, below the 1% bar but wanted: Angel Beats and AM are avmenc's targets. Its time
    (1.3-2.4x) is the reruns, not the idea: its spec does both in one trellis pass, a path starting at any position,
    all zero as one more end. Written: 2026-11-06-rd-zeroing-design.md.
  - W7c off, W7d not built, `--bq` stays 16, each a row in tried-not-kept. chika's -3.4% at `--bq` 8 and 0 goes in the
    report as a lead for its own look: AM's +11% rules out a change for all.
  - The "16-frame mini-GOP" row stands: G1 with W7a + b is +4.14% against G0, W7's own -1.01% included.

**W7f, one more run.** avmenc's leaf is a fine quantizer with few coefficients. The runs so far were not that point: x2
and x4 ran at `--bq` 16 (coarse and zeroed) and on every frame, `--bq` 8 and 0 at x1 (fine, not zeroed).
- W7a's bits x2 and x4 in the frames below the anchors only (key frames and anchors x1), with W7b, at `--bq` 8 and 0:
  four runs, 33 frames, the eight sources.
- 0.5 points or better than W7a + b's -0.80%: the ladder goes into the default candidate's spec, with the user's viewing.
  Otherwise zeroing is finished, and the row says so.

**Result:** `--bq` 8 x2, x4 +1.46, +5.00%; `--bq` 0 +5.10, +9.17%. Zeroing is finished. So avmenc's fine quantizer with
few coefficients is not a point CelV2 can reach by choosing levels: its residual is smaller before quantization, or its
coefficients are worth more. Next: prediction and references, by L0, then L2.

**Without W1 and W4** (33 frames, against off; time under load):

| Run | AIR | DM | AM | chika | AB | 吉村 | cr | pa | Mean | Time |
|---|---|---|---|---|---|---|---|---|---|---|
| W2, 3, 5, 6 | -4.38 | -3.68 | -4.43 | -3.97 | -8.02 | -3.30 | -1.92 | -2.30 | -4.00 | 25.88x |
| + W7a + b | -3.81 | -3.90 | -6.27 | -4.58 | -10.99 | -4.45 | -2.04 | -2.95 | **-4.87** | 37.17x |

- **The ranked list so far:**

  | Fix | Gain | Time | Where |
  |---|---|---|---|
  | W3 | -2.93% | 4.76x | `--preset max` has it |
  | W7a + b | -0.80%, +0.87 points on the row above | to measure in one pass | default candidate, spec written |
  | G5 | 吉村 -5.60% | none | default for no tune; anime without a tune still to measure |
  | W5, W2, W6 on W3 | -0.37, -0.68, -0.15 points; together -1.07 | 2.5x, 5.3x, 1.9x; together 5.4x | no spec yet |

- W2, W5, W6: under the bar one at a time, and W2's gain is nearly all Angel Beats. Their specs wait for L0 and L2: if
  the gap is prediction, W2's search belongs in the GPU candidates, not on the CPU.
- Against the about -27% that Dragon Maid and Angel Beats need, every W together is -3.90 and -10.99%. CelV2's own
  search does not reach avmenc: stage 3 opens.

**W4 loses on every plane** (Y / U / V: AM +23.50 / +39.15 / +33.13, Angel Beats +5.11 / +5.93 / +7.27, crowd_run
+6.25 / +5.70 / +6.56). Bits moved to chroma would gain there, so the decisions themselves are worse. From reading
rd_pass.cpp, not measured: chroma is charged only on blocks of 8 or more on both sides, so an 8x8 pays for it and its
four 4x4s do not, and intra blocks pay with the luma mode and no CfL, where the coder searches its own. Both shift the
comparison rather than inform it. W4 stays out: W3 already costs chroma by the real coder. No run; the two shortcuts go in
the report as the likely cause, and any future chroma in the pass starts by removing them.

### G5: the mini-GOP without a tune

吉村 gained -6.2 to -6.7% in G1 since no tune means 8 frames, and it reaches 16 only under `--tune film`, which
also turns on `--deltaq` and `--grain`. Run `--bframes 15 --gop-still 0.7` alone against G0 on 吉村, crowd_run and
park_joy, 120 frames, q100-190 (about 15 minutes in parallel). If 吉村 keeps most of the gain and the other two
stay within +0.2% (the rule left them at 8 frames on 2026-10-30), it goes in the ranked list as a default for
no tune. No new code.

## Stage 3: replaying avmenc's decisions (only if)

**When:** with every W on, CelV2 still 10% or more larger than avmenc cpu-used 0 on an anime source
where avmenc is level with AV1. The rest is then in decision rules or allocation, not candidates.

**2026-11-06:** Dragon Maid and Angel Beats would need about -27% to fall under that (avm0c -33.66%). The best so far
is W2 + W3: mean -3.61%, Angel Beats -7.85%. So stage 3 opens when the run without W1 and W4 lands, unless it reaches
that. **L0 alone first:** the frame list only, no `ImportEngine`. It says whether the frames below the anchors at
1.6-2.3 times avmenc's bytes come from its frame order, references and qindex. The qindex alone has been tried
(tried-not-kept: `--bq 8 --mq -8 --aq -8`, keyframe offsets; G1's avmenc-like mids), so L0 may well gain little, and it
then points at L2. It also covers Angel Beats' cut: CelV2 codes a key frame at frame 3 (190 kB, a third of the clip),
avmenc none. Read L0, then decide L1-L4 here. Opened 2026-11-06: every W that helps together is -4.87%, Dragon Maid
-3.90, Angel Beats -10.99.

**L0's result** (ad879d3; docs/reports/2026-11-04-av1-gap-breakdown.md). Against avm0c on the same frames: +22.63 to
+79.87%, AIR -0.80%. avmenc's frames cost CelV2 on every source, so frame order, references and qindex do not explain the
gap, unless the temporal filter below changes that.

**Where the bytes go on the same leaves** (q130, the same frames, L0 / avm0c, kB; `l0cmp.py`, shared note
l0-leaves-2026-11-06.txt):

| Source | Coefficients | End of block | All else | Inter area with no coefficient | Intra area |
|---|---|---|---|---|---|
| crowd_run | 627 / 90 | 78 / 19 | 225 / 187 | 47 / 88% | 0.1 / 0.4% |
| AM | 39.5 / 7.9 | 10.2 / 2.4 | 25.1 / 13.4 | 44 / 74% | 0.8 / 1.0% |
| Angel Beats | 113 / 31 | 9.6 / 6.1 | 24.8 / 23.4 | 38 / 48% | 2.1 / 26.6% |
| Dragon Maid | 39.0 / 16.5 | 5.7 / 2.7 | 26.5 / 18.5 | 98 / 75% | 0.1 / 24.3% |

- **The step is the same.** Read from the streams (`gq.c`): avm0c has TCQ on every frame, no quantizer matrices, no DC or
  chroma deltas, delta q on key frames only. So "coarser than its qindex suggests" is not the quantizer.
- **Coefficients and ends of block are 76-98% of the byte gap.** Partitions, modes and MVs cost about the same on
  crowd_run and Angel Beats, 1.4-1.9 times on Dragon Maid and AM.
- **crowd_run's blocks outside skip mode:** 4% of CelV2's area has no coefficient, 77% of avmenc's. avmenc leaves most of
  its residual out and loses only 1.6 dB. W7 shows CelV2 cannot: x2 and x4 lose. So avmenc's prediction must be closer to
  the source where it drops the residual, by this table's bytes and W7's result together.
- **A mechanism to test.** avm0c's key frames and anchors are temporally filtered. A leaf predicted from an unfiltered
  reference carries two frames' noise in its residual, from a filtered one about one frame's. Half the residual energy would
  make zeroing cheap in avmenc and dear in CelV2, and it would look like this table. It fits crowd_run and park_joy better
  than clean anime, so it may not be all of it. Tried-not-kept has temporal filtering twice, in CelV2's own structure.
- **A second lead:** a quarter of the leaf area is intra in avm0c on Dragon Maid and Angel Beats, under 3% in CelV2.

**Decided: L1-L4 are not built yet. F0 first.** CelV2 replays against the unfiltered source, so decisions taken on filtered
frames would not transfer: the replay needs these streams whatever they show.
- **F0:** avm0c's settings plus `--arnr-strength=0 --enable-keyframe-filtering=0`, on Dragon Maid, Angel Beats, AM and
  crowd_run, q100-190. 16 streams, about 3 hours in parallel, in the background beside other work (no timing runs meanwhile).
- Then: their frame lists through `X_L0`, and the same tables (`l0_cls`, `l0cmp.py`).
- **Reading**, on L0 against avmenc, filter off beside filter on, Dragon Maid and Angel Beats first:
  - Under half the gap left: the filter is the main cause. L1-L4 are not built. A spec for filtered references, its
    tried-not-kept rows reopened: their trigger was a structure where more frames use one anchor.
  - Over three quarters left: the gap is decisions inside the frame. Build the tool on F0's streams, L1 and L2 first. L3
    and L4 only if L2 leaves a gap.
  - Between: back here with the tables.

**F0's result** (69cee4f, fca792a): 99% of L0's gap stays on Dragon Maid, 94% on Angel Beats. Over three quarters: the tool
on F0's streams, L1 and L2 first. AM came later: 58% left, "between", read below.

**128 blocks, now that they are 85, 48, 42% of F0's leaf area** (Dragon Maid, Angel Beats, AM, q130). The replay still
splits them, for these reasons:
- **Nothing is lost in the coefficients.** A 128 block's transforms are 64x64 units, and it cannot split them. Four unsplit
  64x64s with its mode and MV have the same prediction and the same transforms.
- **The split always exists.** Three-way and four-way partitions are invalid at 128 (`subsize_lookup`), so no block of that
  class crosses the 64 grid.
- **What it costs is side bytes:** about 12, 3, 6 kB, 9, 1, 2% of the byte gap (rough: three more blocks at the mean side
  bytes of a block, about 1 byte, all 33 frames at q130). `l0cmp` on each level's streams shows the real figure.
- **Size alone is a small part.** Area in blocks of 64 or more, all frames, F0 / L0: 94 / 91, 78 / 70, 83 / 74%.

Also different in a split, and left in L4's remainder: the deblocking of edges inside a 128 block, contexts, and any motion
mode CelV2 cannot copy.

**F64, what 128 blocks are worth, measured apart.** The only number so far is at 16K, and tried-not-kept reopens on a source
with large flat or still areas, which Dragon Maid is.
- **Only if needed:** the split's side bytes measured at q130 are +0.9, +0.5, +1.1 kB, about 0.5, 0.1, 0.3% of F0's bytes,
  a tenth of the estimate above. A split stream is one a 64 cap allows, so F64 can only reach +1% if the split does. `side.py`
  on q100, q160, q190 first: under 1% of F0's bytes at all four points on Dragon Maid and Angel Beats, F64 is not run, and those
  numbers go in tried-not-kept's row.
- F0's settings plus `--max-partition-size=64`, Dragon Maid and Angel Beats, q100-190. Eight streams in the background,
  after F0's last streams and the speed runs of the zeroing spec, beside the tool's work. The streams are not replayed.
- **Reading,** BD-rate of F64 against F0:
  - Under +1% on both: 128 blocks are not a lead. The numbers go in tried-not-kept's row.
  - +1% or more on either: that number is 128 blocks' share of the gap, and it comes back here for a spec. It starts from
    the merge the intra gap spec describes for inter frames, and from why `compose_sb()`'s test came out 30% larger.

**L1 and L2's result** (e545fbd; L2 against F0 +30.16, +53.38, +13.07, Dragon Maid, Angel Beats, AM). L2 leaves a gap, so L3
and L4 are built. At the same modes, at q130: coefficients and ends of block 172.9 / 104.4, 531.0 / 281.2, 472.2 / 273.4 kB
at +1.86, +1.38, +1.06 dB, and blocks with no coefficient at the leaves 66 / 68, 31 / 49, 40 / 74%. So what is left looks like
the quantizer or the coder, and L4's remainder would not tell them apart. Hence two more levels below, and these changes
from L3 on:
- **One base, L2b:** L2 with one tile and `--key-filter 0`. Tiles are measured (`l2t`: -1.66, -1.30, -2.40%), and
  `--tune animation` filters the key frame where F0 does not. L3 and on are read against L2b.
- **L2z:** L2b with `--rd-zero 1`, one flag, no tool. The ladder has run with the option off. W7 lost on CelV2's own modes;
  this shows what the one pass is worth on avmenc's.
- **L4 is read with care:** avmenc's filter settings fit its reconstruction, not CelV2's, so a loss at L4 does not clear the
  filters. L6o below reads them on avmenc's reconstruction.
- **L5 and L6 only if L4 stays over +5% against F0 on any of the three sources.**

**Reading L5 and L6:**
- **L6's check first:** per frame, MSE between L6's reconstruction and F0's under a tenth of F0's against the source, on the
  one-tile run. Otherwise the levels do not belong to the prediction and the reading is void: the report lists what differs,
  and it comes back here.
- **L6 against F0, same reconstruction:** what is left is the coder. 2% or more: a spec for the coder, a gain with no quality
  cost.
- **L4 to L5 to L6:** the quantizer. L5 is which transform blocks are zero, L6 the levels in the others. Back here with a
  table of where L4 and L6 differ on the same block, by frame class, transform size and intra or inter. A gain at L5 reopens
  zeroing, closed on CelV2's own modes.
- **L6o against L6:** the filters.

**AM's filter** (F0 against avm0c +10.15%; L0's gap +31.44 to +18.11, 58% left). "Between", so read here:
- **The ladder stays on F0.** The filter's share stays a separate row of the report, outside the levels. L1 and L2's results
  stand: they compare with F0, so they say nothing on it.
- **No filter spec yet.** Tried-not-kept has it twice, `--tune animation` already filters the key frame, and one source does
  not show which half of avmenc's filter this is. crowd_run's F0 comes first.
- **F1 and F2:** AM, q100-190, F0's settings with only `--arnr-strength=0` (key frame filter on), then only
  `--enable-keyframe-filtering=0` (anchors filtered). Eight streams in the background, after the speed runs. Also L0 on AM
  with `--key-filter 0`, four quick streams, for what CelV2's own key filter gives in avmenc's frames.
- **Reading:**
  - The key frame's half: compare it with CelV2's filter of the 2 frames after it against avmenc's wider one. Back here.
  - The anchors' half: tried-not-kept's trigger was more frames on one anchor, and L0 has avmenc's structure. A spec to
    try filtered anchors on it.
  - Neither alone, or under 3% each: the row stays, no spec.

**F1, F2's result (6033113): the key frame's half gets no spec.** The report's -8.3% read frame by frame from the streams (AM, PSNR-Y; key-frame-2026-11-07.txt in the shared notes):

| q130 | First unit | Frame 0 | Frames 1-32 |
|---|---|---|---|
| F2, key frame not filtered | 160.6 kB | 49.30 dB | 42.19 dB |
| avm0c | 131.2 kB | 43.33 dB | 42.24 dB |
| CelV2, `--key-filter 0` | 73.4 kB | 43.21 dB | 42.13 dB |
| CelV2, `--key-filter 1` | 68.8 kB | 41.94 dB | 42.12 dB |

- **The filter cures an excess CelV2 does not have.** Without it avmenc's frame 0 stands 2.9 to 8.8 dB above the rest over q100-190; with it 0.4 to 1.7 dB. The frames after it do not miss those dB. CelV2's stands -1.1 to +2.6 dB
  above the rest without any filter, so there is little to take off: at q130 it is at avm0c's PSNR for 56% of the bytes. So AM's +10.15 is mostly F0's key frame, not a part of CelV2's gap. The row stays in the report, read this way.
- **-8.3% is a 33-frame number.** Frames 17-32 alone: -2.66%. Repeating them, an extrapolation for one scene: -4.9% at 129 frames, -3.3% at 513.
- **What is left is the first anchor,** -16% in avmenc, -0.8% from CelV2's 2 frames, and it does not last: the second anchor -2.4%. Up to 7 frames lost +1.04%. 12 frames are not run: they need reading ahead past `bframes + 2`, for this one chance.
- **crowd_run's F0:** if F0 against avm0c passes 3%, compare frame 0's height above the rest in F0 and in CelV2. CelV2's within 3 dB at all four points: no spec. Otherwise back here.

**L3 to L6's result** (23e456f). L4 against F0 +23.62, +26.22, +17.15; L5 no gain; L6 against L5 -19.59, -25.91, -13.91; L6
against F0 +0.55, +1.04, +0.33, every frame bit for bit F0's. By the reading: no coder spec, the gap is the quantizer's levels.
- **The split was wrong for L6.** "Same prediction" above does not hold at that level: BAWP's template and the deblocking inside
  a 128 block do not survive a split (L6 split +1.92, +40.75, +38.73). L4 to L6 are read with whole 128 blocks.
- **F64 stays queued:** split side bytes pass 1% at q160 and q190 (up to 2.36%).
- **`--key-filter` is a no-op under `X_L0`,** so AM's L0 with `--key-filter 0` above is the same stream: not run. For the key
  filter's part, L0 needs the filter first.

**One tile as a fix, T1.** `l2t`'s -1.30 to -2.40% is a lead of its own, and the row workers may have made the second tile
unneeded. CelV2 at its defaults, `--tile-columns 0 --tile-rows 0` against the default, the eight sources, q100-190, BD-rate and
time, after the speed runs. 1% or better on average: the ranked list, under the speed rule.

**The tool:**
- `tools/oracle/`, built only with the CMake option `CELV2_ORACLE` (off by default); not in the release.
- It links AVM's inspection build (`_build_inspect`: the decoder with inspection and accounting), the
  decoder only (no `av2/encoder` header).

**Flow:**
1. avmenc cpu-used 0 (CelV2's tools) codes the 33 frames.
2. The tool reads that stream through the inspection decoder:
   - per frame: type, shown or hidden, order, refreshed slots, reference list, qindex, per-superblock
     delta q, filter settings;
   - per block: partition tree; intra mode, angle delta, MRL, CfL type; inter references, MVs,
     interpolation, compound, motion mode; transform split and type.
3. CelV2 codes again from them:
   - `ImportEngine`, a third `DecisionEngine` implementation, returns avmenc's partitions and modes as
     the decision map;
   - the tool drives `Pipeline` directly in avmenc's frame order, references and qindex;
   - a measurement switch stops the coder's own refinements (MV refinement, mode re-scoring), so the
     imported decisions are used as they are.

**Levels** (each a run; the difference between neighbours is that decision's share):

| Level | Taken from avmenc | Shows the loss in |
|---|---|---|
| L0 | frame structure and qindex | GOP and allocation |
| L1 | + partitions | partitioning |
| L2 | + modes and MVs | prediction choices |
| L3 | + transform split and type | transform choices |
| L4 | + filter settings | in-loop filter choices |
| L5 | + which transform blocks are all zero | zeroing, at avmenc's prediction |
| L6 | + avmenc's coefficient levels, CelV2's quantizer bypassed | the levels |
| L6o | L6 with CelV2's own filter settings | the filters, on avmenc's reconstruction |

L6 against avmenc itself: coding (the CDFs' source, primary reference, CDF averaging). What L6 needs to reconstruct avmenc's
frames, IBP, the CfL downsampling filter and the key frames' delta q, comes from the stream where the coder can take it; the
rest is printed with its area.

**Not expressible:**
- 128 blocks: CelV2's decisions are 64x64. They are split into 64x64s, each with the block's mode and MV from L2 on, and
  their number and area are printed. F64 above measures what they are worth.
- Frames whose slot use CelV2 cannot reproduce: coded with CelV2's own structure and counted.

## Report

`docs/reports/<date>-av1-gap-breakdown.md`:

1. Per source, CelV2 (default, `--preset max`), avmenc cpu-used 0 (all tools, CelV2's tools),
   SVT-AV1 preset 0 and libaom cpu-used 0 side by side.
2. G0-G3, G5 (stage 1b): BD-rate, time, anchor share and coefficient bytes. W1-W7: each alone and all
   together, BD-rate and time factor.
3. L0-L6 (if built): the share of each decision.
4. The ranked list of fixes worth 1% or more on average, each with its time factor and whether it
   suits `--preset max` or the default. Each becomes a design spec in the design session.

## Tests

- **W1-W7:**
  - each on: AVM decodes equal to `--recon`;
  - all off: the same bytes as before;
  - `--workers 1` and `2` give the same bytes.
  - W7a: a block with one lone coefficient just over the threshold codes it with the switch off, codes
    none with it on, and AVM decodes equal to `--recon`.
- **The tool** (tests only in a `CELV2_ORACLE` build):
  - **Self-replay:** a CelV2 stream read through the inspection decoder and coded again at L4, and at L6, gives the
    same bytes as the original. This proves the extraction and the injection exactly, and needs no
    avmenc.
  - **avmenc replay:** F0's streams coded again at each level decode equal to `--recon` (`avm_testsrc2.ivf`
    has extended partitions, so it cannot be replayed).
  - The counts and the area of the blocks and frames it could not express are printed, by kind.
- **The product build:** with the option off, `celv2enc` is unchanged; every existing test passes.

## Speed

The switches and the tool are for measurement: their speed does not matter. When a measured gain
becomes a feature, its own spec carries the speed bound and the rule: a slowdown past the bound is
offset by a speed gain elsewhere, else reported to the user for a decision.

## Not in this

Any fix itself (each its own spec, from the report's list); importing AV1 encoders' decisions (a
different syntax).
