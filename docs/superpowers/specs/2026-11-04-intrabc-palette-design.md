# `--screen 0|1|auto`: IntraBC and palette for screen content and halftone (design)

2026-11-04. Staged: A (CPU, exact matches and few-colour blocks) first; B1 (CPU, approximate copies along
a halftone's period) only if A leaves the halftone scenes short; B2 (GPU) only if B1 does too.

**Updated 2026-11-05 from docs/reports/2026-11-05-screen.md.** Stage A and B1 are built, in the coder.
New here: the gate as built, B1 in the RD pass, and the conditions for `auto` as the default. Palettes over 8
values and near-match IntraBC for captures like Flash are not decided yet.

## Why

- Flash (Cirno, 1440x1080, avmenc cpu-used 6, docs/reports/2026-11-03-flash-cirno.md): palette +
  IntraBC -13.4% over 30 frames; over a stretch with scene changes palette alone -12 to -16%, IntraBC
  alone -22%, both -22 to -30%.
- AM demo (a visual novel full of UI and text): leave-one-out IntraBC +1.74%, palette +0.41% when off.
- Regular anime: within +-0.3% (docs/design/tried-not-kept.md, 2026-10-30): hence a per-frame gate.
- Dragon Maid's halftone scenes trail x264 by 3-7 dB (docs/superpowers/specs/2026-11-04-halftone-analysis-design.md);
  repeating dots are what IntraBC copies.

## AV2's tools (AVM v1.0.0 decoder)

- **IntraBC:** a block copies an already coded area of the same frame by an integer block vector.
  - *Local* (`allow_local_intrabc`): the current superblock and its neighbourhood
    (`av2_is_dv_in_local_range()`); allowed **in inter frames too**.
  - *Global* (`allow_global_intrabc`): anywhere already coded, under a delay rule; key and intra-only
    frames only; with the in-loop filters on it raises the decoder's memory (level.c).
  - Blocks below 64x64 (64x64 and larger excluded), luma tree (not a chroma-only partition).
  - Frame header: `allow_intrabc`, then on key / intra-only frames `allow_global_intrabc` and (if
    global) `allow_local_intrabc`, then the frame's max BVP DRL bits when the sequence allows it.
- **Palette:** luma only in AV2, blocks 8x8 to 64x64, under the frame's `allow_screen_content_tools`.
- Sequence header: `force_screen_content_tools` 0 / 1 / 2 (2: each frame chooses), `force_integer_mv`.

Today CelV2 writes `force_screen_content_tools = 0` (seq_header.cpp), `allow_intrabc = 0` (codec_state.cpp)
and throws on a palette-capable block if the tools were on (block_writer.cpp).

## Stage A: CPU, exact matches and few-colour blocks

### Per-frame gate (keeps regular content as it is)

In the read-ahead, per frame, on luma:

- the share of 16x16 blocks using at most 8 distinct values (palette-like);
- the share of 8x8 blocks whose hash occurs elsewhere in the frame (IntraBC-like).

The screen tools are on for a frame when either share passes its threshold (kPaletteShare,
kRepeatShare; starting values 0.15 each), else off. Logged per frame under `CELV2_GOP_LOG`.

**As built:** the palette share alone, kPaletteShare 0.04, or a key frame with 40% or more periodic 16x16s.
The repeat share does not open it: Flash's values overlap the regular sources'.

### Signalling

- With `--screen 1` or `auto`: sequence `force_screen_content_tools = 2`. `force_integer_mv` stays 2 and
  frames write 0 (no forced integer MVs).
- With `--screen 0`, the default: the sequence header as now, the same bytes.
- Frames the gate passes: `allow_screen_content_tools = 1` and `allow_intrabc = 1`.
  - Key frames: global and local.
  - Inter frames: local.
  - Frames it does not pass: both 0.

### Block decisions (candidates added on the CPU; the GPU engine unchanged)

**Palette.** Blocks 8x8 to 64x64 whose luma has at most 8 distinct values:
- a 2-8 colour palette by k-means (the exact values when there are 8 or fewer);
- the colour index map, coded as AV2 codes it;
- compared by RD cost, residual included, with the block's intra / inter choice in the RD pass and the
  coder.
- Chroma takes its usual modes.

**IntraBC.** Blocks 8x8 to 64x32 / 32x64:
- candidate vectors from a hash table of the source's 8x8 blocks (exact repeats), kept when the target
  is valid (local range, or global under the delay rule on key frames);
- scored in the RD pass on source predictions, then by the coder on the actual reconstruction;
- vectors coded through AV2's block vector predictors (BVP DRL), integer only.

### With the rest

- Workers: the gate and the hashes are per chunk, so any worker count gives the same bytes.
- `--paq`, psy-rd: independent; palette and IntraBC blocks take the same qindex and weights.

## Stage B1: CPU, copies along a halftone's period (only if)

**When:** after stage A, Dragon Maid's halftone frames (33, 55, 77, 100) have not improved, or the
halftone diagnosis lands on its branch 4. A diagnostic count over those scenes is then made: the share of
16x16 blocks with no exact hash match but a copy at an integer multiple of the local period vector
better than the chosen prediction by 20% or more in SATD. B1 goes in when that share is 10% or more of
the halftone area.

**What:**
- The `--paq` map's periodicity search already finds each block's autocorrelation peak; its lag (the
  period vector) is exposed as well (an amendment to the `--paq` spec).
- For blocks with a strong period, IntraBC candidates are the valid positions at small integer
  combinations of the period vector(s), 10 to 20 per block. They are scored by SATD, then by the
  RD pass and the coder as in stage A.
- They need not match exactly.

**Cost:** an exhaustive local search would compare about 16,000 positions per 8x8 block: about 53 G
cycles a 1080p frame, about 0.5 s on this machine against about 0.1 s a frame today (about 5x on gated
frames; estimate, not measured). Period candidates are about a thousandth of that: a few milliseconds
a frame, within a few percent.

**As built (0d9d6cf):** up to 3 copies a block, in the coder only, with a 128-bit charge (kNearCopyCharge)
for the chroma its luma cost leaves out. Dragon Maid's halftone frames did not change.

## Stage B1 in the RD pass: the copy has a way back

**Why.** These frames need the RD pass to choose the copy: with it they gained +0.34 to +1.01 dB at q160
and Dragon Maid 120 frames -0.48%, but Angel Beats lost +46.68% (x1), +17.74% (x1.5), +1.10% (x2.5). Two
causes are named in the report: the pass scores a copy of the source, which flatters a copy of what will
be reconstructed; and a block given to a copy the coder declines stays an intra DC. The second is a
structural fault, and the simpler to remove.

**Step 1: a way back.**
- When the RD pass gives a block to a near copy, it also keeps the block's best choice without near
  copies, at the same block size.
- The coder scores the copy on the reconstruction as now. If it declines, it codes the kept choice, not
  an intra DC.
- Logged per frame: copies chosen by the pass, and declined by the coder.
- Measured at weights x1, x1.5, x2.5 on Angel Beats, Dragon Maid 120 frames and Flash.

**Step 2, only if step 1 misses the goals:** the pass's cost of a near copy adds the noise of the area it
copies: kCopyNoise x the quantizer's step squared / 12 per sample, kCopyNoise starting at 1, in place of
the x1-x2.5 weight.

**The gate.** `auto` leaves these frames shut today, since only key frames take the periodic share. Once
step 1 or 2 is kept, it also opens inter frames with 40% or more periodic 16x16s. Frames 33 to 100 have
about 50-70%, and the 120 frames' mean is about 11%. In inter frames the copies stay local IntraBC.

**Kept if, under `auto`:**
- Dragon Maid's frames 33, 55, 77, 100 gain 0.2 dB or more at q130 and q160, on at least three of the four;
- Dragon Maid 120 frames -0.3% or better;
- Angel Beats no worse than the coder-only -7.78% by more than 0.2 points, Flash likewise against -3.81%;
- the regular sources within +0.2%.

If neither step meets this, B1 stays in the coder and stage B2 does not open: B2 finds more copies, which
is not what is short here.

## Stage B2: GPU (only if)

If B1 still leaves many near-matches unused (the same diagnostic count, with an exhaustive offline
search, well above what B1 takes), an IntraBC candidate in the GPU engine:
- an integer local search by SATD;
- in Vulkan and the CPU reference, bit-identical.

Its own spec.

## Interface

- `--screen 0|1|auto`: 0 off (the default; the same bytes as now), 1 the tools on every frame, auto the
  per-frame gate.
- The default becomes auto when both hold (decided 2026-11-05). Under auto every frame header carries
  one more bit, so even regular sources' bytes change slightly.
  - **Speed:** on a quiet machine, S1-S4's CPU time a frame within 3% of `--screen 0`, entered in the
    speed ledger. These sources' gate stays shut, so this is the gate's own cost. Sources where it opens
    (Flash read 1.12-1.18x, under load) report their slowdown beside their gain, and are not held to
    the 3%.
  - **The periodic threshold 0.4:** it rests on one frame (Angel Beats 0.617; the next highest 0.291). On
    the four 600-frame sources (S1 and S2's three), with every key frame's share logged, each stays
    within +0.2%. The stretches with scene cuts are what give it more key frames.
- A `--tune screen` (for Flash-like captures) and the GUI setting: after the measurements.

## Measure

- **Where it should gain:**
  - Flash: the 60-second stretch, the report's 30 frames and 240 frames;
  - AM demo;
  - Dragon Maid (the halftone scenes).
- **Where it must not lose:** AIR OP, chika, Angel Beats, 吉村, the cut-heavy AM 600 frames.
- **Settings:** `--refs 7`, qindex 100 / 130 / 160 / 190; `--screen` 0, auto, 1.
- **Measures:** BD-rate by global-MSE PSNR (these tools code the same picture in fewer bits),
  SSIMULACRA2 reported; the gate's per-frame decisions logged, its thresholds tuned on them.
- **Goals:**
  - Flash -10% or better (avmenc: -13.4%);
  - AM demo -1% or better;
  - Dragon Maid's frames 33, 55, 77, 100 higher in PSNR (B1 measured on them separately);
  - `--screen auto` on the regular sources within +0.2% and within 3% of the speed, the condition for
    auto becoming the default.
- **Speed** (the 3% bound above, and B1's few percent): a slowdown beyond it does not by itself drop the feature: the implementing session profiles the
  encoder and looks for an equal speed gain elsewhere (another stage made faster) to offset it, and
  reports the two together; if none is found, it reports the
  loss to the user and asks them to decide; it does not drop the feature for speed on its own.
- Every threshold or part dropped gets a row in docs/design/tried-not-kept.md.

## Tests

- **Palette:** a synthetic picture of 4-colour glyphs and shapes on a flat ground, `--screen 1`: AVM
  decodes equal to `--recon`, and it is smaller than with `--screen 0`.
- **IntraBC:** a synthetic picture of repeated identical tiles, `--screen 1`: decodes equal to
  `--recon` and is smaller than `--screen 0`, on key frames (global) and inter frames (local).
- **Gate (unit):** off on natural synthetic frames (`scene_frame`), on for the glyph and tile pictures.
- **Unchanged and deterministic:**
  - `--screen 0` gives the same bytes as before;
  - `--workers 1` and `2` give the same bytes with `--screen auto`;
  - `--device cpu` and the GPU give the same bytes;
  - 10-bit input decodes equal to `--recon`.
- **B1 (when built):** a synthetic halftone with period (6, 6): the IntraBC candidates include integer
  multiples of it, and the stream decodes equal to `--recon`.
- **B1's way back:** a frame where the coder declines a near copy the RD pass chose: the block codes the
  kept choice, not an intra DC, and the stream decodes equal to `--recon`.

## Not in this

- Chroma palette (not in AV2); IntraBC morph prediction and other AV2 IBC refinements; forced integer
  MVs; the GPU search (B2, its own spec).
