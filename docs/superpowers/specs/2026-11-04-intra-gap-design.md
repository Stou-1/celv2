# Closing the intra gap to avmenc, round 2: CfL by RD with explicit alpha, honest transform splits, 128 blocks (design)

2026-11-04. Follows docs/reports/2026-11-03-intra-rd.md (64x64 intra blocks and RD mode choice, 613f72c:
-1.32% over 120 frames; one frame against avmenc cpu-used 6 from +44-58% to about +30% on Flash and from
+47% to about +20% on AIR). Three fixes, one at a time, each kept or dropped on its own numbers.

## Why

A keyframe still costs CelV2 far more than avmenc with the same tools. That report lists three
remaining differences:

1. **Chroma (CfL).** avmenc uses CfL on 95% of AIR's area and 69% of Flash's; CelV2 on 36% and 0%.
   Without CfL avmenc loses +26.9% on AIR.
   - CelV2 offers only two of AV2's three CfL types: the derived alpha (`CFL_DERIVED_ALPHA`) and MHCCP
     (`CFL_MULTI_PARAM`), not the signalled alpha (`CFL_EXPLICIT`, AV1's form).
   - It scores CfL by a SATD from the source luma, times 1.2 (block_writer.cpp, the chroma mode
     choice).
2. **Luma transform splits.** avmenc's intra blocks hardly split their transform; CelV2 splits 17-21% of
   the area, and on Flash codes about twice avmenc's transform blocks (39.9k ends of block against
   18.9k).
3. **128x128 / 128x64 blocks.** avmenc uses them on 17% of AIR's area; CelV2 decides in 64x64 units.

## Stage 0: measure the gap again

- **Frames** (1 each; the copies are in
  `scratchpad\me\tmp\`,
  a temporary folder: copy them somewhere stable first):
  - `iv_flash_f20_0.y4m` and `iv_flash_f100_0.y4m` (Flash, 1440x1080);
  - `iv_airop240_60.y4m` (AIR OP frame 60);
  - `iv_g_am30_0.y4m` (AM), `iv_g_md_0.y4m` (Dragon Maid);
  - frame 0 of `g_ck.y4m` (chika) and of `g_yo.y4m` (吉村, live action), cut out.
- **Runs:** CelV2 after 613f72c against avmenc cpu-used 6 with CelV2's tools, matched in PSNR-Y, with
  the previous tools (`gap/acct.c`, `gap/qmatch.py` in the same scratchpad).
- **Recorded per frame:**
  - the bytes difference and the bits by syntax element (coefficients, modes, partitions);
  - the area under CfL, split per CfL type for avmenc;
  - the area with split luma transforms;
  - the area of 64x64 and larger, and of 128 blocks;
  - the area per intra transform type, IDTX included (avmenc without IDTX: +5.8% on Flash).
- The table sets the order of the fixes below: the largest share of the gap first. Report in
  `docs/reports/<date>-intra-gap-2.md`.

## Fix 1: CfL by RD, with the signalled alpha

1. **Signalled alpha.** For each chroma block, the least-squares alpha of the source chroma against the
   reconstructed luma's AC, per plane; rounded to the nearest signalled value, then its neighbours one
   step either side tried. Coded as AV2 codes `CFL_EXPLICIT` (joint sign, per-plane index).
2. **Exact prediction.** All three CfL types predicted from the reconstructed luma (coded before the
   chroma, so the decoder's prediction exactly); the source-luma estimate and its 1.2 factor go.
3. **RD choice.** The chroma mode's SATD shortlist re-scored by quantized U + V cost plus kRdBitWeight
   (0.05) times the mode bits, as the luma modes are since 613f72c. The shortlist is the 4 best non-CfL
   modes and the three CfL types (MHCCP's best direction).
- **Live action:** CfL is on only under `--tune animation` today (live action +0.26 / +1.22%, Y / YUV,
  with the SATD choice). After this fix it becomes the default if live action passes (below).
- **Switch:** `--cfl-rd 0|1` (1 once kept), for comparisons.

## Fix 2: transform splits costed like the modes

- `choose_intra_tx_partition` compares split against unsplit by squared error plus 0.2 times the split
  symbol's bits; the coefficient bits are not counted the way the mode choice counts them.
- Both sides are costed by the mode choice's RD: squared error in steps plus kRdBitWeight times every
  bit, per transform block: its end of block, its all-zero flag, its levels, and the partition symbols.
- **Transform types:** if stage 0 shows IDTX (or another type avmenc uses) rare in CelV2's intra blocks,
  the type shortlist (`residual_code`: DCT against the most compact other type) gains it as a candidate.
- **Switch:** `--intra-txp-rd 0|1`.

## Fix 3: 128x128, 128x64 and 64x128 intra blocks

- **Where:** intra frames, as `--intra64`.
- **How:** after the RD pass decides each 64x64, one more comparison per 128x128 superblock.
  - Only where all four 64x64s (two for 128x64 / 64x128) chose one unsplit block.
  - A single large block, its mode from the top candidates and its transform in 64x64 parts, against
    the sum of the parts.
  - The large block's cost times kIntra128Ratio (starting value 1.1, as kIntra64Ratio).
- **Coder:** the decision map gains a way to hold a 128 block. `compose_sb()` today only lays 64x64
  decisions under a 128x128 superblock.
- **Switch:** `--intra128 0|1`.
- **Inter frames (conditional):** first, avmenc with 128x128 blocks against a 64x64 cap on
  - `a4k.y4m` (4K, 30 frames) and `g_am30_4k.y4m` (4K, 120);
  - `a8k.y4m` (8K, 9 frames);
  - `a16k.y4m` (16K, 9 frames; the 8K clip is it scaled down);

  all from the same folder, cpu-used 6, q100-190. The only number so far is 16K over 3 frames, -0.31%
  (docs/design/tried-not-kept.md).
  - At -1% or better at any resolution, CelV2's inter frames get 128 blocks too, by the same merge:
    four unsplit 64x64s with the same mode and MV, or all skipped. Then CelV2's own BD-rate is checked
    at that resolution.
  - Otherwise intra only, with the numbers in tried-not-kept.

## Measure

- **One frame**, each fix in turn, on stage 0's seven frames at qindex 100 / 130 / 160 / 190:
  - the BD-rate against before the fix;
  - the bytes gap to avmenc at matched PSNR-Y.
  - Goal: Flash from about +30% to +15% or less, AIR from about +20% to +10% or less, once all three are
    in.
- **120 frames**, each fix: `g_air`, `g_md`, `g_am30`, `g_ck`, `g_ab` (720p), `g_yo`, `g_cr`, `g_pa` from
  the same folder; `--refs 7`, qindex 100-190, BD-rate by global-MSE PSNR-Y.
  - Kept if the mean is negative and no source is worse than +0.5%.
  - CfL by default: if 吉村, crowd_run and park_joy are within +0.2% in Y and negative in YUV (6:1:1).
- **Speed:** within 5% per fix. A slowdown beyond it does not by itself drop the fix: the implementing
  session profiles the encoder and looks for an equal speed gain elsewhere (another stage made faster)
  to offset it, and reports the two together. If none is found, it reports the loss to the user and
  asks them to decide; it does not drop the fix for speed on its own.
- Every fix or variant dropped gets a row in docs/design/tried-not-kept.md.

## Tests

- **CfL:**
  - a chroma block built as alpha x luma AC + c: the least-squares alpha rounds to the signalled value
    nearest alpha;
  - on the existing `chroma_follows_luma` synthetic source, `--cfl-rd 1`: AVM decodes equal to
    `--recon`, smaller than `--cfl-rd 0`.
- **Transform splits:** a smooth gradient block is not split; the stream decodes equal to `--recon`.
- **128 blocks:** a flat synthetic keyframe picks 128 blocks, is smaller than `--intra128 0`, decodes
  equal to `--recon`; the same on inter frames if they get them.
- **All:**
  - `--workers 1` and `2` give the same bytes;
  - `--device cpu` and the GPU give the same bytes;
  - 10-bit input decodes equal to `--recon`.

## Not in this

- avmenc cpu-used 0's further -28.6% on Flash (its wider search); the GPU engine's intra candidates; an
  oracle that imports avmenc's decisions into CelV2 (built only if these fixes leave the gap unexplained).
