# Chroma loop restoration per unit (design)

2026-11-12. The next CelV2 schedule item after C2 part 3. The week 1 gates' side finding
(docs/reports/2026-11-11-week1-gates.md, gate 2): on Angel Beats CelV2's chroma restoration, one WIENER_NONSEP frame
filter per plane in 64x64 units, loses to F0's per-unit chroma taps. Encoder only, no format change: AV2 codes all of it
already.

## The gate (oracle, L4)

`X_ORACLE_FILT ru` (new letter `u`, oracle build only): luma is CelV2's restore() at HEAD e49513c (per unit and per
class), chroma keeps F0's restoration as it is: its frame type, unit size, unit types and per-unit taps. Against
`ru`'s parent `r` (CelV2's restore() on every plane), md / ab / am30 / cr, 33 frames, q100-190, BD-rate %:

| ru against r | md | ab | am30 | cr | mean |
|---|---|---|---|---|---|
| PSNR-YUV 6:1:1 | +0.26 | -4.76 | +0.19 | -0.66 | **-1.24** |
| PSNR-Y | -0.05 | +0.46 | +0.17 | +0.09 | +0.17 |

The bar (-0.5% YUV mean) holds, so this is built. These are F0's taps, fit on F0's own reconstruction, not refit on
CelV2's (a refit needs the per-unit syntax deferred past the tiles, which is what this step builds): read it as a floor
on what F0's choices are worth rather than the ceiling. For reference, r against L4 (F0's whole restoration) is now YUV
+0.87 / Y -0.78% (CelV2's luma is ahead of F0's; chroma is the gap), ru against L4 YUV -0.41 / Y -0.62%. Every one of the
48 streams decodes equal to `--recon`.

F0's chroma (lrstat.py on the dumps): always per-unit taps (frame_filters_on 0), one class, unit sizes 64, 128 and 256
(by source and frame), and many frames without chroma restoration.

## What is searched

Per frame, the two chroma planes (AV2: chroma has WIENER_NONSEP only, one class, and one unit size for both planes,
64, 128 or 256 with 128x128 superblocks):

- **Unit size** S in {64, 128, 256}, shared by U and V: the least J of the two planes' best modes at each S.
- **Plane mode** per plane at that S: NONE, a frame filter (today's: one filter in the frame header, a flag a unit), or
  per-unit taps (frame_filters_on 0: each unit on or off and, when on, its own taps coded against the unit bank).
- **Per-unit taps**: for each unit, in coding order (tiles, superblocks in raster order, the units whose corner is in
  the superblock) with the bank as the decoder keeps it (av2_reset_wienerns_bank() at each tile, LR_BANK_SIZE 4): off;
  a bank filter as it is (exact bit); or the unit's least-squares taps of each subset coded against the bank filter that
  codes them in the fewest bits. The best by estimated gain less bits.
- **Not searched**: classes (chroma has one), temporal prediction of the frame filter, the per-unit choice between
  frame filter and own taps (AV2 has no such unit type), symmetric taps (the least-squares taps are coded as they come).
- **The tile rule** (decode_restoration_mode()): max(luma unit, chroma unit in luma samples) divides every tile but the
  last of each row and column; a size that does not is not searched (64, the superblock, always qualifies).

## The cost: J = -gain + sse_per_bit x bits (chroma's sse_per_bit, x4 as today)

- **Gain**: per 64x64 unit (at every S a larger unit is a union of 64 units: units start at each tile, with the
  4-row offset, and the last one takes the rest), the normal equations of today's fit (wiener_sums(), integers),
  summed per unit; the gain of taps t is wienerns_gain() less 1/12 a sample (the filter's rounding, as the luma part
  has it). The frame filter's sums are the sum of the units', the same integers as today.
- **Bits**: the frame filter's header as wienerns_syntax() counts it today; a unit's flag 1 bit in the search; per-unit
  taps as read_wienerns_filter() reads them: the exact bit and the bank index (literals), the subset (wienerns_length_cdf),
  the symmetry bit (wienerns_uv_sym_cdf) and each tap's 4-part code (wienerns_4part_cdf, then its literal), priced at the
  frame's starting CDFs.
- **Then measured**: one filter pass with the chosen plan (with luma's first pass, as today), each unit's exact error,
  a unit kept where its exact gain beats its bits (flag and taps), the plane off where the kept units' gain does not pay
  for its bits and the flags' entropy (as today). The units' bank coding is redone on the final units (a unit turned off
  leaves the bank; the taps of the units after it stay as filtered).

## Coding (deferred past the tiles)

Today each chroma unit's flag is a SymbolLog::deferred() symbol logged at tile time. The unit size and the taps are now
known only after the tiles, so BlockWriter::lr_units() logs one position entry a superblock (as CDEF's index is), after
luma's deferred symbols; SymbolLog::replay() hands it to a callback that codes, for each chroma plane, the units whose
corner is in that superblock at the final size: the flag (wienerns_restore_cdf) and, for per-unit taps, the planned
syntax. lr_flags() keeps luma's four entries a unit only. A frame that keeps today's plan (frame filter, 64 units) codes
the same symbols in the same order as before.

## CPU (S1 1080p, about 1.76 s CPU a frame; +1% is 17.6 ms)

- The sums: the same samples and products as today's chroma fit (about 31 ms CPU a frame), split by 64 unit instead of
  by row band: about the same.
- The search: 2 planes x 3 sizes, 672 units in all at 1080p (510 + 128 + 34), each 3 subset solves (18x18) and a few bank
  candidates' bits: about 2-4 k solves, an estimated 3-8 ms CPU, the six searches in parallel.
- The pass, the exact errors and the decisions: as today.

Expected +5-10 ms CPU a frame (0.3-0.6%), inside the rule without offsets.

## Checks

- test_cli `cli_lr_units` (768x512, 8 and 10 bits, B frames) extended: CELV2_GOP_LOG's chroma `lr` line shows per-unit
  taps on more than one unit and a unit size other than 64 in the run, AVM decodes equal to `--recon`, `--workers 1 / 2`
  and `--device cpu` the same bytes.
- test_wiener_sums: the rectangle sums against the reference.
- The multi-tile streams of test_ippp and test_cli's CDEF test exercise the tile rule (the first build, without it, sent
  units that crossed tiles and AVM refused the streams).

The tried-not-kept row "Chroma loop restoration (WIENER_NONSEP) variants" (2026-11-01) named cheaper filter syntax as
its revisit condition: this is that. Its variants (filter always sent, plane always restored, chroma bits weighed x1 / x2
/ x8 / x16 against today's x4) were judged on PSNR-Y; with PSNR-YUV 6:1:1 as the measure the lighter weights (x1, x2:
"YUV better in every row") may deserve a re-measure on top of this step. Not re-measured here.

## Results (2026-11-12): BD passes, within the speed rule

**At the gate's level** (L4, md / ab / am30 / cr, 33 frames, X_ORACLE_FILT r with this build against r at e49513c),
BD-rate YUV / Y %: md -0.01 / -0.02, ab -4.88 / +0.17, am30 -0.59 / +0.15, cr -1.08 / +0.09, mean **-1.64 / +0.10**:
past F0's own chroma (`ru`, -1.24), which is now +0.40% YUV against this. Against L4 (F0's whole restoration) CelV2's
restoration is now YUV -0.81 / Y -0.68%. All 64 streams decode equal to `--recon`.

**BD-rate against e49513c** (tools/c17/bd_run.py, 8 sources, q100-190), PSNR-YUV 6:1:1 / PSNR-Y, %:

| | air | md | am | ck | ab | yo | cr | pa | mean | worst |
|---|---|---|---|---|---|---|---|---|---|---|
| 33 frames | -0.16 / +0.03 | +0.01 / +0.11 | -3.12 / +0.17 | +0.05 / +0.36 | -3.47 / +0.40 | -0.21 / -0.11 | -1.14 / +0.08 | -1.16 / +0.08 | **-1.15 / +0.14** | ck +0.05 |
| 120 frames | -0.73 / -0.56 | -0.16 / -0.01 | -2.58 / +0.12 | -0.25 / -0.06 | -1.66 / +0.16 | -0.35 / -0.15 | -1.18 / +0.07 | -1.43 / +0.05 | **-1.04 / -0.05** | md -0.16 |

The bar holds at both lengths (YUV mean -0.5% or better, no source above +0.3%); no source is worse than +0.4% on Y.

**Speed against e49513c** (speed_ab.ps1's S1 5 rounds, S2air and yo 3, medians, paired in brackets, the shell at
normal priority):

| | CPU a frame | fps |
|---|---|---|
| S1 | -1.10% (-0.09%) | -0.43% (+0.02%) |
| S2air | -1.47% (-1.14%) | -1.35% (-1.84%) |
| yo | +0.13% (+0.13%) | -0.60% (-0.60%) |

The first build (the units' fits inside the six sequential bank passes, the symbol costs taken from the CDFs at each
call) was S1 CPU -0.25% / fps -1.39% over 10 rounds (paired -0.29 / -1.73%); CELV2_PROFILE on S2air 150 frames had the
LR fit stage at 44.9 -> 49.8 ms CPU a frame, its wall 4.3 -> 5.4 ms, the LR pass 4.3 -> 3.3 ms and LR decide 5.2 -> 4.7
ms (fewer chroma units filtered). Its own cut, the same bytes (5 streams md5-equal, 33 frames: ab, am30, cr, S2air, pa):
every unit's least-squares taps and gains for every plane and size in one parallel pass before the bank passes, and the
symbols' bit costs taken once a frame.
