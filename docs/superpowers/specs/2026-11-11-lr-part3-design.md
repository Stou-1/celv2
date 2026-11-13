# C2 part 3: luma loop restoration per unit and per class (design)

2026-11-11. Week 3 of the CelV2 schedule. The week 1 gate (docs/reports/2026-11-11-week1-gates.md, gate 2) put F0's luma
restoration types, unit types and class count, with CelV2's own least-squares taps (oracle `X_ORACLE_FILT rf`, ec816ba), at
-2.90% BD-rate YUV 6:1:1 on md / ab / am30 / cr (Y -3.41%) against CelV2's LR today. Encoder only, no format change: AV2
codes all of it already; CelV2 sends luma PC_WIENER in 128x128 units, each on or off, and its sequence header turns luma
WIENER_NONSEP off.

The target is the gate's `rf` path: per class, taps by least squares on CelV2's reconstruction over the units that take the
filter. What CelV2 lacks is the choices `rf` took from F0: which units take WIENER_NONSEP, PC_WIENER or nothing, and the
class count. This step searches those; the unit size stays 128.

## What is searched

Per frame, luma only (chroma as today):

- **Unit type**, per 128x128 unit: none, PC_WIENER or WIENER_NONSEP (frame filters: no per-unit taps).
- **Frame restoration type**: what the unit types need. Both filters used: SWITCHABLE; one of them: that type with its
  flag per unit; no unit filtered: NONE (today a PC_WIENER frame codes every flag even when all are 0).
- **Class count** C in {1, 2, 3, 4, 6, 8, 12, 16} (encode_num_filter_classes()), the taps of each class by least squares
  over the units taking WIENER_NONSEP, each class's tap subset and dictionary match by error less bits (the gate's `rf`
  rule), plus the option of a dictionary filter as it is (exact match), as the chroma fit has it.
- **Not searched**: the unit size (128, as today; 256 and 512 would reuse the same per-unit sums, a later step if the
  per-unit types leave a gap), per-unit taps (not measured by the gate), temporal prediction of the frame filters.

The smallest search that keeps the gate's two parts: the unit types (WIENER_NONSEP where PC_WIENER loses: Angel Beats'
-9.28% and Dragon Maid's -0.75% in `rfk`) and the class count (AM's -1.98% and crowd_run's -1.12%).

## How it runs (CodecState::restore())

1. **Pass 1, as today**: every luma unit PC_WIENER, chroma its frame filter, one filter pass on lf_workers_. It gives each
   unit's exact squared error with PC_WIENER and without (the kept copy), and the PC_WIENER classes of every 4x4 block
   (`mi_params.wiener_class_id[0]`, which WIENER_NONSEP's classes are: av2_wiener_nsfilter_stripe() classifies with the
   same apply_pc_wiener_highbd()).
2. **Per (unit, class) normal equations** (core/wiener_sums: `wiener_class_sums()`): on the kept luma before restoration,
   the gate's features (a symmetric pair's two samples less twice the centre, 16 of them for wienerns_filter_y) and target
   (source less the centre), summed per 16-class index in integers, as wiener_sums() does for chroma (int16 features,
   _mm256_madd_epi16, 32-bit lanes widened every 128 samples), on every other row (as built: -0.10% YUV against every
   row, half the CPU; every fourth row lost 0.16%). Each row's features for a unit are computed vectorised in 16 bits,
   then each 4-sample group (one 4x4 block's row: one class) is copied to its class's buffer. 17 x 18 / 2 = 153 sums a
   (unit, class), in parallel over units. The 16-class map is a refinement of every smaller one (checked on the four sets' sub-classifier tables),
   so any C's sums are sums of these; the merge map is built from get_pc_wiener_sub_classifier() at run time.
3. **Search** (no pixel pass): for each C, fit over every unit, decide the units, refit over the units that took
   WIENER_NONSEP, decide again. A unit's WIENER_NONSEP error is its error without restoration less the estimated gain
   (2 t.v - t.A.t) / 128^2 summed over its 16 class indices with their class's taps, plus 1/12 a sample (the filter's
   rounding; without it Angel Beats' and AIR's low rates took units the filter did not pay for). Per class: the subset
   by J at match 0 (the zero filter), then of the 3 dictionary filters nearest the fit (L1) the one that codes it in the
   fewest bits, or one of those 3 as it is. The 8 class counts run on 8 threads, each with its own filters and
   dictionary (frame_filter_class_bits() takes them); kept: the least J over C and frame type, in count order.
4. **Pass 2**: the kept luma back, WIENER_NONSEP on the units that took it (chroma types NONE for this pass), then the
   PC_WIENER units' pass-1 output copied back. Each WIENER_NONSEP unit's exact error is measured and the units are decided
   again on exact errors (WIENER_NONSEP only where measured); if that plan's J is not below the best plan without
   WIENER_NONSEP (PC_WIENER flags or NONE, exact already), that one is taken (the fit's in-sample estimate is optimistic:
   without this check AIR lost 0.59% YUV). The frame type follows the final unit types.

## The cost: J = SSE + sse_per_bit x bits, the bits as AVM codes them

- **Unit symbols** (loop_restoration_read_sb_coeffs()): PC_WIENER frame: pc_wiener_restore_cdf, one binary symbol a unit;
  WIENER_NONSEP frame: wienerns_restore_cdf, one; SWITCHABLE: switchable_flex_restore_cdf[0] (is it none?), then for a
  unit that is not, switchable_flex_restore_cdf[1] (is it PC_WIENER? else WIENER_NONSEP). Adaptive CDFs, so each symbol is
  priced at the frame's own odds of it: decide with 1 bit a symbol, count, decide again with -log2 of the counted odds;
  the frame's symbol bits are n H(p) per symbol, as the chroma decision prices its flags today.
- **Frame filter header** (read_match_indices_hdr() and read_wienerns_framefilters_hdr()): class by class, the match
  index (group bit(s) and refsubexpfin within the group, predicted from the classes before), the exact bit, and for a
  filter that is not exact its subset (unary), symmetry bit and taps (refsubexpfin against the matched filter). Counted
  per class exactly as the writer writes them (`frame_filter_class_bits()`), the dictionary built once per C
  (set_frame_filter_dictionary(): zero filter, the earlier classes, the references' frame filters, sampled PC_WIENER
  filters). The total is checked against the written header (a mismatch throws). Plus frame_filters_on, temporal_pred_flag
  and the 3-bit class count. The frame type's own 2 bits are the same for every type and not counted.

## CPU (estimate from the gate's code, S1 1080p, 1757 ms CPU a frame at HEAD; +1% is 17.6 ms)

- The gate's refit summed doubles per pixel (171 products) over F0's WIENER_NONSEP units plus one more full filter pass for
  the class map: on every unit that is about 2 M pixels x 171 = 350 M double MACs, roughly 120 ms CPU a frame.
- Here: the class map is pass 1's (free); the sums are integer SIMD as chroma's (the chroma fit, LR fit stage, is 31 ms CPU a
  frame for 1 M chroma samples with 190 sums and a luma downsample), so about 30-45 ms for 2 M luma samples with 153 sums
  and the class scatter; the search about 2-5 ms (units x 16 x 153 per gain estimate, 8 class counts x 2 rounds, parallel
  over units); pass 2 at most the luma share of today's LR apply (28 ms CPU for all planes), only on frames and units that
  take WIENER_NONSEP. Expected +40-70 ms, 2.5-4% CPU: over the +1% rule, so it needs an offset (bc1b5f5 names the CDEF
  set / damping choice and the CCSO search) or a cheaper fit (rows subsampled in the sums).

## Coding the unit flags (deferred symbols)

Today BlockWriter::lr_units() logs one SymbolLog::deferred() symbol a unit with the CDF of the plane's type at tile time;
restore() fills lr_flags() and kLrSkip drops a symbol. The luma type is now known only after the tiles, so for each luma
unit lr_units() logs four deferred symbols: [PC_WIENER flag (pc_wiener_restore_cdf), WIENER_NONSEP flag
(wienerns_restore_cdf), SWITCHABLE first (switchable_flex_restore_cdf[0][0]), SWITCHABLE second ([1][0])], lr_flags()
holding four entries a luma unit. restore() writes the ones the final type codes and kLrSkip the rest, so replay() codes,
per unit, the decoder's symbols in its order. Chroma keeps one entry a unit. The logged CDFs adapt as if each were 0; they
only steer bit estimates, as before.

Header: the sequence header enables luma WIENER_NONSEP (lr_tools_disable_mask[0] = 0, so luma has 4 frame types and SWITCHABLE;
chroma's mask unchanged, sent explicitly when --lr-uv 0); the frame header writes decode_restoration_mode() in full (the
type, frame_filters_on, temporal_pred_flag 0, the class count, the unit sizes, the frame filters), the writer the oracle
path has used since ec816ba, now the only one.

## Checks

- test_cli `cli_lr_units`: a 768x512 clip (6x4 units; left half crossed sine waves, right half a ramp), 8 and 10 bits,
  B frames, q100; CELV2_GOP_LOG's `lr` line shows WIENER_NONSEP units and PC_WIENER units (SWITCHABLE frames); AVM
  decodes equal to --recon; `--workers 1`, `--workers 2` and `--device cpu` give the same bytes. (The CCSO test's clip
  has no unit where PC_WIENER pays, so no SWITCHABLE frame: a broken switchable path would pass there.)
- test_wiener_sums: wiener_class_sums() against a per-pixel reference.

## Results (2026-11-11): BD passes; within the speed rule with its own cuts and the filter offsets

BD-rate against e1f29da (tools/c17/bd_run.py, 8 sources, q100-190), PSNR-YUV 6:1:1 / PSNR-Y, %:

| | air | md | am | ck | ab | yo | cr | pa | mean | worst |
|---|---|---|---|---|---|---|---|---|---|---|
| 33 frames | -0.80 / -1.04 | -0.69 / -1.09 | -1.22 / -1.68 | -0.31 / -0.37 | -4.19 / -4.97 | -0.82 / -1.01 | -0.39 / -0.53 | -0.74 / -0.87 | **-1.15 / -1.45** | cr -0.39 |
| 120 frames | -0.55 / -0.75 | -3.01 / -3.72 | -1.26 / -1.65 | -0.63 / -0.75 | -2.36 / -2.83 | -0.73 / -1.05 | -0.40 / -0.54 | -1.18 / -1.35 | **-1.26 / -1.58** | cr -0.40 |

The BD bar holds at both lengths (YUV mean -0.5% or better, no source above +0.3%, AB and AM past -1%); no source is
worse on Y. The gate's -9.49% on Angel Beats was F0's unit types on F0's decisions; CelV2's own search gets -2.4 to -4.2%.

Variants, 33 frames, YUV mean: sums on every row -1.05; every fourth row -0.99 (Dragon Maid -0.21); class counts 1, 4,
16 only -1.08; also the chroma fit on every other row (an offset candidate, -15 ms CPU) -1.06 (120 frames -1.24, AIR
-0.29); the first build, without the rounding term and the exact plan check, -0.84 with AIR +0.59.

Speed of the first build (S1, speed_ab.ps1's S1 runs, 5 rounds, medians): CPU a frame +2.0%, fps -8.1%; with the
chroma fit on every other row CPU +3.4%, fps -5.9%. From the profile (150 frames): the LR fit stage 33.7 -> 72.8 ms
CPU a frame, LR apply 30.1 -> 34.5 ms; on the filter chain LR decide 1.8 -> 10.1 ms wall.

### Its own cuts (the same bytes)

The bytes are the first build's: all 32 BD points at 120 frames (8 sources x 4 q) md5-equal, so the BD-rate above
stands. S1 150 frames, CELV2_PROFILE:

- The class sums' scratch (150 KB a unit) kept per thread, the per-unit sums and pass 2's copy kept between frames:
  a fresh allocation a unit from 48 threads, and fresh pages each frame, cost more than the sums. Sums 36 -> 18 ms
  CPU a frame, 4.1 -> 0.85 ms wall.
- madd_pairs() (each feature row loaded once for four pairs, shared with the chroma fit) and runs of groups of one
  class copied at once: sums 18 -> 14 ms CPU.
- Pass 2's two whole-plane copies on 32 tasks: 1.2 -> 0.2 ms wall.
- Each class count's unit gains four units at a time (four chains of additions, each in one unit's order): the
  search 1.5 -> 1.3 ms wall.

LR decide 10.1 -> 4.6 ms wall; the LR fit and restoration stages add about 20-25 ms CPU a frame over e1f29da's.

### The offsets (each its own commit, the same bytes)

| Commit | What | Measured (S1 150 frames unless said) |
|---|---|---|
| 6058509 | Wiener sums: f_i loaded once for four pairs | chroma fit about 36 -> 33 ms CPU |
| 9a77721 | LR unit errors unit by unit | LR decide 1.8 -> 1.3 ms wall |
| 20f4877 | CDEF set choice: pair-major table, AVX2 min-sum, early stop | set choice 2.7 -> 1.6 ms wall (bench 2.8 -> 1.2 ms) |
| 249457c | CDEF damping pass with the first pass's directions | about -3 ms CPU, within the runs' spread |
| 40866cd | CCSO unit stats in strips, the planes at once | CCSO search 10.4 -> 8.6 ms wall |
| 7ebb173 | CCSO counting tables kept between frames, one add | CCSO search about -1 ms wall; S1 600 fps +3.3% with LR part 3 |
| 290bc0c | CCSO counting: 16 samples' bins and cells at once | CCSO search 84-88 -> 62-63 ms CPU, 6.8 -> 5.7 ms wall |

Not used: fewer class counts, an early exit over the counts, or the first pass's sums reused (step 2's byte-changing
options): the plans are 2-3 ms CPU, not where the cost is.

### Speed against e1f29da (LR part 3 and the offsets)

speed_ab.ps1 runs S1 (5 rounds), S2air and yo (3 rounds), medians (paired medians in brackets), the shell at normal
priority:

| | CPU a frame | fps |
|---|---|---|
| S1 | -0.74% (+0.13%) | +2.16% (+2.75%) |
| S2air | -2.51% (-0.71%) | +8.38% (+9.28%) |
| yo | -2.11% (-2.23%) | +0.71% (+0.53%) |

The runs move by 1-2% CPU and 2-4% fps between sessions: before 290bc0c, 15 S1 rounds over three sessions gave CPU
+1.2% and fps +1.2% (paired medians); 290bc0c's 23 ms is what puts CPU under the bar.
