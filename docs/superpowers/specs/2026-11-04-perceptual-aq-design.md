# `--paq S`: perceptual adaptive quantization (design)

2026-11-04. Perceptual plan item 3 (after `--target-vmaf` and `--grain`, before psy-rd). Opt-in. No
neural network: the proposal of 2026-11-04 (a neural importance map steering QP and partition pruning)
starts here with a hand-made map, and a learned one replaces it only if it beats this.

## Why

The qindex is the same over a whole frame unless TPL (`--deltaq`, off by default) moves it per
superblock by how later frames reference it. The eye is not uniform: flat areas, gradients, line art,
text and faces show coding error, busy texture hides it. Spending bits where the error shows, and taking
them where it hides, raises the perceived quality at the same size; PSNR falls by construction.

## Decisions (agreed 2026-11-04)

- **Measures that decide:** BD-rate by SSIMULACRA2 and by XPSNR. Global-MSE PSNR and VMAF are reported,
  not deciding (PSNR falls with any AQ; VMAF is lenient here). The user's viewing has the last word.
- **Staged (plan C):** the importance map is built at 16x16 from the start; stage A uses it through the
  existing per-superblock delta qindex (`FrameSpec::sb_dq`, coded per 128x128 superblock); stage B,
  per-block qindex through AV2 segmentation (up to 8 levels), only if stage A gains and its grid is shown
  to lose most of the map (below).

## Stage A

### The importance map (`src/core/perceptual_aq.{h,cpp}`)

From the source luma only (the denoised frame under `--grain`), per 16x16 block:

1. **Energy E**: the variance of its samples (8-bit scale).
2. **Coherence C** (0..1): of its structure tensor, ((l1 - l2) / (l1 + l2))^2 over the block's
   gradients (central differences); near 1 for one dominant direction (an edge, line art), near 0 for
   texture.
3. **Brightness L**: its mean.
4. **Periodicity P** (0..1): the largest normalized autocorrelation of its zero-mean samples over the
   lags 2..6 pixels away (half-plane: dy >= 0; lags with dy = 0 only for dx > 0), each lag over the
   samples where the block overlaps its shifted self. A halftone (screentone) dot field repeats at its
   dot period, so one lag correlates strongly (P near 1); random texture does not (P near 0). Computed
   only for blocks that would be coarsened (positive d below, C < 0.5), so flat areas and edges cost
   nothing (an edge correlates along itself, which C already covers).

The offset, in qindex (24 qindex double the quantizer step):

- `d = S * k_e * (log2(E + 1) - mean over the frame of log2(E + 1))` (x264's aq-mode 2 shape):
  flat finer, busy coarser.
- **Line art kept:** a positive d is scaled by (1 - k_c * C): an edge block is busy by variance but not
  texture, and plain variance AQ coarsens anime's lines.
- **Halftone kept:** a positive d is scaled by (1 - k_p * p), p = clamp((P - P0) / (P1 - P0), 0, 1): a
  screentone field is busy by variance and has no dominant edge, but it covers large areas regularly and
  its breaking up shows at once (Dragon Maid OP frames 33, 55, 77, 100: CelV2 q130 3-7 dB below x264
  crf 26 there, at about the same overall quality).
- **Dark areas:** `d -= S * k_l * max(0, (L0 - L) / L0)` (x264's aq-mode 3 idea), a small bias.
- **Zero mean:** the frame's mean offset is subtracted, so a frame's bits stay about the same (no clash
  with `--target-size`, `--target-vmaf`, two-pass).
- Clamped to +-35 (`9 * kDeltaQRes - 1`, the existing sb_dq range).

k_e, k_c, k_l and L0 are named constants set by the measurements below, each with its evidence in a
comment; not options. Starting values: k_e = 4 (x264's aq-mode 1 adds about 1.04 QP per doubling of
the variance, and 1 QP is about 4 qindex), k_c = 1 (no coarsening at all on a pure edge), k_l = 4,
L0 = 64, k_p = 1, P0 = 0.3, P1 = 0.7. At 10 bits the map reads the 8-bit planes (`Frame420::y`), as
the decisions do.

The map keeps, per 16x16, besides its offset: log2(E + 1), C, P (0 where not computed) and the lag
(dx, dy) where P peaked (the period vector; (0, 0) where not computed). psy-rd
(docs/superpowers/specs/2026-11-04-psy-rd-design.md) weights its blocks from the first three, and its
halftone rule reads P; IntraBC's stage B1 (docs/superpowers/specs/2026-11-04-intrabc-palette-design.md)
takes its copy candidates along the period vector.

Cost: P is about 80 lags x 256 samples per block computed, on the coarsened texture blocks only: at
most about 170 M multiply-adds a 1080p frame (every one of its 8160 blocks), far fewer in practice;
measured with the rest, and vectorized or computed on 8x8-averaged lag sums if it shows.

### Superblock offsets

- `aq_sb64()`: each 64x64 of the `sb_dq` grid takes its 4x4 blocks' offsets weighted to the sensitive
  side: the mean of their 25th percentile and their mean (a plain mean coarsens a sky superblock that a
  person's edge just enters).
- CodecState averages four of them into each coded 128x128 superblock, as now.
- **Spread statistic** (for the stage B decision): per frame, the mean variance of the 16x16 offsets
  within each 128x128 superblock, and the variance between the superblocks' means; logged under a
  verbose flag or returned for the bench.

### Where it applies

- `chunk_coder` builds each frame's `FrameSpec`: every coded frame gets `sb_dq` from its own source
  frame: keyframes, anchors, mids **and the non-reference B frames** (TPL offsets reach reference frames
  only; AQ is for what is shown).
- With `--deltaq` on, the TPL offsets and the AQ offsets are summed, then clamped to +-35.
- `--target-vmaf` probes and `--pass 1` run with the same AQ, so their picks include it.
- Deblocking follows each block's qindex already (AV2 derives it from base + delta).

### Required of the coder (check first when implementing)

- The RD pass and the coder's decisions use each superblock's own qindex for both the quantizer and the
  lambda (encoder.cpp already resets the intra lambda per superblock qindex; the RD pass and the inter
  lambda must be checked). A coarse quantizer under the frame's lambda mis-decides.
- The GPU candidate search keeps the frame's qindex: the CPU RD pass re-picks.

### Interface

- `--paq S`: strength, 0 (off, the default) to 2; 1 the nominal. (`--aq` is the anchor qindex.)
- Default on or in a `--tune`: after the measurements. The GUI setting: after that too.

## Measure

- Sources: AIR OP, Dragon Maid, AM demo, chika, Angel Beats (720p), 吉村 (live action), crowd_run,
  park_joy; 120 frames each, `--refs 7`, qindex 100 / 130 / 160 / 190; `--paq` 0.5, 1, 1.5 against 0.
- SSIMULACRA2: every 4th frame (30) as PNG, mean score per point; the tool (libjxl's `ssimulacra2`,
  BSD-3, or `ssimulacra2_rs`, BSD-2) must be downloaded: the implementing session shows the user its
  file name, source and size and asks first.
- XPSNR: ffmpeg's `xpsnr` filter over every frame (ffmpeg 9.0.2 has it).
- Reported too: BD-rate by global-MSE PSNR and by VMAF, the time.
- **Speed:** within 3%; a slowdown beyond it does not by itself drop the feature: the implementing session profiles the
  encoder and looks for an equal speed gain elsewhere (another stage made faster) to offset it, and
  reports the two together; if none is found, it reports the
  loss to the user and asks them to decide; it does not drop the feature for speed on its own.
  Judged by the speed ledger's CPU time a frame (`--workers auto`), the agreed whole-video measure, not a
  single encode's time. Result, decided 2026-11-05: `--paq 0.25` passes (S1 +0.9%, S2 +0.6 / +1.0%, 吉村
  +2.1%). The report's 1.03-1.14x is not the measure.
- **Kept if:** at some strength, the mean BD-rate over the sources is negative by both SSIMULACRA2 and
  XPSNR, and no source is worse than +1% by either; then the user views AIR OP, 吉村 and Dragon Maid's
  halftone scenes (frames 33, 55, 77, 100) side by side (decoded files at one mid rate, with and
  without) before it goes into a tune or the default.
- The halftone term on its own: Dragon Maid with k_p 1 against 0 at the chosen strength, PSNR and
  SSIMULACRA2 of those four frames reported besides the BD-rates (the term must not make them worse than
  `--paq 0`).
- Every strength or term dropped gets a row in docs/design/tried-not-kept.md.

## Stage B (only if)

Started when stage A is kept and the spread statistic shows the 128x128 grid losing most of the map: the
mean within-superblock variance of the 16x16 offsets above the between-superblock variance on most
sources. Then: AV2 segmentation (segmentation params in the frame header, a segment id per block, one
qindex per segment, the RD pass and coder per segment), the same 16x16 map quantized to the segments'
levels. Its own spec.

## Tests

- `test_perceptual_aq`:
  - a flat frame: every offset 0;
  - half flat, half noise: the flat side negative, the noisy side positive, the mean 0;
  - thick straight lines on a flat ground (line art): the line blocks' offsets below the noise blocks'
    at the same variance;
  - two flat areas of the same variance, one dark: the dark one lower;
  - a halftone field (round dots of radius 2 on a period of 6 pixels, 45-degree grid) against random
    noise of the same variance: P above 0.7 on the dots and below 0.3 on the noise, and the dot blocks'
    offsets below the noise blocks' by 2 or more;
  - the map's per-16x16 log2(E + 1), C and P exposed, P 0 on flat and edge blocks;
  - every offset within +-35;
  - `aq_sb64()`: the sb_dq grid's size, and a 64x64 with one sensitive 16x16 among busy ones pulled
    below the plain mean.
- `test_cli`:
  - `--paq 1` on a mixed synthetic source with B frames: AVM decodes it equal to `--recon` (delta
    qindex on non-reference B frames included), and the bytes differ from `--paq 0`;
  - `--paq 0`: the same bytes as before the change;
  - `--workers 1` and `2`: the same bytes with `--paq 1`;
  - out-of-range values refused.

## Not in this

- Temporal masking (motion hides error): TPL already moves reference frames by how they are used; added
  only after stage A's spatial result, to keep the two apart.
- Chroma in the map; a learned map (the neural proposal); psy-rd (plan item 4, its own spec).
- The rest of the halftone problem: the scene-cut and still-share analyses work on 8x8-averaged
  luma and TPL at half scale, where a dot field averages to flat colour, so a new halftone picture may
  pass for an unchanged one; and IntraBC / palette (off) suit repeating dots. Each needs its own design.
