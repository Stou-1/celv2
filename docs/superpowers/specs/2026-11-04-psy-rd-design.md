# `--psy-rd S` / `--psy-trellis T`: keeping texture and line energy in the RD decisions (design)

2026-11-04. Perceptual plan item 4 (after `--target-vmaf`, `--grain`, `--paq`). Opt-in. Builds on the
`--paq` importance map (docs/superpowers/specs/2026-11-04-perceptual-aq-design.md).

## Why

Every RD decision in CelV2 weighs squared error in quantizer steps against bits (no psychovisual term
anywhere: survey of 2026-11-04). Squared error prefers the smooth candidate: a blurred texture, a
softened line or a halftone dot field melted into flat colour costs fewer bits for little squared
error, and looks worse. x264's psy-rd adds the difference in AC energy between source and
reconstruction; this does the same in CelV2's transform domain, weighted per block by content.

## Decisions (agreed 2026-11-04)

- **Measures that decide:** BD-rate by SSIMULACRA2 and XPSNR, as `--paq`; the user's viewing last.
- **By content:** live action keeps texture energy, anime keeps line art (and not grain on flat cels),
  switched by `--tune`.
- **Staged:** A (the psy term in prediction, mode, partition and transform decisions) first; B (psy in
  quantization, "psy-trellis") measured on top of A; C (in-loop filter choices) not here: its own spec
  if A and B leave the filters as the smoothing step.
- **Where the search runs:** the GPU candidate engine keeps SATD (a wide search by a cheap measure, also
  less smoothing-prone than squared error); the psy term goes into the CPU stages that re-score the
  engine's candidates and decide the rest (an accurate measure over a narrow set).

## The psy term

For a luma block and transform size, with S = the source's transform, c = the residual's coefficients
(source minus prediction), q = their dequantized levels, all in quantizer steps:

- the reconstruction's transform is R = S - (c - q) (prediction + dequantized residual), so
- `psy = | sum over AC of |S_k|  -  sum over AC of |R_k| |` (DC excluded),

and a decision's cost becomes `J = D + bit_weight * bits + S_psy * kPsyScale * w * psy`, with D and
bits as now. Both blurring (AC lost) and added grain or ringing (AC gained) cost.

- S is computed once per block position and transform size (the source does not change between
  candidates) and cached per superblock; no pixel reconstruction is needed for the term.
- kPsyScale: a named constant, starting value 1, set by the measurements with its evidence in a comment.

## The block weight w (0..1)

From the `--paq` map's per-16x16 measures (energy E = variance, coherence C), which the map module then
exposes besides its offsets (a small addition to that module), averaged over the 16x16s a block covers:

- **texture rule** (the default and `--tune film`): `w = t(E) * (1 - C)`, t rising from 0 at flat
  (log2(E + 1) <= kFlatLog) to 1 at busy (>= kBusyLog): texture weighted, flat areas and edges not.
- **line rule** (`--tune animation`): `w = C * t(E)`: edges of drawn lines weighted, flat cels 0 (no
  grain or ringing pushed onto them), texture 0.
- A third rule for halftone (screentone) fields is left room for: it needs a periodicity measure in the
  map, planned as a `--paq` spec amendment, and gets its weight then.
- kFlatLog, kBusyLog: named constants, starting values 4 and 8 (variance 15 and 255 in 8-bit samples).

## Where it goes (stage A)

- **RD pass** (core/rd_pass.cpp): `tx_cost` / `residual_cost` and so every comparison built on them:
  partition (`square`), intra mode (`intra_cost`), MV and reference (`inter_cost`), compound pairs.
- **Coder**: `pred_rd` (encoder.cpp; every inter choice in block_writer.cpp: MV refinement, NEWMV
  precision and AMVD, NEAR index, GLOBALMV, BAWP, inter-intra, skip-mode index), `intra_rd` (the intra
  mode re-scoring with MRL), `choose_tx_partition`, `choose_intra_tx_partition`, and the transform type
  choice in `residual_code`.
- **Not:** the GPU engine (SATD), chroma (luma only, as the RD pass), IST and CCTX choices, the in-loop
  filters.

## Stage B: psy-trellis

- The level choice (quant.cpp: nearest, nearest - 1 or 0 by `(e - l)^2 + kBitWeight * bits`) and the TCQ
  Viterbi gain `S_trellis * kTrellisScale * w * | |e| - |l| |` per AC coefficient (e the coefficient, l
  the level, in steps): a linear term outweighs the squared one below one step, so small AC levels stop
  dropping to 0 as readily (fine texture and dots kept).
- w reaches the quantizer as an argument (it sees coefficients only).
- In both the RD pass's estimate and the coder's quantization, so decisions and the coded result agree.
- kTrellisScale: starting value 1, set by the measurements.

## Interface

- `--psy-rd S` (0 off, the default, to 2): stage A's strength. `--psy-trellis T` (0 off, the default,
  to 2): stage B's. The rule follows `--tune` (animation: lines; else texture).
- The importance map is built per frame when either is on, with `--paq 0` too (it is cheap); FrameSpec
  carries a shared pointer to the frame's map, used by AQ and psy alike.
- Default on, or in a tune: after the measurements. GUI: after that.

### With the other features

- `--paq`: AQ moves the qindex, psy keeps the energy at that qindex: independent, usable together.
- `--grain`: the source compared against is the denoised one, so psy does not rebuild the removed noise.
- `--target-vmaf` / `--target-size`: their probes run with psy, so their picks include it.
- Workers, devices: psy runs only in CPU stages in a fixed order: the same bytes at any worker count,
  and `--device cpu` equal to the GPU's.
- Speed: one source transform per block position and size; measured, kept within +10% or made cheaper;
  a slowdown beyond it does not by itself drop the feature: the implementing session profiles the
  encoder and looks for an equal speed gain elsewhere (another stage made faster) to offset it, and
  reports the two together; if none is found, it reports the
  loss to the user and asks them to decide; it does not drop the feature for speed on its own.

## Measure

- Sources: the `--paq` eight (AIR OP, Dragon Maid, AM demo, chika, Angel Beats, 吉村, crowd_run,
  park_joy), 120 frames, `--refs 7`, qindex 100 / 130 / 160 / 190, anime with `--tune animation`.
- Stage A: `--psy-rd` 0.5, 1, 1.5 against 0. Stage B: `--psy-trellis` 0.5, 1 on the best A.
- Measures as `--paq`: SSIMULACRA2 (every 4th frame), XPSNR (all frames) deciding; global-MSE PSNR,
  VMAF and the time reported.
- **Kept if:** the mean BD-rate over the sources is negative by both SSIMULACRA2 and XPSNR, no source is
  worse than +1% by either, and the time rises 10% at most; then the user views decoded files: Dragon
  Maid's halftone scenes (frames 33, 55, 77, 100, where CelV2 q130 trails x264 crf 26 by 3-7 dB),
  吉村 and AIR OP.
- **GPU candidates:** if stage A gains little on live action, it is measured again with `--rd-intra 1`
  (every intra mode re-scored on the CPU); a much larger gain there means the engine drops the
  candidates psy wants, and a psy term in the engine (Vulkan and the CPU reference) gets its own spec.
- Every strength or rule dropped gets a row in docs/design/tried-not-kept.md.

## Tests

- `test_psy`:
  - a reconstruction equal to the source: psy 0;
  - a textured block with every AC level zeroed: psy equal to the source's AC sum;
  - R from S - (c - q) equal, within rounding, to the forward transform of the actual pixel
    reconstruction (prediction + inverse transform of q): the identity the design rests on;
  - weights: a flat block 0 under both rules; a texture block near 1 under the texture rule and 0 under
    the line rule; an edge block above 0.5 under the line rule;
  - psy-trellis: an AC coefficient of 0.6 steps that strength 0 zeroes is kept at strength 2.
- `test_cli`:
  - `--psy-rd 1 --psy-trellis 1`: AVM decodes equal to `--recon`, and the bytes differ from both off;
  - both 0: the same bytes as before the change;
  - `--workers 1` and `2`: the same bytes with both on;
  - `--device cpu` against the GPU with both on: the same bytes;
  - out-of-range values refused.

## Not in this

- The in-loop filters' choices (stage C); a psy term in the GPU engine; chroma; the halftone rule
  (after the periodicity measure); psy in IST / CCTX choices.
