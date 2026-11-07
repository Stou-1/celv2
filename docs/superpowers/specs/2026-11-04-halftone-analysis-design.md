# Halftone (screentone) scenes: find why CelV2 trails, then fix the analysis that misses them (design)

2026-11-04. Diagnose first, then fix by the branch the measurement points to.

## Why

Dragon Maid OP (120 frames 1080p), CelV2 q130 against x264 crf 26 (about the same overall quality):
where a new picture with large halftone (screentone) fields appears, CelV2 trails by 3-7 dB:

| Frame | CelV2 | x264 | x264's frame |
|---|---|---|---|
| 33 | 35.9 dB | 42.8 dB | 84 KB |
| 55 | 34.2 dB | 41.8 dB | 125 KB |
| 77 | 39.4 dB | 41.8 dB | 102 KB |
| 100 | 34.3 dB | 37.5 dB | 157 KB |

The pictures arrive through iris wipes (a circle opening or closing over several frames).

What the code shows (read 2026-11-04, not yet measured):

- The scene-cut test and the still share (`core/scene_cut.cpp`) compare 8x8 cell averages: a dot field
  with a 4-8 pixel period averages to flat colour there, so a new halftone picture of similar colours
  can pass for an unchanged one.
- The scene-cut test also rejects a cut when over half the cells are unchanged, so a wipe, which
  replaces the picture over several frames, is never a cut, halftone or not.
- TPL (`core/tpl.cpp`) costs blocks by SAD at half scale, where the dots are gone too.
- Mini-GOPs under `--tune animation` run to 16 frames; a non-reference B frame codes at `--bq` (+16
  qindex). A new picture landing on a B frame gets the fewest bits of any frame.

Which of these causes the loss is not known; this design measures first.

## Stage 1: measure

### Logging (code, small)

`CELV2_GOP_LOG=1` makes celv2enc print to stderr:

- per frame, from the scene-cut test: `celv2enc: gop frame <display>: mad <m> unchanged <u> hf <h> cut <0|1>`
  - m: the mean absolute difference of the 8x8 cell averages from the previous frame;
  - u: the share of cells within 2 levels;
  - h: the mean absolute difference of the cells' high-frequency energy, i.e. each cell's mean absolute
    deviation from its own average;
  - cut: the test's answer.
- per mini-GOP, from ChunkCoder: `celv2enc: gop group <first display> len <m> still <s> anchor_q <q> r0 <r>`
  - s: still_share(first, last);
  - q: the anchor's qindex delta;
  - r: the anchor's TPL r0 (blank when TPL did not run).

### Runs

- CelV2: Dragon Maid OP, 120 frames, `--tune animation --refs 7 --qindex 130 --psnr --progress --recon`,
  `CELV2_GOP_LOG=1`. Per frame: type, qindex and bytes (the `progress` lines), PSNR and SSIMULACRA2
  (from the recon).
- x264: the same source at crf 26, with the settings of the comparison above (recorded in the report,
  the preset included). Per frame: type, QP and bytes (`ffprobe -show_frames`), PSNR.
- A table per frame, CelV2 beside x264, over frames 25-41, 47-63, 69-85, 92-108 (each scene ±8). It
  answers: the type CelV2 gave each new-picture frame, whether x264 made it I or P, and whether the
  byte difference sits on those frames or spreads around them.

Report: `docs/reports/<date>-halftone-diagnosis.md`, ending with the branch or branches that hold.

## Stage 2: fix by branch

The branches may hold together; the one with the largest loss goes first, each measured on its own.

### 1. A new picture lands on a B frame

Sign: the new-picture frames are B (`--bq`) in CelV2 and I or P in x264.

Fix: **end the mini-GOP at a large change.**
- While reading ahead, count the share of 16x16 blocks changed from the previous frame. "Changed"
  uses branch 3's cell test, so halftone counts.
- The mini-GOP ends before the frame where the share summed since the group's start passes kChange
  (starting value 0.3), so that frame becomes the next group's anchor (a reference, at the anchor's
  qindex).
- A wipe spreads its change over frames, hence the running sum. No new boundary within kMinGroup
  (starting value 4) frames of the last.

### 2. The type is right but the bits are too few

Sign: the types match x264's, but CelV2's qindex is higher and its bytes fewer, and the log shows the
anchor's TPL r0 or still-share boost misjudging the halftone frames.

Fix: **TPL costs at full scale where the half scale loses detail.**
- The TPL blocks whose high-frequency energy is large (the dots) are costed by full-scale SATD instead
  of half-scale SAD.
- This is the "finer TPL (full-scale SATD)" that docs/design/tried-not-kept.md names as the revisit
  trigger of `--deltaq`; measure `--deltaq 1` with it as well.

### 3. The analysis misses the halftone

Sign: in the log, `mad` stays small on the new-picture frames while `hf` is large.

Fix: **each 8x8 cell keeps its high-frequency energy beside its average.**
- A cell is unchanged only when both are close: the average within 2 levels, and the energy within
  max(2, 10%).
- The scene-cut difference adds the energy difference to `mad`.
- The still share then sees halftone changes, and so do everything it drives: `--gop-still`'s
  mini-GOP length, and the keyframes' and anchors' qindex boosts.

### 4. Same bits as x264, still worse

Sign: the allocation matches x264's frame by frame, yet CelV2's quality there is lower.

Not this design's: the coding tools are short. The candidates are:
- IntraBC and palette (off; repeating dots copy from within the frame);
- psy-trellis (docs/superpowers/specs/2026-11-04-psy-rd-design.md, stage B);
- the `--paq` halftone term (docs/superpowers/specs/2026-11-04-perceptual-aq-design.md).

The diagnosis report becomes the evidence for their designs.

## Measure (stage 2)

- **Goal:** on Dragon Maid frames 33, 55, 77, 100, CelV2 within 1 dB of x264 crf 26 at the same total
  bytes, by PSNR and by SSIMULACRA2; and Dragon Maid's whole-clip BD-rate negative (global-MSE PSNR,
  SSIMULACRA2).
- **No harm elsewhere:** AIR OP, Dragon Maid, AM demo, chika, Angel Beats, 吉村, the cut-heavy AM 600
  and Flash; 120 frames (600 for AM 600), `--refs 7`, qindex 100-190.
  - Global-MSE PSNR BD-rate within +0.5% on each.
  - SSIMULACRA2 BD-rate not positive on any.
  - Speed within -5%; a slowdown beyond it does not by itself drop the feature: the implementing session profiles the
    encoder and looks for an equal speed gain elsewhere (another stage made faster) to offset it, and
    reports the two together; if none is found, it reports the
  loss to the user and asks them to decide; it does not drop the feature for speed on its own.
- **Scene cuts:** the anime clip `scene_cut.h` was tuned on (49 cuts in 5087 frames, none on fast action
  or flashes) keeps its count, with no cut on fast action or flashes.
- Every fix or threshold dropped gets a row in docs/design/tried-not-kept.md.

## Tests (for the branches implemented)

- `test_scene_cut`:
  - a halftone field replaced by another of the same mean colour is "changed" (still share below 0.5);
  - the same halftone field with a little noise is "unchanged";
  - the existing cases still pass.
- **Branch 1:** a synthetic iris wipe (a circle opening over 6 frames onto a new picture): the frame
  where the summed change passes 0.3 is coded as an anchor (`progress` line `type=A`).
- **Branch 2:** a halftone block's full-scale TPL intra cost clearly above its half-scale one; on flat
  and smooth blocks the two about equal.
- **`CELV2_GOP_LOG`:** a CLI test that the per-frame and per-group lines appear with the fields above.

## Not in this

- IntraBC, palette and coding-tool changes (branch 4).
- Scene cuts at gradual transitions other than wipes (fades are handled already).
- Detecting halftone as a content class for a tune.
