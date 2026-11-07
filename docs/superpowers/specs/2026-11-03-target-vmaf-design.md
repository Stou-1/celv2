# `--target-vmaf`: a quality target per scene (design)

2026-11-03. Perceptual plan item 1 (then denoise + grain, perceptual AQ, psy-rd). Opt-in.

## What it does

`celv2enc --target-vmaf V` codes each scene at the qindex whose VMAF (mean over the scene's frames,
the default `vmaf_v0.6.1` model) is closest to V without going below it, instead of one qindex for the
whole video. Still scenes then take fewer bits, busy ones more. The GUI gets a 「画質指定（VMAF）」 field
next to 「サイズ指定（MB）」; the two exclude each other.

## How

1. **Scenes.** The input is cut into chunks at scene cuts as `--target-size` with workers does
   (a ChunkCoder per worker, a keyframe at each chunk's start, 48 frames at least). With
   `--target-vmaf` chunking is on at any worker count, and the read-ahead is the multi-worker one
   (`--chunk-memory`).
2. **Probes.** Each probe codes the chunk on the worker's own coder at one qindex, with the run's own
   settings, and scores it with ffmpeg's libvmaf (reconstruction and source written as y4m to the
   temporary folder). A chunk of 128 frames at most is probed whole: kept (FrameQueue::keep), coded,
   rewound; the probe at the pick is then the chunk's output, so its VMAF is the one measured. A
   longer chunk is probed by a copy of its first 96 frames and coded once more at the pick (four
   probes of n frames cost less than four of 96 plus n only up to 128).
3. **Search** (`celv2::vmaf_search`). First probe at `--qindex`; the second where the usual slope
   says (log of 100 - VMAF rising 0.017 a qindex, 4 to 48 away); then where the line through the two
   nearest probes on either side of V meets it; at most 4 probes (a 5th while the best is over V + 1),
   stopping at one within 0.15 above V.
   The pick is the highest probed qindex at or above V (the lowest probed when none is).
4. Chunks do not wait for each other's picks, so workers probe at once and any worker count gives the
   same bytes.

## Interface

- `--target-vmaf V` (0 < V <= 100). With `--pass` / `--target-size`: refused.
- `--ffmpeg PATH`: the ffmpeg with libvmaf the probes run (default `ffmpeg` on the PATH). Refused at
  start when it does not run or has no libvmaf filter.
- stderr per scene: `celv2enc: scene N (frames a-b): qindex Q (P probes of F frames[, then coded]:
  coding x s, VMAF y s)`.
- Tests: `CELV2_TEST_VMAF=c,d[,odd]` replaces ffmpeg with VMAF 100 - c x 2^(q / d) (+ odd on odd
  chunks).

## Why these choices (AM / Dragon Maid / 吉村, 120 frames 1080p, V = 93)

| Probe | Each scene's VMAF | Time against one `--qindex` |
|---|---|---|
| Every 2nd frame (48 at most), pass 1's settings | 92.3-95.0; one Dragon Maid scene 89.9 | about 2-3x |
| Every 4th frame (24 at most) | 91.4-91.7 | |
| Runs of 8 or 16 frames from keyframes | 92.9-94.7 | |
| First 96 frames whole, 8x8 tiles, RD pass on | 93.13-93.24 | about 5x |
| Whole chunk (128 at most) with the run's settings, the pick's probe kept | **93.00-93.23** | **3.4-4x** |

- Skipping frames makes each coded frame cost more bits at the same qindex (the gaps predict worse), so
  a strided probe overrates the qindex: Dragon Maid frames 0-72 at qindex 172 gave VMAF 89.9 coded
  whole and 94.9 every 2nd frame.
- Pass 1's settings (`--rd-pass 0`) score about 1.8 lower than the real pass at the same qindex; 8x8
  tiles change nothing in VMAF but are slower than the reuse.

## Measured

docs/reports/2026-11-03-target-vmaf.md: 8 sources at V = 93, every scene 93.0-93.55; against one
qindex whose worst scene is 93, -5 to -72% on the sources with several scenes, within 2% on one-scene
ones; 1.4-5.1x the time. V = 95 and other models not yet measured.

## Not in this

- Other models (NEG, 4K), percentiles instead of the mean.
- Probing at reduced resolution; fewer probes from a model of the scene.
