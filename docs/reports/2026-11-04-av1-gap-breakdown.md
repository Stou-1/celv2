# Where CelV2 loses to AV1 (stage 1 of the gap breakdown, 2026-11-04)

Spec: docs/superpowers/specs/2026-11-04-av1-gap-breakdown-design.md. **Draft: the avmenc rows are still
running; numbers marked (n/4) have n of the four points.**

## Stage 1: the same 33 frames through every encoder

- The eight sources (AIR, Dragon Maid, AM demo, chika, Angel Beats (720p), 吉村, crowd_run, park_joy), the
  first 33 frames.
- CelV2 default (`--paq 0`; anime `--tune animation`) and `--preset max`, qindex 100-190; avmenc cpu-used 0 with
  every tool (avm0) and with CelV2's tools (avm0c); SVT-AV1 preset 0 and libaom cpu-used 0, crf 20-56.
- Global-MSE PSNR-Y at 10 bits, BD-rate against CelV2 default (negative: that encoder is smaller at the same
  PSNR). Script: the session's gap33.py; results in celv2-bench\gap33.

| Source | `--preset max` | avm0 | avm0c | SVT-AV1 p0 | libaom c0 |
|---|---|---|---|---|---|
| AIR | +1.84% | +120.48% | +139.79% | +38.01% | +32.93% |
| Dragon Maid | -2.27% | -39.95% | -33.66% | -29.42% | -33.60% |
| AM demo | -0.28% | -27.46% | -21.53% | -6.08% | -11.18% |
| chika | -0.56% | +1.42% | +7.09% | -25.58% | -27.85% |
| Angel Beats | +0.68% | -54.25% | -33.66% | -7.86% | -18.27% |
| 吉村 | -1.53% | +7.68% | +16.26% | -22.84% | -20.92% |
| crowd_run | -0.53% | (1/4) | -22.61% | -18.59% | -24.43% |
| park_joy | -0.58% | (2/4) | -16.13% | -16.25% | -21.66% |

Time per encode (mean over the points): CelV2 8 s, `--preset max` 16 s, SVT-AV1 40 s, libaom 997 s, avm0c
15,784 s, avm0 41,726 s so far (one thread each for avmenc, up to 44 at once, so not a speed comparison).

First reading (to confirm when the rows are complete):
- On 33 frames, AV1 at its slowest presets is 6-34% smaller than CelV2's default on seven of the eight; AIR is the
  exception (CelV2 33-38% smaller).
- avmenc is a target on Dragon Maid, AM demo and Angel Beats (level with or ahead of AV1), not on AIR, chika or
  吉村 (behind AV1, or behind CelV2).
- `--preset max` moves CelV2 by +1.8 to -2.3% only: the gap is not in the RD pass's trial coding.

### The key frame against the other 32 frames

Frame 0 and the other 32 scored apart, piecewise-linear BD-rate on the shared PSNR range, against CelV2 default:

| Source | libaom key | libaom rest | avm0c key | avm0c rest | CelV2 key share | avm0c key share |
|---|---|---|---|---|---|---|
| AIR | +88.76% | +10.41% | +148.22% | -42.17% | 40-57% | 73-94% |
| Dragon Maid | -19.44% | -32.08% | -20.72% | -33.09% | 4-9% | 3-5% |
| AM demo | +60.95% | -27.90% | +27.38% | -45.22% | 10-30% | 22-59% |
| chika | +6.68% | -9.09% | +37.79% | -39.11% | 54-65% | 57-87% |
| Angel Beats | +32.20% | -15.52% | +83.41% | -43.22% | 4-6% | 11-28% |
| 吉村 | +54.41% | -41.76% | +130.42% | -62.30% | 18-29% | 48-91% |
| crowd_run | +34.91% | -30.29% | +14.82% | -32.54% | 6-10% | 13-32% |
| park_joy | +15.90% | -24.19% | +4.61% | -24.19% | 6-10% | 10-28% |

- The key frame is not the gap: CelV2's is smaller than libaom's at the same PSNR on seven of eight. The other 32 frames are:
  libaom is smaller on seven of eight.
- The two columns are not independent (a better key frame makes the rest cheaper), and the key ranges overlap by only 2-3 dB on
  AIR, Angel Beats and 吉村 against avm0c.
- avm0c against libaom, whole clip: Dragon Maid +2.37%, crowd_run +2.64%, park_joy +7.24%, AM -6.05%, Angel Beats -14.68%: CelV2's
  tools with avmenc's search are level with libaom. All tools (avm0 against avm0c): -4.08 to -9.85%, Angel Beats -33.19%.
- AIR, chika and 吉村: avmenc spends 57-94% of the bytes on the key frame, up to 9 dB above the rest, so its whole-clip curve is
  flat and says little; its rest frames are what to compare.

Checked and wrong: CelV2's key premium on anime (`--kq -8 --key-boost 64`) as the cause of chika's gap. Without it, 33 frames, HEAD:
chika +16.43%, AIR +6.36%, AM +1.98%, Angel Beats +1.79%, Dragon Maid -1.22%. The premium pays even on 33 frames.

## Stage 1b: the mini-GOP's length (2026-11-06)

The eight sources, 120 frames, q100-190, the usual settings, BD-rate by global-MSE PSNR-Y at 10 bits against G0 (today's choice):

| Run | AIR | Dragon Maid | AM | chika | Angel Beats | 吉村 | crowd_run | park_joy | Mean of the moving four |
|---|---|---|---|---|---|---|---|---|---|
| G1 step 2 | +0.50 | +6.86 | +9.48 | +2.68 | +8.29 | -6.19 | +3.37 | +4.30 | +6.36 |
| G1 step 4 | +0.52 | +6.20 | +9.87 | +2.94 | +6.86 | -6.60 | +2.94 | +3.98 | **+5.91** |
| G1 step 8 | +0.97 | +6.70 | +11.93 | +4.08 | +7.45 | -6.71 | +3.61 | +4.81 | +6.95 |
| G2 | +1.24 | +6.53 | +12.23 | +4.35 | +7.42 | -6.53 | +3.58 | +4.81 | +7.01 |
| G3 | +5.14 | +6.69 | +6.63 | +10.23 | +11.73 | -0.22 | +4.09 | +5.17 | +6.91 |

- **Verdict by the spec: worse, so not the length at these offsets.** The best run on crowd_run, park_joy, AM and Angel Beats is
  +5.91%. G4 (32 frames) does not open; the gap rests on stage 2.
- 吉村 alone gains (-6.19 to -6.71%): it has no tune, so G0 is 8 frames, and `--tune film` already takes 16 where the still share allows.
- Anchors at q130 / q160: 15-18 to 8-11, their share of the bytes down 3 to 14 points (crowd_run 35.4 / 40.9% to 24.5 / 29.6%, G1 step 4).
  The bytes the anchors save come back in the frames between them. Coefficient bytes move with the total. Streams and gacct's dumps:
  celv2-bench\gop1b.
- The time column (0.39-0.56x) is not usable: G0 ran first, under more load.

**G5, the mini-GOP without a tune** (`--bframes 15 --gop-still 0.7` against G0, 120 frames, q100-190): 吉村 -5.60%, crowd_run and
park_joy +0.00% (the same bytes at q130 and q160: no mini-GOP reaches the still share). 吉村 keeps most of G1's -6.2 to -6.7% and the
other two stay within +0.2%, so by the spec it goes in the ranked list, as a default for no tune. Its time column (3.30x) is load, not the
option: G0 ran alone earlier; the other runs had 16-frame mini-GOPs faster.

## Stage 2: CelV2's own search widened (W1-W7)

W3 (real cost), 04eb471: per 64x64, the main tree and three others of the RD pass (weight:bits 0.0125:12, 0.05:12, 0.025:24) coded for
real, chosen by CDF bits and Y + U + V squared error. 33 frames, q100-190, against off:

| Lambda scale | AIR | Dragon Maid | AM | chika | Angel Beats | 吉村 | crowd_run | park_joy | Mean | Time |
|---|---|---|---|---|---|---|---|---|---|---|
| x1 | -2.89 | -2.39 | -3.14 | -3.64 | -3.97 | -2.81 | -1.02 | -1.54 | -2.67 | 3.70x |
| x2 | -2.40 | -2.68 | -3.82 | -2.56 | -4.52 | -3.24 | -2.05 | -2.17 | **-2.93** | 4.76x |
| x4 | -1.44 | -1.54 | -3.26 | -1.35 | -3.27 | -1.59 | -0.58 | -1.62 | -1.83 | 4.85x |

- Every global weight of the pass lost on average (weight 0.0125 / 0.05 / 0.1: +2.39 / +2.12 / +6.58%; block bits 24 / 48: +1.79 / +7.37%), so
  the default is a good single weight, and it is the choice per superblock that gains.
- About 3 points of a 16-34% gap. Whole trees per 64x64 only: not an upper bound for real-cost decisions. Times under load.

W1, W2, W4-W6 (e19b0b7), each alone, with W3 at x2, and all together. 33 frames, q100-190, against off. W1 per-reference MVs, W2 full search,
W4 chroma in the decisions, W5 every transform type, W6 every pair:

| Run | AIR | Dragon Maid | AM | chika | Angel Beats | 吉村 | crowd_run | park_joy | Mean | Time |
|---|---|---|---|---|---|---|---|---|---|---|
| W1 | +0.27 | +1.04 | +0.05 | -0.11 | -0.11 | +0.98 | +0.87 | +0.59 | +0.45 | 1.47x |
| W2 | -0.43 | -0.97 | +0.06 | -1.07 | -4.58 | +0.53 | +0.76 | +0.16 | -0.69 | 20.98x |
| W4 | +3.86 | +1.57 | +23.50 | +2.16 | +5.11 | +0.64 | +6.25 | +7.20 | +6.29 | 1.18x |
| W5 | -0.51 | -0.36 | -0.51 | -0.74 | +0.89 | -0.17 | -0.44 | -0.19 | -0.25 | 2.49x |
| W6 | -0.01 | -0.03 | -0.64 | -0.20 | -0.48 | +0.09 | -0.06 | +0.15 | -0.15 | 1.81x |
| W1 + W3 | -2.67 | -1.74 | -3.60 | -2.77 | -4.48 | -2.51 | -1.42 | -1.76 | -2.62 | 7.22x |
| W2 + W3 | -3.31 | -3.45 | -3.81 | -3.55 | -7.85 | -3.11 | -1.57 | -2.21 | **-3.61** | 25.29x |
| W4 + W3 | -1.55 | -1.77 | +7.17 | -1.60 | -0.64 | -2.74 | +0.91 | +1.43 | +0.15 | 5.76x |
| W5 + W3 | -3.69 | -2.89 | -4.25 | -2.86 | -4.69 | -3.42 | -2.36 | -2.21 | -3.30 | 11.84x |
| W6 + W3 | -2.51 | -2.51 | -4.37 | -2.68 | -5.28 | -3.15 | -2.11 | -2.07 | -3.08 | 8.96x |
| All but W3 | +4.31 | +0.96 | +25.68 | +0.30 | -0.76 | +1.05 | +9.82 | +10.80 | +6.52 | 29.35x |
| All | -2.54 | -2.44 | +7.50 | -3.25 | -6.22 | -2.94 | +1.91 | +2.59 | -0.67 | 49.25x |

- W3 alone holds nearly all of it; the rest add under a point on W3, Angel Beats' -7.85% with the full search aside.
- W4 loses on every plane, so it is not bits moved to chroma. BD-rate by PSNR-Y / U / V / YUV 6:1:1: AM +23.50 / +39.15 / +33.13 / +27.86, Angel Beats
  +5.11 / +5.93 / +7.27 / +5.34, crowd_run +6.25 / +5.70 / +6.56 / +6.31, AIR +3.86 / -0.56 / +1.86 / +3.12. AM at q130: 24% more bytes, Y level, U and V
  0.3 dB down. It is what spoils "All". Cause, from reading the code and not measured (design session): chroma's cost is counted only on blocks of 8 and over, and intra without CfL. This says the switch as built loses, not that chroma cannot help.
- Times under load.

W7, zeroing residual by RD (bbc3ed5): a, the whole transform block as zero; b, earlier ends; c, bit weights by depth. Same runs:

| Run | AIR | Dragon Maid | AM | chika | Angel Beats | 吉村 | crowd_run | park_joy | Mean |
|---|---|---|---|---|---|---|---|---|---|
| W7a x1 | +0.03 | +0.12 | -1.14 | -0.16 | -2.75 | +0.94 | +0.25 | -0.25 | -0.37 |
| W7a x2 | +1.43 | +1.06 | +2.74 | +0.60 | -1.35 | +6.84 | +3.32 | +2.47 | +2.14 |
| W7a x4 | +4.77 | +3.56 | +11.31 | +3.92 | +3.04 | +19.08 | +9.42 | +8.83 | +7.99 |
| W7b | -0.15 | +0.04 | +0.59 | -0.96 | -1.79 | +0.44 | -0.06 | -0.00 | -0.24 |
| W7c x1 | -0.06 | +0.27 | +0.21 | +0.07 | +0.29 | +0.02 | -0.06 | +0.10 | +0.11 |
| W7c x2 | -0.06 | +1.04 | +0.48 | +0.66 | +0.07 | -0.04 | +0.11 | +0.29 | +0.32 |
| **W7a + b** | -0.04 | -0.06 | -1.45 | -1.13 | -3.44 | +0.21 | +0.00 | -0.52 | **-0.80** |
| W7a + c x1 | +0.11 | +0.61 | -0.48 | +0.38 | -2.60 | +0.83 | +0.77 | +0.35 | -0.00 |
| W7a + c x2 | +0.10 | +1.58 | +0.35 | +0.60 | -2.13 | +0.81 | +1.14 | +0.93 | +0.42 |
| W7a + b, `--bq` 8 | -0.64 | -0.95 | +2.52 | -3.41 | -2.05 | +2.13 | -0.49 | -0.17 | -0.38 |
| W7a + b, `--bq` 0 | +0.44 | +0.83 | +11.18 | -3.55 | +3.85 | +4.44 | +3.19 | +3.82 | +3.02 |

- 120 frames, the moving four (crowd_run, park_joy, AM, Angel Beats): W7a + b +0.33, -0.75, -1.16, -2.46, mean -1.01%. With G1 step 4 against G0:
  +2.53, +2.88, +8.15, +3.02, mean +4.14%, so the "16-frame mini-GOP" row stands.
- Time of W7a + b: 1.3-2.4x run in turn; it is the reruns. The default candidate's spec does both in one trellis pass.
- Between the spec's readings, so design decided: W7a + b a default candidate, written once the run with W3 shows what it adds there. W7c off, W7d not
  built (a-c did not reach 1%), `--bq` stays 16.
- chika's -3.41 and -3.55% at `--bq` 8 and 0 are a lead for its own look. AM's +11% rules out a change for all.

W7f (a0614a7), avmenc's leaf of a fine quantizer and few coefficients: W7a's bits x2 and x4 below the anchors only, with W7b, at `--bq` 8 and 0:

| Run | AIR | Dragon Maid | AM | chika | Angel Beats | 吉村 | crowd_run | park_joy | Mean |
|---|---|---|---|---|---|---|---|---|---|
| x2, `--bq` 8 | -0.29 | +1.06 | +6.90 | -2.03 | -0.90 | +2.55 | +1.92 | +2.44 | +1.46 |
| x4, `--bq` 8 | -0.09 | +5.41 | +14.25 | -0.10 | +3.27 | +3.29 | +6.21 | +7.78 | +5.00 |
| x2, `--bq` 0 | +0.91 | +2.89 | +15.94 | -2.32 | +5.51 | +5.03 | +6.03 | +6.82 | +5.10 |
| x4, `--bq` 0 | +0.64 | +7.59 | +24.55 | +0.20 | +10.35 | +5.93 | +11.09 | +13.03 | +9.17 |

All worse than W7a + b's -0.80%, and worse than x1 at the same `--bq` (-0.38, +3.02%). By the spec zeroing is finished: no ladder goes into the default candidate.

Bytes at q130 by true distance (`truelv2.py`, `truelv3.py` in celv2-bench\gapstreams: a frame's distance from the coding order). `lv.py` buckets by display
order, and after a cut the pyramid is shifted: Angel Beats' "L1" held the key frame at frame 3 (190 kB), the anchor at 11 (90 kB) and the mids. Use these
figures, not `lv.py`'s, in anything that follows.

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

- Below the anchors CelV2 spends 1.6-2.0 times avm0c's bytes at q130 and 1.8-2.3 at q160 (AIR aside) on every source. Key frames and anchors together are
  within about a third either way, AIR aside. Same base qindex, not the same quality.
- Zeroing is not the cause: on crowd_run W7 takes the area from 59 to 69% and 8.6% of those bytes for +0.00%.

Every switch that helps together, W1 and W4 left out (33 frames, against off, times under load):

| Run | AIR | Dragon Maid | AM | chika | Angel Beats | 吉村 | crowd_run | park_joy | Mean | Time |
|---|---|---|---|---|---|---|---|---|---|---|
| W2, W3, W5, W6 | -4.38 | -3.68 | -4.43 | -3.97 | -8.02 | -3.30 | -1.92 | -2.30 | -4.00 | 25.88x |
| W2, W3, W5, W6, W7a + b | -3.81 | -3.90 | -6.27 | -4.58 | -10.99 | -4.45 | -2.04 | -2.95 | **-4.87** | 37.17x |

- Dragon Maid -3.90 and Angel Beats -10.99% against the -27% they would need (avm0c -33.66, -33.7%): with every W on, CelV2 is still well over 10% above avmenc, so stage 3 opens.
- W7a + b adds 0.87 points on the rest (Angel Beats 2.97, AM 1.84), about what it gives alone.

`X_W7P` (15a9b81), zeroing by RD in one trellis pass (docs/superpowers/specs/2026-11-06-rd-zeroing-design.md): W7a + b's reruns replaced by paths that start at any
end over the threshold. 33 frames, q100-190, against off. W3 at x2 (`CELV2_WIDE=3 X_TL=2`):

| Run | AIR | Dragon Maid | AM | chika | Angel Beats | 吉村 | crowd_run | park_joy | Mean |
|---|---|---|---|---|---|---|---|---|---|
| W7a + b | -0.04 | -0.06 | -1.45 | -1.13 | -3.44 | +0.21 | +0.00 | -0.52 | -0.80 |
| R1: one pass | +0.22 | +0.19 | -1.23 | -1.65 | -3.64 | -0.20 | -0.91 | -1.55 | **-1.09** |
| R2: W3 | -2.40 | -2.68 | -3.82 | -2.56 | -4.52 | -3.24 | -2.05 | -2.17 | -2.93 |
| R2: W3 + one pass | -2.73 | -2.55 | -5.55 | -3.59 | -7.32 | -4.86 | -3.10 | -3.66 | **-4.17** |
| R3: W3 + `X_W7P=2` | -2.54 | -2.32 | -3.75 | -2.06 | -7.53 | -3.72 | -3.33 | -3.90 | -3.64 |

- R1 passes: -1.09% against the -0.70% bar, and no source 0.3 points worse than W7a + b (worst AIR and Dragon Maid +0.26, AM +0.22). The fallback was not used: the
  cost returned keeps the end's term.
- R2: on W3 it adds 1.24 points, more than alone. What `--preset max` would gain, on every source but Dragon Maid (+0.13).
- R3 fails: AIR loses 0.19 points where it had to get 0.3 back, and the mean loses 0.53 where the bar is 0.1. AIR gains with the one pass on W3 (-2.40 to -2.73): the loss
  the run was for was on W2, W3, W5, W6 together, not here. Dropped.
- Times are under load (avmenc on the same machine) and not usable.

R4, the one pass alone (`CELV2_WIDE=7 X_W7P=1`), 120 frames, q100-190, against off, as `--psy-rd`'s measures (SSIMULACRA2 every 4th frame, XPSNR-Y every frame):

| BD-rate | AIR | Dragon Maid | AM | chika | Angel Beats | 吉村 | crowd_run | park_joy | Mean |
|---|---|---|---|---|---|---|---|---|---|
| PSNR-Y, global MSE | +1.04 | -0.40 | -0.84 | -0.83 | -3.19 | +0.88 | -0.39 | -2.16 | -0.74 |
| SSIMULACRA2 | +0.61 | -2.26 | -2.37 | +0.49 | -5.88 | -1.13 | +2.36 | -0.13 | -1.04 |
| XPSNR-Y | +1.25 | -0.28 | -2.08 | -0.36 | -1.06 | +3.72 | +7.92 | +2.63 | +1.47 |

- R4 fails. PSNR's mean (-0.74%) and SSIMULACRA2's (-1.04%) pass; XPSNR's mean is +1.47%, and over +1%: XPSNR AIR, 吉村, crowd_run, park_joy, SSIMULACRA2 crowd_run. To the
  user with these numbers. R5's viewing files not made, no default changed.
- XPSNR loses most on the three live-action sources, where SSIMULACRA2 is mixed and PSNR gains on two of them. Why is not measured.
- A first XPSNR pass was wrong and is not in this table: avmdec's y4m says 30 fps, and on sources at another rate ffmpeg paired different frames. Both inputs on one
  timebase now; AIR off at q100, q130 gives 41.87, 39.20, as the `--paq` bench's 41.88, 39.20. Sizes on the encodes run twice were equal, all 32.
- Speed bound: not measured yet, the machine is running F0. Nothing goes in the ledger until it is.

R4b, the one pass with `--psy-trellis` 0.1 and 0.25 (spec fa7f53c, no build), 120 frames, q100-190, against R4's off rows, six sources:

| BD-rate | crowd_run | park_joy | 吉村 | AIR | AM | Angel Beats | Mean |
|---|---|---|---|---|---|---|---|
| R4, no psy: PSNR-Y | -0.39 | -2.16 | +0.88 | +1.04 | -0.84 | -3.19 | -0.78 |
| SSIMULACRA2 | +2.36 | -0.13 | -1.13 | +0.61 | -2.37 | -5.88 | -1.09 |
| XPSNR-Y | +7.92 | +2.63 | +3.72 | +1.25 | -2.08 | -1.06 | +2.06 |
| 0.1: PSNR-Y | -0.35 | -0.89 | +0.82 | +1.50 | -0.85 | -3.16 | -0.49 |
| SSIMULACRA2 | +3.56 | +1.51 | -0.70 | +1.35 | -0.10 | -5.68 | -0.01 |
| XPSNR-Y | +7.38 | +3.39 | +4.12 | +1.76 | -1.58 | -1.22 | +2.31 |
| 0.25: PSNR-Y | +0.45 | +1.10 | +0.90 | +1.90 | +0.61 | -3.22 | +0.29 |
| SSIMULACRA2 | +5.43 | +4.62 | +1.06 | +1.93 | +3.12 | -5.56 | +1.76 |
| XPSNR-Y | +7.37 | +5.08 | +4.41 | +2.10 | +0.81 | -1.38 | +3.07 |

- R4b fails at both strengths. 0.1: XPSNR's mean +2.31%, PSNR's -0.49% misses -0.5%, and over +1% are XPSNR crowd_run, park_joy, 吉村, AIR and SSIMULACRA2 crowd_run, park_joy, AIR. 0.25: all
  three means fail.
- It is worse than no psy on the same six by all three means. crowd_run's XPSNR moves 0.55 points at most, so the charge does not reach what XPSNR sees there.
- The streams spend more for it (crowd_run q160: 2925, 3025, 3149 kB against off's 3202) and buy back little: its SSIMULACRA2 goes 12.30, 13.56, 14.90 against 18.45.

R6, the user's choice of anime only as an option (`--rd-zero`, 0921e4a; off is the bytes before it, on those of `X_W7P=1`, W7's switches gone): `--tune animation` and the default `--paq`,
`--rd-zero` 0 against 1, 120 frames, q100-190, the five anime sources:

| BD-rate | AIR | Dragon Maid | AM | chika | Angel Beats | Mean |
|---|---|---|---|---|---|---|
| PSNR-Y, global MSE | +1.04 | -0.42 | -0.93 | -0.86 | -3.42 | -0.92 |
| SSIMULACRA2 | +0.61 | -2.04 | -2.43 | -0.03 | -5.15 | -1.81 |
| XPSNR-Y | +1.25 | -0.17 | -2.14 | -0.19 | -1.23 | -0.49 |

- R6 passes: all three means better than 0. Over +1%: AIR alone, PSNR +1.04 and XPSNR +1.25, not past R4's +1.25, so it does not stop step 4. To the user with these numbers.
- AIR's rows are R4's, byte for byte: the default `--paq` moves nothing there. The other four move by under 0.8 point from R4, all three measures.
- Speed (step 3) not measured, the default (step 4) not changed.

## Stage 3: replaying avmenc's decisions

Opened by stage 2's last table. L0 first; L1-L4 decided from it.

**L0, avmenc's frames** (ad879d3, `X_L0`, `--scenecut 0`; each source and qindex takes its own avm0c stream's list: order, slots refreshed, qindex; avm0c's shown copy
of its hidden key frame left out). `tools/oracle/` was not built for it: the switch keeps every setting of the off runs. The references of every frame match
avm0c's, 33 of 33, on all 32 streams. 33 frames, q100-190:

| BD-rate | AIR | Dragon Maid | AM | chika | Angel Beats | 吉村 | crowd_run | park_joy |
|---|---|---|---|---|---|---|---|---|
| L0 against off | +121.46 | +2.73 | +4.88 | +32.12 | +19.25 | +53.63 | +1.63 | +3.03 |
| avm0c against off | +138.90 | -33.64 | -21.32 | +6.97 | -33.21 | +16.00 | -22.59 | -16.02 |
| L0 against avm0c | -0.80 | +49.06 | +31.44 | +31.05 | +79.87 | +38.03 | +31.32 | +22.63 |

- **Frame order, references and qindex are not the gap.** They cost CelV2 on every source (+1.63 to +121.46%): its own structure and allocation are the better ones for its coder. With
  the frames equal the gap is wider, 23-80%, so it lies in what is done inside a frame. AIR alone closes: there avmenc's whole loss is its structure.
- Leaves as references (avmenc keeps every leaf in a slot, CelV2 few) come with this list, so they are not the cause either.
- The same frames at the same qindex now compare directly (`l0_cls.txt` in the shared notes). q130, CelV2 against avm0c, bytes and PSNR-Y: leaves 2.6-3.7x (Dragon Maid 1.9x), +1.0 to +2.0 dB (Dragon Maid +2.7);
  mids 1.7-2.7x, +0.8 to +2.3 dB; anchors 1.4-2.3x, +1.8 to +3.2 dB. At one qindex CelV2 keeps far more. Not the quantizer: avm0c's streams have TCQ on every frame, no quantizer matrices and no delta q off the key frames, and at the same step 76-98% of the leaves' gap is coefficient and end of block bytes (design session, l0-leaves-2026-11-06.txt in the shared notes). F0, avm0c without its temporal filter, is next.
- Not equal even so: avm0c's key frame is hidden and comes back as a 31-byte shown copy, and it and the anchors look temporally filtered (吉村's frame 0: 46.8 dB at q42, CelV2 57.1 dB), which
  avmenc does by default and CelV2 does not (tried-not-kept, twice). Its key frames also carry delta q (吉村 mean 47.4 at base 42). avm0c codes 8 bits, CelV2 10 (+48, the same step).

**F0, avm0c without its temporal filter** (`--arnr-strength=0 --enable-keyframe-filtering=0`, 33 frames, q100-190; `X_L0` from F0's own lists). Dragon Maid, Angel Beats and AM are in; three of crowd_run's streams are still running.

| BD-rate | Dragon Maid | Angel Beats | AM |
|---|---|---|---|
| F0 against avm0c | +0.09 | +2.53 | +10.15 |
| L0 against avm0c (filter on) | +49.06 | +79.87 | +31.44 |
| L0 against F0 (filter off) | +48.78 | +75.18 | +18.11 |
| Gap left with the filter off | 99% | 94% | 58% |

- **On Dragon Maid and Angel Beats the filter is not the cause:** over three quarters left, the spec's reading is decisions inside the frame, so the tool is built on F0's streams, L1 and L2 first. **AM is different:** 58% left, the spec's "between". The filter is worth 10% to avmenc there, and that much of AM's gap is not decisions. This was read on the first two sources, as the spec says; AM came later and goes back to the design session.
- CelV2's eight streams are byte for byte the L0 streams: the frame lists did not change, and F0's references match 33 of 33.
- Same frames, same qindex, CelV2 against F0 (`l0f_cls.txt` in the shared notes), q130: leaves 1.93x (Dragon Maid), 2.59x (Angel Beats), +2.66, +1.03 dB; mids 1.70x, 1.95x, +2.33, +1.25 dB; anchors 1.72x, 1.82x, +1.72, +1.79 dB. Nearly the same as against avm0c. The key frame moves: its PSNR-Y rises by 2-4 dB without the filter (Dragon Maid q130 60.22 to 62.38, Angel Beats 51.97 to 56.02), and the other frames stay.
- The leaves with the filter off (`l0cmp`, q130, F0's column; AM also, its q130 stream is done). Coefficients 16.5, 30.3, 7.7 kB against CelV2's 39.0, 113.2, 39.5 (Dragon Maid, Angel Beats, AM). What stands out in the area shares is what CelV2 does not do: 128 blocks are 85%, 48%, 42% of avmenc's leaf area, none of CelV2's, and intra is 32%, 28%, 1% against 0.1%, 2.1%, 0.8%. Without the filter Dragon Maid's skip mode falls from 41% to 6%, and its intra rises from 24% to 32%, at the same bytes.

**L1 and L2, avmenc's partitions, then its modes and MVs** (`tools/oracle/`, only with `CELV2_ORACLE`; F0's streams and frame lists, the l0f set's settings, 33 frames, q100-190, BD-rate by global-MSE PSNR-Y). L1: the RD pass keeps avmenc's tree and picks each leaf's mode and MV; the coder re-scores as at L0. L2: no engine, no RD pass, no re-scoring: each block coded as avmenc chose it, with CelV2's transforms, quantization and filters. All 52 streams (L0, L1, L2 and the one-tile run below) decode equal to `--recon`, the frame list 33 of 33; L0's eight Dragon Maid and Angel Beats streams are the l0f set byte for byte. crowd_run has F0's q190 only so far, so no BD-rate.

| BD-rate | Dragon Maid | Angel Beats | AM |
|---|---|---|---|
| L0 against F0 | +48.78 | +75.18 | +18.11 |
| L1 against L0: partitions | +14.30 | +3.06 | -0.75 |
| L2 against L1: modes and MVs | -25.87 | -15.04 | -3.47 |
| L2 against F0 | +30.16 | +53.38 | +13.07 |
| L0's gap closed by L2 (log rate) | 34% | 24% | 26% |

- **Decisions are a quarter to a third of the gap.** The rest is what L2 leaves CelV2's: transforms, quantization, filters, entropy coding.
- **The partitions alone do not pay.** +14.30% on Dragon Maid, whose 128 blocks (81% of the area) are 52% NEARMV and 37% intra DC: CelV2's own modes lose on that tree. The gain comes with the modes, so L1 and L2 read together: -13.92, -12.30, -4.23% against L0.
- **Side information is closed.** At L2 partitions, modes and MVs cost what F0's do, all 33 frames at q130 (partition + modes + other, kB, L2 / F0): 50.0 / 50.5, 53.5 / 56.8, 75.3 / 78.8.
- **What is left is coefficients at the same modes.** Coefficients and ends of block, all frames, q130, L2 / F0: 172.9 / 104.4, 531.0 / 281.2, 472.2 / 273.4 kB, at +1.86, +1.38, +1.06 dB. L2 leaves a gap: L3 and L4 by the spec's rule.
- **Blocks with no coefficient** (share of the leaves' area, L2 / F0): 66 / 68, 31 / 49, 40 / 74%. Dragon Maid's agree; Angel Beats' and AM's do not, on the same prediction, so there it is not the mode. The leaves' coefficients at L2 are still 1.9, 2.8, 4.6 times F0's.

The leaves, the frames of `l0cmp` (q130, kB, `gacct` on each level's stream):

| Source, level | All | Coefficients | End of block | Tx type | Tx split | Partition | Modes, MVs | Other | Intra area | No coefficient | Blocks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dragon Maid L0 | 71.2 | 39.0 | 5.7 | 2.5 | 1.1 | 3.8 | 17.0 | 2.0 | 0.1% | 98% | 22951 |
| L1 | 71.7 | 42.7 | 5.6 | 2.5 | 1.4 | 3.3 | 14.3 | 1.9 | 0.1% | 98% | 18403 |
| L2 | 57.3 | 32.0 | 4.8 | 2.5 | 1.3 | 3.3 | 11.7 | 1.6 | 32.0% | 66% | 18403 |
| F0 | 37.6 | 16.5 | 2.6 | 1.8 | 0.1 | 3.2 | 11.8 | 1.6 | 32.0% | 68% | 13660 |
| Angel Beats L0 | 147.6 | 113.2 | 9.6 | 4.0 | 1.2 | 3.7 | 14.0 | 1.9 | 2.1% | 38% | 15476 |
| L1 | 149.0 | 111.7 | 11.9 | 4.4 | 1.9 | 3.5 | 13.6 | 2.0 | 0.3% | 34% | 14808 |
| L2 | 119.9 | 84.5 | 11.5 | 4.7 | 2.4 | 3.5 | 11.9 | 1.4 | 27.3% | 31% | 14808 |
| F0 | 59.9 | 30.3 | 6.0 | 4.9 | 1.3 | 3.5 | 11.9 | 1.9 | 27.8% | 49% | 13765 |
| AM L0 | 75.3 | 39.9 | 10.2 | 5.4 | 2.1 | 4.9 | 12.4 | 0.3 | 0.8% | 43% | 21143 |
| L1 | 66.0 | 38.3 | 9.5 | 4.8 | 2.0 | 2.8 | 8.4 | 0.2 | 0.1% | 39% | 15607 |
| L2 | 62.6 | 35.2 | 9.1 | 4.6 | 2.3 | 2.8 | 8.6 | 0.1 | 0.9% | 40% | 15607 |
| F0 | 23.6 | 7.7 | 2.4 | 1.6 | 0.3 | 3.1 | 8.3 | 0.4 | 0.9% | 74% | 13731 |

**Split 128 blocks, measured.** On the ground of F0's 128 leaves, all 33 frames at q130 (accounting symbols by the block they are read in, so a partition symbol counts to the block before it), partition + modes + other, F0 / L2: 1.3 / 2.2, 1.7 / 2.2, 2.9 / 4.0 kB, so +0.9, +0.5, +1.1 kB. The rough estimate was 12, 3, 6. All bytes on that ground, F0 / L2: 1.3 / 3.9, 4.0 / 11.7, 9.2 / 25.2 kB: 81%, 39%, 30% of the area, under 3% of F0's bytes.

**Not expressible** (q130, all 33 frames, blocks and share of the frames' area):

| Kind | Dragon Maid | Angel Beats | AM |
|---|---|---|---|
| 128x128 split into 64x64s | 3140, 75.18% | 466, 22.50% | 371, 8.88% |
| 128x64 split | 516, 5.48% | 286, 7.70% | 1160, 13.63% |
| 64x128 split | 37, 0.44% | 376, 9.18% | 602, 7.21% |
| VERT superblock, its 64x64s in raster order | 90, 0.54% | 364, 4.66% | 1152, 6.90% |
| L2: NEAR, no candidate holds the MV | 187, 0.41% | 307, 1.09% | 1071, 2.15% |
| L2: skip mode, no candidate holds the pair | 42, 0.20% | 80, 0.35% | 557, 1.12% |
| L2: other compound mode lost | 1, 0.00% | 13, 0.06% | 76, 0.14% |
| L2: AMVD difference not in the set | 50, 0.02% | 72, 0.32% | 151, 0.30% |
| L2: block MV precision lost | 36, 0.01% | 38, 0.21% | 163, 0.29% |
| L2: MV moved by 1/8 pel, no predictor on its grid | 143, 0.63% | 138, 0.98% | 1279, 2.57% |

Same-reference compound, CfL, BAWP: none. Frames whose slots could not be reproduced: none. crowd_run q190 adds 6047 MHCCP blocks, 2.2% of the area, coded chroma DC: its avmenc has MHCCP and no CfL, CelV2's sequence the reverse.

- **Tiles are most of the L2 rows.** CelV2 codes these sources in two tile columns, avmenc in one, and on Dragon Maid and Angel Beats the first misses lie on that edge. With `--tile-columns 0 --tile-rows 0` Dragon Maid's MV misses fall from 0.46% of the area to 0.01%. That run against L2: -1.66, -1.30, -2.40%; against F0: +27.87, +51.26, +9.66%. The table above keeps L0's tiles.
- **The import holds.** L2's streams read back through the inspection decoder, leaves against F0's (q130, share of the area differing): MV 0.46, 0.80, 2.07%; mode 0.71, 1.23, 2.64%. The self-replay test, on a CelV2 stream with no tile edge, reads back identical.
- **Different from avmenc even so,** besides L3, L4 and the quantizer: delta q on key frames and the primary reference (L0 has neither), IBP, the CfL downsampling filter, CDF averaging, the tiles.
- **Departures from the spec.** ChunkCoder's `X_L0` loop drives `Pipeline`, as for L0, so no second driver holds the settings. `oracle_dump` is its own program: the inspection build's structures are not `avm.lib`'s. The switch is compiled out of a normal build (`#ifdef CELV2_ORACLE`). L2 also takes from the stream what its blocks need: MV precision and interpolation filter per frame, reference mode, skip mode flag, DRL limit, no global motion. Leaves the engines have no candidates for (4:1, 8:1, 4x4) take the square around them at L1. The spec's two tests are for L4: self-replay runs at L2 for now; `avm_testsrc2.ivf` cannot be replayed, it has extended partitions.

**L2b to L6, what is left after the modes** (one tile, from L4 on 8 bits and whole 128 blocks, see the departures; the same 12 streams, 33 frames, q100-190, BD-rate by global-MSE PSNR-Y):

| BD-rate | Dragon Maid | Angel Beats | AM |
|---|---|---|---|
| L2b against L2: one tile | -1.66 | -1.30 | -2.40 |
| L2z against L2b: `--rd-zero 1` | -1.73 | -7.48 | -2.95 |
| L3 against L2b: transform splits and types | -1.16 | -7.82 | +3.43 |
| L3 on 8 bits against L3 | -1.23 | +3.91 | +8.90 |
| L4 against L3 on 8 bits: filter settings | +0.15 | -7.77 | -2.35 |
| L4, whole 128 blocks, against L4 | -0.92 | +0.77 | -1.38 |
| L5 against L4: which blocks are zero | +0.75 | +4.12 | -0.68 |
| L6 against L5: the levels | -19.59 | -25.91 | -13.91 |
| L6o against L6: CelV2's own filters | +9.02 | +82.51 | +17.66 |
| **Against F0:** L2b | +27.87 | +51.26 | +9.66 |
| L2z | +25.93 | +39.69 | +7.07 |
| L3 | +26.22 | +34.33 | +12.74 |
| L4 (whole 128) | +23.62 | +26.22 | +17.15 |
| L5 | +25.07 | +36.39 | +16.53 |
| L6 | +0.55 | +1.04 | +0.33 |

- **The gap is the levels.** L4 stays +17 to +26% over F0, so L5 and L6 ran; L6 closes it to +0.33-1.04% with no other change. On every source the step that does it is L5 to L6, the levels in the blocks that are not zero.
- **The coder is not the gap.** L6 against F0 is under 2% on all three, so no coder spec by the spec's rule. By point, L6 / F0 bytes: Dragon Maid +0.2 to +1.2%, Angel Beats +0.3 to +2.0%, AM +0.1 to +0.9%, crowd_run q190 +0.2%. All frames, q130, kB, L6 / F0: coefficients 91.4 / 91.3, 250.8 / 250.6, 229.3 / 229.3; ends of block 13.2 / 13.1, 30.6 / 30.6, 44.1 / 44.1. Of what remains on Angel Beats, 1.9 of 2.5 kB is "other".
- **L6's check holds.** Whole 128 blocks: every frame of all 13 streams (crowd_run q190 too) reconstructs bit for bit as F0's, ratio 0, and equals `--recon`. With 128 blocks split, as the spec first said, it fails (L6 against F0 +1.92, +40.75, +38.73): BAWP's template and the deblocking inside a 128 block do not survive a split. Nothing else is left out: at L6 all the mode and MV kinds of the table above are 0.
- **Zeroing does not pay on avmenc's modes.** L5 against L4: no gain, so zeroing stays closed. It takes 10-25% of the coefficient bytes (q130, kB, L4 / L5: 131.7 / 111.3, 458.2 / 414.5, 382.6 / 288.2) and gives it back in quality. Read with care: with CelV2's levels, a forced zero cannot correct a neighbour one grey step off, and on flat areas the step copies on through 64x64s of zero blocks (Dragon Maid's key frame, q130: 53.94 dB at L5, 61.67 at L4, 62.36 at F0, 97% of the added error in blocks F0 leaves zero). `--rd-zero 1` remains worth -1.73 to -7.48% on avmenc's modes, where the one pass decides by itself.
- **8 bits cost CelV2.** L3 on 8 bits is +3.91 and +8.90% on Angel Beats and AM, so from L4 on the rows hold that much besides the gap. L6 is free of it: its levels are avmenc's.
- **L6o is one-sided.** Levels fitted to avmenc's filtered references do not fit CelV2's, so the error builds up through the frames and +9 to +83% is mostly that drift, not the filters. The clean point is the key frame, the levels exact before the filters: CelV2's own settings against avmenc's lose 0.00-0.32 dB of PSNR-Y at q100-190 (Dragon Maid 0.00, 0.12, 0.32, 0.23; Angel Beats 0.00, 0.01, 0.03, 0.05; AM 0.04, 0.11, 0.08, 0.13; crowd_run q190 0.11).
- **L3 and L4.** avmenc's transform splits and types pay on Angel Beats alone; AM loses at them. Its filter settings pay on Angel Beats and AM (-7.77, -2.35%) even on CelV2's reconstruction.
- **`--key-filter` is a no-op here.** `X_L0` never filters the key frame, so L2b's streams are byte for byte `l2t`'s. AM's `--key-filter 0` L0 would be the same stream as L0.

**Where L4 and L6 differ, the same transform blocks** (L6's levels are F0's; Dragon Maid, Angel Beats and AM together, whole 128 blocks; `dt.py`; luma, by the frame's class, transform's long side, intra or inter; share: of the extra sum of |level| over all planes; SSE: decoded against the source over the block; full tables, chroma and L5's in `l4-l6-blocks-2026-11-06.txt` in the shared notes):

| q130 | Area | F0 zero, L4 not | Nonzero, L4 / F0 | Sum of levels | SSE | Share |
|---|---|---|---|---|---|---|
| Hidden, inter, 32 | 2.8% | 31.4% | 2.12 | 1.60 | 0.70 | 22.8% |
| Shown, inter, 64 | 25.5% | 28.8% | 4.15 | 2.67 | 0.82 | 11.9% |
| Hidden, intra, 32 | 1.2% | 8.9% | 1.61 | 1.32 | 0.74 | 11.7% |
| Hidden, inter, 64 | 15.7% | 23.8% | 2.42 | 1.73 | 0.82 | 11.4% |
| Hidden, inter, 16 | 1.1% | 30.2% | 1.52 | 1.33 | 0.78 | 9.3% |
| Shown, inter, 16 | 0.5% | 25.1% | 2.09 | 1.70 | 0.70 | 5.6% |
| Hidden, inter, 8 | 0.5% | 20.0% | 1.20 | 1.12 | 0.85 | 5.5% |
| Shown, inter, 8 | 0.2% | 24.8% | 1.67 | 1.39 | 0.71 | 5.2% |
| Shown, inter, 32 | 1.5% | 31.7% | 2.94 | 2.12 | 0.75 | 5.0% |
| Key, intra, 8 | 0.7% | 4.8% | 1.10 | 1.05 | 1.02 | 4.1% |
| Hidden, intra, 8 | 0.2% | 15.1% | 1.23 | 1.15 | 0.82 | 4.1% |
| All planes | 100% | 15.2% | 1.44 | 1.24 | 0.87 | 100% |

| Sums, luma, L4 / F0 (L5 / F0) | Area at L4 | Nonzero q130 | Sum of levels q130 | SSE q130 | Nonzero q190 | Sum of levels q190 | SSE q190 |
|---|---|---|---|---|---|---|---|
| Hidden, inter | 20.0% | 1.68 (1.37) | 1.39 (1.21) | 0.79 (0.84) | 2.29 (1.78) | 1.81 (1.49) | 0.82 (0.77) |
| Shown, inter | 27.8% | 2.44 (1.58) | 1.84 (1.35) | 0.79 (0.83) | 3.84 (2.31) | 3.11 (2.01) | 0.85 (0.82) |
| Hidden, intra | 10.3% | 1.45 (1.29) | 1.25 (1.14) | 0.81 (1.04) | 1.96 (1.61) | 1.60 (1.38) | 0.80 (1.37) |
| Shown, intra | 6.6% | 1.55 (1.27) | 1.24 (1.12) | 0.82 (1.34) | 1.78 (1.49) | 1.55 (1.36) | 0.78 (1.57) |
| Key, intra | 2.0% | 1.14 (1.07) | 1.07 (1.03) | 0.97 (1.21) | 1.34 (1.16) | 1.21 (1.09) | 0.88 (1.15) |
| All luma | 66.7% | 1.58 (1.30) | 1.30 (1.15) | 0.79 (0.88) | 1.86 (1.49) | 1.51 (1.28) | 0.83 (0.96) |
| Chroma | 33.3% | 0.81 (0.77) | 0.90 (0.87) | 1.12 (1.38) | 0.81 (0.73) | 0.92 (0.85) | 1.19 (1.44) |

- **CelV2 keeps more coefficients at the same step, everywhere in luma,** and most in inter frames: at L4 58% more nonzero, 30% more level, for 21% less error. Over a quarter of the inter area is zero in F0 and not in L4. The excess grows with qindex (q190: 86%, 51%).
- **Same blocks nonzero, it still keeps more:** at L5 luma has 30% (q190 49%) more nonzero coefficients, so the rest of the gap is inside blocks. That is quantization: avmenc's levels at the same step, what its RDOQ with TCQ chooses, against `quantize()`.
- **By area it is 64 transforms, by the levels' excess 8 to 32.** 64s are 41% of the area and 23% of the extra level; 32 and smaller are 7% and 53% (inter).
- **Chroma goes the other way:** 19% fewer nonzero, 12-19% more error, so avmenc gives chroma what CelV2 gives luma. The BD figures here are luma only, so they favour CelV2 on this.
- Beside these: L4's references are not F0's, so its residuals differ too, and a block with no record on either side (skip_txfm at L5) is out of the table.

**Split side information** (`side.py`, Stage 3's F64 condition; split 128 blocks' side bytes as a share of F0's, q100, q130, q160, q190): Dragon Maid 0.19, 0.55, 1.52, 2.36%; Angel Beats -0.01, 0.16, 0.50, 1.74%; AM 0.02, 0.29, 0.87, 2.05%. Not under 1% at all four points, so F64 ran.

**F64, 128 blocks** (2026-11-07; F0's settings plus `--max-partition-size=64`, q100-190, BD-rate against F0): Dragon Maid +0.86%, Angel Beats +0.78%. Under +1% on both, so by the spec 128 blocks are not a lead; the numbers are in `tried-not-kept.md`.

**F1 and F2, AM's filter by halves** (2026-11-07; 33 frames, q100-190, BD-rate):

| Stream | Against avm0c | Against F0 |
|---|---|---|
| F0, both off | +10.15 | |
| F1, `--arnr-strength=0` | +9.88 | -0.24 |
| F2, `--enable-keyframe-filtering=0` | +9.10 | -0.96 |

- **F1 does not isolate the anchors.** `arnr_strength` is the strength of every temporal filter in avmenc, the key frame's too (`temporal_filter.c`), so F1 is both halves off: its key frame is 159.8 kB at q130 against F0's 160.6 and avm0c's 131.2. Only the hidden frame's structure stays on, and it is worth nothing. No flag turns the anchors off alone.
- **The anchors' half is about 1%:** F2 against F0, -0.96%, under the spec's 3%. No spec for filtered anchors.
- **The key frame's half is the rest:** avm0c against F2, -8.3%, with the anchors on in both. At q130 the key frame goes 160.6 to 131.2 kB (-18%), the first anchor 76.8 to 64.8 kB (-16%), the whole stream 374.5 to 329.9 kB.
- **CelV2's filter of the 2 frames after it:** the candidate d86dac0, AM, same frames, `--key-filter 1` against 0: +0.39%. It saves 7.6% of the key frame at q100 and 2 to 6% above, and loses as much in PSNR. So on AM CelV2's filter gives nothing where avmenc's gives -8.3%.
- **Two differences stand out.** Width: at AM's noise of 0.29 (`tried-not-kept.md`) avmenc's code asks for 13 frames, the key frame and 12 after it, against CelV2's 2; the count was read in the code, not logged. Share: avmenc's key frame is 40% of the stream at q130 (131.2 of 329.9 kB), CelV2's 18% (68.8 of 380.1 kB), so the same cut buys avmenc about twice as much.
- **Width alone did not do it before:** AVM's method on the key frame, up to 7 frames, was +1.04% on AM (2026-10-30, `tried-not-kept.md`), with the key frame's share as the reason it gave. Nothing here tests 12 frames, or a larger key frame with the filter.
- **Design's reading: no spec.** The share above is not the whole reason: the filter cures avmenc's frame 0 standing 2.9 to 8.8 dB above the rest, and CelV2's stands -1.1 to +2.6 dB without one. See "F1, F2's result" in the gap spec.

**crowd_run:** F0's q100, q130 and q160 streams have not landed, so no BD rows. q190 alone, bytes: L4 558194, L5 415182, L6 339280, F0 338538.

**Departures from the spec, L3 on:**
- From L4 on `--bit-depth 8`: avmenc's Wiener filter sets go by base qindex, and at 10 bits the same step is 48 higher, so avmenc's filters run on the wrong classifier (L4 at 10 bits breaks).
- Whole 128 blocks from L4 on (`X_ORACLE_128`), the spec's split rows kept for L4, L5, L6 as `l4e`, `l5e`, `l6e` (L4 +24.80, +25.04, +19.14; L5 +26.50, +36.10, +20.00 against F0). L6 needs them, and the table above needs the same blocks in L4, L5 and L6.
- L5 takes the prediction's tools with it (delta q, IBP, the CfL filter, FSC, TMVP's step), which the spec gives L6: it says "at avmenc's prediction", and without them Dragon Maid's L5 falls to 37-42 dB.
- TCQ's levels are not in the inspection buffers: `oracle_dump` links a copy of `decodetxb.c` CMake patches in the build directory; the AVM tree is untouched.
- CfL over more than one 64x64: the decoder predicts chroma after the last unit's luma, and the levels are written after the first. The replay reconstructs the luma first, unwritten.

**Tests** (`CELV2_ORACLE` build): `cli_oracle_replay` now runs L1 to L6; L4 and L6 give the first stream's bytes, every level decodes to its `--recon`. avmenc replay is the runs above: F0's 13 streams at every level, frames, references and `--recon` equal on all (no failure in 221 rows). The self-replay does not reach whole 128 blocks, which CelV2 does not code; F0's streams do.

**Q0, levels by the coder's real costs** (docs/superpowers/specs/2026-11-06-real-cost-trellis-design.md; `X_ORACLE_Q` at L4, 8 bits, whole 128 blocks, oracle build only; the same 12 streams, explicit `--rd-zero 0 --screen 0`: L4's 13 streams came back byte for byte from HEAD's oracle build. "Closed": L4 to L6 by log rate):

| Against F0, closed | Dragon Maid | Angel Beats | AM |
|---|---|---|---|
| L4, PSNR-Y | +23.62 | +26.22 | +17.15 |
| Qa: real costs, starting CDFs, avmenc's weights | +13.62, 40.8% | +13.99, 45.8% | +8.58, 49.0% |
| Qb: Qa and depth factors | +13.77, 40.2% | +14.47, 43.9% | +9.41, 44.1% |
| **Qc: Qa, CDFs each superblock** | +13.07, **43.2%** | +10.98, **57.9%** | +7.94, **52.8%** |
| Qd: Qc and the plain path | +12.37, 46.2% | +13.91, 46.1% | +12.93, 23.7% |
| Qa and the plain path | +12.99, 43.5% | +17.03, 34.0% | +13.72, 19.2% |
| L6 | +0.55 | +1.04 | +0.33 |
| L4, PSNR-YUV 6:1:1 | +29.74 | +35.46 | +30.39 |
| Qa | +16.58, 42.0% | +17.61, 48.2% | +14.85, 48.4% |
| Qb | +16.03, 43.8% | +17.28, 49.2% | +14.95, 48.1% |
| Qc | +16.14, 43.4% | +14.51, 57.3% | +14.11, 50.9% |
| **Qd** | +11.99, **57.7%** | +15.11, **55.5%** | +12.67, **55.7%** |
| L6 | +0.55 | +1.05 | +0.32 |

Same levels, share of the area (each stream's transform blocks, all planes, q100, q130, q160, q190):

| | Dragon Maid | Angel Beats | AM |
|---|---|---|---|
| L4 | 97.5, 97.8, 97.9, 98.7 | 37.3, 45.8, 56.5, 64.8 | 40.0, 49.6, 66.6, 79.5 |
| Qa | 98.2, 98.4, 99.1, 99.4 | 47.9, 56.9, 66.4, 73.0 | 56.1, 67.3, 78.9, 87.3 |
| Qb | 98.3, 98.5, 99.1, 99.4 | 50.8, 58.4, 67.5, 73.4 | 64.3, 71.3, 81.2, 88.9 |
| Qc | 98.2, 98.6, 99.0, 99.4 | 47.3, 56.9, 67.6, 73.9 | 57.2, 67.6, 79.1, 87.8 |
| Qd | 98.3, 98.7, 99.1, 99.4 | 45.1, 54.6, 67.5, 73.6 | 55.0, 67.0, 79.2, 87.8 |

- **By PSNR-Y, the spec's reading, no row reaches half on every source:** Qc is the best, 57.9% and 52.8%, with Dragon Maid at 43.2%. By the spec that brings `dt.py`'s table back rather than Q1. **By YUV 6:1:1, Qd does:** 55.5-57.7% on all three.
- **The shape of the price was most of it.** One change, real costs, takes 41-49% with no tuning. Nonzero levels at q130, luma, L4 to Qc: 1.31 to 1.17, 1.67 to 1.25, 1.60 to 1.11 times F0's, at 0.85 to 1.02, 0.70 to 0.93, 0.86 to 1.02 of its error.
- **Depth factors lose on Y** on all three (-0.6, -1.9, -4.9 points) and about tie on YUV. Bringing levels nearer F0's (AM 71.3% the same at q130 against 67.3%) does not pay here.
- **Refreshing pays on Angel Beats and AM,** +12.1 and +3.8 points on Y, +2.4 on Dragon Maid.
- **The plain path moves bits from luma to chroma.** Y loses on Angel Beats and AM (57.9 to 46.1, 52.8 to 23.7), YUV gains on all three. Chroma overshoots F0: nonzero 0.81 to 1.18 times F0's, error 1.13 to 0.98 at q130. So its weight is not yet avmenc's in effect (see the departures).

Where Qc and Qd still differ from F0, the three sources together (share: of the extra sum of |level| over all planes; luma but the last rows; full tables per source and qindex in `q0-blocks-2026-11-06.txt` in the shared notes):

| q130, Qc | Area | Same levels | F0 zero, Qc not | Nonzero, Qc / F0 | Sum of levels | SSE | Share |
|---|---|---|---|---|---|---|---|
| Hidden, inter | 20.0% | 67.7% | 9.4% | 1.16 | 1.10 | 0.98 | 39.4% |
| Hidden, intra | 10.3% | 74.6% | 4.1% | 1.22 | 1.15 | 0.99 | 37.2% |
| Shown, inter | 27.8% | 73.8% | 13.7% | 1.43 | 1.26 | 0.98 | 27.2% |
| Key, intra | 2.0% | 63.3% | 1.8% | 1.06 | 1.04 | 1.15 | 13.4% |
| Shown, intra | 6.6% | 83.9% | 3.9% | 1.28 | 1.13 | 0.98 | 2.7% |
| 32, inter | 4.3% | 66.9% | 8.4% | 1.32 | 1.18 | 0.95 | 23.5% |
| 8, intra | 0.9% | 44.2% | 4.0% | 1.10 | 1.06 | 0.99 | 21.2% |
| 32, intra | 1.7% | 39.1% | 5.4% | 1.23 | 1.14 | 0.97 | 19.0% |
| 64, inter | 41.2% | 72.3% | 12.5% | 1.36 | 1.21 | 0.99 | 15.2% |
| 8, inter | 0.7% | 47.9% | 8.1% | 1.13 | 1.08 | 0.97 | 14.3% |
| 16, inter | 1.6% | 63.9% | 7.7% | 1.18 | 1.12 | 0.96 | 13.5% |
| 16, intra | 1.2% | 55.5% | 7.4% | 1.12 | 1.08 | 1.03 | 9.3% |
| 64, intra | 15.1% | 84.3% | 3.3% | 1.28 | 1.14 | 1.03 | 3.8% |
| All luma | 66.7% | 72.7% | 9.6% | 1.18 | 1.11 | 0.98 | 119.9% |
| Chroma (Qc) | 33.3% | 89.5% | 2.8% | 0.81 | 0.90 | 1.13 | -19.9% |
| Chroma (Qd) | 33.3% | 86.9% | 6.6% | 1.18 | 1.12 | 0.98 | 15.0% |

- At the same error Qc still keeps 18% more nonzero luma, 11% more level (q190: 23%, 17%, SSE 1.02). Shown inter frames keep the most (q190: 1.85 times). The spec's first suspect, avmenc's rate multiplier per superblock from TPL, was not measured.
- All planes: same levels 68.6 to 78.3% at q130, 84.7 to 90.0% at q190.

**Departures in Q0:**
- Rates are -log2 of the CDFs, with no cost tables: the same numbers, and the row's CDFs need no build step.
- Each path carries its whole levels, so the contexts are exact; avmenc's diagonals are approximate. Slow, and marked as a shortcut.
- The weight leaves out `rd_boost_factor` (by `gfu_boost`, x1 to x1.5, not in the stream), as does the spec's table; and the transform type's bits in the all-zero choice (avmenc has them).
- The plain path is one path, round and one less by real costs, any end. avmenc starts from its own quantizer's output and only lowers it, so its chroma can only lose levels; this is the likely cause of the overshoot. Inter forward skip coding stays rough.
- Qc copies the row's CDFs at each superblock. The depth is rebuilt from the frame list (one past the deeper neighbour coded before, 5 at most).

**Tests:** on fixed CDFs the path's rate equals `SymbolLog`'s bits of the symbols written, to 1e-6, for every transform block, and again on the same coefficients x64, past the range and into the Golomb steps (`X_ORACLE_QCHECK`, in `cli_oracle_replay`). All 25 transform sizes, both quantizers, TCQ and plain, all three planes, on the test clip and on AM q130, Angel Beats q100, Dragon Maid q190. Every Q row decodes equal to `--recon`, 33 of 33 frames.

## The next fixes, ranked

Filled from stages 1-2: each fix worth 1% or more on average, its time factor, and whether it suits
`--preset max` or the default.
