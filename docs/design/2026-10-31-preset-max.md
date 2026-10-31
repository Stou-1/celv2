# `--preset max`: speed ignored for quality and size (stage 1)

2026-10-31. Roadmap step 7. Spec: `docs/superpowers/specs/2026-10-31-celv2-preset-max-design.md`;
plan: `docs/superpowers/plans/2026-10-31-celv2-preset-max-stage1.md`. The GUI's 「とても遅いが最高画質」
passes `--preset max`.

## What it sets

Defaults only: an option given on the command line wins, as with `--tune`.

| `--tune` | `--gop-trial` | `--gop-still` | `--deblock-search` |
|---|---|---|---|
| none | on | 0.5 | on |
| `animation` | on | 0.5 | off |
| `film` | off | film's 0.7 | on |

- `--gop-trial`: a mini-GOP that passes the still threshold is coded both as one 16-frame mini-GOP
  and as two of 8; the one of lower J = SSE (Y+U+V, 10-bit values) + 0.025·step²·bits is kept. The
  trial codes on its own thread while the main pipeline codes the other way (same bytes as coding
  them one after the other).
- `--deblock-search`: per frame, the deblocking delta in {-4, -2, 0, +2, +4} of the lowest squared
  error (Y, U, V), chosen before CDEF and CCSO search.

## Measurements

Twelve sources: 120 frames 1080p each of AM (from 30 s), Dragon Maid OP, AIR OP, chika ED (from
30 s), Angel Beats OP (720p, from 40 s), 吉村, crowd_run and park_joy, and four 600-frame stretches
that mix still and moving scenes (AM from 0 s, Angel Beats from 155 s, AIR from 54 s, Dragon Maid
from 65 s). q100-190, `--refs 7`; anime with `--tune animation`, live action with `--tune film`.
BD-rate against the same tune without `--preset`, PSNR-Y and VMAF NEG (`vmaf_v0.6.1neg`); negative
is smaller.

| source | everything on: PSNR | VMAF | as shipped: PSNR | VMAF |
|---|---|---|---|---|
| AM | +0.02% | +0.41% | +0.00% | +0.00% |
| Dragon Maid | -1.04% | +3.82% | -1.12% | -0.34% |
| AIR | -0.12% | +0.56% | +0.00% | +0.00% |
| chika | -1.02% | +0.02% | +0.00% | +0.00% |
| Angel Beats | -0.29% | +1.60% | +0.00% | +0.00% |
| AM mixed | -2.47% | -1.18% | -2.39% | -1.10% |
| Angel Beats mixed | -2.33% | +2.58% | -1.94% | -0.83% |
| AIR mixed | -2.77% | +0.18% | -2.40% | +0.03% |
| Dragon Maid mixed | -2.14% | +0.52% | -2.01% | +1.15% |
| 吉村 | +0.80% | +2.10% | -0.07% | +1.36% |
| crowd_run | -0.02% | -0.34% | -0.02% | -0.34% |
| park_joy | -0.21% | -4.06% | -0.21% | -4.06% |

"As shipped" is the table above: anime without the deblocking search, live action without the GOP
trial. On the 120-frame anime clips the trial kept the default GOP every time (bytes identical);
its gains are in the mixed stretches, where still scenes meet moving ones.

How the per-tune split was chosen (the rule: PSNR BD-rate down and VMAF NEG not worse):

- The deblocking search lowers PSNR BD-rate on anime but raises VMAF NEG BD-rate: Dragon Maid
  -1.04% / +3.82% with it against -1.12% / -0.34% without, Angel Beats mixed -2.33% / +2.58% against
  -1.94% / -0.83%. On live action it is the other way round (park_joy VMAF -4.06%). Off for
  `--tune animation`.
- The GOP trial on 吉村 takes 8+8 where 16 was the default: 13% more bits for 0.6 dB, a loss in
  BD-rate (+0.80% PSNR, +2.10% VMAF). Weighting the trial's bits ×2 changes nothing; ×4 gives 吉村
  -0.75% PSNR but +2.46% VMAF and shrinks the anime gains (Dragon Maid +0.15%, AM mixed -0.52%). The
  weight stays ×1 and the trial is off for `--tune film`.
- The deblocking search on live action, as shipped, is better in PSNR on all three and in VMAF NEG
  on crowd_run and park_joy, but 吉村 is +1.36% in VMAF NEG (it takes 9-13% fewer bits at each q for
  0.4-0.5 dB less; VMAF 76-97, not saturated). Kept for `--tune film` on the average (PSNR -0.10%,
  VMAF -1.01%); revisit if more calm live-action sources (the kind `--tune film` is for) go the
  way of 吉村.
- Dragon Maid mixed, as shipped, is -2.01% in PSNR but +1.15% in VMAF NEG. Frame by frame the trial's
  frames are no worse in either (frame 592: VMAF 94.47 → 95.00, PSNR-Y 46.03 → 47.34 dB at the same
  q; no visible difference): VMAF NEG saturates at 95-97, so a choice of more bits and higher quality
  reads as a BD-rate loss there.

All three are rows in `tried-not-kept.md`.

Time: the twelve sources at four q took 2505 s by default and 3783 s with everything on (1.5×),
with the trial coded after the main pipeline. Coding the two at once is 1.1× (AM) to 1.3-1.4×
(吉村) faster than that, with the same bytes (four files checked).

## Next

- Stage 2 (its own spec): block decisions by the real coding cost instead of the RD pass's
  estimate. Most of avmenc's gap between cpu-used 6 and 0 (14-20% on anime) is there.
- CPU use: measure one encode's CPU use on an idle machine, then scene-keyframe chunks in parallel
  (tested for the same bytes).
