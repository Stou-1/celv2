# C3: a partial scene change is a P frame (2026-11-09)

Candidate C3 (variant B) of docs/reports/2026-11-08-improvement-candidates.md. Against f17a2da (C7(a)), RTX 5090.

## Verdict

**Kept. BD-rate (global PSNR-Y, q100-190, `--refs 7`, `--tune animation`): S1 -1.18%, S2 AIR +0.04%; no CPU.**

| Source | Frames | BD-rate |
|---|---|---|
| S1 (AM 600, mxam600) | 600 | -1.18% |
| S2 AIR (s2_air600) | 600 | +0.04% |
| AIR, Dragon Maid, AM, chika, Angel Beats, 吉村, crowd_run, park_joy | 120 | the same bytes (no cut leaves 25-50% still) |

The candidates report estimated S1 -0.5 to -3% and S2 AIR -0.1 to -0.3%.

## The change (scene_cut.cpp)

- A frame was a cut only when at most half of its 8x8 cells stay within 2 levels of the frame before.
- Now at most a quarter: a change that leaves 25-50% of the picture still is coded as a P frame of the same shot.
- S1 has 15 cuts; 10 of them left 0.23-0.48 still (frames 223, 226, 402, 440, 515, 523, 533, 535, 545, 551; `CELV2_GOP_LOG=1`).
- Test: `partial_change_is_not_a_cut` (60% of the picture changed).
- The default build's S1 q190 stream equals the measured one (md5).

## Not checked

- VMAF and XPSNR. At the same q S1 is 2.1-3.9% smaller and 0.05-0.16 dB lower in PSNR-Y.
- C3 step 2 (a shot coming back to an earlier one) stays dropped: 0-2 returns a source.

## Reproduce

```bash
CELV2_GOP_LOG=1 celv2enc -i mxam600.y4m -o s1.ivf --qindex 190 --refs 7 --tune animation 2> gop.txt
BD_SOURCES=s1:mxam600.y4m:1,s2air:s2_air600.y4m:1 python -I tools/c17/bd_run.py old.exe new.exe sources work 600
```

The spike read the threshold from `X_CUT_STILL` (`NEW_ENV=X_CUT_STILL=0.25`); it is not in the tree.
