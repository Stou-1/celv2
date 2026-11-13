# C7(a): a short shot's key frame boost scaled by its length (2026-11-09)

Candidate C7(a) of docs/reports/2026-11-08-improvement-candidates.md. Against c9c7e18 (C4(A) and G2.1), RTX 5090.

## Verdict

**Kept. BD-rate (global PSNR-Y, q100-190, `--refs 7`): S1 -5.97%, S2 AIR -1.91%, Angel Beats -1.13%; no CPU.**

| Source | Frames | BD-rate |
|---|---|---|
| S1 (AM 600, mxam600) | 600 | -5.97% |
| S2 AIR (s2_air600) | 600 | -1.91% |
| Angel Beats | 120 | -1.13% |
| AIR, Dragon Maid, AM, chika, 吉村, crowd_run, park_joy | 120 | the same bytes (no shot under 9 frames) |

The candidates report estimated S1 -1.5 to -3% and S2 AIR -0.5 to -1.5%; the measured gain is about twice that.

## The change (chunk_coder.cpp, the key frame's qindex)

- A key frame's qindex goes down by TPL's delta (`--tplq`), up to about -72, for the frames after it that reuse it.
- A shot shorter than 9 frames (`ahead` < 8 before the next key frame) now takes (ahead + 1) / 9 of that delta.
- Why: S1's two-frame shots had keys of 143 and 138 kB at q130, 6.4% of the file, for one frame of reuse.
- The tried-not-kept row 47 (key depth by q) named this as its revisit condition.

## Not checked

- VMAF and XPSNR (the candidates report asks for them). At the same q, S1 is 3.5-8% smaller and 0.05-0.09 dB lower in
  global PSNR-Y; whether the short shots' keys look worse was not looked at.
- A shot of 9 frames or more keeps the full boost: the scale is a step at 9.

## Reproduce

```bash
BD_SOURCES=s1:mxam600.y4m:1,s2air:s2_air600.y4m:1 python -I tools/c17/bd_run.py old.exe new.exe sources work 600
```
