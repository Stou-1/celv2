# Target size (two passes)

2026-10-31. Spec: `docs/superpowers/specs/2026-10-31-target-size-design.md`; plan:
`docs/superpowers/plans/2026-10-31-target-size.md`.

`celv2enc --pass 1 --stats F` writes each IVF frame's bytes; `--pass 2 --stats F --target-size B`
picks one qindex for the whole video from them (24 qindex = half the bytes, 98% of B aimed at) and
moves it by at most 2 a mini-GOP as the bytes come in; over B it exits with 3 and the line
`celv2enc: over --target-size by N bytes` (`even at qindex 255` when every frame was at 255). The
GUI's 「サイズ指定（MB）」 runs pass 1, makes the audio, gives pass 2 the size less the audio and
MP4's boxes (1024 + 5 a frame), and when the whole is over, runs pass 2 again with the budget less
the excess and 1% (3 times at most; a retry over its smaller budget but within the target is kept).

## The quick look (pass 1)

Pass 1 turns off what costs most unless given: `--gop-trial 0 --deblock-search 0 --rd-pass 0`.
Measured at q150 on four 600-frame mixed anime stretches and 吉村, with `--preset max` (anime
`--tune animation`, 吉村 `--tune film`): the real pass's bytes over the candidate's, per 120-frame
stretch, max / min.

| pass-1 settings | AM | Angel Beats | AIR | Dragon Maid | 吉村 | time |
|---|---|---|---|---|---|---|
| trial and deblocking search off | 1.157 | 1.063 | 1.053 | 1.086 | 1.000 | 1682 s |
| and the RD pass off (kept) | 1.225 | 1.083 | 1.080 | 1.169 | 1.000 | 878 s |
| and `--refs 4` | 1.129 | 1.592 | 1.073 | 1.159 | 1.000 | 994 s |
| the real pass | | | | | | 1812 s |

The second came out as close to the target as the first (below), at half the time; `--refs 4`
changes the GOP structure.

## Results

Twelve sources with the shipped `--preset max` (anime `--tune animation`, live action `--tune film`):
eight of 120 frames 1080p and four 600-frame mixed stretches; three targets each, 0.9 x the
constant-qindex size at q100, q130 and q190 (the curves of `docs/design/2026-10-31-preset-max.md`).
Pass 1 at q150. Size as % of the target after the GUI's retries; ΔPSNR-Y against the constant-qindex
curve at the same size (log-size interpolation). The mixed stretches with the RD pass off in pass 1:

| source | q100 target | q130 target | q190 target |
|---|---|---|---|
| AM mixed | 95.7%, +0.14 dB | 97.3%, +0.01 dB | 98.8%, -0.10 dB |
| Angel Beats mixed | 98.1%, -0.15 dB | 97.9%, +0.06 dB | 98.1%, +0.13 dB |
| AIR mixed | 97.7%, +0.01 dB | 97.5%, +0.05 dB | 98.0%, -0.04 dB |
| Dragon Maid mixed | 99.2%, +0.08 dB | 98.9%, +0.03 dB | 98.7%, -0.10 dB |

No retries; pass 1 took 0.25-0.45 of pass 2's time (1.25-1.45x in all). With the RD pass on in pass
1 the same stretches came out at 96.0-99.1% and within -0.30 to +0.16 dB.

The 120-frame clips (4-5 s) are harder: 86-99.6% of the target, all within it, but 6 of 21 needed a
retry (up to 3 more pass 2s) and the q100 targets lost up to 0.83 dB (AM -0.83, AIR -0.41,
Dragon Maid -0.38 dB). Two causes:

- Bytes halve every 16 to 35 qindex across these sources (q100-190), not 24; from pass 1's q150 to a
  q100 target the estimate is off by up to a third, and a short clip has few mini-GOPs to correct it.
- A keyframe is a large share of a short clip, and it is coded before any correction.

The scene-chunk plan (`docs/superpowers/specs/2026-10-31-chunk-parallel-design.md`) fits the halving
step to the video and re-codes only the last chunk when over; a pass 1 at a qindex near the target
is the other candidate.

## Found by the measurement

The GUI counted a retry over its smaller budget as a failure even when the whole output fitted the
target (AM and chika at q190: 92.6% / 87.6% after 3 retries); fixed in 26910d8.
