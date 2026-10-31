# Scene chunks coded at once (`--workers`)

2026-10-31. Spec: `docs/superpowers/specs/2026-10-31-chunk-parallel-design.md`; plan:
`docs/superpowers/plans/2026-10-31-chunk-parallel.md`.

The reader classifies frames in order and starts a chunk at a keyframe once the one before holds 48
frames; `--workers N` ChunkCoders take the chunks in order and the output is written in chunk order.
Without a target size the bytes are those of one worker (tests: 1 / 2 / 4 workers, one scene only,
a scene longer than `--chunk-memory`, `--frames` ending inside a scene, `--preset max`).

Memory is capped at 80% of the installed RAM (`--memory-limit`), from a model of measured peaks
(AM 120 frames, `--refs 7`, peak private bytes):

| | default | `--parallel-b 0` |
|---|---|---|
| 1080p | 4.6 GiB | 1.9 GiB |
| 1080p `--preset max` | 8.3 GiB | 3.0 GiB |
| 2160p | 13.8 GiB | 5.0 GiB |
| 2160p `--preset max` | 26.0 GiB | 9.2 GiB |

Most of it is the copies of the encoder for parallel B frames (the same bytes without them), so the
cap first takes fewer workers, then turns parallel B frames off, then the GOP trial.

## The target size across chunks

The chunks can share the target-size control: chunk k sees the chunks up to k - workers, the
halving step is fitted to the coded frames (the first chunks start 0, +1, -1 qindex apart to
measure it), the last chunk is coded last and re-coded when the file is over. Measured with
`--workers auto` (3 here) on the target-size sources (q100 / q130 / q190 targets, % of the target,
ΔPSNR-Y against constant qindex at the same size), against the control in order:

| | in chunks | in order |
|---|---|---|
| 600-frame mixed anime | 83-99.9%, retries in 7 of 12, AIR -1.90 / -1.13 dB at q100 / q130 | 95.7-99.2%, no retries, within -0.15 to +0.14 dB |
| 120-frame clips | 30-99.1% (吉村 q100 30%, crowd_run q100 52%) | 86-99.6% |

Aiming at 99% or 99.5% instead of 98% did not change that. The fitted step and each chunk's start
mix up scene-to-scene differences with the qindex's effect. So with a target size `--workers auto`
(what the GUI passes) is one worker; an explicit `--workers N` still shares the control. Speed (one
worker against several on an idle machine) is still to be measured, and with it `auto`'s rule
(one worker per 16 hardware threads for now).

The team review of the branch found that whether the last chunk was re-coded, and which chunks ran
the GOP trial after a failure, depended on thread timing, that a failure's wake-up could be missed,
and that later chunks' output was held without bound; fixed in 3e43dba.
