# The speed ledger

docs/superpowers/specs/2026-11-04-speed-budget-design.md. Whole-video speed with `--workers auto
--refs 7 --qindex 130`, the CPU about full; the main number is the process's CPU time a frame
(`CELV2_PROFILE=1`). Machine: Xeon 658X (48 threads), RTX 5090. Sources in celv2-bench\sources.

| Run | Source | Settings |
|---|---|---|
| S1 | `mxam600.y4m`: AM, 600 frames 1080p, 6 scenes | `--tune animation` |
| S2 | `s2_air600.y4m` (AIR OP from 54 s), `s2_md600.y4m` (Dragon Maid OP from 65 s), 600 frames 1080p each; Angel Beats from 155 s: its source is not on this machine | `--tune animation` |
| S3 | `g_am30_4k.y4m`: AM 4K, 120 frames | `--tune animation` |
| S4 | as S1 | `--tune animation --preset max` |
| ref | `g_yo.y4m`: 吉村, 120 frames (one scene) | `--tune film` |

## Baseline (132a801, 2026-11-04, `--paq 0`)

| Run | fps | CPU a frame | GPU busy |
|---|---|---|---|
| S1 | 10.85 | 3.19 s | 37% |
| S2 AIR / Dragon Maid | 10.50 / 8.54 | 3.43 / 3.91 s | 35 / 36% |
| S3 | 2.29 | 14.86 s | 23% |
| S4 | 2.46 | 18.22 s | 15% |
| ref 吉村 | 5.71 | 4.31 s | 24% |

CPU time per stage, share of the process's (S1 / S2 AIR / S2 DM / S3 / S4 / 吉村):

| Stage | Share |
|---|---|
| quantization (TCQ included) | 36.2 / 26.3 / 35.5 / 23.9 / 41.0 / 24.6% |
| transforms (forward) | 30.2 / 34.8 / 31.8 / 34.9 / 35.4 / 29.9% |
| RD pass (its own work) | 17.0 / 22.3 / 19.6 / 28.4 / 19.5 / 32.8% |
| coder (its own work) | 3.5 / 2.9 / 2.9 / 3.4 / 0.5 / 2.6% |
| CDEF | 3.4 / 3.2 / 3.0 / 0.9 / 1.1 / 2.8% |
| loop restoration | 3.1 / 2.8 / 2.4 / 2.9 / 1.0 / 2.3% |
| CCSO | 2.2 / 2.0 / 1.7 / 1.9 / 0.7 / 1.9% |
| decision (CPU side) | 1.2 / 1.0 / 1.0 / 0.8 / 0.3 / 0.9% |
| TPL | 0.7 / 0.8 / 0.8 / 1.3 / 0.2 / 1.1% |
| reading, analysis, global motion, deblocking, entropy coding | under 0.5% each |

The stages add up to 98% of the process's CPU time. Stage A's order follows: quantization and the
forward transforms (55-76% together), then the RD pass's own scoring; the in-loop filters together
are 3-9%.

## Spent

| Change | Commit | S1 | S2 AIR / DM | S3 | S4 | Note |
|---|---|---|---|---|---|---|
| `--paq 0.25` by default | 1502e61 | +0.9% | +0.6 / +1.0% | -0.1% | -0.5% | CPU a frame; 吉村 +2.1%. The map (stage "analysis") 26-58 ms a frame, about 1%; the rest is coding the offsets. fps 10.84 / 10.27 / 8.54 / 2.31 / 2.48 |
| Vulkan as the default GPU (fps, not CPU a frame; under load) | 97fbf2b | -3.3% | -4.8 / -2.6% | -1.3% | -1.1% | a16k -0.7%. No offset: the user waived the 5% gate on 2026-11-05. docs/reports/2026-11-05-vulkan-only.md |
| `--rd-zero 1` under `--tune animation` (on one build, 0 and 1 alternating, 3 rounds, medians) | 0921e4a | +0.5% | +1.0 / +0.8% | not run | 0.0% | CPU a frame, against a bound of +2% on S1 and S2; S4 reported only. fps -0.4, +0.5, +0.4, +0.3% |
| `--screen auto` by default, against `--screen 0` (one build, alternating, 3 rounds, medians, with `--rd-zero 1`) | aa3631c | -0.7% | +0.2 / +0.6% | not run | 0.0% | CPU a frame, against the spec's 3%. fps +0.3, -3.5, -3.9, -0.6%. Sizes +61 to +73 bytes in 4-6 MB. Angel Beats 600: its source is not on this machine, unconfirmed |

## Saved

| Change | Commit | S1 | S2 AIR / DM | S3 | S4 | Note |
|---|---|---|---|---|---|---|
| quantize_tcq()'s Viterbi on AVX2, levels and previous states carried beside the costs | 2509e94 | -10.1% | -7.0 / -10.3% | -7.2% | -9.9% | CPU a frame, the same bytes on all six; 吉村 -6.7%; fps +3.7 to +8.1%. Under load, old and new alternating. The RD pass calls it per transform: 36% of the CPU before |

## Against v0.1.0 (2026-11-06, aa3631c, quiet machine, 2 rounds alternating)

| Run | v0.1.0 fps | Now fps | Time | Size at q130 |
|---|---|---|---|---|
| S1 | 15.24 | 10.13 | x1.50 | -30.47% |
| S2 AIR | 16.14 | 9.47 | x1.70 | -24.69% |
| S2 Dragon Maid | 11.65 | 8.18 | x1.42 | -20.32% |
| ref 吉村 | 7.86 | 5.01 | x1.57 | -30.35% |

Size at one qindex is not BD-rate: the quality differs. 吉村's row is not compression: `--tune film` now takes `--grain 25` (2fd4713), so it codes the denoised picture, and its time holds the denoising. v0.1.0 printed no CPU a frame. BD-rate over 33 frames, without `--rd-zero`: mean -10.25%.

Not in this ledger, default changes with no row yet: `--cfl-rd` (ec9911d), MHCCP with CfL (f87378e), film's `--grain 25` (2fd4713). The rows after the baseline do not add up to today's fps by 1-11% (design session's estimate): an ablation on one build is planned for when the CPU is free.

## Balance against the baseline

Spent: about +1% CPU a frame (`--paq`), +0.5 to +1.0% (`--rd-zero` on anime); up to 4.8% fps at 1080p (Vulkan, waived). Saved: 7-10% CPU a frame (the TCQ search). Stage A starts with quantization and the
forward transforms.
