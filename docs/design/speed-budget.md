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
| `--cfl-rd 1` by default (from the ablation below) | ec9911d | +0.4% | +1.0% / not run | not run | not run | CPU a frame; 吉村 -1.5%, inside the noise. fps -1.6, -2.4% |
| MHCCP with CfL under `--tune animation` (from the ablation below) | f87378e | +0.1% | +0.5% / not run | not run | not run | CPU a frame; fps -0.3, -1.2% |
| `--grain 25` under `--tune film` (from the ablation below) | 2fd4713 | | | | | 吉村 only: CPU a frame +1.0%, fps -5.8%; `--grain 0` is +25.59% bytes at q130. Denoising 335 ms a frame, and coding the denoised picture saves 291 ms |
| The RD pass's side information priced from the frame's CDFs (partition, inter flags, reference, mode), bit weight 0.03 | (this commit) | +0.3% | -0.4% / not run | not run | not run | CPU a frame against 27dfe00, no offset counted. S1: all 9 rounds pooled (medians), fps -0.1%; 6 rounds with the order alternating, load 72%: CPU -0.1%, fps -0.2%; a first 3 rounds under 87% load gave CPU +0.8%, fps -5.4% (the load's spread wider than the change). S2 AIR (3 rounds, 87%) fps +7.0%; 吉村 CPU +0.4%, fps +4.0% |

## Saved

| Change | Commit | S1 | S2 AIR / DM | S3 | S4 | Note |
|---|---|---|---|---|---|---|
| quantize_tcq()'s Viterbi on AVX2, levels and previous states carried beside the costs | 2509e94 | -10.1% | -7.0 / -10.3% | -7.2% | -9.9% | CPU a frame, the same bytes on all six; 吉村 -6.7%; fps +3.7 to +8.1%. Under load, old and new alternating. The RD pass calls it per transform: 36% of the CPU before |
| The RD pass's `tx_cost()` memo, keyed by the residual (speed-first C20; memos kept for the process: one per thread cost -20% fps) | ff0d03b | -13.8% | -5.8 / -6.8% | -3.3% | -16.1% | CPU a frame, the same bytes on all six; 吉村 -1.8%. fps +2.1, +3.4 / +2.3, +3.1, +10.5%, 吉村 -0.7% (noise). One exe, `CELV2_TX_MEMO=0` against on, alternating, 3 rounds (S3, S4: 1). fps moves less than CPU: S1 ran 28.8 of 48 threads busy |
| Intra prediction a block at a time from its edges, in the RD pass (speed-first C18 c) | 50b683d | -0.1% | -2.4 / -1.7% | -2.3% | -22.6% | CPU a frame, the same bytes on all six; 吉村 -0.5%. fps -0.8, +2.6 / +0.1, +2.7, +20.3%, 吉村 0.0%. Old and new exe alternating, 3 rounds (S3, S4: 1). S4 gains most: `--rd-intra` scores every mode of every block |
| The RD pass's residuals 4 or 8 at a time and the compound average by `_mm_avg_epu8`; the cost sums' divisions 4 at a time, added in order; the level bits from a table here (speed-first step 3) | 89ca48c | -7.2% | -10.1 / -9.2% | -10.6% | -11.2% | CPU a frame, the same bytes on all six; 吉村 -8.6%. fps -2.9 (rounds overlap: 9.96-10.80 against 9.70-10.80), +0.2 / +6.3, +6.6, +13.1%, 吉村 +2.3%. With the `CELV2_RD_TIMES` timers built in (off) |
| `fwd_dct()`: the lane count fixed when compiled (4-64), 4x4 transposes where 8x8 do not divide, the int16 to float conversion and the final rounding 4 or 8 at a time (speed-first step 3) | e1782ee | -6.5% | -8.5 / -7.1% | -8.2% | -6.8% | CPU a frame, the same bytes on all six; 吉村 -6.6%. fps +2.6, +3.6 / +1.2, +5.1, +3.6%, 吉村 +5.0%. `CELV2_RD_TIMES` on S1: fwd_dct 540 to 368 ms a frame |
| The Vulkan motion search with a workgroup of 64 threads a block, for frames up to 4K (speed-first C23; `me_wide.slang`, the same fields; larger frames keep one thread a block) | (this commit) | 5090 -0.5% / B50 +13.6% fps | | 5090 +1.6% / B50 +7.3% fps | | fps, the same bytes; against a fresh build of bd3fad8, alternating, 3 rounds (S3: 2). engine_bench 1080p 10 frames: me 1600 to 700 ms (B50), 84 to 26 (5090). A reference binary built earlier from the same source read about 15% faster on the B50 at 16K than a fresh build: compare fresh builds only |
| Vulkan's waiting (the wait spec's V1, V2 with one queue per GPU, V3) | 84617c4 | gap 0.4% | gap 3.1% / +5.2% | +1.4% | | fps. S1 and AIR: the gap to CUDA, down from 6.7 and 7.6%; DM and S3 against V1; the B50 +59%. The user accepted the 3.1% on 2026-11-08 and CUDA was removed. docs/reports/2026-11-05-vulkan-only.md |
| `fwd_dct()` with AVX-512 for sizes with both sides >= 16, chosen by CPUID (`CELV2_SIMD=avx2` turns it off) | 636e9a6 | -10.0% | -11.5% / not run | not run | not run | CPU a frame, the same bytes (5 md5s, 9 sizes in the unit test); 吉村 -7.7%. fps 10.23 to 11.08 (S1). Banked: not an offset for any capacity item (user, 2026-11-12); those are measured against 636e9a6 or later. Machines without AVX-512 do not get it |

## Against v0.1.0 (2026-11-06, aa3631c, quiet machine, 2 rounds alternating)

| Run | v0.1.0 fps | Now fps | Time | Size at q130 |
|---|---|---|---|---|
| S1 | 15.24 | 10.13 | x1.50 | -30.47% |
| S2 AIR | 16.14 | 9.47 | x1.70 | -24.69% |
| S2 Dragon Maid | 11.65 | 8.18 | x1.42 | -20.32% |
| ref 吉村 | 7.86 | 5.01 | x1.57 | -30.35% |

Size at one qindex is not BD-rate: the quality differs. 吉村's row is not compression: `--tune film` now takes `--grain 25` (2fd4713), so it codes the denoised picture, and its time holds the denoising. v0.1.0 printed no CPU a frame. BD-rate over 33 frames, without `--rd-zero`: mean -10.25%.

## One tool off at a time (2026-11-07, aa3631c, one build, quiet machine)

Default against each tool off, ledger's settings, each run twice, tools in turn and reversed, medians. The default's four runs spread by 1-2% in fps and under 1% in CPU a frame on S1 and S2, 2% and 4% on 吉村. CPU a frame / size at q130 with the tool off, against the default (size is not BD-rate):

| Off | S1 | S2 AIR | 吉村 |
|---|---|---|---|
| Default: fps, CPU a frame | 10.18, 2.94 s | 9.50, 3.34 s | 4.92, 4.27 s |
| `--lr 0` | -2.9%, +0.03% | -2.2%, +0.21% | -1.2%, +0.35% |
| `--lr-uv 0` | -2.7%, -0.17% | -2.9%, -0.10% | -1.2%, -0.12% |
| `--intra64 0` | -2.8%, -0.02% | -5.4%, +0.21% | -1.1%, +0.14% |
| `--key-filter 0` | -1.8%, +2.66% | -2.7%, +0.36% | not on |
| `--cctx 0` | -0.6%, +0.06% | -0.8%, +0.10% | -0.2%, -0.02% |
| `--cfl-rd 0` | -0.4%, +0.23% | -1.0%, +0.02% | +1.5%, +0.12% |
| `--mhccp 0` | -0.1%, +0.21% | -0.5%, +0.01% | not on |
| `--intra-txp 0` | +0.6%, +0.14% | -0.1%, +0.26% | +0.8%, +0.16% |
| `--ist 0` | +0.1%, -0.20% | -0.4%, -0.86% | +0.5%, -0.61% |
| `--paq 0` | +0.8%, +0.39% | -0.2%, -0.25% | +1.3%, +1.16% |
| `--screen 0` | +0.4%, -0.00% | -1.1%, -0.00% | -1.3%, -0.00% |
| `--rd-zero 0` | +0.3%, +7.62% | -0.1%, +5.10% | not on |
| `--grain 0` | not on | not on | -1.0%, +25.59% |

No tool costs more than 5.4% CPU; the heavy stages are the coder's own transforms and quantization, not any one option.

CPU time per stage, ms a frame, 132a801 against aa3631c at `--paq 0` (吉村's holds `--grain 25`: denoising 335 ms a frame):

| Stage | S1 | S2 AIR | 吉村 |
|---|---|---|---|
| quantization | 1155 -> 793 | 902 -> 689 | 1060 -> 664 |
| transforms | 963 -> 944 | 1194 -> 1230 | 1289 -> 1293 |
| RD pass | 542 -> 607 | 765 -> 781 | 1414 -> 1414 |
| coder | 112 -> 123 | 99 -> 116 | 112 -> 125 |
| CDEF | 108 -> 102 | 110 -> 107 | 121 -> 127 |
| loop restoration | 99 -> 96 | 96 -> 98 | 99 -> 105 |
| CCSO | 70 -> 66 | 69 -> 66 | 82 -> 86 |
| decision (CPU side) | 38 -> 11 | 34 -> 11 | 39 -> 10 |
| TPL | 22 -> 28 | 27 -> 30 | 47 -> 52 |
| Share: quantization + transforms | 66.4 -> 58.7% | 61.1 -> 57.6% | 54.5 -> 45.2% |

## Why fps fell while CPU a frame fell too

132a801 to aa3631c: S1 CPU a frame -8%, fps -6%; S2 AIR -3%, -10%. The rows above did not add up to that. aa3631c on one build, S1 and S2 AIR, 2 rounds alternating, medians (off: `--paq 0 --rd-zero 0 --screen 0 --cfl-rd 0 --mhccp 0`, which gives 132a801's bytes exactly: 4406720 and 4290984):

| | S1 fps, cores busy | S2 AIR fps, cores busy |
|---|---|---|
| 132a801 | 10.85, 34.6 | 10.50, 36.0 |
| aa3631c off, CUDA | 11.27, 33.1 | 10.83, 35.2 |
| aa3631c off, Vulkan | 10.59, 30.7 | 9.95, 32.4 |
| aa3631c default, CUDA | 10.92, 32.5 | 10.22, 34.0 |
| aa3631c default, Vulkan | 10.19, 29.9 | 9.44, 31.7 |

- **The rows now add up.** The same bytes on CUDA are +3.9 and +3.1% over the baseline: the TCQ row's gain. From there Vulkan takes 6.0 and 8.1%, the new defaults 3.1 and 5.6%.
- **Vulkan costs 6.7 and 7.6% today** (default, CUDA against Vulkan), not the row's 3.3 and 4.8%, which was measured under load. Above the 5% the user waived on 2026-11-05. The CPU time is the same, so it is waiting: 2.3 to 2.6 cores less busy, not more work.
- **The new defaults cost 3.8 and 5.1% fps for 1.1 and 3.2% CPU a frame.** Wall time moves more than CPU here: chunks on 3 workers wait for the longest one, so a small change shifts the end.

## Balance against the baseline

Spent: about +1% CPU a frame (`--paq`), +0.5 to +1.0% (`--rd-zero` on anime), the other new defaults under 1% each; 6.7-7.6% fps at 1080p on a quiet machine (Vulkan; after the wait spec 0.4 and 3.1%, accepted by the user on 2026-11-08, and CUDA removed). Saved: 7-10% CPU a frame (the TCQ search); the speed-first spec's steps 2, 1 and 3 (2026-11-08/09, ff0d03b to e1782ee, the same bytes): S1 2.995 to 2.129 s a frame (-29%), S2 AIR 3.384 to 2.563 (-24%), S2 DM 3.712 to 2.758 (-26%), 吉村 4.204 to 3.231 (-23%), S3 14.5 to 10.8 (-26%), S4 16.7 to 9.04 (-46%), from the memo A/B's off rows to the DCT A/B's new rows (separate runs). Stage A starts with quantization and the
forward transforms.
