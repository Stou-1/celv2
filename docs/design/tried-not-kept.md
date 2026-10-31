# Tried and not kept

Every measure that was built or measured and then dropped, with why, and what would make it worth
trying again. Add a row whenever something is dropped; look here before planning, and again after
any change named in a "revisit when" cell. BD-rate: PSNR-Y, negative is better. Details are in the
note named in the last column.

## Coding tools

| Measure | Result | Why not kept | Revisit when | Notes |
|---|---|---|---|---|
| Local warp, WARP_CAUSAL on compound blocks | anime B -0.03%, live action +0.39% | No gain; the decoder's projection from the neighbours rarely fits | The warp search improves (see the next rows) | bit-allocation "local warp" |
| Single-reference warp, WARP_NEWMV + WARP_DELTA (branch feat/warp, `--warp 1/2`) | -0.24% / +0.04%; with MV moves +0.06% / -0.26% | Within noise, up to ~40 RD costs a block. avmenc loses +1.5% (anime) / +3.2% (live action) without warp-delta, so CelV2's search, not the tool, is short | Warp models from the GPU motion field, or the parallel RD pass makes many trials cheap | bit-allocation, feat/warp |
| CfL with the derived scale | anime B Y -0.12%, live action U +1.5%, V +2.7% | The uv mode is coded before the luma is reconstructed, so the choice used the source luma and erred | The coder codes a block's luma before writing its mode info | bit-allocation "CfL" |
| Temporal MVs (enable_ref_frame_mvs, `--tmv`) | 2026-10-27: anime A -2.1%, two other anime sources +0.9 / +1.2%, clips +0.7%. 2026-10-30 (pyramid, 128x128): anime A -3.8%, kaguya -0.4%, anime AM 0%, live action +0.6%, anime D +2.2% | Mixed by source | The engines or the RD pass see the temporal candidates (today only the coder's NEARMV / DRL use them), or a per-sequence choice; avmenc's study says +0.61% on live action | bit-allocation "flexible MV precision" |
| Transform size choice, first try (TX_MODE_SELECT, 2026-10-26) | -0.5 to +0.4%, 38% slower | Under 1% for 0.62x speed | Done later with partition costs (kept); listed for the history | reports 2026-10-26 |
| NEWMV DRL predictor by difference bits | live action -0.20%, anime B +0.10%, chroma +0.4% | Mixed, chroma worse | The MV cost model is calibrated (see RD pass rows) | bit-allocation "compound MV refinement" |
| Joint MVD without its 2-bit charge | anime -0.46%, live action chroma +0.63% | The charged version is steadier | Never by itself | bit-allocation "joint MVD" |
| DRL reorder (`enable_drl_reorder` 1 / 2) and implicit masked blending (`enable_imp_msk_bld`): sequence flags only, the decoder's own derivations | 2026-10-30, 120 frames 1080p q100-190, am / live action / kaguya / Dragon Maid / AIR: reorder 1 +0.06 / +0.11 / +0.62 / -0.03 / +0.50%, reorder 2 -0.03 / +0.13 / +0.11 / -0.12 / +0.17%, masked blending -0.02 / +0.16 / +0.14 / +0.01 / +1.01% (decoded output matches) | No gain; avmenc's study at cpu-used 6 had them at about 0 too | The coder's NEARMV / DRL choice or compound decisions change | |

## Bit allocation and GOP

| Measure | Result | Why not kept | Revisit when | Notes |
|---|---|---|---|---|
| Temporal filtering of the anchor (ARNR-like) | +8 to +37% at 3 B frames; bframes 7 -2.2% live action / +8.8% anime | The overlay recodes the removed noise; without it the filtered frame's error shows. Too few frames use one anchor | Mini-GOPs of 16+ frames win on their own, or TPL gives the anchor the right quality | bit-allocation "Temporal filtering"; the filter is in the scratchpad |
| Temporal filtering again, AVM's method (per-sample 5x5 error with the block's, noise-scaled; quarter-pel bilinear motion; 10-bit samples), of the anchor (shown as is) and / or the keyframe (from up to 7 frames after it) | 2026-10-30, 30 frames q100-190 Y, am / live action / kaguya / Dragon Maid / AIR: anchor at 7 references +0.67 / -0.76 / +1.51 / +1.08 / +2.01%; keyframe +1.04 / +0.36 / +2.21 / +0.18 / +0.62%; both +1.16 / -0.94 / +1.75 / +1.16 / +2.91%; live action chroma -7 to -8% | Anime loses; the noise estimate does not tell am (0.29) from live action (0.29). avmenc gains -4.1% (am) / -11.1% (live action) from filtering, mostly the keyframe's, because it spends 3x CelV2's bits on the keyframe (256 against 86 KB on live action q130) and filtering takes that noise out; CelV2 codes the keyframe coarser and is already smaller | CelV2 spends far more on keyframes / anchors (higher-quality presets, long mini-GOPs), or a chroma-only filter is wanted | patch: docs/reports/temporal-filter-2.patch |
| Mini-GOP length from motion (4 frames when motion is high) | fast clips kept, but anime B 60 s -4.9% against -8.2%, live action -8.5% against -9.3% | Lost on 120 frames | A better motion measure, or TPL decides the length | bit-allocation "Mini-GOP length" |
| 16-frame mini-GOP | still anime A lost 0.64 dB at equal size | Its anchor needs more quality than the fixed offsets give | TPL / per-frame qindex from reference use exists | bit-allocation |
| Frame qindex from TPL (`--tplq`, step = B frames' * r0^0.7, as libaom's constant-q GOPs) | 2026-10-30, 120 frames 1080p q100-190: keyframes only (1) AM -3.5%, Dragon Maid -1.1%, AIR -0.5%, live action +0.7%, kaguya +0.9%; keyframes and anchors (2) +0.1% on average; anchors only +1.0%; on 30-frame clips 2 gave -2.3% (the keyframe weighs more there) | The still-share boosts win on long clips: the model sees one mini-GOP, not the chain of anchors after it (adding the last anchor's weight as the next one's: +0.1 / +1.5% at half / full) | A quantizer-aware TPL (libaom's rate and reconstruction error at the frame's q), or a lookahead over two mini-GOPs; then 16-frame mini-GOPs | main.cpp `--tplq` |
| Mids' qindex from their own TPL share (`--tplq 2` variant) | 30 frames: worse than by level on 4 of 5 (+2.2% live action, +0.2% AIR) | Their share swings with the motion | The TPL model improves | |
| Superblock qindex from TPL (`--deltaq 1`, AVM's objective delta q) | live action Y -1.4% (U / V -6%), anime +0.1 to +2.3%; four variants (finer only, zero mean, beta^0.25, both) no better on anime; replacing the still-share boost worse | Anime's held cels and flat areas: the model's SAD costs at half scale call moving characters unreferenced and coarsen them; avmenc's own: anime AM -0.42%, live action +0.05% | A finer TPL (full-scale SATD, cel-aware motion), content detection (live action only), perceptual allocation in step 7 | bit-allocation "Superblock qindex from TPL" |
| Smaller B / hidden offsets at high quality (`--bq 8 --mq -8 --aq -8`) | +3.8 to +7.1% | Worse | The GOP structure changes | bit-allocation |
| Keyframe qindex offset -24 to -32 | -0.2 to -1.0%, but clip f +9% | One bad source; kept as `--kq` (default 0) | TPL | reports 2026-10-27 overnight |
| B / hidden qindex offset tweaks, compound cost, ME penalty (2026-10-27) | within +-1% | Noise | Never by itself | reports 2026-10-27 overnight |
| B frames 3 / 4 in the flat structure | -0.5% / +1.2% (high quality +0.7 / +4.3%) | Superseded by the pyramid | (done differently) | reports 2026-10-27 overnight |
| 8-frame mini-GOPs where fine noise covers a still window (over T of the samples change by 1-2 between frames; AM's still windows 6.2%, kaguya 3.8%, Angel Beats / AIR 1.2%) | 2026-10-31, 600 frames, against the anime tune (0.85, `--level-step-16 2`), mixed AM / Angel Beats / AIR, kaguya: T 4% -2.36 / 0 / 0 / +12.62%, 5% -2.36 / 0 / 0 / +8.16%, 7% -1.56 / 0 / 0 / +2.22% | kaguya's still windows change as much and need the 16 frames | A measure that tells AM's noise from kaguya's held cels (amplitude, spatial spread) | tunes note |
| Keyframe offset from the still share alone (`--key-start`, no fixed `--kq`; 0 / 0.25 starts, boosts 64 to 96) within `--tune animation` | 2026-10-30, 600 frames q100-190 against the tune: fast stretches, without the fixed -8 kaguya +2.65%, chika +0.69%; start 0.25 boost 80 kaguya -4.34%, chika -0.61%, Angel Beats +1.58%; on stretches mixing quiet and fast scenes every variant worse (start 0.25: +0.45 to +4.23%) | The fixed -8 helps moving scenes too; deeper keyframes lose on mixed scenes | The 16-frame GOP choice is fixed for mixed scenes (the tune itself loses there: AM +7.2%, from its 16-frame GOP: -4.4% without it) | commit reverted on feat/scene-key; tunes note |
| `--preset max`'s GOP trial with `--tune film` | 2026-10-31, 120 frames 1080p q100-190: 吉村 PSNR +1.1%, VMAF NEG +2.4%; crowd_run / park_joy unchanged (no still mini-GOPs) | The trial's J picks two 8-frame mini-GOPs (+13% bits, +0.6 dB) where 16 does better on the whole curve | The trial's J sees the anchor's later use (the next mini-GOP's first frame, or stage 2's look-ahead) | docs/design/2026-10-31-preset-max.md |
| GOP trial bits weighed x2 / x4 (`--gop-trial-lambda`) | 吉村 PSNR +0.80 / -0.75% (VMAF +2.46% at x4), Dragon Maid -1.04 / +0.15%, AIR -0.12 / +0.50%, mixed AM -2.54 / -0.52%, mixed AIR -2.55 / -0.69%, against x1 +0.80 / -1.04 / -0.12 / -2.47 / -2.77% | x2 changes nothing; x4 gives anime's gain away and still loses VMAF on live action | As the row above | same |
| `--preset max`'s per-frame deblocking search with `--tune animation` | PSNR -0.4 to +0.1%, VMAF NEG Dragon Maid +4.2%, mixed Angel Beats +3.4%, Angel Beats +1.6% | The lowest squared error per frame smooths less than VMAF likes on anime (the tune's offset is 0 already) | The search weighs a perceptual measure | same |
| `--refs 4` in the target size's pass 1 | 2026-10-31, q150, 120-frame stretches of 600-frame mixed anime: real pass over pass 1 varied up to 1.59 (Angel Beats) against 1.08 without it, at 994 s against 878 s | It changes the GOP structure (refs 4: 3 B frames), so pass 1 no longer predicts pass 2 | A pass 1 that keeps the GOP with fewer references | docs/design/2026-10-31-target-size.md |
| The target size shared between scene chunks as the default (`--workers auto` with `--target-size`) | 2026-10-31, 600-frame mixed anime, 3 workers: 83-99.9% of the target, retries in 7 of 12, AIR -1.90 dB at q100; in order 95.7-99.2%, no retries | The halving step fitted per video and each chunk's start qindex mix up scene-to-scene differences with the qindex's effect | A control that fits the step within a scene, or chunks long enough to correct in | docs/design/2026-10-31-chunk-parallel.md |

## Decisions (GPU engine, RD pass)

| Measure | Result | Why not kept | Revisit when | Notes |
|---|---|---|---|---|
| MV difference bits in the GPU decision | +6 to +11% | More intra and splits | The engines only hand candidates (the RD pass decides) | reports 2026-10-27 overnight |
| NEARMV choice weight x2 / x0.5 | +6.1% / -0.3% | Worse / noise | Never by itself | reports 2026-10-27 overnight |
| RD pass: each reference's best MV per block | +0.5 to +1.5% live action, +0.3 to +0.5% anime; slower | The pass's MV / reference cost misses the coder's (ref MV stack, contexts) | The RD pass models the real MV cost (log estimate against coded bits per block) | cpu-rd-pass Stage 2; handover note |
| RD pass: chroma in the estimate | worse alone and with per-reference MVs | Not why references differ | As above | cpu-rd-pass |
| RD pass: a bit weight per qindex | +0.1 to +6.6% | Worse | The estimate is recalibrated | cpu-rd-pass |
| RD pass: rough MV cost (log2 distance to the 64x64 MV) | no gain | Not the missing part | As above | cpu-rd-pass |
| RD pass: quarter-pel re-search of each block's best inter MV (8 neighbours, 1 / 3 rounds; avmenc's cpu-used 6 -> 2 gains -14% on am, much of it fuller sub-pel search) | 2026-10-30, 30 frames q100-190, Y: am -0.19 / -0.25%, live action -0.85 / -0.98%, kaguya +1.21 / +0.67%, Dragon Maid -0.36 / -0.35%, AIR -0.69 / -1.32%; chroma mostly worse | Within noise, kaguya loses; the luma-only estimate trades chroma | The pass's costs come from the coder's CDFs (mode, MV, coefficients) | |
| RD pass: every intra mode, not only the engine's best (all frames / keyframes only) | same runs, Y: am -0.08 / -0.20%, live action -0.12 / -0.07%, kaguya -0.61 / +1.10%, Dragon Maid +0.07 / +0.19%, AIR -0.63 / -1.08% | -0.3% / 0% on average | As above | |
| RD pass: level bits priced by the coder's own syntax and the frame's starting CDFs (skip flag, eob, levels with their contexts, signs; the writer's coefficient code made a template shared by both, byte-exact) | 2026-10-30, 30 frames q100-190, Y: am -0.10%, live action +0.28%, kaguya +1.81%, Dragon Maid +0.26%, AIR +0.64%; bit weight 0.018 / 0.035 and block bits 6 / 20 all worse (+0.6 to +1.6% on average) | Exact level bits do not rank blocks better than the per-level table; the estimate's residual bits were already within 5-6% | The side bits (partition, mode, reference, MV) are priced from CDFs too, or decisions move into the coder | patch: docs/reports/rd-pass-cdf-bits.patch |

## Speed

| Measure | Result | Why not kept | Revisit when | Notes |
|---|---|---|---|---|
| More threads for the CCSO search | slower | 1 MB allocations per frame and thread | The search's buffers are kept between frames | speed-plan |
| AVX-512 instead of AVX2 | AVX2 faster in the first test | Clock drop / no gain on that kernel | Measure per kernel on new hot code | gpu-candidates-direction memory |
| CelV2's own inter prediction (plan step 6) | not built: 0.47 s of anime A's ~9 s tile time; AVM is already AVX2 | 2-3% at 1080p for about half a day | The tile coder's trials stop dominating, or MC moves to the GPU | avm-dependencies |
| The rest of AVM's common code (intra pred, inverse transforms, MV prediction, contexts) | under 1% each at 1080p | Nothing to gain for speed; the AVM library stays linked anyway | Licence or portability needs AVM gone entirely | avm-dependencies |
| Fewer RD-pass threads once the coder runs rows in a wavefront (`tile_threads()` counting wavefront rows: 1080p 44 -> 32 pass threads) | 2026-10-30: 0.98-1.05x on the five FHD sources, same bytes | No gain; the pass and the coder already share the cores well | The coder or the pass changes its share of the work (deeper decisions, GPU candidates) | parallel-coder |

## AVM tools never built because the study found little

256x256 superblocks (2026-10-30): avmenc at cpu-used 6, q100-190, 256 against 128: 4K (9 frames)
-0.20%, live action 1080p 0.00%; `--sb-size=dynamic` picks 128 on both. CelV2 also never picks a
block over 64x64 (the engines decide 64x64), so 256 would buy less still. 16K (2026-10-30, 3
frames, cpu-used 6, q100-190): 256 against 128 +0.18%, and 128x128 blocks against a 64x64 cap
-0.31%, so neither pays at 16K either. Revisit only if a source shows large flat or still areas
where 128x128 blocks win more than this.

intrabc and palette (2026-10-30): avmenc's leave-one-out put intrabc at +1.74% and palette at
+0.41% on anime AM, a visual-novel demo full of UI and text. On regular anime they are worth
nothing: anime A (30 frames, cpu-used 6, q100-190) loses +0.02% without intrabc and gains -0.17%
without palette; kaguya's sizes move under 0.3% without intrabc (its PSNR measurement failed, so no
BD-rate). Revisit for screen content (game captures, UI, slides), where the AM numbers apply.

From avmenc's leave-one-out study (each tool off against avmenc defaults, cpu-used 6, 33 frames,
anime B / live action): TIP -0.90 / +0.04%, SDP -0.67 / +0.31%, IST, inter-IST, CCTX, LR, QM,
intra-DIP, parity hiding about 0 (several are unused at cpu-used 6). Revisit when the cpu-used 0
study (oracle_cpu0) shows them used and worth more.
