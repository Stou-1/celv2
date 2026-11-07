# Levels by the coder's real costs (design)

2026-11-06. From the AV1 gap breakdown's L6 (docs/superpowers/specs/2026-11-04-av1-gap-breakdown-design.md, report 23e456f).
Encoder only, CPU only, no format change. The user's word: v0.2.0 first, so Q1 to Q3 are for v0.3.0; Q0 goes ahead.
Q0 fell short of its bar by PSNR-Y (2dbdd6b): Q0b and Q0c come before Q1. See "Q0's result".

## Why

| BD-rate, PSNR-Y, 33 frames, q100-190 | Dragon Maid | Angel Beats | AM |
|---|---|---|---|
| L4 against F0: avmenc's modes, transforms, filters, CelV2's levels | +23.62 | +26.22 | +17.15 |
| L5 against L4: which blocks are zero | +0.75 | +4.12 | -0.68 |
| L6 against L5: avmenc's levels | -19.59 | -25.91 | -13.91 |
| L6 against F0 | +0.55 | +1.04 | +0.33 |

The coder is not the gap. The gap is which levels are chosen inside blocks that are not zero: at q130 CelV2's luma has 58%
more nonzero levels than avmenc's for 21% less error, 86% more at q190. So the two spend bits at different prices.

## What differs

From reading `core/quant.cpp` and avmenc's `trellis_quant.c`, not measured:

| | CelV2 | avmenc |
|---|---|---|
| Rate of a level | `level_bits()`: 0.6, 2.6, 3.6, 4.4 bits, the same anywhere | the CDFs, by the neighbours' levels on this path, position, quantizer |
| End of block | a threshold on the first level (`--rd-zero`: rough bits) | any position, by its real cost |
| All zero | rough, with `--rd-zero` only | the real flag, always |
| Weight of a bit, in squared steps | 0.05 (luma, trellis), 0.2 (plain path) | 0.082 inter, 0.087 intra luma; 0.051, 0.067 chroma; of the DC step |
| Weight by frame | one | x1 to x1.41 by depth |
| Refresh | none | every superblock |
| Chroma, 1-D and identity transforms | plain path, no TCQ (the format's rule) | no TCQ either, `av2_optimize_txb_new`, real costs |

avmenc's weight: 84/32 x plane factor / 4 / 128, from `RDMULT_FROM_Q2_NUM`, `plane_rd_mult`, `rshift` 2 and `RDCOST`.

**The price's shape, not its level, is the likely cause.** Every test of the level has lost: `kTcqBitWeight` swept at 0.03-0.2,
W7a x2 and x4, W7c's depth factors +0.11%. One rough price cannot be right for both: a lone 1 in the high frequencies of a
sparse block costs several bits, and a 1 among dense low frequencies costs little more than the 0 it replaces. Raise the price
and the dense ones go, lower it and the lone ones stay. CelV2's excess fits this: most in inter frames, whose adapted CDFs
make nonzero dear; hardly any on key frames (1.14 times; 96% of key 64s have the same levels). Chroma goes the other way (19%
fewer nonzero, 12-19% more error), which fits the plain path's 0.2 against avmenc's 0.05-0.07. This is a reading. Q0 tests it.

## Design

- **Cost tables:** built from the frame's starting CDFs, read-only for its rows. Output stays independent of `--threads`
  and `--workers`. Same contents as `av2_fill_coeff_costs()`, written in CelV2: base, low-frequency base, range, end of block,
  all-zero flag, DC sign.
- **The trellis:** beside `quantize_tcq()`, in `core/quant.*`, same eight states and candidates.
  - Each state carries its path's levels nearby, and a level's rate comes by the contexts `block_writer.cpp` writes with
    (`get_lower_levels_ctx*`, `get_br_ctx*`), plus its quantizer.
  - A path may end at any position, by its real end of block cost. All zero is one more end, by the real flag.
  - A position where every state can only take zero takes a short way.
- **Where:** only the last quantization of a transform block, once the coder has settled its type. The RD pass, the type
  choices and every other `quantize_cost()` keep the rough model. They are most of quantization's 24-41% of CPU, and this is
  one call per transform block.
- **The plain path:** the same tables for chroma and non-2-D transforms, levels chosen as `av2_optimize_txb_new()` does, in
  Q0's last row.
- **`--rd-zero`:** the new trellis ends its own paths, so at its call the option has no effect. The rough pass keeps its other
  callers. The tune's `--rd-zero` step goes ahead as decided.

## Runs

**Q0, in the oracle, no product change.** A measurement switch at L4 (8 bits, whole 128 blocks), Dragon Maid, Angel Beats, AM,
q100-190, against L4, L6 and F0. Twelve quick streams a row. `dt.py` on each.

| Row | What |
|---|---|
| Qa | real costs, starting CDFs, avmenc's weights by plane and intra or inter |
| Qb | Qa and the depth factors (1, 1, 1.125, 1.25, 1.25, 1.41) |
| Qc | the better one, tables refreshed each superblock from the row's CDF copy |
| Qd | the best, and the plain path by real costs; YUV 6:1:1 beside Y |

Reading: the share of L4 to L6 closed, by log rate, and the area with the same levels (now 68.6% at q130).
- Half or more on each source: Q1, with the simplest row within 1% of the best.
- Less: `dt.py`'s table back here. First suspect: avmenc's rate multiplier per superblock from TPL, which the frame's weight
  does not hold.

**Q0's result (2026-11-06, 2dbdd6b; report's Q0 section).** Share of L4 to L6 closed, Dragon Maid, Angel Beats, AM:

| | PSNR-Y | YUV 6:1:1 |
|---|---|---|
| Qa | 40.8, 45.8, 49.0% | 42.0, 48.2, 48.4% |
| Qb | 40.2, 43.9, 44.1% | 43.8, 49.2, 48.1% |
| Qc | 43.2, 57.9, 52.8% | 43.4, 57.3, 50.9% |
| Qd | 46.2, 46.1, 23.7% | 57.7, 55.5, 55.7% |

- **The bar stays PSNR-Y, so Q0 is short of it.** This spec's table is PSNR-Y and YUV was to stand "beside Y". Qd's YUV pass is not taken in its place: it was seen after the
  runs, and it comes from chroma overshooting F0 (nonzero 0.81 to 1.18 times F0's) by a known departure, at 23.7% by Y on AM.
- **The trellis is not dropped.** Qc against L4 is about -8.5, -12.1, -7.9% in the oracle: the largest measure in hand. The product will get less (L2z's -1.73 to -7.48% became about -1%).
- **The first suspect was wrong, by reading.** avmenc's rate multiplier per superblock from TPL needs `deltaq_mode` 0 and only runs on ARF and KFFLT frames; the default is
  `DELTA_Q_OBJECTIVE`, whose qindex per superblock is in the stream and in Q0's weight. F0's flags are to be checked, not assumed.
- **New suspect, the frame's weight.** `av2_compute_rd_mult()` adds `rd_boost_factor` by `gfu_boost` on every frame but key frames under lookahead: x1 to x1.5 on top of the depth factor.
  Qb had the depth factor alone, by a rebuilt depth. This fits the excess: most in inter frames, 1.06 times on key frames. Unmeasured.
- **The weight cannot be all of it.** Qa on Dragon Maid has 18% more nonzero luma and 3% more error than F0 (q130). A weight only trades one for the other. Closed-loop drift may blur this;
  Q0b has none.

**Q0b, the same residual.** At L6 the reconstruction is F0's bit for bit, so each block's residual is F0's. There, beside the coding and changing no byte, run `real::levels()` on each
luma TCQ block with the row's CDFs (as Qc), at weights x0.7, 0.85, 1, 1.19, 1.41, 1.68, 2. Log per frame and per 64x64: the area with F0's levels, and nonzero against F0's, by weight.
Dragon Maid, Angel Beats, AM, q130, q190. For blocks that differ at the best weight, also log where they differ: end of block past F0's, the same end with other levels, F0 all zero.

- The best weight per frame halves or better the differing area at x1, and varies little within a frame: the frame's weight. Q0c takes it.
- It varies within frames: a weight per superblock. First check F0's `deltaq_mode`.
- No weight halves it: not the weight. Read the differing blocks against what `av2_trellis_quant()` starts from: the first quantizer's end of block, `skip_trellis`, `perform_coeff_opt`'s thresholds.

**Q0c, two rows by the same table as Q0.**
- Qe: Qc with Q0b's weight per frame, read from a file. A ceiling: the product would have to derive that weight itself.
- Qf: Qc with the plain path started from avmenc's quantizer and only lowered, as `av2_optimize_txb_new()` does (the report's departure).

Reading: as Q0, half or more by PSNR-Y on each source in one row, YUV 6:1:1 beside. Passed: Q1, with that row. Short again: to the user with the table, for Q1 at what is shown or a stop.

**Q1, the product.** `--real-trellis 0|1`, default 0 under every tune, Q0's row. 33 frames, the eight sources, against off,
PSNR-Y and YUV. CelV2's own modes: L2z's -1.73 to -7.48% became about -1% there, so expect less than Q0.

**Q2, as R4 of the zeroing spec.** 120 frames, eight sources: global-MSE PSNR, SSIMULACRA2, XPSNR. Pass: PSNR mean -3% or better,
SSIMULACRA2 and XPSNR means 0 or better, no source over +1% by either. R4 failed here on live action, losing detail in
middling blocks, and this trellis drops more, so read live action first. Short of the pass: to the user with the table and
the choices, as for R4.

**Q3, viewing.** The user views before any default changes, as R5, by tune if Q2 parts anime from live action.

**After Q2, its own spec:** the RD pass still prices residual by rough bits, so with cheaper real residuals its choices lean
the wrong way. Not before the trellis has shown what it is worth.

## Speed

Measured at Q1: `--real-trellis` 0 against 1 on one build, S1, S2, bound +2% CPU a frame, S4 reported, old and new alternating,
result in the speed ledger. avmenc's trellis is far heavier than the rough one, so this may well pass the bound. Past it: an
offset first, the share of the last quantization by `CELV2_PROFILE` showing where. No offset found: to the user, with the gain
beside the cost.

## Tests

- **Rate:** on fixed CDFs, a path's rate equals the bits of the symbols the coder writes for those levels (`symbol_log.h`),
  over 4x4 to 64x64, levels past the range and Golomb steps, and each quantizer. This is what proves the contexts.
- **Option off:** the same bytes as before.
- **Option on:** AVM decodes equal to `--recon`, 8 and 10 bits; `--threads 1` and `--workers 1`, `2` give the same bytes.
- **Edges:** all zero, end of block 1, every coefficient nonzero, a 64 transform's coded 32x32.

## Not in this

The RD pass's cost model, `--rd-zero`'s other callers, the temporal filter, and chroma's weight in the mode choice.
