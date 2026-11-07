# Zeroing residual by RD in one trellis pass (design)

2026-11-06. From the AV1 gap breakdown's W7 (docs/superpowers/specs/2026-11-04-av1-gap-breakdown-design.md). A default
candidate: encoder only, no format change, the user views before the default changes.

**Decided 2026-11-06, after R5:** the user's word, 「アニメのみオンとします」. On with `--tune animation` only, by `--rd-zero`
(below, "After the user's word").

## Why

| 33 frames, q100-190, PSNR-Y BD-rate | AIR | DM | AM | chika | AB | 吉村 | cr | pa | Mean |
|---|---|---|---|---|---|---|---|---|---|
| W7a + b against off | -0.04 | -0.06 | -1.45 | -1.13 | -3.44 | +0.21 | +0.00 | -0.52 | **-0.80** |
| W2, 3, 5, 6 | -4.38 | -3.68 | -4.43 | -3.97 | -8.02 | -3.30 | -1.92 | -2.30 | -4.00 |
| W2, 3, 5, 6 + W7a + b | -3.81 | -3.90 | -6.27 | -4.58 | -10.99 | -4.45 | -2.04 | -2.95 | **-4.87** |

- It adds 0.87 points on the wider search, on every source but AIR. 120 frames, the moving four: -1.01%.
- Below the 1% bar alone, kept because Angel Beats and AM are avmenc's targets.
- It is not the cause of the AV1 gap, and harder zeroing loses (tried-not-kept: x2, x4, W7c, W7f, `--bq` 8 and 0). Only x1
  stays.
- W7b reruns the trellis up to five times a block: 1.3-2.4x alone, 25.88x to 37.17x on the wider search (under load).
  That is the measurement's shape, not the idea's.

## The trellis

`tcq_levels()` starts at `tcq_prep()`'s end of block in state 0 and must keep a level there. `quantize_tcq()` reruns it
per end. At x1 the bits of the comparison are the trellis's own (`m` = `weight`), so W7a + b's J is the path's J plus
`weight * end_bits(eob)`, an additive term, and one Viterbi pass finds the least over the ends.

- **A path may start at any position.** Going down the scan from `prep`, at position c, after the step of the states:
  - S(c) = `tail(c)` + `weight * end_bits(c + 1)`, `tail(c)` being `dropped()` summed over the positions from c + 1 up
    to `prep` - 1, kept as a running sum.
  - The fresh path takes Q0's nonzero candidates at c, the ones `tcq_cands()` already made (the even one, the odd one):
    S(c) + its `d2` + its `wb` is compared into state 0 (even level) and state 4 (odd level).
  - A continuing path wins a tie.
  - The first position needs no case of its own: every path is inf until a fresh one enters, and `tcq_cands()` loses its
    `last` argument.
- **Fresh starts only where `tcq_prep()`'s threshold passes** (`e >= first`, `first_ac` on AC). Elsewhere one branch and
  the running sum. Most positions of a sparse block are below it.
- **All zero is one more end:** J = `tail(-1)` + `weight * kNoneBits` against the least path, the levels winning a tie.
  `prep` 0 returns the same J.
- **The trace:** a fresh start stores previous state 8 in `step[c]`. The walk from the DC end stops there, and that gives
  eob. The dequantization along the states is as now, starting in state 0.
- **The cost returned holds the end term.** `tcq_cost()` then needs no eob: the same merge on its vectors, no `step`. Only
  tests ask it for eob, and those calls go through the level path.
  - This differs from W7a + b, whose returned cost left the term out. The RD pass does not read this cost (it takes the
    levels and adds its own end of block, `tx_cost()`). The coder's transform choices do: 0.05 a rough bit, the
    weight tried-not-kept's row on intra transform splits found about right.
  - If R1 misses its bar, take the term out once by carrying each state's eob beside its cost in `tcq_cost()`.
- It also tries every end above the threshold, where W7b tried three, so the bytes will not match W7a + b's.

**The plain path** (chroma, 1D luma, `--tcq 0`) keeps `zero_wins()` at x1: one sum per block, already one pass.

## Runs

Behind `CELV2_WIDE=7` with `X_W7P=1` until the default changes; off, the bytes are unchanged.

| Run | What | Reading |
|---|---|---|
| R1 | One pass against off, 33 frames, the eight sources, with the time | Mean -0.70% or better, no source 0.3 points worse than W7a + b. Else the fallback above, once |
| R2 | W3 alone with and without it | `--preset max` has W3 without W2, 5, 6, so this is what max gains. Never measured |
| R3 | `X_W7P=2`: key frames and anchors left out (`g_depth >= 2`), with W3 | AIR is the one source that loses on the wider search (-4.38 to -3.81), and it spends nearly everything on those frames. Kept if AIR gets 0.3 points back and the mean loses under 0.1 |
| R4 | The best of them, 120 frames, the eight sources: global-MSE PSNR, SSIMULACRA2 (every 4th frame), XPSNR, as `--psy-rd`'s measures | PSNR mean -0.7% or better, SSIMULACRA2 and XPSNR means 0 or better, no source over +1% by either. Else to the user with the numbers |
| R4b | Added 2026-11-06 from R4. `CELV2_WIDE=7 X_W7P=1 --psy-trellis` 0.1 and 0.25, against off, 120 frames, as R4: crowd_run, park_joy, 吉村, AIR, AM, Angel Beats. No build | XPSNR and SSIMULACRA2 means 0 or better, no source over +1% by either, PSNR mean -0.5% or better. Reads with R5, it changes nothing alone |
| R5 | The user's viewing, off against on: AM, Angel Beats, chika (flat gradients, where zeroing can band), AIR, at q130 and q160, and from R4 crowd_run at q160 | Stop here. Nothing changes before the user's word |

## R1-R4 results (2026-11-06)

Numbers: docs/reports/2026-11-04-av1-gap-breakdown.md, stage 2. R1 -1.09% passes. R2: W3 -2.93 to -4.17%. R3 dropped. R4 fails:
XPSNR's mean is +1.47%, and over +1% are XPSNR AIR, 吉村, crowd_run, park_joy and SSIMULACRA2 crowd_run.

| R4 means | PSNR-Y | SSIMULACRA2 | XPSNR-Y |
|---|---|---|---|
| Anime, five | -0.84 | -1.88 | -0.51 |
| Live action, three | -0.56 | +0.37 | +4.76 |

Read from R4's streams (q130, q160; throwaway, not the bench's method):

- **Not the frames.** Per-frame and global PSNR fall alike (crowd_run q160 -0.374, -0.371 dB), and XPSNR falls alike at every
  position in the group (-0.75 to -0.87 dB). So this is not `bench-global-psnr`'s per-frame averaging.
- **Live action loses detail, anime does not, for the same error.** The added squared error is about the same (+9% on crowd_run
  and AM). The high-pass energy kept against the source, in the blocks of middling activity where it falls most, at q160:
  crowd_run 0.374 to 0.336 and 0.261 to 0.230, park_joy 0.336 to 0.298, 吉村 0.610 to 0.548. That is 10-12%. AM and Dragon Maid
  lose about 1% in any band. The smoothest blocks are not where it happens: their error rises least (+3%).
- **Not settled:** my rebuild of XPSNR's weights missed ffmpeg's level (crowd_run 32.86 against 28.63), so how much of XPSNR's
  loss sits in those blocks is not known. Only the plain error and detail above stand.

So zeroing takes small coefficients that are texture on live action and little on anime. Whether that shows is the viewing's
question, so crowd_run q160 joins R5.

**R4b, the one thing to try first.** `--psy-trellis` already charges a dropped AC coefficient by the block's map weight, in
`dropped()`, so with the one pass it raises zeroing's price exactly where the map says texture (lines with `--tune animation`).
No new code. Alone it lost (tried-not-kept: SSIMULACRA2 worse at every strength), so it may well lose here. If it does, tried-not-kept
gets a row.

**R4b's result** (2b4f992): fails at both strengths, worse than no psy by all three means on the six sources (PSNR-Y / SSIMULACRA2 /
XPSNR-Y: none -0.78 / -1.09 / +2.06, 0.1 -0.49 / -0.01 / +2.31, 0.25 +0.29 / +1.76 / +3.07). crowd_run's XPSNR hardly moves (+7.92,
+7.38, +7.37). At 0.25 crowd_run q160 is back to within 2% of off's bytes (3149 against 3202 kB) with SSIMULACRA2 at 14.90 against
18.45: the term also changes which levels are kept, so it is no clean price on zeroing, and this run says nothing for or against
the detail reading above. No more runs for live action here: tried-not-kept's trigger is a price that tells texture from noise.

**The choices after R5 are the user's:**

1. On for `--tune animation` only (recommended): the five anime sources gain by all three means, live action stays as it is.
   AIR's XPSNR +1.25% and PSNR +1.04% stay over the line, so AIR's viewing decides this one. Anime without the tune gains nothing.
2. On everywhere: crowd_run's XPSNR +7.92% accepted because the viewing shows nothing.
3. `--preset max` only, or not at all.

R2's -4.17% is PSNR-Y only: choice 3 needs R4's three measures on W3 first.

## After the user's word

**When it is on.** `--tune animation` only, at any preset. `--tune film`, `--tune still` and no tune stay off. Anime without
the tune gets it by `--rd-zero 1`.

**The form: `--rd-zero 0|1`,** as `--rd-intra` and `--tmv`: default 0, 1 with `--tune animation`, the command line winning.
- In main.cpp's `--tune animation` block: `if (unset("--rd-zero")) a.rd_zero = true;`, with its line in `--tune`'s text.
- An option, not an inner setting: every setting of that tune has one, and off must stay reachable for tests and benches.
- It reaches the quantizer on `--psy-trellis`'s route (main, the chunk coder's args, the encoder's config), and a
  thread-local set where `g_bit_factor` and `g_depth` are set now (encoder.cpp, pipeline.cpp).
- It covers the plain path's `zero_wins()` at x1 too, as `X_W7P=1` does, so chroma, 1D luma and `--tcq 0` follow it.
- It stays on in `--pass 1`: the quick look predicts pass 2's size.
- **No GUI change.** The GUI passes `--tune`, and it shows none of that tune's other settings.

**What goes.** Everything of W7 but the one pass:
- `struct W7`, `g_w7`, `CELV2_WIDE=7`, `X_W7A`, `X_W7B`, `X_W7C`, `X_W7F`, `X_W7P` (with its value 2).
- `zero_scale()`, `zeroing_here()`, `one_pass()`'s env reading, `quantize_tcq()`'s reruns.
- `bit_factor()`, `g_bit_factor`, `Encoder::bit_factor_`, `g_depth`, `depth_`: only W7c and W7f read them. Off they are 1, so
  `kTcqBitWeight` and `kBitWeight` stand alone again.
- Stays: `force_eob` and `tcq_end_cost()` as the test's oracle, `zero_wins()` without its scale, `tcq_fresh()`. The off path is
  `tcq_levels()` without the pass, the code from before W7. W1-W6's switches stay until their own specs.

**The order.** The default does not change before its speed is measured.

| Step | What | When |
|---|---|---|
| 1 | `--rd-zero`, default 0 under every tune, and the removal above. Off: the bytes of 5db707c without `CELV2_WIDE`. On: the bytes of `CELV2_WIDE=7 X_W7P=1`, on two clips each | Now |
| 2 | R6: `--tune animation` and the default `--paq`, `--rd-zero` 0 against 1, 120 frames, the five anime sources, R4's three measures | Now, beside F0: it times nothing |
| 3 | Speed: the section below, S1, S2 and S4 | After F0 |
| 4 | `--tune animation` turns it on: the one line, tests or hashes that pin bytes change with this spec as the reason, docs/design/2026-10-30-tunes.md gets R6's table | After R6 and speed pass |

- R6's reason: R1-R4 ran `--paq 0`, and the default is 0.25 now. Reading: means 0 or better by all three. A source over +1% goes
  to the user with the numbers, AIR included: the viewing has already seen AIR, so it stops step 4 only if it grows past R4's
  +1.25%.
- docs/design/tried-not-kept.md: "W7a + b x1 stays a default candidate" becomes the decision. A new row for live action: R4's
  table, trigger a price on zeroing that tells texture from noise.

## Speed

- Quantization is 24-41% of the CPU (speed ledger), most of it this trellis.
- Bound: +2% CPU a frame on S1 and S2 against the ledger's method, old and new alternating. The result goes in the
  ledger. S4 is measured too, reported, no bound: `--preset max` takes it with the tune. All three are `--tune animation`
  scenes, so the scenes the option changes are the ones measured: `--rd-zero` 0 against 1 on one build.
- Expected at or under the bound, not measured: the reruns are gone, the added work is a branch and a sum per position
  and a merge per position over the threshold, and a block that goes to zero skips its inverse transform and its
  coefficients' writing.
- Past the bound: an offset elsewhere first, else the user decides.

## Tests

- **The oracle:** random blocks, every transform size, at a few qindex, psy 0 and 2: one pass's levels and J equal the
  least over forcing each end above the threshold and all zero, the old `force_eob` run. Ties by the rule above.
- `quantize_cost()` with eob null equals `quantize()`'s cost (test_transform's check passes `&eob` today, which would now
  skip the cost-only path), and with eob given it equals `quantize()`'s eob.
- A lone coefficient 0.6 steps at scan position 1: coded off, none on (x1 is enough there: J about 0.50 against 0.41).
- AVM decodes equal to `--recon`, luma and chroma; `--workers` 1 and 2 give the same bytes; off gives the same bytes as
  before until the default changes.

## Not in this

- `skip_txfm` as a choice (W7d: a-c did not reach 1%).
- Real bits in the trellis: `level_bits()` and `end_bits()` stay rough.
- The trellis's bit weight: `kTcqBitWeight` was tuned without zeroing. Retune only if R4 suggests it, as its own row.
- chika's -3.4% at `--bq` 8 and 0: a choice per source, its own look.
