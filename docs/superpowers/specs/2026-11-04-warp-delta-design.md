# `--warp 0|1`: warp-delta blocks fitted from the engine's MVs on the CPU (design)

2026-11-04. Staged: a check on avmenc first, then stage A (CPU: fit, refine, compare); stage A2 (CPU: warp
reaches more blocks and its models spread); stage B (a warp search in the GPU engine) only if A2 shows
model quality is what limits it.

**Updated 2026-11-05 from docs/reports/2026-11-05-warp.md.** The check and stage A are done: avmenc gains
on every source, CelV2 on none (warp on 0.01-0.07% of the area). Stage A2 and the changed goals below
are new; the rest is the original design.

## Why

On Flash (docs/reports/2026-11-03-flash-cirno.md; avmenc cpu-used 6 with CelV2's tools, frames 40-69
of the 60-second stretch, q145-190):

| Added | BD-rate |
|---|---|
| warped motion only, or with warp causal | +-0% |
| warp-delta | -19.6% |
| warp-delta with warp-extend | -19.5% |
| warp-delta with 6 parameters | -19.9% |
| every warp tool | -20.3% |

Rotation and zoom, which a translational MV cannot follow, are what it codes.

Measured since, q100-190, 33 frames (Flash 30), with `--enable-warped-motion` and `--enable-warp-delta`:
Flash -9.62% (-12.40% over q145-190; the rest of the gap to -19.6% is not explained), AIR -4.28%, 吉村
-3.08%, Dragon Maid -3.01%, Angel Beats -2.98%, crowd_run -1.87%, AM demo -1.63%, park_joy -0.95%, chika
about -0.5% (two points). Without WARPMV: Flash -7.11%, 吉村 -2.30%, AIR -3.42%.

CelV2 uses no warp motion mode today. block_writer.cpp sets every warp field to 0. Only per-frame
global motion follows rotation and zoom: a rotzoom model for each of the two nearest references.

## AV2's warp-delta (AVM v1.0.0, decodemv.c `read_warp_delta`)

- A block's motion mode `WARP_DELTA` carries a rotzoom (4-parameter) or affine (6-parameter) model,
  coded as a difference from a base model (`av2_get_warp_base_params`: from the warp candidate stack,
  neighbours' warps or global motion).
- Signalled per block:
  - a precision index (`warp_precision_idx`), which sets the step and the largest coded value;
  - per parameter (2-3, or 2-5 for affine), the delta's magnitude and sign;
  - the default 6-parameter flag is derived.
- Its availability is per frame (the frame's enabled motion modes) and per block size
  (`is_motion_variation_allowed_bsize`).

## Check first: avmenc with and without warp-delta

- avmenc cpu-used 6 with CelV2's tools, with and without warp-delta.
- **Sources:** the eight 1080p sources (33 frames) and Flash's 60-second stretch (30 frames); q100-190,
  BD-rate by global-MSE PSNR-Y; the area coded with warp-delta.
- **Within +-0.3% on every regular source:** warp-delta goes only into a Flash-like setting (a
  `--tune screen`, if `--screen`'s measurements make one), not the default.

## Stage A: CPU, fit from the engine's MVs

1. **Frame gate (speed).** Searched only in frames with rotation or zoom: a reference's global motion
   (④ in docs/design/pipeline-map.md) is not translational, or the engine's 8x8 MVs vary within blocks
   beyond translation (the residual of a per-block translational fit above kWarpGate). Other frames:
   warp-delta off in the frame header, no search.
2. **Fit.** For each inter block where AV2 allows warp, a rotzoom model by least squares from the 8x8
   MVs it covers (from ③, the decision engine). The 6-parameter model is fitted only where the rotzoom
   one won.
3. **Quantize and refine.**
   - The base model built as AV2 builds it.
   - The difference rounded to the coded steps at the precision that fits it.
   - Each parameter tried one step either way; the best kept.
4. **Compare with translation.**
   - The RD pass (⑤) predicts the warp from the source and scores it against the block's translational
     choice.
   - The coder (⑥) re-scores it on the real references.
   - Warp-delta wins where its RD cost is lower.
5. **Signalling.** The frame header enables the warp-delta motion mode in gated frames; blocks write the
   motion mode, the precision index, and the parameters' deltas and signs.

**Cost** (estimate, not measured): about ten warp predictions a block, each about as heavy as today's
sub-pel interpolation (AVM's warp predictor has an AVX2 version). About 2 G operations a 1080p frame
over three block sizes, 0.1-0.3 CPU seconds against the 4.9 CPU seconds a 1080p frame takes today:
2-6% on gated frames, less on average.

**Interface:** `--warp 0|1`, 0 by default during the measurements. Default on or in a tune after them.

**As built (1ced58a):** the coder alone searches. The frame gate, the RD pass's comparison and the
6-parameter model are not in: the RD pass's scoring lost (+0.48%, +1.70%), and there was no gain to gate.

## Stage A2: CPU, warp reaches more blocks and its models spread

**Why this before a finer search.** At Flash q160 the fit beats the block's translation in 2,209 blocks
and the +-4 search in 3,000, but only 11 win once about 30 bits of model are counted. A better model
adds few winners; the bits are what stop them. avmenc avoids those bits: without WARPMV, 72% of its warp
on Flash (吉村 58%) inherits a neighbour's model at `warp_ref_idx` 1 and codes no delta, and only 16% (吉村
34%) codes its own. In CelV2 the list stays nearly empty, so inheriting never starts. Also, the search
reaches 19% of Flash's area: skip-mode and compound blocks, 64.5% of the inter area, never try warp.

**First, two counts (no encoder change):**
1. A throwaway avmenc with inheritance off (`warp_ref_idx` 0 only), Flash, 吉村, AIR: what inheriting is
   worth.
2. Over avmenc's WARP_NEWMV area at Flash q160: the share where CelV2's blocks are smaller than avmenc's.
   Its warp sits in 32 and 64 blocks, and CelV2's engine decides on translation, which splits a rotating
   area.

If count 1 shows inheriting worth under a quarter of the warp gain, the design session looks again
before step 2 below is built.

**Then, in the coder, on the real references:**
1. **Reach.** Blocks the decision made skip mode or compound also try single-reference WARP_NEWMV, with
   their full RD cost as the bound.
2. **Seeds.** A model's bits pay for the blocks that inherit it, so a block's first model gets a credit:
   - Coming neighbours: the blocks right and below, not yet coded. A neighbour fits when the model
     predicts its 8x8 engine MVs within half a pel.
   - With at least kWarpSeedNeighbours of them fitting (starting value 2), the model may lose to the
     block's choice by up to kWarpSeedCredit bits (starting value 30, about a model) and still be coded.
   - The engine's MVs for the whole frame are already there, so this needs no second pass.
3. **Inheriting.** Every block in reach tries the list's models at `warp_ref_idx` 1 and up first (no
   delta). The flaw noted in the report (`choose_warp()`'s partial sums) is fixed here.
4. **The frame gate** (stage A's step 1), now that a wider search has a cost to save. Frames it closes
   give the bytes of `--warp 0`.

**Only if count 2 is large (a half or more) and steps 1-4 fall short of the goals:** a merge candidate in
the RD pass: a 32 or 64 block whose children's MVs fit one model. Stage A's RD pass scoring lost, so this
gets its own measurement and is dropped if it loses again.

## Stage B: a warp search in the GPU engine (only if)

The stage A conditions held (48-53% against 20%), but they count before the bits, so they say little:
see stage A2's first paragraph. They are replaced.

**When both hold, after stage A2:**
- Flash is short of its goal below.
- Counted **after the bits**, the share of blocks trying warp where a +-4 search wins and the fit loses
  reaches 20%.

**Then:** the decision engine (③) searches warp candidates itself: Vulkan and the CPU reference,
bit-identical, one shader version. Its own spec.

## Measure

- **Sources:** Flash's 60-second stretch, the sources the check found gaining, and the eight 1080p
  sources as the must-not-lose set. `--refs 7`, q100-190, BD-rate by global-MSE PSNR-Y.
- **Goals (changed 2026-11-05):** half of avmenc's gain, the measure stage B's condition used.
  - Flash -5% or better (avmenc -9.62%, or -7.11% without WARPMV; the old -10% asked for all of it);
  - the eight sources' mean -1% or better (avmenc's about -2.3%), each within +0.2%;
  - the gate's per-frame decisions logged, its threshold tuned on them.
  - The area coded with warp and the share of it that inherits, reported per source beside avmenc's.
- **Default:** `--warp` goes on by default, or in a tune, once the goals are met, within the speed bound.
- **Speed:** within 5%. A slowdown beyond it does not by itself drop the feature: the implementing
  session profiles the encoder and looks for an equal speed gain elsewhere (another stage made faster)
  to offset it, and reports the two together. If none is found, it reports the loss to the user and
  asks them to decide; it does not drop the feature for speed on its own. The speed ledger
  (docs/design/speed-budget.md) records it.
- Every part or threshold dropped gets a row in docs/design/tried-not-kept.md.

## Tests

- **Fit (unit):** an MV field made from a known rotation and zoom (2 degrees, 3%): the fitted model is
  that model within one coded step.
- **Coding:** a synthetic rotating, zooming picture with `--warp 1`: AVM decodes equal to `--recon`,
  and it is smaller than `--warp 0`.
- **Gate:** a purely translating synthetic source: the gate stays off, and the bytes equal `--warp 0`'s.
- **Seeds and inheriting:** a synthetic picture rotating as a whole: more than half of the warp blocks
  inherit at `warp_ref_idx` 1, it decodes equal to `--recon`, and it is smaller than with
  kWarpSeedCredit 0.
- **Everything else:**
  - `--warp 0` gives the same bytes as before;
  - `--workers 1` and `2` give the same bytes;
  - `--device cpu` and the GPU give the same bytes;
  - 10-bit input decodes equal to `--recon`.

## Not in this

Warp causal, warp-extend, compound warp (each within 0.7% on Flash beyond warp-delta); the GPU search
(stage B, its own spec).

WARPMV is not in stage A2 either, but the old "within 0.7%" was wrong for it: it adds 0.8-2.5%, and it is on
whenever warp-delta is on. Identity models are worth about 0.2% of that. It is the next step once stage
A2 meets its goals.
