# Bit allocation and GOP structure

2026-10-28. What the avmenc comparison leaves after the CPU RD pass, and what was tried.

## Why

The avmenc comparison (33 frames, `docs/design/2026-10-28-cpu-rd-pass.md` Status) after the RD pass:
anime B C -7.4%, B -21.6%; live action C +5.3%, B +8.6% (CelV2 ahead). B differs from C by
TPL, temporal filtering of the alt-ref (ARNR) and avmenc's own GOP (up to 16 frames, a pyramid of
several levels); on anime that is the largest gap left, about 14%.

Again on 2026-10-29 with main after this note's work and the coding tools (2d65ad9), against the
same stored avmenc runs (`--good --cpu-used=6`, 33 frames, q100-190; avmenc's bits against
CelV2's, minus where avmenc needs fewer):

| | C (CelV2's tools and GOP) | B (CelV2's tools) | A (avmenc defaults) |
|---|---|---|---|
| anime B | -0.5% (was -19.8%) | -13.5% (was -33.4%) | -22.5% (was -40.2%) |
| live action | +17.0% (was -0.7%) | +23.9% (was +2.1%) | +12.1% (was -7.0%) |

("was": the CelV2 build the comparison first ran against.) With the same tools and GOP, CelV2 now
codes anime as well as avmenc and live action 17% better. What is left on anime is avmenc's GOP,
TPL and temporal filtering (B against C, about 13%) and the tools CelV2 lacks (A against B, about
9%). avmenc at cpu-used 6 is not AV2's ceiling: the AOM test conditions use slower settings.

CelV2's mini-GOP today (`src/main.cpp`): up to `--bframes` (3) + 1 frames; the last one coded first,
hidden (qindex offset `--aq` -16), with `--pyramid` the middle one next as a hidden reference
(`--mq` 0), the others non-reference B frames (`--bq` +16) predicting from both sides, then the
anchor shown with show_existing_frame.

## Temporal filtering of the anchor (tried, not kept)

Motion-compensated average of the anchor with up to 2 neighbours each side (16x16 full-pel search
in two scales, weight exp(-SSE per pixel / (k x step^2)) / distance, chroma along), coded hidden in
place of the anchor. 30 frames q100-190, BD-rate (luma) against HEAD, live action / anime:

| shown as | k | bframes 3 | bframes 7 |
|---|---|---|---|
| the filtered frame (show_existing) | 1 | +8.0 / +9.4% | +0.9 / +14.1% |
| an overlay (the source coded from it), qindex offset 0 | 0.5 / 1 / 2 / 4 | +10.6 to +17.8% | |
| an overlay at the anchor's offset (-16) | 0.25 / 1 | +18 to +37% | +24 / +25% |
| no filtering | | (HEAD) | -2.2 / +8.8% |

Both decode correctly. The overlay costs nearly a frame: bringing the anchor back to its source
means coding again the noise the filter took out. Without one, the filtered frame's own error is
shown. With 3 B frames the filtered anchor serves too few frames to pay for either; it needs long
GOPs, and the current structure loses on long GOPs (bframes 7: -2.2% live action, +8.8% anime).
The filter is kept outside the tree (scratchpad) for when the GOP is longer.

## Next: a multi-level pyramid

A mini-GOP of 8 (later 16) frames coded as a binary pyramid: the anchor, then the middle, then the
quarter points (each hidden and kept as a reference with its own qindex offset), and the remaining
frames as non-reference B frames, so every frame predicts from references at most 1-2 frames away.
Then temporal filtering again on top of it.

## Multi-level pyramid (done)

`--pyramid` now splits recursively: the middle of every gap between references that holds 3 or
more frames is coded hidden and kept, coarse to fine, while the hidden frames leave a slot for the
last shown one (5 slots for an 8-frame mini-GOP's 4). Level 1 takes `--mq`, each level below 8 more.
`--bframes` defaults to 7 with `--refs` 5 or more (else 3, as before).

30 frames, against HEAD (4-frame mini-GOPs), live action / anime:

| mini-GOP | levels | level step | `--aq` | BD-rate |
|---|---|---|---|---|
| 8 | 1 | | -16 | -2.2 / +8.8% |
| 8 | 2 | 4 / 8 / 12 | -16 | -3.7 / -0.8%, -4.4 / -0.9%, -4.5 / -0.3% |
| 8 | 2 | 8 | -24 | -4.8 / +3.4% |
| 16 | 3 | 4 / 8 | -16 | -6.0 / +1.9%, -6.6 / +2.4% |
| 16 | 3 | 8 | -24 / -32 | -6.2 / +6.2%, -3.4 / +13.0% |

120 frames 1080p q100-190 (anime A 240), against HEAD:

| source | 8 frames, 2 levels | 16 frames, 3 levels |
|---|---|---|
| live action | -9.3% | -13.5% |
| anime B 30 s / 60 s | -2.2 / -8.2% | +2.7 / -5.1% |
| anime C | -3.4% | -4.7% |
| anime D | +1.0% | +5.0% |
| anime A | -3.3% | +11.3% |
| anime A q130 speed | 7.7 -> 8.2 fps | 9.1 fps |

16 frames wins on still or slow material and loses where motion is fast.

## Mini-GOP length: by qindex, not by motion

- At high quality 8 frames lose (30 frames live action / anime, against 4): q40-100 +1.5 / +2.3%,
  q70-130 -0.2 / +1.4%, q100-190 -4.4 / -0.9%. Smaller offsets there (`--bq 8`, `--mq -8`,
  `--aq -8`) did worse (+3.8 to +7.1%). So 8 frames only at `--qindex` 100 or more; the GUI's
  高画質 preset (q60) keeps 3 B frames.
- Three fast 1080p50 Xiph clips (crowd_run, park_joy, ducks_take_off) lost +0.4 to +2.1% on 30
  frames but are even on 120 (-0.5 / +0.3 / -0.2%): the short clips misled.
- A per-GOP choice of 4 frames where the motion-compensated difference at 1/8 scale over 4 frames
  is over 8 (fast clips 8-16, live action ~2, anime B ~5) kept the fast clips at the
  4-frame bytes but lost elsewhere on 120 frames (anime B 60 s -4.9% against -8.2%, live
  action -8.5% against -9.3%). Not kept; anime D (+1.0%) is the one source 8 frames lose on.
- Then: TPL-like qindex offsets from how much each frame is referenced (the 16-frame GOP lost
  0.64 dB at equal size on still anime A: its anchor needs more quality than the fixed offsets give).

## GUI presets (done)

標準 (q170) and 小容量 (q200) now use 7 references and celv2enc's B frames (`BFrames` -1 = no
`--bframes`), so they get the 8-frame pyramid; 高画質 (q60) keeps 3. 120 frames q170-230 against 4
references and 3 B frames (merged build c5e2ac7): anime B -7.3 / -9.9%, live action -9.1%,
anime A -9.3%, anime C -1.4%, anime D -0.9%, crowd_run -4.0%, park_joy -6.3%, ducks_take_off
+4.5%; anime A q170 8.9 -> 9.2 fps. The ducks_take_off loss is the 8-frame mini-GOP's (7 references with
3 B frames: +0.07%); it was even at q100-190 (-0.2%), and the motion statistic that would flag it
does as much for crowd_run and park_joy, which gain.

## Scene cuts in held animation (done)

After held (identical) frames any change is far over the median of recent differences, so a new
drawing over part of the picture was coded as a keyframe (anime C: frames 2 and 9 of 120, 16-21 KB
each). A cut now also needs more than half of the 8x8 cells to change by over 2 levels: anime C
-36.6%, anime B 60 s -0.8%, the other sources the same bytes; 107 -> 81 cuts over the
3010 frames of anime B (the dropped ones checked by eye: wipes, partial changes).

## Anchor quality from the still share (done)

SVT-AV1 spent 3x CelV2's bits on anime C's keyframe and almost nothing after. The anchor's qindex now
goes down by 24 x (still share - 1/2) x 2, the still share being the part of the 8x8 cells within 2
levels between the mini-GOP's first frame and its anchor (anime C 0.74-1.0, live action ~0.75, anime A
~0.7, anime B 0.4-0.75, fast Xiph clips 0.1-0.4).

| boost | floor | anime C | live action | anime D | AM 30 / 60 s | anime A | fast clips |
|---|---|---|---|---|---|---|---|
| 12 | none | -3.0% | -2.8% | -1.2% | +0.4 / +1.3% | -2.5% | +0.2 to +0.6% |
| 16 | 1/2 | -5.6% | -2.5% | -1.7% | -1.8 / -0.5% | -2.8% | |
| **24** | **1/2** | **-6.1%** | **-2.4%** | **-2.3%** | **-2.2 / -0.4%** | **-4.3%** | **same bytes** |
| 32 | 1/2 | -6.9% | -1.2% | -2.9% | -2.4 / +0.2% | -4.9% | |

120 frames q100-190 against the scene-cut fix; q40-100 (30 frames live action / anime): +0.1 / +0.1%.

Keyframes the same way, from the still share to the frame up to 8 ahead, by up to 48 (after the
anchors' boost; 120 frames q100-190, then q40-100 on 30 frames):

| boost | AM 30 / 60 s | anime D | anime A | live action | anime C | fast clips |
|---|---|---|---|---|---|---|
| 16 | -0.5 / -1.2% | -0.9% | +0.1% | +0.1% | +0.3% | |
| 32 | -1.4 / -2.2% | -1.4% | +0.1% | -0.1% | -0.1% | same bytes |
| **48** | **-2.0 / -2.2%** | **-2.2%** | **-0.5%** | **-0.2%** | **-0.1%** | **same bytes** |
| 64 | -2.6 / -2.2% | -2.7% | -0.6% | -0.6% | +0.2% | |

48 at q40-100: live action -1.2%, anime -0.1%.

## Coding tools: CCSO (done)

A leave-one-out study on avmenc (each AV2 tool CelV2 lacks turned off against avmenc's defaults, 33
frames, anime B q130-190, 11 of 52 tools so far) put CCSO first (+1.55% without it), then
MRLS (+1.19%); TIP lost nothing (-0.95% without it).

CelV2 now codes CCSO on every plane with every unit's flag on (the flags are in the tile data, coded
before the reconstruction is final); after CDEF, each plane's classification (band offsets only, or
band and edge classes of one of the 7 supports, every threshold and classifier) and offsets are the
least squared error plus 0.025 squared quantizer steps per header bit, from per-bin sums of source -
reconstruction. Per-unit flags would add 10-20% to the luma gain (the units the offsets hurt).
33 frames q100-190 against `--ccso 0`: anime B Y -2.04%, U -4.03%, V -2.47% (6:1:1
-2.36%); live action Y -0.78%, U -2.05%, V -0.96% (-0.95%). `--psnr` now prints chroma too.

## Coding tools: MRLS (done)

Directional intra luma modes now also try reference lines 1-3, alone or averaged with line 0, in
the coder's luma mode search (SATD plus weighted bits, after the mode is chosen on line 0). 33 frames
q100-190 against `--mrls 0` (both with CCSO): anime B Y -0.31%, U -0.46%, V -0.29%; live
action Y -0.05%, U -2.26%, V -2.19%. The decision engines still pick modes on line 0 only.

## Coding tools: local warp (tried, not kept)

AV2 gives a single-reference block local warp only through its own inter mode (WARP_NEWMV, which
needs WARP_DELTA on); a NEW_NEWMV compound block can code WARP_CAUSAL, the projection the decoder
fits to the neighbours' motion. Chosen in the coder by its luma RD cost (with the flag's bits),
33 frames q100-190 against no warp: anime B Y -0.03%, live action Y +0.39%. Not kept
(the patch is outside the tree); WARP_NEWMV is the next step for warp.

## Coding tools: CfL with the derived scaling factor (tried, not kept)

CfL needs no alphas with CFL_DERIVED_ALPHA (the decoder fits the scale to the neighbours), only
two flags; the uv mode is coded before the block's luma is reconstructed, so the coder chose it by
chroma SATD with the source luma standing in for the reconstruction. 33 frames q100-190 against
no CfL: anime B Y -0.12%, U -0.84%, V -0.19%; live action Y +0.02%, U +1.48%, V +2.72%.
Requiring CfL to win by 10%: -0.10 / -0.66 / -0.06% and +0.05 / +1.22 / +4.32%. Not kept (patch
outside the tree); choosing it on the real reconstruction needs the luma coded before the mode
info is written.

## Coding tools: single-reference warp (tried, not kept)

WARP_NEWMV blocks with WARP_DELTA models: a rotation / zoom from the warp candidate list, index 0
with coded steps on its two terms, searched one step at a time from the wanted MV (also moving the
MV, and again at the finer step). Decoded output matches. In the AVM study, leaving warp-delta out
of avmenc costs +1.51% on anime B and warp-causal -0.10% (the projection the decoder fits
to the neighbours gave CelV2 no wins either). 33 frames q100-190 against no warp: steps only,
anime B Y -0.24%, live action Y +0.04%; with MV and finer steps +0.06% / -0.26%; on top
of the MV refinement below +0.17% / -0.23%. Not kept (patch outside the tree): within noise, and
up to ~40 RD costs per block.

## Coding tools: NEWMV MV refinement (done)

The warp search's MV steps, tried on plain NEWMV: the coded MV (rounded to the MV precision) moves
one step at a time while the coder's RD cost with the difference's bits falls. The decision MV's
search sees neither. 33 frames q100-190: anime B Y -0.38%, live action Y -0.59%.

## Coding tools: adaptive MVD (done)

A NEWMV block may code its difference from the set 0, +-2, 4, 6, 8, 16, 32, 64, 128 (1/8 pel per
component: a joint type, an index and a sign), behind a use_amvd flag on every NEWMV / NEW_NEWMV
block. The coder starts from the components nearest the refined NEWMV MV and moves one index on one
component at a time while the RD cost with the symbols' bits falls. 33 frames q100-190 against
`--amvd 0`: anime B Y -0.11%, U -0.42%, V -0.42%; live action Y -0.26%, U -0.53%,
V -0.67%. Compound blocks always code use_amvd 0.

## Coder search: compound MV refinement (done), NEWMV predictor by bits (not kept)

NEW_NEWMV pairs went out as the engine chose them; the coder now moves one MV of the pair one step
at a time while its RD cost with both differences' bits falls (as NEWMV's refinement does).
33 frames q100-190: anime B Y -0.19%, U +0.13%, V +0.14%; live action Y -0.66%,
U -0.09%, V -0.63%. Choosing NEWMV's DRL predictor by difference plus index bits instead of L1
distance, on top: live action Y -0.20% more, anime B Y +0.10%, chroma +0.4%; not kept.

## Coding tools: joint MVD (done)

A compound pair may code JOINT_NEWMV: one difference, for the farther reference, the other's
projected from it by the references' distances and scaled by one of five modes. The coder tries
it after refining the NEW_NEWMV pair: the best scale mode on the refined base difference, then
one step at a time, with 2 bits' charge (luma cost only). 33 frames q100-190 against NEW_NEWMV
alone: anime B Y -0.34%, U -0.35%, V -0.24%; live action Y +0.05%, U +0.14%, V +0.07%
(no charge: -0.46 / -0.18 / -0.57% and +0.04 / -0.18 / +0.63%).

## Superblock qindex from TPL (built, off by default)

`--deltaq 1`: a temporal dependency model of each mini-GOP's source frames (`core/tpl`: 32x32
blocks, SADs on half-scale luma, motion searched at quarter scale +-64 samples then refined, the
intra cost the least of DC / vertical / horizontal from the source's neighbours, each block's
saving 1 - inter / intra passed back to what its motion covers, as libaom / AVM's tpl_model) gives
the keyframes, anchors and mids a qindex offset per superblock as AVM's objective delta q (beta = r0
/ r, step / sqrt(beta), at most +-35, coded in steps of 4). The coder, the RD pass and the intra
lambda use each superblock's qindex; streams decode equal to `--recon` (`test_random_decisions`:
random offsets near both qindex ends).

33 frames 1080p q100-190 against `--deltaq 0`, Y (U, V):

| | anime AM | anime A | anime kaguya | anime D | live action |
|---|---|---|---|---|---|
| as AVM (sqrt(beta)) | +0.72% | +2.27% | +0.67% | +0.06% | -1.42% (-6.2, -6.0%) |
| finer only | +0.36% | +0.56% | | | -0.67% |
| zero mean | +1.28% | +2.53% | | | -1.12% |
| beta^0.25 | -0.01% | +1.15% | | | -1.15% |
| finer only, beta^0.25 | +0.18% | +0.42% | | | -0.44% |

Not double counting with the anchors' still-share boost: without the boost, TPL offsets lose more
(against the default: anime AM +1.13% / +0.19% without either, anime A +7.6% / +4.0%, live action
-0.7% / +0.5%). avmenc's own delta q (deltaq-mode 1 against 0, cpu-used 6, q100-190) gains -0.42% on anime
AM and +0.05% on live action. So `--deltaq` stays off; the plumbing (FrameSpec::sb_dq, the delta
qindex syntax, per-superblock quantizers) stays for perceptual or low-bitrate allocation (plan
step 7).

## Coding tools: flexible MV precision (done), temporal MVs again (not kept)

`--flexmv` (on): a NEWMV block not using adaptive MVD codes its MV at the frame's 1/4 pel or at 1,
4 or 8 pel (AV2's precision set under a quarter-pel frame), the predictor rounded to that precision
plus the difference in its steps. The coder tries the coarser three after the NEWMV refinement, by
RD cost with the difference's and the precision symbols' bits; compound blocks code the frame's.
33 frames 1080p q100-190 against `--flexmv 0`, Y (U, V): anime AM -0.49% (-0.65, -0.57%), live
action -0.12% (+0.26, +0.33%), anime A -1.26% (-1.14, -0.72%), anime kaguya -1.48% (-0.55,
-1.28%), anime D -0.60%; the same speed.

`--tmv` (off): temporal MVs again with today's GOP and 128x128 superblocks (the header lacked the
TMVP sampling-step bit those need; fixed): anime A -3.77%, kaguya -0.44%, anime AM +0.02%, live
action +0.55%, anime D +2.17%.
