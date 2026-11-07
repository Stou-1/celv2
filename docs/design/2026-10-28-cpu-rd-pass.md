# CPU RD pass over GPU candidates (stage 1)

2026-10-28. Why, what, and how to tell it worked.

## Why

Against the AVM reference encoder restricted to CelV2's coding tools and GOP (avmenc "C",
scratchpad `me/oracle.py`, anime B 1080p, 33 frames), CelV2 needs about 20% more bits:
the loss is in its decisions (partition, mode, MV). On live action the same comparison is about
even. Bit allocation (temporal filtering of anchors about 4%, TPL about 3%) and tools (about 7%)
come after it.

The GPU decides with SATD plus fixed bit charges and the CPU only patches the result (NEAR /
GLOBALMV / BAWP checks). Profiling (tools/sampler) shows 96% of the CPU's samples waiting: at 1080p
two tiles keep two of 48 threads coding.

## What

1. The GPU (and its CPU mirror, CpuEngine) hands over, next to today's tree, its best candidates
   for every block of the superblock: the 425 blocks of 64x64 down to 8x8 with their halves, each
   with the best inter choice (reference, MV, compound pair) and the best intra mode, and their
   SATD costs. `SbCandidates` beside `SbDecision` in the DecisionMap.
2. A CPU pass (`core/rd_pass`) re-scores those candidates with the encoder's estimate - the
   prediction (intra from source neighbours as the GPU does, inter from the 8-bit reference
   reconstructions at quarter pel), the luma DCT at the block's largest transform, quantize_cost()
   and dropped_cost(), plus rough mode / MV / partition bits - and picks the partition bottom-up,
   writing an ordinary SbDecision. Superblocks are independent here (no reconstruction, no entropy
   state), so the pass runs on a thread pool, row by row as the GPU publishes them.
3. Tile coding is unchanged: it reads the pass's SbDecision instead of the GPU's.

## Not in stage 1

Mode choice beyond the GPU's per-block best, MV refinement, chroma, real entropy contexts (stage 2);
AVX2 of the estimate (after it pays off); 256x256 superblocks.

## Measure

BD-rate on anime A (240 frames), anime B 30 s and 60 s, anime C, anime D and
the 1080p live-action clip, against HEAD; speed on anime A q130; the avmenc C gap again on AM.
Done when the anime sources gain with no large loser; the GPU and CPU engines must still give the
same bytes (test_ippp cpu_pipeline_matches_gpu).

## Status

- 2026-10-28, stage 1a done: both engines hand over the candidates (commit 95bd4c2).
- Stage 1b, first estimator (uncommitted, behind CELV2_RDPASS): the pass reproduces the engine's
  bytes exactly when it scores with the engine's own costs, so the plumbing is right; with
  the luma DCT + quantize_cost() estimate it loses on live action at every bit weight tried
  (0.1 to 6.4 per bit, 0 to 200 bits per block: +2% to +29% bytes for -0.4 to +0.07 dB). The
  estimate ranks partitions differently from what coding them costs. Next: log, per block, the
  estimate against the bits and squared error the coder actually spends, and calibrate from that.
- Stage 1b done (on by default, `--rd-pass 0|1`). The per-block log (estimate against the bits
  and luma squared error the coder spent) found two faults: the 8x4 / 4x8 halves had no transform
  scale, so their distortion read as 0 and they were picked everywhere; and level bits were
  weighed by the RDOQ weight while side bits used another. Now each transform is quantized as the
  coder does and every bit takes one weight; the estimate's squared error lands within 5% of the
  coder's for every block size, its bits 4-43% low (inter side information). Weight 0.025 per bit
  in squared steps, 12 bits per block. 120 frames q100-190 against HEAD: anime B -16.0 /
  -11.1% (30 s / 60 s), anime C -14.1%, anime D -13.4%, live action -10.0%, anime A (240 frames)
  -9.0%; anime A q130 10.2 -> 8.2 fps.
- Tried after: a weight per qindex (x (qstep(120) / qstep)^a, a = -0.4 / 0.4 / 0.8: +0.1 to +6.6%
  on 30 frames), extra side bits for inter blocks (5 / 10 / 20: 0 to +4.2%), plain quantization in
  the estimate instead of TCQ (+3.6 / +6.0%): all worse.
- Each block also tries the inter MVs the engine found for the square's own blocks, its parent and
  its children: anime B -2.1 / -1.4%, live action -1.5%, anime D -0.8%, anime C -0.6%,
  anime A -1.9%. The pass's motion compensation is now separable (same values; it was 43% of the
  CPU's samples): anime A q130 is 10.2 fps without the pass, 9.9 with it before pooling, 7.7 with it.
- Stage 2, first try (not kept): the engines also handed over each reference's best MV per block
  and the pass tried those within a ratio of the best one's engine cost: 1.1 / 1.3 / 2 / all gave
  +0.5 / +1.0 / +1.5 / +1.5% (live action) and +0.3 to +0.5% (anime) on 30 frames, and anime A q130
  7.7 -> 6.0 fps with all. Charging MV bits by the distance to the 64x64 block's MV of that
  reference (1 / 3 bits per log2 step) did not change it (-0.0 to +1.7%). The luma-only estimate
  picks other references where coding them costs more than it sees (MV prediction, chroma); a
  model of that comes before more candidates.
- Chroma in the estimate (both planes, bilinear 1/16-pel prediction, plain quantization; intra by
  DC of the source neighbours): +0.4 / +5.9% luma BD-rate at full weight, +0.0 / +0.5% at 0.25;
  with each reference's best tried as well still +0.8 to +7.4%. Not kept: chroma is not why other
  references lose.
- The avmenc comparison again (33 frames, the stored avmenc runs), against CelV2 at db6d28a:
  anime B C -7.4% (was about -25%), B -21.6%, A -29.7%; live action C +5.3%, B +8.6%,
  A -1.2% (CelV2 now codes live action at least as well as avmenc with the same tools). On anime
  the decisions' share is down to about 7%; temporal filtering, TPL and the GOP (B against C,
  about 14%) are now the largest gap.
- Stage 2, each reference's best MV (the GPU / CPU engines now hand it over, bit-exact). A per-block
  log of the estimate against the coder (tell_frac around the mode info and the block, luma squared
  error) showed the estimate is as close for those blocks as for the pool's: side bits 15.8 coded
  against 18.4 estimated (no neighbour shares the reference in 79% of them), residual bits and squared
  error within 5-6% of the coder's, as for the engine's own choice. Yet as single candidates they lose
  (30 frames live action / anime: +1.46 / +0.40% at ratio 2). Tried and not kept: an MV cost from the
  engine choices of the 8x8 cells left of and above the block (exact copy free, else the difference's
  bits; with the reference's best: +0.32 to +0.64% / +0.22 to +0.35%, alone +0.47 / +0.35%; with 8
  fewer bits per block +0.07 / +0.35%), extra bits where no neighbour has the reference (+0.31 /
  +0.15%, on those candidates only +0.10 / +0.01%), the nearer references only (refs 0-1: -0.03 /
  +0.17%), a flat charge on them (4 / 8 / 16 / 24 / 32 bits: +0.51 / -0.01 / -0.06 / -0.14 / +0.09%
  and +0.11 / -0.03 / +0.04 / +0.03 / +0.08%), one by size (+0.11 / +0.10%). A charge on the pool's
  candidates instead (+0.13 / +0.25%) says the pool is not over-picked.
- Used as compound pairs they gain: the cheapest reference's best averaged with each other reference's
  best within 1.2x its engine cost, 36 bits plus the indices per compound block in the pass, and no
  compound 8x4 / 4x8 in the estimate (AV2 has none: the coder predicted those from the first
  reference; alone -0.15 / +0.09%). Ratio 1.5 and 24 bits looked best on 30 frames (-1.48 / -1.76%)
  but lost on anime C / anime D / anime A over 120-240 frames (+0.10 / +0.16 / +0.46%, mostly at
  q190); 1.2 / 36 is the one that gains everywhere. 120 frames against HEAD: anime B
  -1.21 / -1.31% (30 s / 60 s), live action -2.32%, anime C -0.03%, anime D -0.17%, anime A
  (240 frames) -0.08%; anime A q130 7.7 -> 6.9 fps.
- 2026-11-02, DC fix (a886385): intra_cost() took intra_pred()'s -1 (it leaves DC to its caller) as
  the DC prediction, so every DC block was scored against a flat -1. Fixed (tests/test_rd_pass.cpp). 120 frames
  1080p q100-190, global PSNR-Y against e535cda: am30 -0.27%, Dragon Maid -0.13%, AIR +0.07%, chika -0.15%,
  Angel Beats -0.21%, live action -0.07%, crowd_run / park_joy -0.03%. Small: one keyframe per clip. The
  intra-in-inter charge stays 32 bits (16 / 48 are worse, tried-not-kept). Intra results measured before
  the fix (every intra mode in the pass) are re-run.
