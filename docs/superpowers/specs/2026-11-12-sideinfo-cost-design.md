# The RD pass prices side information from the frame's CDFs (design)

2026-11-12. After-schedule plan item 5 / after-capacity plan 3.2 item 1. Encoder only, CPU only, no format change.

## Why

- The RD pass (core/rd_pass.cpp) prices everything that is not residual with constants: partition 2 bits for any split
  (`kPartBits`), a block 12 bits (`block_bits`), an inter block 3 + its reference index (`inter_bits()`), a compound
  block 36 + both indices, an intra mode 1 / 3 / 9 (+ 32 in inter frames). No MV bits at all.
- W3 (`X_ALTS`: four trees coded for real, the cheapest kept) gave YUV -4.06% on 33 frames after C11: the pass's trees
  are often not the cheapest the coder could have, at the same candidates.
- tried-not-kept "level bits priced by the CDFs": revisit when the side bits (partition, mode, reference, MV) are priced
  from the CDFs too.

## Design

- **Tables, not coder calls.** Once per frame, right after `Encoder::begin()` and before the collector starts, the
  encoder builds `SideCosts` (core/rd_pass.h) from the frame's starting CDFs (`cm->fc`): all in 1/512 bit, each symbol's
  bits averaged over its contexts (the pass has no coded neighbours: superblocks run in any order).
  - Partition: for each square 64 / 32 / 16 / 8, NONE / HORZ / VERT / SPLIT; for each 2:1 half, NONE / into its two
    squares. The symbols `BlockWriter::partition()` writes (do_split, do_square_split, rect_type), where AVM's
    `init_allowed_partitions_for_signaling()` at an inner position says they are coded.
  - Inter frames, every block: skip_mode 0 (where the frame allows it). Intra: is_inter 0 (the mode's bits and the 32
    "intra in inter" bias stay: the coder re-picks every intra mode, and the bias is about the pass's residual
    estimate, not syntax). Inter: is_inter 1, skip_txfm 0, comp flag, the reference (truncated unary on
    single_ref_cdf), the mode (NEARMV where the MV equals the predictor, else NEWMV), the first DRL bit, and the MV
    difference: shell set and class from `nmvc`, the offset bits averaged, the pair and column bits approximated
    (about one bit a class), a sign a nonzero component.
  - Compound: comp flag, both references' tree bits, NEW_NEWMV, both differences, plus a bias for what the constant
    36 held beyond syntax (env, swept). As built this is an approximation, not the writer's symbols: each reference is
    priced as a single one (the single_ref table twice), not by the compound reference tree the writer codes, and the
    bias (`kSideCompBias`, 20) absorbs the difference.
- **The MV predictor** (shortcut: the coder's is its coded neighbours' stack). The engine's best MV for the same
  reference on the enclosing block: a square's own block takes its parent square's, a half takes its square's, the
  64x64 its own. No such MV: (0, 0).
- **What stays:** the residual price (rough table, C11 not in), `bit_weight`, `block_bits` (re-swept: the per-block
  syntax it stood for is now priced), intra mode bits on key frames, palette and IntraBC.
- **Switch:** `X_SIDE=1` (env) until it passed; `X_SI_*` knobs for the sweeps. Kept as the default; `X_SIDE=0` gives
  the constants back.

## CPU estimate

- Tables: about 300 `log2` a frame (partition 12 x 4 contexts x 3 symbols, references 6 x 3, MV classes 11,
  the flags): under 0.1 ms a frame against about 2000 ms CPU a frame on S1.
- The pass: each `inter_cost()` adds two table reads, a bit scan and a few adds per MV; each `square()` reads its
  partition costs instead of a constant. Against the 1300 transforms a superblock costs, under 0.1%.
- The real risk is indirect: different trees change the coder's work (more small blocks or more NEWMV refinements in
  `inter_block_mode_info()`) and the pass's own (more inter_cost calls do not change: the candidate set is the same).
  Measured by speed_ab.ps1 against 27dfe00, not estimated.

## Bar

120 frames, 8 sources, BD-rate PSNR-YUV 6:1:1 mean -1% or better, no source worse than +1% on PSNR-Y; S1 CPU a frame
+1% / fps -2% at most against the commit before.

## Measured (2026-11-12, against 27dfe00)

33 frames, 8 sources, BD-rate PSNR-YUV 6:1:1 / PSNR-Y means (worst Y), bit weight 0.025 unless named:

| Variant | YUV | Y (worst) |
|---|---|---|
| All of the design (partition, intra flags, inter with MV differences) | +0.04% | +0.57% (cr +1.67%) |
| The same, block bits 6 | +0.49% | +0.61% (air +1.83%) |
| Partition only | -0.40% | -0.26% (pa +0.02%) |
| Inter only (with MV differences) | +0.14% | +0.51% (air +1.58%) |
| Partition and inter, no MV difference bits | -0.70% | -0.43% (ab +0.30%) |
| The same with the intra flags | -0.61% | -0.35% (ab +0.34%) |
| The same as the row before last, block bits 16 | -0.48% | -0.04% (cr +0.84%) |
| ... compound bias 30 | -0.46% | -0.14% (ab +1.10%) |
| ... bit weight 0.03 (**kept**) | -0.98% | -0.36% (ck +0.59%) |

Partial runs (stopped once clearly behind): MV bits at half weight, 4 sources -0.56 / -0.16% (ck Y +0.90%); MV
against the parent and the squares left and above, 3 sources -0.74 / -0.28% (air Y +0.94%); block bits 9, 4
sources -0.11 / -0.03%; bit weight 0.035, 4 sources -0.98 / -0.17% (ck Y +1.42%).

What was kept: partition, the inter blocks' flags, reference, NEARMV / NEWMV and NEW_NEWMV from the CDFs, no MV
difference bits, intra blocks on the constants, bit weight 0.03.

- 120 frames (m4's settings, 8 sources): YUV -1.04%, Y -0.40% (worst pa +0.38%), XPSNR -0.67%,
  SSIMULACRA2 -1.68% (every source better). YUV per source: air -0.28, md -1.36, am -1.57, ck -0.70, ab -1.32,
  yo -2.66, cr -0.58, pa +0.12%.
- Speed (speed_ab.ps1's way, the base and this build alternating, another session's encodes on the machine):
  S1 6 rounds, the order alternating, load 72%: CPU a frame 1824 -> 1822 ms (-0.1%), fps 12.51 -> 12.49 (-0.2%).
  A first run under 87% load, 3 rounds: S1 CPU +0.8%, fps -5.4% (old 8.8-10.0 fps, new 8.7-9.1: the load moved
  more than the change); S2 AIR CPU -0.4%, fps +7.0%; 吉村 CPU +0.4%, fps +4.0%. All 9 S1 rounds pooled: CPU +0.3%,
  fps -0.1% (medians).

## Fix: 8x4 / 4x8 priced as the writer codes them (2026-11-12, after a9d5670)

Two over-charges found in review, about 3.8 bits each 8x4 / 4x8 inter block (2.2-4.7), so an 8x8 HORZ / VERT about
7.5 bits too dear:

- The partition prices took `INTRA_REGION`; inter frames code their trees in `MIXED_INTER_INTRA_REGION`, where an
  8x4 / 4x8 allows only NONE and codes no partition symbol (it was charged 0.48-0.65 bit). Now the frame's region.
- An 8x4 / 4x8 inter block was charged skip_mode, is_inter and the comp flag; the writer codes none of them
  (skip_mode needs a compound-capable size, is_inter is forced where `offset_started`, `is_comp_ref_allowed()` is
  false). Now skip_txfm, the reference and the mode only. A compound candidate of that size was already priced as
  single, as the writer codes it.

33 frames, 8 sources, BD-rate YUV / Y against a9d5670 (bd_run.py's settings, no real trellis), means (worst Y):

| Variant | YUV | Y (worst) |
|---|---|---|
| The fix, bit weight 0.03 (**kept**) | -0.02% | -0.05% (md +0.17%) |
| ... bit weight 0.025 | +0.35% | -0.02% (yo +0.50%) |
| ... bit weight 0.035 | +0.28% | +0.39% (ck +0.76%) |
| ... compound bias 15 | -0.02% | -0.08% (md +0.35%) |
| ... compound bias 25 | +0.01% | -0.00% (yo +0.22%) |

Bit weight 0.03 and compound bias 20 stay.

120 frames (m4's settings, 8 sources) against a9d5670's streams (817a46e's si run, md5 equal on ab q130 and pa q160):
YUV -0.08%, Y -0.10% (worst md +0.00%), XPSNR +0.01%, SSIMULACRA2 +0.29% (cubic; piecewise linear +0.21%, ck +2.42 /
+1.11% the worst). YUV / Y / SSIMULACRA2 per source: air -0.10 / -0.05 / +0.05, md +0.19 / +0.00 / +0.48, am -0.17 /
-0.11 / -0.39, ck -0.28 / -0.34 / +2.42, ab -0.08 / -0.11 / -0.37, yo -0.09 / -0.01 / -0.24, cr -0.13 / -0.09 / -0.03,
pa -0.02 / -0.06 / +0.38%. Low / high half of the rate range (piecewise linear): YUV -0.09 / -0.05%, Y -0.09 / -0.07%.
A small gain on PSNR, within the noise of one run, and a small loss on SSIMULACRA2 (ck); kept as the correct price.
- Speed (speed_ab.ps1's measurement, the order alternating, a quiet machine: no other encoder running): S1 6 rounds
  against a9d5670, CPU a frame 1819 -> 1828 ms (+0.5%), fps 12.61 -> 12.61 (+0.0%) (medians).
- a9d5670 against 27dfe00 again on the quiet machine, 4 rounds each, medians: S1 CPU +0.2%, fps -0.6%; S2 AIR CPU
  +0.6%, fps -1.2%; 吉村 CPU -0.2%, fps -0.3%.
