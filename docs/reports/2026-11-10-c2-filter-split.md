# C2: what each per-unit filter tool is worth, by the oracle (2026-11-10)

Candidate C2 of docs/reports/2026-11-08-improvement-candidates.md: the in-loop filters per unit. The oracle at L4 (avmenc's
decisions and filters, whole 128 blocks, 8 bits, 33 frames, q100-190) with one part of F0's filters cut to the form CelV2
codes (`X_ORACLE_FILT`, oracle build only). BD-rate against L4 with all of F0's filters (global PSNR-Y; + is what the part
is worth). Every stream decodes equal to `--recon`.

| Cut to CelV2's form | Dragon Maid | Angel Beats | AM | crowd_run |
|---|---|---|---|---|
| c: CDEF one strength (F0's most used) | +0.67 | +1.66 | +1.28 | +1.75 |
| o: CCSO on every unit | +2.97 | +1.18 | +0.16 | +0.54 |
| d: no deblocking deltas | +0.07 | +0.28 | +0.25 | +0.06 |
| s: no CDEF on skip blocks | 0 | 0 | 0 | 0 |
| k: one LR class (class 0's taps) | +0.30 | +17.98 | +41.39 | +10.38 |
| w: luma Wiener units as PC Wiener | +4.40 | +66.20 | +6.93 | -0.77 |
| all six | +7.40 | +68.21 | +8.65 | +1.35 |
| L3e (CelV2's own filters) | +0.78 | +9.98 | +4.14 | +0.36 |

## Reading

- **CDEF strengths per 64x64 (c) are worth 0.7-1.8%, on every source** (mean 1.3%): the first tool to build.
- **CCSO flags per unit (o) are worth 0.2-3.0%** (mean 1.2%), most on Dragon Maid.
- Deblocking deltas 0.1-0.3%; CDEF on skip blocks nothing (F0 does not use it). Not worth building.
- **k and w are not measures:** they keep F0's taps, fitted for other classes or another filter, so they overstate. All six
  together lose far more than CelV2's own filters (L3e), which refit. LR's worth needs CelV2's own fit at L4, part by part.
- These are F0's choices; CelV2's own search for them will get part of it. The candidates report's bar (1% or more) is met
  by c on three sources and o on two.

## Next

- C2 build order: CDEF strengths per 64x64 (the frame's strength set and an index per unit, searched with the strength
  search of 492344e), then CCSO flags per unit. CPU +1-2.5% estimated: needs an offset.

## Reproduce

In celv2-bench/l1l2: `SRCS=md,ab,am30,cr LEVELS=l4fce,l4fse,l4foe,l4fde,l4fke,l4fwe,l4fcsodkwe python -I l12.py 8`, then BD
against `l4we` by `tables.py`'s `pts()` and `bd()`.
