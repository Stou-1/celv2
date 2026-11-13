# C1 Q1 and Q2: the real-cost trellis in the product (2026-11-10)

`--real-trellis 1` (9e7caea, f9e4b01): the coder's TCQ levels by the bits of the symbols it writes, on the live tile CDFs
(docs/superpowers/specs/2026-11-06-real-cost-trellis-design.md, row Qc; the user chose to build it on 2026-11-10). Against
`--real-trellis 0` on the same build, RTX 5090. Default still 0.

## Verdict

**Better on every metric, every source within the bar, but short of the spec's PSNR bar (mean -3%): to the user.**

| 120 frames, q100-190, BD-rate on against off | AIR | DM | AM | chika | AB | 吉村 | crowd_run | park_joy | mean |
|---|---|---|---|---|---|---|---|---|---|
| global PSNR-Y | -1.20 | -0.91 | -2.21 | -0.77 | -3.02 | -0.65 | -1.35 | -2.78 | **-1.61** |
| SSIMULACRA2 | -0.98 | +0.24 | -0.26 | +0.41 | -1.14 | -0.04 | +0.32 | -1.82 | **-0.41** |
| XPSNR-Y | -1.27 | -0.75 | -2.00 | -0.76 | -3.40 | -0.49 | -0.03 | -1.62 | **-1.29** |

Q2's bar: PSNR mean -3% or better (missed: -1.61%), SSIMULACRA2 and XPSNR means 0 or better (met), no source over +1% by
either (met: worst chika +0.41% SSIMULACRA2). Live action, where R4 (`--rd-zero`) lost detail, gains here too.

## The weight

Avmenc's weight (Q0's Qa) over-zeroes in CelV2: its decisions are made with the rough costs, not avmenc's. 33 frames, 8
sources, global PSNR-Y:

| weight | x1 | x0.7 | x0.5 | x0.35 | x0.25 |
|---|---|---|---|---|---|
| mean | +1.91% | -0.83% | **-1.91%** | -1.74% | -0.85% |
| worst | +4.80 (crowd_run) | +0.92 (AIR) | **-1.00 (chika)** | -0.26 | +0.79 |

## Speed (`tools/c17/speed_ab.ps1 -NewArgs "--real-trellis 1"`, medians)

| | S1 | S2 AIR | S2 DM | 吉村 | S3 | S4 |
|---|---|---|---|---|---|---|
| CPU a frame | -0.2% | +1.0% | +1.3% | +1.9% | -0.4% | -0.1% |
| fps | -1.3% | -2.2% | -4.5% | -2.7% | -1.0% | +2.1% |

Within the spec's bound (+2% CPU). fps loses more than CPU: the coder is on the frame chain.

## How it works

- core/real_trellis: the oracle's Viterbi in fixed point (bits in 1/512, J in int64); each TCQ path keeps only the levels
  its contexts can still read, by scan index. Equal to the oracle's double version on 99.6-99.8% of 20000 random blocks,
  the rest ties within the rounding (test_real_trellis).
- The coder keeps its type, IST and CCTX choices, then quantizes the chosen coding again; a type the new levels no longer
  allow falls back as the decoder reads. Blocks `quantize()` left all zero stay so.
- Rulings: live tile CDFs instead of Qc's per-superblock copy (they are what the symbols are coded with; no copy).
- Tests: cli_real_trellis (AVM decodes equal to `--recon` at 8 and 10 bits; workers and devices give the same bytes);
  off gives the same bytes as before.

## Left

- The spec's next steps: Q3, the user views before any default changes. Then its own spec: the RD pass still prices
  residual by rough bits.

## Reproduce

```bash
NEW_ARGS="--real-trellis 1" python -I tools/c17/bd_run.py enc.exe enc.exe sources work 33
python -I rz_r6.py OUT 3 "off=enc.exe" "on=enc.exe,A:--real-trellis:1"   # Q2 harness (scratchpad), SSIMULACRA2 every 4th frame
```
