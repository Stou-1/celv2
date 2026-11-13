# The in-loop filters' wall time (2026-11-09)

The fps-cap report (docs/reports/2026-11-09-fps-cap.md) found the filters 52 ms of a reference frame's 230 ms chain
at 3 workers, and turning them off +20% fps. This note cuts their searches' CPU with the same bytes. Against 64d49c4,
RTX 5090, Xeon 658X (24 cores, 48 threads).

## Verdict

**Kept: the CDEF strength search and the chroma Wiener fit rewritten, the same bytes; filters 51 to about 38 ms.**

| Run (medians; S1, S2, 吉村 3 rounds, S3 and S4 one) | CPU a frame | fps |
|---|---|---|
| S1 | 2119 -> 2019 ms (-4.7%) | 12.34 -> 12.97 (+5.1%) |
| S2 AIR | 2433 -> 2334 ms (-4.1%) | 11.79 -> 12.25 (+3.9%) |
| S2 Dragon Maid | 2764 -> 2667 ms (-3.5%) | 10.27 -> 10.60 (+3.2%) |
| 吉村 (`--tune film`) | 3214 -> 3159 ms (-1.7%) | 6.32 -> 6.30 (-0.3%) |
| S3 (AM 4K) | 10579 -> 10343 ms (-2.2%) | 2.73 -> 2.79 (+2.1%) |
| S4 (`--preset max`) | 9031 -> 8769 ms (-2.9%) | 4.01 -> 4.31 (+7.6%) |

The same bytes: s1cut cfe1e813 at `--workers 1`, `2`, `auto` and `--device cpu`; 吉村 33 frames dd461b96 on Vulkan 1,
auto and the CPU; S1 600 f6199489 before and after.

## Where the filters' time was

`CELV2_PROFILE=1` now prints the filters' parts (`filter wall`, ms a frame where the part runs) and the three searches'
CPU as stages of their own (S1 600, 3 workers):

| Part | wall before | wall after | CPU before | CPU after |
|---|---|---|---|---|
| deblocking | 0.9 | 0.9 | | |
| CDEF strength search | 16.3 | 9.4 | 108 | 35 |
| CDEF | 1.0 | 0.9 | | |
| CCSO search | 10.6 | 10.9 | 68 | 70 |
| CCSO | 1.7 | 1.7 | | |
| LR fit (chroma Wiener) | 12.2 | 6.2 | 69 | 32 |
| LR | 4.5 | 4.5 | | |
| LR decide | 2.8 | 2.7 | | |

- The searches were 80% of the filters, the filters themselves (AVM's, on their workers) fast.
- With one worker the filters take 29.5 ms: the rest at 3 workers is waiting for cores (24 near full).
- The searches depend on each other (CDEF, then CCSO on its output, then LR on that), and each was on many threads
  already, so only less CPU shortens them.

## What changed

- **CDEF search** (src/core/cdef_search.cpp): a block's error at all 63 strengths from the filter's parts.
  - Built once: the primary sum at each level's adjusted strength, and the secondary sum at each strength, along the
    block's direction and along 0 (AVM passes direction 0 when the level is 0).
  - Then each strength is a sum, rounding and clamp (the clamp only with both parts on), as
    cdef_filter_block_internal(), and its error. Before, it was 63 whole filters a block.
  - Equal to av2_cdef_filter_fb() on 20000 test blocks (8, 10 and 12 bits, luma and chroma, the frame's edge,
    the visible part).
- **Dampings in one job**: the three other dampings' runs in one fork-join instead of one each.
- **CDEF input rows** copied by memcpy instead of a test a sample.
- **LR fit** (src/core/wiener_sums.cpp): the normal equations' sums by `_mm256_madd_epi16` on int16 features, 128
  samples at a time, widened to 64 bits.
  - The doubles they replace were exact integers, so the values are the same.
  - Equal to a scalar reference on every size, at 8, 10 and 12 bits.

## Tried, not kept (rows in tried-not-kept)

- **Deblocked-only references for the next frame's decisions** (`X_DBREF=1`). The filters' searches could then run
  beside the next frame's GPU work, but BD-rate is +0.92% mean, worst +1.42%.
- **The same for the GPU engine only** (`X_DBREF=2`): +0.28% mean, worst +0.71%. Over the gate.
- **The pool's filter jobs first**: no change. The filters wait for cores, not for the pool.
- **CCSO counts summed in runs**: CCSO's CPU 70 to 90 ms. The cells change too often.

`prepare()` already starts the next frame's reference-free GPU work (its upload and intra costs) before this frame's
filters end. Everything else the next frame's engine does reads this frame's filtered picture, so with the same bytes
the filters cannot overlap the next frame.

## Left

- CCSO search, now the largest part (about 11 ms wall, 70 ms CPU): 21M scattered increments a 1080p frame into
  per-task tables of 488 KB. A vectorised class index with privatised histograms is the next thing to try.
  - Tasks of 256K samples at least instead of 64K: CPU 70 to 59 ms, but wall 10.9 to 11.6 ms at 48 threads (fewer
    tasks). Not kept; worth it where the cores are the limit (8-16 threads).
- LR (PC Wiener luma) and its decision, about 7 ms: AVM's filter.

## Reproduce

```bash
CELV2_PROFILE=1 celv2enc -i sources/mxam600.y4m -o s1.ivf --workers 3 --refs 7 --qindex 130 --tune animation --device vulkan
powershell -File tools/c17/speed_ab.ps1 -Old head64.exe -New filt1.exe -Src sources -Out speed.csv
NEW_ENV=X_DBREF=1 python -I tools/c17/bd_run.py same.exe same.exe sources work 120
```

`NEW_ENV` (bd_run.py) sets environment variables for the new build's runs only. The X_DBREF spike is not in the tree;
it was a copy of the frame after deblocking handed to Pipeline::keep().
