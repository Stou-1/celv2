# The encode pipeline in order: where each stage runs, and where the 2026-11-04 designs go

2026-11-04. Today only the decision engine (③) runs on a GPU; everything else is CPU. Keep this table
current when a stage moves or a design lands.

| # | Stage | Runs on | Planned for the GPU |
|---|---|---|---|
| ① | Reading; denoising (`--grain`) | CPU (the reader thread) | Denoising: speed budget stage B, after a fixed-point rewrite |
| ② | Per-frame analysis: the importance map (`--paq`), the screen-content gate (`--screen`), scene cuts, still share; chunking; mini-GOP length; TPL; qindex allocation | CPU | TPL: speed budget stage B |
| ③ | **Decision engine:** downsampling, motion search (5-level pyramid), sub-pel refinement, intra costs, partition candidates by SATD | **GPU** (Vulkan; CUDA until retired; a CPU engine too) | Coarse ME on the sources ahead (speed budget stage C); IntraBC search (`--screen` B2); warp search (warp-delta stage B) |
| ④ | Global motion estimation (rotzoom per reference) | CPU | — |
| ⑤ | **RD pass:** per 64x64, candidates re-scored by quantization and bits; partitions and modes chosen | CPU | — |
| ⑥ | **Coder:** intra mode re-scoring, MV refinement, transform split and type, IST, CCTX, chroma modes, TCQ, entropy coding | CPU | — |
| ⑦ | In-loop filters: deblocking, CDEF, CCSO, loop restoration | CPU | Applying CDEF and loop restoration: speed budget stage B |
| ⑧ | Output: OBUs (film grain model OBUs included), IVF | CPU | — |

## Where the 2026-11-04 designs act

| Stage | Designs (docs/superpowers/specs/2026-11-04-*) |
|---|---|
| ① | `--grain` (2026-11-03 spec) |
| ② | `--paq` map; `--screen` gate; halftone analysis fixes (high-frequency cells, mini-GOP ends at large changes, full-scale TPL) |
| ③ | speed budget stage C; `--screen` B2 and warp-delta B (only if) |
| ⑤ | psy-rd (stage A); `--screen` palette and IntraBC candidates (A, B1); 128 intra blocks (intra gap fix 3); warp-delta stage A (fit from ③'s 8x8 MVs and ④'s global motion, refine, compare with translation); the AV1-gap breakdown's W1-W6 switches |
| ⑥ | CfL by RD with the signalled alpha (intra gap fix 1); transform splits costed like the modes (fix 2); psy-trellis (psy-rd stage B); `--rd-zero` (zeroing by RD in the trellis); warp-delta re-scored on the real references |
| ⑦ | psy-rd stage C (filters), not designed yet |

All GPU work is Vulkan only (CUDA retirement: 2026-11-04-retire-cuda-design.md), one shader version for
every GPU, bit-identical to a CPU reference.

## Transfers and VRAM (rules from the speed budget spec)

- **One round trip per frame:**
  - a frame goes up once, before ① when denoising runs on the GPU, else at ③;
  - GPU stages ① to ③ chain on the resident copy;
  - only results come down: the denoised frame for ⑤ and ⑥, TPL's table, the decision map;
  - transfers overlap computation.
- **VRAM budget:** frames between ① and ③ stay resident only as far as the free VRAM, less the engine's
  share, allows; the rest come down and go up again before ③.
  - At 16K with `--refs 7` (a16k, 9 frames) peak VRAM measured 16.48 GB with 16 quarter-pel planes per
    reference and 7.59 GB with the 4 half-pel planes; docs/reports/2026-11-04-vram-16k.md.
  - Decided 2026-11-05: the 4 half-pel planes are the Vulkan default, the other 12 phases interpolated on
    read; no planes at all as the fallback (the speed budget spec).
  - Estimate with all seven references held: about 8.3 GB, so a 16 GB GPU (Arc Pro B50) keeps the
    read-ahead frames too (about 3.4 GB at 8 bits).
