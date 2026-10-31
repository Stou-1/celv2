# AVM code CelV2 uses (2026-10-29)

Plan step 6 replaces AVM code step by step. This lists what `src/` calls today, by area: 142
distinct `av2_*` / `avm_*` functions plus unprefixed helpers. All of it is AVM's normative common
code (`av2/common`, `avm_dsp`), never `av2/encoder`. The order to replace it in comes from the speed
step's profile, not from this list.

| Area | What CelV2 calls | Where | Notes |
|---|---|---|---|
| Entropy coding | `avm_write_symbol` / `_bit` / `_literal`, `avm_start_encode`, the `avm_wb_*` header writers; the CDF and context pickers (`av2_get_*_context`, `av2_get_drl_cdf`, `av2_get_pred_cdf_*`) | block_writer, headers | Every symbol: a hot path. The writer can be replaced on its own. The context rules must stay bit-exact. |
| Inter prediction | `av2_build_inter_predictors` (translation, warp, BAWP), `av2_setup_pre_planes`, scale factors | encoder (coding and every RD-cost trial) | Hot: the coder's MV, AMVD and joint searches call it several times a block. Candidate for own SIMD, and later Vulkan. |
| Intra prediction | `av2_predict_intra_block_facade`, CfL store, MRLS lines | encoder (`intra_satd`, coding) | Called per mode and line tried. |
| Transforms and quantization | `av2_inverse_transform_block`, `av2_init_stxfm_kernels`, `av2_get_tx_scale`, `av2_iqmatrix`, quant tables | encoder, quant | The forward DCT is CelV2's own (`core/transform`); the inverse is AVM's. |
| MV prediction and modes | `av2_find_mv_refs`, `av2_find_mode_ctx`, ref-MV / warp banks, `motion_mode_allowed`, `av2_findSamples`, warp model helpers | block_writer | Large and exact (the decoder derives the same lists). Replace last. |
| Loop filters | `av2_loop_filter_frame(_mt)`, `av2_cdef_frame(_mt)`, `ccso_frame` | codec_state (`end_frame`) | Frame-level, so a natural GPU (Vulkan) target at 8K / 16K. |
| Frame state | buffer and context allocation, tiles, `av2_setup_motion_field`, `av2_copy_frame_mvs`, `set_offsets`, mi maps | codec_state, encoder | Bookkeeping that mirrors the decoder. Cheap; replacing it buys nothing until the rest is CelV2's. |
| Verification | `avm_codec_*` (the AVM decoder) | verify/avm_decode, tests | Test-only; it stays as the reference. |

## Profile (2026-10-29)

Timers around each AVM call (summed over threads), q130, 10-bit, RTX 5090. The CPU was shared
with 9 avmenc runs, so the shares matter more than the totals. 1080p: anime A 30 frames (4.4 s),
live action 33 (7.1 s); 16K: 9 frames (33 s).

| AVM call | anime A | live action | 16K | Where it runs |
|---|---|---|---|---|
| `av2_build_inter_predictors` | 0.47 s | 0.92 s | 5.1 s | the tile coders (1080p: 2 threads, the critical path) |
| `av2_predict_intra_block_facade` | 0.06 s | 0.08 s | 3.6 s | the tile coders |
| `av2_inverse_transform_block` | 0.02 s | 0.03 s | 0.50 s | the tile coders |
| `av2_find_mv_refs` | 0.02 s | 0.07 s | 0.26 s | the tile coders |
| `ccso_frame` | 0.07 s | 0.08 s | 2.0 s | end of frame, one thread |
| deblock, CDEF (already multithreaded) | 0.02 s | 0.03 s | 0.24 s | end of frame |

The whole end of frame (deblock, CDEF search and filter, CCSO search and filter, copies) took 1.0 s
of anime A's 4.4 s and 4.8 s of the 16K run's 33 s, and it holds up the next frame. Most of it
is CelV2's own CDEF and CCSO searches (1080p 0.55 / 0.24 s, 16K 0.40 / 1.35 s), which are the
speed plan's, not this step's.

## Replaced

1. **CCSO filter** (`core/ccso_apply`): CelV2's own AVX2 filter for its subset (filtering across
   tiles, no lossless segments, no BRU), the rows of 32x32 blocks of all planes on threads (one
   per million samples, up to the cores). The padded luma copy it reads is made on threads too,
   into a buffer kept between frames. `test_ccso_apply` checks it against AVM's C filter on every
   filter shape; the streams are byte-identical (1080p 8 and 10 bits, 16K). 16K: the filter 2.0 ->
   0.15 s, the end of frame 4.8 -> 2.3 s, 0.274 -> 0.295 fps. 1080p: the filter 0.072 -> 0.043 s.

## Next, by the profile

2. Inter prediction (translation, compound, BAWP, global and local warp): the largest AVM cost on
   the 1080p critical path. AVM's own convolutions are already AVX2, so the gain comes from
   specialising (CelV2's filters and block sizes, no scaling, no padding copies), not from SIMD.
3. Intra prediction (16K's second largest).
4. The rest (inverse transforms, MV prediction, entropy contexts, frame state) costs under 1% of
   the time; replacing it buys nothing for speed.
