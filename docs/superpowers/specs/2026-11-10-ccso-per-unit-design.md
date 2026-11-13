# C2 part 2: CCSO flags per unit (design)

2026-11-10. From docs/reports/2026-11-10-c2-filter-split.md: the oracle puts per-unit CCSO flags at 0.2-3.0% BD-rate (mean 1.2%,
most on Dragon Maid). Encoder only, no format change: AV2 already codes a flag per CCSO unit and plane (read_ccso(), avm
av2/decoder/decodemv.c:161); CelV2 codes 1 on every unit (BlockWriter::ccso(), block_writer.cpp:513) and turns no plane off.
Same route as part 1 (docs/superpowers/specs/2026-11-10-cdef-per-unit-design.md): log the flag's place while the tiles are
coded, decide it in CodecState::ccso(), code it at SymbolLog::replay().

## The one difference from CDEF: the context's neighbours

av2_get_ccso_context() (avm av2/common/pred_common.c:701) reads `xd->neighbors[0]` and `[1]`, which fetch_spatial_neighbors()
(av2_common_int.h:3732) fills from bottom-left / above-right / left / above in that order, depending on availability and the
superblock top edge. Rebuilding that at replay is error-prone, so **log the two pointers** when the block is logged (they are
valid then: the current code already calls av2_get_ccso_context() there). MB_MODE_INFOs live in the frame's mi grid until the
frame ends, so the pointers stay valid; the W3 trial truncates the log and relogs, so kept entries point at final blocks.
At replay the flags those neighbours carry must be their final ones (see Propagation).

## Pieces

1. **SymbolLog: a `kCcso` entry.** `void ccso(int plane, MB_MODE_INFO* first, const MB_MODE_INFO* n0, const MB_MODE_INFO* n1)`
   pushes `{plane, first, n0, n1}` to a side vector and logs `{kCcso, index}`. No CDF adapts while logging (nothing estimates
   bits from ccso_cdf). truncate() leaves the side vector alone (indices stay unique; unreferenced rows are harmless).
   replay() takes one more optional callable `(avm_writer*, int plane, MB_MODE_INFO* first, const MB_MODE_INFO* n0,
   const MB_MODE_INFO* n1)` and throws on a kCcso entry without it.
2. **BlockWriter::ccso(), non-oracle path only.** At a unit's start, for each plane with ccso_enable[p]: instead of
   `w_->symbol(1, ...)`, call `w_->ccso(p, first, xd->neighbors[0], xd->neighbors[1])`. Keep the rest as it is (the
   provisional 1 written into xd, `first`, sb_filter_control and the non-first blocks' copy): CodecState::ccso() overwrites
   them. The CELV2_ORACLE path stays exactly as now (oracle flags coded directly), so the oracle build still builds and
   gives the same bytes.
3. **Replay callback (encoder.cpp, next to cdef_index ~1710).** Per tile with `xd = st_->xd(t)`: return if
   `!ccso_frame_flag || !ccso_enable[plane]` (a plane the search turned off has no flags in the syntax); set
   `xd->neighbors[0] = n0, xd->neighbors[1] = n1` (xd->tile is the tile's already); v = first's flag for the plane;
   `avm_write_symbol(w, v, xd->tile_ctx->ccso_cdf[plane][av2_get_ccso_context(cm, xd, plane)], 2)`. sb_reuse_ccso is always 0.
4. **The search (CodecState::ccso() and core/ccso_search).**
   - **Units and groups (CodecState).** Unit size `log2 = get_ccso_unit_size_log2_adaptive_tile(...)` luma samples, `>> ss`
     in chroma (as apply_ccso() maps them). A unit's first MB_MODE_INFO is `mi_grid_base` at its top-left; units with the same
     pointer (a block larger than the unit) are one group and take one flag, exactly as CDEF's groups.
   - **Classification and offsets as now** (`search()` on the whole plane): keeps quant, support, edge_clf, bands, scale.
   - **Per-unit error changes** for that one classification: one more pass over the plane, the class of each sample computed
     exactly as ccso_apply.cpp classifies it (ccso_class_row(), shared with the filter), summing per (unit, class) the
     squared-error change of each of the 8 offsets `kOffset[i] x scale` (`CcsoDelta{d[8]}`), the filtered sample clipped to
     [0, 2^bd - 1] as the filter clips it (`n o^2 - 2 o s` from sums, plus a correction on the samples an offset clips).
     Only the classes the choice uses (at most 72 edge, 64 band), compacted. Parallel over unit rows; then summed by group.
   - **Flags then offsets, twice:** a group is on iff its error change at the current offsets is < 0; then each class's
     offset is re-picked (least error change over the on groups + its unary bits); repeat once more (or stop when the flags
     do not change).
   - **J of three ways, take the least:** (a) per-unit: the on groups' error change + sse_per_bit x (header bits as search()
     counts them + the refitted offsets' unary bits + the flags' bits); (b) every unit on, with search()'s offsets or ones
     refitted over every unit, + its flags' bits; (c) plane off: 0. (c) sets `ci->ccso_enable[p] = cur_frame
     ccso_info.ccso_enable[p] = 0`; `ccso_frame_flag = 0` when all three planes are off. **Flag bits** are an adaptive
     coder's estimate: in unit raster order each group's flag at its first unit, the context av2_get_ccso_context() gives
     there: a unit starts at a superblock's top, where fetch_spatial_neighbors() keeps only bottom-left and left, both in
     the left unit (2 if its flag is on, else 0; 0 at the frame's left edge, the tile's left edge not modelled), each
     context's odds counted as it goes (KT, from 1/2). (The first version counted 4 contexts from left and above, which
     the decoder never sees at a unit's corner.) All on pays a few bits too; groups x H(p) gave per
     unit no credit for flags that cluster nor all on its cost. On Angel Beats q160 the estimate is 91% of the coded bits.
   - **Write the result:** each group's flag into its first MB_MODE_INFO (`ccso_blk_y/u/v`), then **propagation**: every
     MB_MODE_INFO in the frame takes the flag of the first MB_MODE_INFO of the unit holding its (mi_row_start, mi_col_start),
     and `cur_frame->ccso_info.sb_filter_control[p]` per unit as span_ccso() leaves it. This is what the decoder's neighbours
     carry when it reads each flag (each block's flag is set once, when it is read or copied, from a unit read before it).
     Then the filters into ci and `av2_copy_ccso_filters()` and apply_ccso() as now; apply_ccso() already skips units whose
     flag is 0.
   Keep CcsoChoice's existing callers working (add an out-parameter or a second function; do not change search()'s result
   for callers that do not ask for units).

## Checks

- Tests: all existing (37/37 before this change); test_cli decode-equals-recon cases must still pass (8 and 10 bits, tunes,
  workers 1/2, `--device cpu` same bytes). Add a unit test of the flag decision (synthetic stats: groups an offset helps on,
  groups it hurts -> mixed flags; nothing helps -> plane off) and a test_cli case like cli_cdef_units: a clip whose halves
  differ gives some frame with a CCSO flag 0 on some unit (log it on the CELV2_GOP_LOG line) and AVM decodes it equal to
  `--recon`, 8 and 10 bits, B frames.
- Oracle build (`build_oracle.ps1` or however cce825d built it) still builds.
- BD-rate: `python -I tools/c17/bd_run.py old.exe new.exe <sources> <work> 33`, the 8 sources,
  global PSNR-Y, old = a build of 3035828. Bar: mean -0.3% or better, no source over +0.3%.
- Speed: `tools/c17/speed_ab.ps1` old vs new. CPU a frame over +1% or fps down over 2% on S1: report it, keep it on by
  default anyway only if the user accepts (as part 1); say what the extra pass costs in ms (CELV2_PROFILE's CCSO search).

## Measured (33 frames, 8 sources, old = 3035828, BD-rate global PSNR-Y; YUV = 6:1:1 of each plane's global PSNR)

| Step | Y mean | Y worst | YUV mean |
|---|---|---|---|
| first attempt, `Acc{sum, n}` stats, groups x H(p) | -0.109 | ab +0.265 | -0.706 |
| 1: exact clipped error changes | -0.109 | ab +0.265 | -0.706 |
| 2: + classification re-searched on the on units (not kept) | -0.100 | ab +0.287 | -0.793 |
| 3: 1 + adaptive flag-bit estimate, 4 contexts | -0.134 | ab +0.272 | -0.723 |
| diagnostic: flags on luma only (chroma as before), with 3 | -0.200 | md +0.060 | -0.169 |
| 4: 3 with the flag bits on the left unit's flag only, 2 contexts (built) | -0.146 | ab +0.268 | -0.737 |

- Step 1 changes 1 stream of 32 (md q100, by 0.0001 dB): almost no sample sits where an offset clips (Angel Beats'
  first frame has luma 64..239 at 8 bits, 256..956 coded); the first attempt's "saturated whites" diagnosis was wrong (dropping samples moved ab by changing the decisions, not by
  fixing a mis-prediction). It is kept as the exact statement of what the filter does.
- Step 2 adds 0.009 Y / -0.087 YUV: under the 0.2% bar, not kept (it would cost a second CCSO search on frames with a unit
  off).
- Step 4 (review finding: the estimate's contexts) per source, Y / YUV: air -0.523 / -0.929, md +0.056 / -0.109, am -0.001 /
  -1.621, ck -0.604 / -1.046, ab +0.268 / -0.223, yo -0.280 / -1.152, cr -0.030 / -0.367, pa -0.056 / -0.446; against 3: Y
  -0.012, YUV -0.014 mean (within noise, md's Y +0.075 the largest loss), kept as the decoder's real contexts.
- The PSNR-Y bar (-0.3% mean) is not met: the chroma flags buy large chroma gains (U/V +0.1..+1.1 dB at q160-190) that
  PSNR-Y cannot see, and cost up to 0.9% rate there (am q190, ab +0.5%); luma flags alone reach about -0.2%.

## Not in this part

CCSO reuse from references (sb_reuse_ccso, reuse_ccso): not touched.
