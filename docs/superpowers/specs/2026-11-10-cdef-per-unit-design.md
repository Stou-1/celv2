# C2 part 1: CDEF strengths per 64x64 unit (design)

2026-11-10. From docs/reports/2026-11-10-c2-filter-split.md: the oracle puts per-unit CDEF strengths at 0.7-1.8% BD-rate
(mean 1.3%, every source). Encoder only, no format change: AV2's syntax already has the frame's strength set and a
per-unit index; CelV2 codes one strength (`nb_cdef_strengths = 1`, no index symbols).

## The problem and the way round it

The index is a block-level symbol (read_cdef(), avm av2/decoder/decodemv.c:74), read at the first non-skip block of each
64x64 unit, but CelV2 searches CDEF after the tiles are coded (CodecState::end_frame() -> cdef()). The tiles are
already logged, not range coded: SymbolLog (src/bitstream/symbol_log.h) is replayed after end_frame()
(src/core/encoder.cpp ~1720), and LR's unit flags already ride on that as `deferred()` symbols. The CDEF index does
the same, with one difference: its CDF context depends on the neighbour units' final indices, so it is worked out at
replay, not at logging.

## Pieces

1. **SymbolLog: a `kCdef` entry.** `void cdef(int mi_row, int mi_col)` logs the block position (v = mi_row << 16 |
   mi_col); no CDF adapts while logging (nothing estimates bits from the CDEF CDFs). `replay()` takes one more optional
   argument, a callable `(avm_writer*, int mi_row, int mi_col)`, and calls it for each kCdef entry (throw if absent).
2. **BlockWriter: log it where read_cdef() reads it.** At both places the oracle calls `import_cdef()`
   (block_writer.cpp ~165 intra, ~867 inter), in the non-oracle path: if `seq_params.enable_cdef` and not
   `coded_lossless`, mirror read_cdef()'s bookkeeping exactly as `import_cdef()` (tools/oracle/block_import.inc:287)
   does: zero `xd->cdef_transmitted` at a superblock's first block, index = av2_get_cdef_transmitted_index(), return if
   transmitted, if skip_txfm[0] (and cdef on skip is off, as cdef() sets it) set nothing and return, else
   `w_->cdef(mi_row, mi_col)` and mark transmitted. The W3 trial (encoder.cpp ~1775) restores the whole MACROBLOCKD,
   so `cdef_transmitted` comes back with it.
3. **Replay callback (encoder.cpp).** Per tile, with `st_->xd(t)`: return if `!cdef_frame_enable` or
   `nb_cdef_strengths == 1`; set xd->mi_row / mi_col; v = mi_grid_base[fetch_cdef_mi_grid_index(cm, xd)]->cdef_strength
   (throw if v < 0); write `v == 0` on `tile_ctx->cdef_strength_index0_cdf[av2_get_cdef_context(cm, xd)]`, then if v and
   n > 2, `v - 1` on `cdef_cdf[n - 3]` with n - 1 symbols. The contexts read the neighbours' final indices, which the
   decoder has also read by then (left and above units are read before the unit in any partition order).
4. **The search (CodecState::cdef(src, sse_per_bit), codec_state.cpp:957).** Keep the coded-unit list and the -1 for
   uncoded units.
   - **Groups:** units whose first-grid MB_MODE_INFO is the same pointer (a block of 128 wide or tall covers two or
     four units, and the decoder reads one index into it) are one group and take one index.
   - **Per-unit errors:** every coded unit (not the 128-unit sample), all 64 strengths, luma and chroma, at the
     default damping, from `cdef_block_errors()` as now, stored per unit (units x 2 x 64 int64), in parallel over units.
   - **Set:** candidates are the luma strengths that are some group's best (most often first, at most 12) plus the
     summed best and 0; chroma likewise at most 6 plus the summed best and 0. For n = 1..8: greedy add the pair that
     lowers sum over groups of min over the set the most, then two refinement passes (each slot re-picked with the others
     fixed). J(n) = error + sse_per_bit x (bits): header 3 + per pair 2 x (1 + (s < 4 ? 2 : 6)); indices by the
     groups' entropy (index-0 flag, then among the rest; 0 bits at n = 1). Order the set by use, the most used index 0.
     Also J of off (all at strength 0, no index bits). Take the least; off, or n = 1 at (0, 0), means
     `cdef_frame_enable = 0`.
   - **Assign** each group its least-error index; write it into the group's MB_MODE_INFO.
   - **Damping:** as now, on the sampled units (every step-th), each filtered at its own pair (the `only` path made
     per-unit) at dampings 3-6 other than the default; keep the least. The set is not re-chosen.
   end_frame() passes its sse_per_bit to cdef().

## Checks

- Tests: test_cli's decode-equals-recon cases (8 and 10 bits, tunes, workers 1/2, `--device cpu` same bytes) must pass;
  add one to test_cli that a 1080p-ish anime clip's stream has nb_cdef_strengths > 1 on some frame (or a unit test of
  the set search: groups whose best strengths differ get two pairs when the error saved is large against sse_per_bit,
  one when it is small) and AVM decodes it equal to `--recon`.
- BD-rate: `tools/c17/bd_run.py`, 33 frames, the 8 sources, global PSNR-Y, against d630ed6. Bar: mean -0.5% or better,
  no source over +0.3%.
- Speed: `tools/c17/speed_ab.ps1` against d630ed6. CPU a frame over +1%, or fps down over 2% on S1: report, do not
  default it (the speed-offset rule: an offset must be found or the user asked).
- Oracle build (`build_oracle.ps1`) must still build; its import_cdef path stays as it is.

## Not in this part

CCSO flags per unit (part 2, the same deferred-symbol route). Deblocking deltas and CDEF on skip: not worth it (the
split report).
