# Parallel tile coding: decide in a wavefront, then replay the symbols (2026-10-30)

The speed plan's largest item. At 1080p the tile coder is the critical path: 2 tiles, so 2 threads
code every frame, and their trials (the NEWMV / AMVD / compound / joint refinements, the RD mode
checks, the transform partition) are most of it (5.3 s of 11 s of tile time on 33 frames).

## Idea

Coding a tile does two kinds of work, and only one of them must be serial:

- **Deciding and reconstructing** a superblock needs its left and above neighbours finished (MV
  candidates, intra edges, BAWP and CfL neighbours, contexts from the mi grid and the above / left
  context arrays), but nothing about the entropy coder's state except the CDFs' probabilities,
  which only steer decisions through bit estimates.
- **Entropy coding** must run in raster order with the CDFs adapting symbol by symbol.

So pass 1 decides and reconstructs the superblock rows of a tile in a wavefront (row r runs two
superblocks behind row r - 1, as HEVC's WPP), each row with its own MACROBLOCKD, left context and a
copy of the CDFs taken from the row above after its second superblock. Every symbol a row writes
goes into a **symbol log** instead of a range coder: the CDF's offset in FRAME_CONTEXT, the number
of symbols, the value (and for literals / bits, the value and width). Pass 2 replays the rows' logs
in raster order into the tile's real range coder with the tile's live CDFs.

The decoder reads the same symbol values in the same order, so the stream is valid. The
reconstruction is pass 1's, because nothing in it depends on the CDFs: the coefficients come from
the TCQ / RDOQ cost model in quantizer steps, not from the adaptive CDFs.

## What changes in the output

Only decisions whose bit estimates read the CDFs (`symbol_bits()` on `ec_ctx`): with WPP-inherited
CDFs instead of the exact raster state they may pick differently. Expect well under 0.5% BD-rate,
to be measured. Stage 1 below must give byte-identical streams.

## Stages

1. **Symbol log and replay, still serial** (one row worker per tile = today's order and CDFs). All
   of BlockWriter's writes go through a `SymbolSink`: `avm_write_symbol` (78 calls),
   `avm_write_literal` (22), `avm_write_bit` (2), `avm_write_primitive_quniform` (1),
   `write_tu_quasi_uniform` (CelV2's, on top of those) and the one `avm_od_ec_encode_cdf_q`. In
   pass 1 the sink logs and updates the row's CDF copy as `avm_write_symbol` would; pass 2 replays
   into the `avm_writer`. Check: streams byte-identical to today on the 1080p sources, 8 and 10
   bits, 16K, and `test_random_decisions`.
   Done (`src/bitstream/symbol_log.h`): byte-identical on AIR OP 10 / 8 bits, AIR with delta-q and
   TMV, AMBITIOUS MISSION at the default q and at qindex 40, 4K and 16K; all tests pass. (The
   design's one `avm_od_ec_encode_cdf_q` was a comment, not a call.)
2. **Row workers in a wavefront.** One TileCoder per superblock row: its own MACROBLOCKD, mc buffers,
   left context and CDF copy; the above context arrays and the mi grid are shared (each row writes
   its own columns / blocks, and reads the row above only where the wavefront has finished it).
   The ref MV and warp banks already reset at each row start. The RD pass's RowGate already hands
   rows over as they finish. Lag 2 superblocks (above-right MV candidates and intra edges).
   Check: AVM decodes every stream equal to `--recon`; BD-rate on the five FHD sources; speed at
   1080p, 4K and 16K.
   One symbol depends on raster order beyond the neighbours: delta_qindex codes the difference to
   `xd->current_base_qindex`, the qindex of the last superblock that coded one, so a row's first
   superblock depends on the end of the row above (and a skipped whole superblock codes none).
   Its log entry must carry the superblock's qindex and the difference be taken at replay.
   Done: all superblock rows of all tiles are one row-major job on the thread pool (a row only
   waits on rows listed before it); row workers come from a pool in CodecState (MACROBLOCKD copied
   from the tile's, own CDFs and scratch), so memory follows the rows coded at once. Output does
   not depend on the thread count (`--threads 1` gives the same bytes). AVM decodes AIR OP 10 / 8
   bits, delta-q + TMV, qindex 40 and 4K equal to `--recon`. BD-rate against stage 1 (Y): AM -0.19%,
   yo -0.13%, AIR +0.57%, kaguya -0.60%, md +0.12%. Time 1.16-1.33x faster on a loaded machine;
   the clean speed run is still to do.
3. **Tiles in a frame and leaf B frames share the pool**: with row workers, a 1080p frame has ~14
   runnable rows instead of 2 tiles; the frame-parallel leaf B frames add more.

## Risks

- Hidden shared state in AVM's MACROBLOCKD / AV2_COMMON written during decisions (scratch buffers
  hung off `cm`, the CfL store, the is_mi_coded map, the TMVP rows of `copy_frame_mvs`). Stage 2
  starts with an audit of what `code_block` writes outside the block.
- A symbol whose presence or CDF choice depends on the range coder's state: none known (AV2 syntax
  never reads probabilities), and stage 1's byte-identity checks it.
- Memory: a log entry is ~8 bytes; a 1080p frame writes a few hundred thousand symbols.

Expected: 1080p ~1.5-2x (the tile path from 2 to ~14 threads), less at 16K, where 32+ tiles
already fill the cores.
