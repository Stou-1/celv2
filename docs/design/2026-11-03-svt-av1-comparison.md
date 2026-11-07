# Structure against SVT-AV1: the weaknesses (handoff)

2026-11-03. A read-only review of CelV2's frame pipeline against SVT-AV1's process structure, for a
later session to pick up. Nothing was built or measured for this note: the figures quoted come from
the design notes named, and several of them predate the wavefront coder (2026-10-30), so **profile
before you plan** (tools/sampler, `--threads`, the 1080p / 16K runs of the speed plan).

Read first: `2026-10-29-speed-plan.md`, `2026-10-30-parallel-coder.md`,
`2026-10-31-chunk-parallel.md`, `2026-10-28-cpu-rd-pass.md`, and `tried-not-kept.md` (look there
before planning; add a row for anything dropped).

## Where CelV2 already matches SVT-AV1

| SVT-AV1 | CelV2 |
|---|---|
| Mode decision as a funnel: MDS0 (cheap metric, many candidates) to the last stage (full RD) | GPU engine (SATD + fixed bit charges) -> CPU RD pass (luma DCT + TCQ estimate) -> tile coder trials |
| Segments handed to the next process as they finish | `RowGate`: rows published as the engine / RD pass finish them |
| EncDec (decide + reconstruct, wavefront) apart from Entropy Coding | Row workers in a wavefront logging symbols, replayed per tile (`encoder.cpp:916-939`) |
| Same output whatever the thread count | Same bytes on CUDA, Vulkan and CPU and at any `--threads` (tests) |

MDS0 on the GPU is something SVT-AV1 does not have. Chunk parallelism (`--workers`) is outside
SVT-AV1 (Av1an does it there).

## The weaknesses, in the order to look at them

### 1. Hidden mid frames are coded one after another

- **SVT-AV1:** pictures of the same temporal layer are encoded at once once their references are
  done; the picture decision process gives each picture its reference list.
- **CelV2:** an 8-frame mini-GOP codes the anchor, then the mids one by one (`chunk_coder.cpp:281-290`),
  then the 4 leaf B frames together (`code_group`, `chunk_coder.cpp:307-309`). With `--refs 7`
  `mids_of` (`chunk_coder.cpp:173-185`) gives frame 3 (level 1), then frames 1 and 5 (level 2): 4
  serial stages for 8 frames. Frames 1 and 5 depend on each other only because frame 5's reference
  slots hold frame 1.
- **Why it is in the way:** AV2 picks references implicitly: `begin_frame` refreshes slots
  (`codec_state.cpp:454-463`) and `av2_get_ref_frames` ranks every slot by display distance
  (`codec_state.cpp:486-487, 509`). Frame 5 therefore sees frame 1 whatever the encoder wants.
- **Expected:** about 20% at 1080p (speed plan item 5, estimated before the wavefront coder: measure
  again). Matters most with one worker (`--target-size` forces `--workers auto` to 1, which is what
  the GUI passes); with several workers the chunks already fill the machine.
- **Directions to check:**
  1. Whether AV2 v1.0.0 syntax lets a frame signal its references explicitly, and what it costs
     in bits per frame.
  2. Else a slot layout where the two level-2 mids cannot see each other (for example frame 1 does
     not refresh a slot that frame 5 ranks), at the cost of one reference for some leaves.
  3. Code frames 1 and 5 like `code_group` does: on a copy of the state, the engine deciding them in
     turn.
- **Done when:** AVM decodes every stream equal to `--recon`; BD-rate (Y) on the five 1080p
  sources within noise (state the bound before running); speed at 1080p with `--workers 1`.

### 2. Each frame's motion search waits for the previous frame's reconstruction

- **SVT-AV1:** hierarchical ME runs on the source pictures, many pictures ahead; only mode decision
  refines against the reconstructions.
- **CelV2:** the engine searches the references' reconstructions (`pipeline.cpp:128-136`), every
  pyramid level (`kMeLevels = 3`, `engine.cu:253-288`). `launch()` for frame N + 1 can only start
  after frame N's `keep()` (`pipeline.cpp:159-160`), that is after its coding, replay and in-loop
  filters. `prepare()` (`engine.cu:208-222`) overlaps only the next source's upload and intra costs.
- **When it matters:** not at 1080p as far as the old figures go (the engine is about 10 ms a frame
  on CUDA against ~170 ms a frame of coding at 6 fps). It does at 16K, where the engine alone took
  about 640 ms a frame (speed plan) and the frame cannot start until the one before is done, and on
  slow GPUs (the Arc Pro B50 ran 126 ms a 1080p frame).
- **Direction:** the coarse levels (and maybe a first full-scale pass) on the sources, queued for
  the frames ahead as soon as they are read; only the last refinement on the reconstruction once it
  exists. Changes the bytes. The CPU engine must give the same bytes as the GPU ones
  (`test_ippp cpu_pipeline_matches_gpu`).
- **First step:** time, at 16K and on the B50, how long each frame's `launch` waits for the frame
  before it, against the engine's own time. If the wait is small, drop this item.

### 3. The copies made for parallel coding take most of the memory

- **CelV2:** parallel leaf B frames code on `shadows_` Encoders (`pipeline.cpp:266-268`); the GOP
  trial runs a second Pipeline. The memory model (`main.cpp:501-530`) puts 1080p at 4.6 GiB with
  parallel B frames and 1.9 GiB without; 2160p 13.8 / 5.0 GiB (`2026-10-31-chunk-parallel.md`). The
  cap turns parallel B frames off first, so a machine with little RAM loses that speed.
- **Correction to the first review:** the reference frames are not simply copied each time.
  `sync_refs_from` (`codec_state.cpp:351-383`) keeps a buffer that already mirrors the same frame
  (`mirror_of_` / `gen_`) and copies only new ones. So it is not known what makes up the ~0.9 GiB a
  shadow costs at 1080p.
- **First step:** measure what one shadow holds (AVM frame buffer pool and borders, MV buffers, row
  worker pool, mc buffers, scratch). Then pick: share read-only reference buffers between states
  (SVT-AV1 keeps reference pictures in reference-counted objects), shrink the per-state pools, or
  leave it.
- **Done when:** the same bytes; peak private bytes at 1080p and 2160p against the table above.

### 4. TPL runs its own coarse motion search instead of using the engine's

- **SVT-AV1:** TPL reuses the motion estimation results.
- **CelV2:** `core/tpl.cpp` searches again on the CPU, SAD on the 8-bit luma at half scale
  (`tpl.h:1-4`), called per mini-GOP from `chunk_coder.cpp:191-214`.
- **Why it matters (quality, not speed):** `--deltaq` was dropped because on anime the half-scale
  SAD model calls moving characters unreferenced; its "revisit when" cell asks for "a finer TPL
  (full-scale SATD, cel-aware motion)" (`tried-not-kept.md`, the `--deltaq` row). The engine already
  computes full-scale SATD costs and quarter-pel motion per block.
- **Catch:** the engine searches reconstructions, one frame at a time, and TPL needs the whole
  mini-GOP's source motion before the anchor is coded. This depends on item 2 (an engine pass on
  sources ahead), or on an engine call on source pairs made only for TPL.
- **Done when:** BD-rate on the anime and live-action sources against HEAD with `--deltaq 1` and
  `--tplq`; the cost in speed.

### 5. In-loop filters at the end of each frame

- **CelV2:** `end_frame` (`encoder.cpp:940`) runs deblocking, the CDEF search and CCSO on the whole
  frame after the replay. SVT-AV1 runs them as processes on segments.
- **Size:** about 0.5 s over 30 frames of 1080p after the 2026-10-29 work (speed plan, "The end of
  frame"). Last in line; look again only if a profile after items 1-2 shows it.

## Not worth taking from SVT-AV1 (already measured)

- 128x128 / 256x256 superblocks: within +-0.3% (`tried-not-kept.md`, "AVM tools never built").
- Temporal filtering of anchors / keyframes: loses on anime, reasons recorded.
- CDF-based rate estimates in the RD pass: level bits tried without gain; only side bits (partition,
  mode, reference, MV) are untried.

## Rules that hold for every item

- `scripts\test.ps1` passes; AVM v1.0.0 decodes every stream equal to `--recon`.
- CUDA, Vulkan and CPU engines give the same bytes.
- A change that alters the bytes needs BD-rate on the 1080p sources (`scripts\bdrate.py`), quality
  first; record the numbers here or in the item's own note, and a `tried-not-kept.md` row if dropped.

## Measured 2026-11-03 (feat/intra-rd 613f72c)

AM 1080p 120 frames (one keyframe, so one chunk: `--workers auto` is the same 5.3 fps), `--qindex 130
--refs 7 --keyint 240 --tune animation --workers 1`, RTX 5090 + Xeon 48 threads. Stage times from
temporary timers in `code_minigop` (not committed); AIR OP and 吉村 have the same shares.

| Stage (15 mini-GOPs) | Time | Share of 23 s | Each |
|---|---|---|---|
| Anchor | 4.6 s | 20% | 0.31 s |
| Mid, level 1 | 3.5 s | 15% | 0.23 s |
| Mids, level 2 (two, one after the other) | 5.3 s | 23% | 0.19 s each |
| The 4 leaf B frames (at once) | 7.3 s | 32% | 0.49 s a group |
| TPL | 0.15 s | 1% | |

- **Item 1, revised estimate: about 7%, not 20%.** Four leaves at once take 2.6x one frame, so two mids at
  once would take about 1.4x one (0.27 s against 0.38 s): about 1.7 s of 23 s.
- **Item 1, direction 1: possible.** AV2 v1.0.0 has `enable_explicit_ref_frame_map` in the sequence
  header and, per frame, a flag, `num_total_refs` (3 bits) and a 3-bit slot per reference
  (`decodeframe.c` read_uncompressed_header): at most about 25 bits a frame.
- CPU: 589 s over 23 s, 25.6 of 48 threads busy on average. tools/sampler (92.9% of samples waiting):
  `fit_wienerns` (chroma WIENER_NONSEP normal equations, `codec_state.cpp`) 31.5% of the busy samples;
  TCQ (quantize_tcq, tcq_cands, tcq_cost, tcq_prep) 11%; the DCT (dct_lanes, fwd_dct, transpose) 11%;
  CDEF filtering 9%.
- **New, ahead of item 1: `fit_wienerns` is 2.5 s of the 23 s (11%)**, twice a frame at its end, where
  the next frame waits (item 5's "end of frame"). It sums 18x18 int64 outer products per chroma sample;
  the sums are exact integers, so doubles with AVX2 FMA (products below 2^24, sums below 2^53) give
  the same bytes.
- Done the same day: the sums in doubles, 4x4 tiles of a row's samples in AVX2 registers. The same bytes
  (four streams, 8 and 10 bits); `fit_wienerns` 2.53 -> 0.95 s, the encode AM 5.26 -> 5.43 fps, 吉村
  5.70 -> 5.81 fps. The 2.5 s was summed over the leaves' concurrent calls, so its wall share was
  below 11%. What is left of it is filling the samples (scalar).
- **Item 1: deferred (2026-11-03).** About 7% at 1080p with one chunk, the same bytes otherwise, for a
  change across the sequence and frame headers (explicit reference maps), the encoder's reference
  ranking (`av2_get_ref_frames` replaced for the second mid), copying the first mid's slot (buffer,
  MVs, CDFs) between two states, and the engine's reference list. Frame 5 would also lose the oldest
  reference (frame 1 takes its slot). Revisit when the other per-frame costs shrink so the mids' share
  grows, or when one-chunk encodes (long scenes, `--target-size`) become the speed target.
- **Item 2, first step (2026-11-03): the wait is large; keep the item.** 16K (a16k, 9 frames, q130,
  `--refs 7 --workers 1`): the encode 5.4 s a frame on CUDA and on Vulkan with the RTX 5090 (the same
  bytes), 32.2 s on the Arc Pro B50. The engines alone (engine_bench, 2 references): CUDA 1.09 s,
  Vulkan 5090 1.21 s, B50 9.68 s a frame. On the B50 the frame time is about the engine's (7
  references) plus the CPU's 4.3 s, so the two barely overlap; moving the coarse search onto the
  sources ahead would hide up to that 4.3 s there (about 13%), and on the 5090 up to the engine's share.
- **Item 3, first step (2026-11-03): the shadows are smaller than the memory model said.** AM 120 frames,
  peak private bytes, 1080p / 2160p (Lanczos upscale): parallel B frames 2.90 / 8.18 GiB (the model
  4.63 / 13.8), without them 1.90 / 5.06 (the model 1.87 / 4.98), with the GOP trial too 2.92 / 8.47
  (the model 8.35 / 26.0, but the trial may not have run much on this clip). Three shadows cost about
  1.0 / 3.1 GiB, 0.34 / 1.04 GiB each. The parallel-B term is now 1.14 + 0.85 Mpx, so a machine with
  little RAM keeps parallel B frames longer; sharing buffers between states is not needed for now.
- **Item 5, measured (2026-11-03, after the Wiener fit change):** `end_frame` 4.3 s summed over AM's 120
  frames (36 ms a frame; the leaves' run at once): deblocking 0.08 s, CDEF 1.06 s, CCSO 1.06 s, loop
  restoration 1.85 s (the fit 0.95 s of it), the rest 0.04 s; 吉村 the same. About 3% of the wall time
  each: left as it is.
- **Where this leaves the review:** item 2 is the one speed item with a large measured wait (16K, slow
  GPUs); item 4 waits on it; items 1, 3 and 5 are done or set aside above.
