# Plan item 4: the in-loop filters off the frame's critical path (design)

2026-11-11. Plan item 4 (reports/2026-11-11-beat-av1-plan.md, chapter 2): start the filters' searches on rows as the
coder finishes them, or overlap the filter tail with work that does not depend on it, with the same bytes. Bar: S1 fps
+5% or better, CPU a frame at most +1%. **Verdict: the honest ceiling is under +5% on S1, so this stops at the design.**

## end_frame() today

The chain per reference frame (Pipeline::code(): launch, RD pass, Encoder::code(): rows, `st_->end_frame()`, replay,
keep(); the next frame's launch reads this frame's filtered picture):

```
rows (wavefront, all tiles) --> deblock --> CCSO pad (deblocked luma)
                                   |
                                   +--> CDEF search: [per-unit errors, 64 strengths] --> [set: cdef_choose_set()]
                                                      --> [damping on 128 sampled units at their pairs] --> CDEF
                                   --> CCSO search (reads the CDEF output) --> CCSO
                                   --> LR fit (chroma Wiener, reads the CCSO output) --> LR --> LR decide
                                   --> copy --> replay (needs CDEF index, CCSO and LR flags) --> next frame's launch
```

Measured on S1 (mxam600, 600 frames, `--workers auto --refs 7 --qindex 130 --tune animation`, CELV2_PROFILE=1,
HEAD 7f10003), ms a frame. The split of the CDEF search and the rows' finishing times come from a timing spike (not in
the tree: scratchpad `timing_spike.patch`).

**Measured with another encoder running** (another session's SVT-AV1 runs through ffmpeg, about 14 cores busy):
use the ratios, not the absolute ms. Three S1 runs (11.06-11.49 fps). The rows below are the spike's second run; the
first run (HEAD's build) gave the same split within 1-2 ms a stage.

| Stage (ms a frame) | Wall | Could start before the last row? |
|---|---|---|
| Reference frame's chain (Pipeline::code, means of 282) | 236.8 | |
| Rows (coder, all frames) | 133.9 | |
| Filters, total (all after the last row today) | 56.8 | |
| deblocking | 1.1 | yes, by SB row |
| CCSO pad | 0.3 | yes, by row |
| CDEF search: per-unit errors (64 strengths, every unit) | 10.5 | yes, by unit row |
| CDEF search: group sums and set (cdef_choose_set(), one thread) | 3.6 | no: all units |
| CDEF search: damping (3 dampings, 128 sampled units) | 7.2 | no: needs the set |
| CDEF | 1.3 | no |
| CCSO search | 17.9 | no: reads the CDEF output |
| CCSO | 1.5 | no |
| LR fit | 6.0 | no: reads the CCSO output |
| LR | 4.6 | no |
| LR decide | 2.8 | no |
| copy | 0.8 | no |
| Range coding (replay) | 2.6 | no: needs the filters' flags |

The rows' finishing times (S1 1080p: 9 SB rows of 128, 2 tiles), ms before the last row ends, means over 500 frames:
the last row 0.1-0.2, the one before 3.1-3.4, then 10.6-11.5, 19.0-20.7, 28.0-30.4, 38.5-41.2.

## What can start before the last row

- **Deblocking**: yes. AVM's MT deblock already runs by superblock row: row r's horizontal edges wait only for the
  vertical edges of rows r-1 and r (thread_common.c, sync_read). So rows 0..r deblock once rows 0..r are coded, and
  SB row r's pixels are final once row r+1 is (its top edge moves row r's bottom lines).
- **CCSO pad**: yes, per row of the deblocked luma.
- **CDEF per-unit errors**: yes. Each 64x64 unit's 64-strength errors (integers, summed per unit) need its deblocked
  pixels and 3 lines of border: the top unit of SB row r is ready when row r is coded, the bottom one when row r+1 is.
  The unit list and the groups (128 blocks) are known from the coded rows; the default damping from base_qindex.
- **Everything else: no.** The set is chosen over all units at once; the damping pass filters each sampled unit at its
  set pair; CCSO's statistics are of the CDEF output (so of the set and the damping); LR's fit is of the CCSO output.
  Speculating on any of them (all 64 strengths at the other three dampings, CCSO statistics per candidate pair) costs
  3x to 64x the CPU, far over the +1% CPU bar.
- **The replay** needs the CDEF indices, CCSO and LR flags; it is 2.6 ms.
- **The next frame**: every reference frame is a reference of the next frame coded (anchor, then the mids, each
  referring to the one before); the leaves already overlap (Pipeline::code_group()), and prepare() already starts the
  next frame's reference-free GPU work during this frame. What is left of it before the references (Encoder::begin,
  about 2.5 ms) is small.

## Expected gain

Of the filters' 56.8 ms, the part that can move ahead of the last row is deblocking, the pad and the CDEF per-unit
errors: 11.9 ms (21%). The rows do not finish evenly: the last two SB rows end within about 3 ms of each other, and 2
of the 17 unit rows wait for the last row whatever is done (their deblocking needs it). So at most about 10.5 ms leaves
a 236.8 ms chain: **+4.6% fps, the ceiling**, if the frame rate were bound by the chain alone and the moved work cost
the rows nothing. Under load the ratio is what counts; the quiet figures of the fps-cap and filter-wall reports
(chain 230 ms, CDEF search 9-17 ms) give the same range.

Neither holds at 3 workers: the 24 cores are near full (fps-cap report: 2.2 s of CPU a frame at 12 fps), so the
per-unit errors' CPU (about 100 ms a frame) moved into the rows' phase competes with the RD pass and the coding of the
same frames. The filter-wall change took 13 ms of wall and 100 ms of CPU off together for +5.1% fps; taking the wall
alone is worth less. The expected gain is about +2-3%.

The plan's +8-10% assumed half of the filters' wall could move. It cannot: after the CDEF per-unit errors, every stage
waits on a frame-global choice (the CDEF set), and CCSO and LR search the outputs of the stage before.

## How the bytes would stay the same (if built)

- Deblocking by row ranges in AVM's order (vertical then horizontal by SB row, as loop_filter_rows_mt()); the same
  edges, the same order per pixel.
- Per-unit errors are int64 sums over a unit, written to that unit's slot; which thread or when does not matter. The
  group sums and the set search stay after the last row, in unit order.
- No floating-point reductions move. The thread count and the order rows finish in do not reach any value.

## What would move fps more

These are the offset candidates for the next CPU cuts:
- The CDEF set and damping choice (10.8 ms after the last row, of which the set choice 3.6 ms on one thread) and the
  CCSO search (17.9 ms): CPU and wall cuts in place, not overlap.
- A cheaper damping choice (on fewer units, or kept from the previous frame at the same level) would change bytes.

## Checks done

None beyond the measurements: nothing was built, so no byte checks, tests or speed A/B. Item 4 is closed at the
design.

## Reproduce

```bash
CELV2_PROFILE=1 celv2enc -i sources/mxam600.y4m -o s1.ivf --workers auto --refs 7 --qindex 130 --tune animation
```

The split of the CDEF search and the rows' finishing times need the timing spike (stderr lines `TMP cdef err`,
`TMP set`, `TMP rows`), kept outside the tree.
