# Speed plan after the 2026-10-29 work (plan step 3 and the Vulkan engine)

Where the time goes now, and what is left to try, ranked by expected gain. Figures are 1080p
anime / live action, 33 frames, q130, on the RTX 5090 + Xeon (24 cores, 48 threads) unless noted.

## Where the time goes

- 1080p codes at about 5.9 fps (5.64 s for 33 frames), from 4.2 fps in the morning, with the same
  bytes.
- The critical path is the tile coder. 1080p has 2 tiles, so 2 threads do the coding. Each
  8-frame mini-GOP codes 4 frames one after the other (the anchor and 3 hidden mid frames) and the
  4 leaf B frames together.
- The tile coder's own time (33 frames, summed over threads, 11 s):
  - the inter mode search's trials (prediction, DCT, TCQ cost): 5.3 s, of which the DCT is 2.1 s,
    the TCQ cost 2.1 s and the motion-compensated prediction 0.85 s;
  - the transform partition choice: 1.4 s;
  - the residual coding: 1.2 s;
  - waiting for the RD pass's rows: 1.4 s.
- The RD pass uses 113 s of CPU over 44 threads (full TCQ 31 s, DCT 24 s, error and bits 9 s,
  prediction 7 s). It is mostly hidden behind the tiles, but it delays each frame's first rows.
- CDEF and CCSO run at the end of every frame, before the next frame can use it as a reference:
  1.5 s over 33 frames before today's changes. They are less now (the planes are searched at
  once and the CDEF search runs on more threads), but they have not been measured again.
- 16K (8 frames) codes at 0.34 fps. The GPU decision alone takes about 640 ms a frame on CUDA,
  more than the 500 ms that 2 fps allows, so 16K needs both the GPU and the CPU faster.

## The end of frame (done, the same bytes)

The in-loop filters' searches and copies at the end of each frame hold up the next one. Timed on
1080p anime A (30 frames) and live action (33), and 16K (9), while 16 avmenc runs shared the CPU:

| | before | after |
|---|---|---|
| anime A: end of frame | 1.15 s | 0.51 s |
| live action: end of frame | 1.24 s | 0.60 s |
| 16K: end of frame | 2.71 s (4.8 s before the CCSO filter) | 1.93 s |
| encode, anime A / live action (against the binary before plan step 6) | 6.17 / 4.34 fps | 6.97 / 4.72 fps |

- A thread pool (`core/threads.h`, `CelV2::parallel`) for every per-frame parallel step: starting up
  to 48 threads several times a frame cost more than 1080p's searches did.
- The CCSO search counts in tasks of 64K to 1M samples into 32-bit tables merged in slices on the
  pool: 1080p chroma had been counted on one thread (the 1 MB tables were merged one by one).
- The CDEF search's squared error is AVX2 for the blocks inside the picture: it cost as much as the
  64 strengths' filtering. The 3 damping trials are 0.5 ms each; the full search ~6 ms a 1080p frame.
- Left: 16K's CCSO counting (2.8 s CPU over 9 frames, scattered adds into the tables).

## Encoder: what is left

Same bytes (free to take):
1. **The DCT core.** The float Lee transform is about 2.5 s on the tile path and 24 s of RD pass
   CPU. Its operation order has to stay the same. Only scheduling is left (fewer passes over
   memory, the 64-point case keeping 32 outputs). Expect 10-20% of the DCT.
2. **The RD pass's error loop.** The divisions and sums must stay sequential to stay exact, so
   restructuring is limited. Its latency shows up as the tiles' 1.4 s of waiting.
3. **The AVM code on the hot path** (inter prediction, the inverse transforms): plan step 6, CelV2's
   own AVX2 versions checked bit for bit against AVM.

The bytes change (needs a BD-rate check on the 1080p sources, quality first):

4. **The parallel RD pass with a serial coding pass** (agreed 2026-10-28). The coder's inter
   trials (MV refinement, adaptive / joint MVD, compound refinement, warp) move into the parallel
   pass, which estimates the entropy contexts. The serial pass then only codes. This removes most
   of the 2-thread limit: the 5.3 s of trials would spread over 44 threads. It is the largest gain
   left, and also the largest change.
5. **Mid frames in parallel.** The two level-2 hidden frames of a mini-GOP are independent if the
   later one does not reference the earlier. AV2 picks references from the slots implicitly, so
   this needs explicit reference signalling. Expect about 20% at 1080p, with a small BD cost to
   measure.
6. **More tiles at 1080p, as an opt-in speed setting.** 4 tiles would nearly halve the serial
   frames' coding time. Expect about +0.5-1% bytes (4K: +0.6% for 2 tiles). Not for the default.

## Vulkan engine: what is left

The engine now queues as CUDA does and matches its bytes on both GPUs. Engine alone, ms a frame:
1080p 13.7 (CUDA 10.4), 4K 37.6 (CUDA 41.2) on the RTX 5090; the Arc Pro B50 126 / 397.

1. **Timestamps first.** Add a `VkQueryPool` of timestamps around each kernel, so the next steps
   are measured per kernel, not guessed.
2. **Narrower barriers.** Every step is followed by a full memory barrier, so the references'
   motion searches, which do not depend on each other, run one after another. Barriers only on
   real dependencies (each reference's pyramid, then its ME chain) let them overlap. This is the
   most likely cause of the 1080p gap to CUDA.
3. **Fewer submits.** Today there is one submit per superblock row (17 at 1080p, 135 at 16K).
   Grouping 2-4 rows a submit after the first rows keeps the first rows early with less driver
   time.
4. **No staging copy on resizable-BAR devices.** Write uploads straight into device-local,
   host-visible memory where the device offers it.
5. **Quarter-pel phases on the fly.** Each reference keeps 16 phase planes (2.1 GB at 16K). This
   is what limits the B50's 16 GB: 7 references fill it. Computing a phase in the cost kernel from
   the integer plane trades memory for arithmetic. Measure on both GPUs; keep the stored phases
   where they are faster.
6. **The B50.** It is about 9x behind the 5090, close to their compute ratio. The kernels already
   sum in 32 bits, so more gain needs less work per candidate (the same lever as CUDA's at 16K),
   or work-group sizes tuned to its subgroups of 16, which needs the timestamps above.

## Order

1. Vulkan timestamps (1), then barriers (2) and submits (3).
2. Encoder: the parallel RD pass (4), the largest gain. Run the mid-frame parallelism (5) and the
   DCT scheduling (1) alongside.
3. 16K: the GPU decision is over budget on both APIs. After (4), profile 16K again and set the
   GPU's per-candidate work against the 2 fps target.
