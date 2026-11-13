# Vulkan kernel speed-ups after the motion search (speed-first step 5, C23): plan and review

2026-11-09. Written by a design agent and checked by a second one trying to refute it (read only, nothing built or run). Verdict: **build with fixes**. Fold in the review before building; measure every step.

**Update, same day.** The premise of item (1), the B50's cost kernels slowing after the new motion search, was most likely an artifact. The reference engine_bench binary (built earlier from the same source) ran about 15% faster on the B50 at 16K than a fresh build of that source. Fresh builds of HEAD and of the new search were equal at 16K. The motion search was then committed for frames up to 4K, judged against a fresh HEAD build. Compare fresh builds only, alternating, and against the A-against-A spread.

## Design

(1) The likeliest cause of the B50's slower cost kernels is a lower GPU clock, from the 70 W card's power or thermal limit, now that the dense 64-thread me keeps every core busy. cost.spv and its inputs are unchanged. cost intra slowed too (+31-37%), and it mostly runs in prepare()'s batch after all 17 row batches (engine_bench.cpp:51-52), so the cache or scheduling state that me leaves cannot reach it. The stamp barriers are the same in both runs, and the B50 runs one dispatch at a time anyway (step0 report:64). Contention from the other measurement now running is the second suspect; one ABAB run with telemetry tells these apart.

(2) While I read, the working tree changed. me.slang is again the committed one-thread kernel, the 64-thread one is now me_wide.slang, and vk_kernels.cpp:199-208 picks it for levels of at most 480x270 blocks (vk_kernels.h:60). That is an allowed by-level rule: keep it, tune the limit on untimed wall time, and make me_wide cheaper so the limit can grow.

(3) The disassembly of build/vk_shaders/cost.spv shows 9 loop- or data-indexed Function arrays, 138 dwords in all. 138 dwords x 4 B x 32 lanes is 17664 B, against the B50's measured 18304 B a thread, so they very likely all live in scratch. Removing them is the top B50 item. The next items, for both GPUs:
- the repeated-candidate scan, about 346k lane-iterations per superblock and reference (estimate), which every lane of a slot repeats;
- kShares, which gives the 5090 only 120 workgroups a row at 1080p;
- finalize's 64x64 compound SATD, run by one lane.

Merging finalize into the cost dispatch is not recommended. Every change here keeps integer results bit-identical.

### Changes

**docs/reports/2026-11-08-speed-step0.md**

Correct lines 86-88. With 10 frames and 7 refs, engine_bench launches only frames 7, 8 and 9 and averages frames 8 and 9 (tools/engine_bench.cpp:44-57), so the CELV2_VK_TIMES totals cover 3 launches. Per launch: 5090 21x10/3 = 70 ms (the 70.1 ms wall), B50 234x10/3 = 780 ms (the 758 ms wall). Both engines are all kernel time; there is no unexplained host time. Today's numbers per launch: B50 me 533 -> 233 ms, cost inter 237 -> 294 ms, sum about 843 -> 613 ms, matching the 820 / 615 ms walls. From now on, run at least 30 frames and divide the totals by (frames - refs).

**(experiment for 1, no code first) engine_bench runs**

Use the working tree's binary. A = CELV2_VK_ME_WIDE_MAX=0 (one-thread me everywhere, vk_kernels.cpp:203-208); B = unset (me_wide on every 1080p level). Run ABAB x3 of engine_bench g33_am30 1080p, 30 frames, refs 7, CELV2_VK_TIMES=1.
- Log the B50's telemetry at 10 Hz or faster: PresentMon --track_gpu_telemetry (GPU frequency, power, temperature, power/thermal/current-limited flags), or GPU-Z's sensor log.
- Log per-process GPU engine utilisation (Get-Counter '\GPU Engine(*engtype_Compute)\Utilization Percentage'), so another process on the B50 is ruled out. GPU time stamps include time sliced to other contexts.

Readout:
- Lower clock during B's cost rows plus a power-limit flag: power. A thermal flag: thermal.
- Other PIDs on the B50: contamination; rerun quietly.
- Same clock, cost still slower: go to the gauge below.
- Also compare untimed walls: if B's shortfall against its me saving (today wall -205 ms against me -300 ms a launch) is the same without CELV2_VK_TIMES, the stamp barriers are not the cause.

**src/vk/vk_engine.cpp, CMakeLists.txt:89 (only if the clock did not move)**

A gauge:
- Add probe_plain to celv2_vk's shader list.
- Under CELV2_VK_GAUGE=iters, dispatch it before each row's costs (vk_engine.cpp:263): a fixed 64 groups x 256 threads, named "gauge". It is ALU only, so its time scales with 1/clock.
- CELV2_VK_ME_GAP_MS=n: wait for the me batch, then sleep, before the row batches. A power average recovers within about 1 s; a thermal limit does not.
- A second costs dispatch in row 0 named "cost inter again" (it rewrites the same values): if only the first is slow, the cause is a cold cache.

None of this changes output bytes. Drop it after the diagnosis.

**src/vk/vk_engine.cpp:19, :262-263**

Choose kShares by frame size: int cost_shares(int sb_cols) { e = getenv("CELV2_VK_SHARES"); return e ? atoi(e) : std::clamp(960 / sb_cols, 4, 32); } for m.k.costs and the per-row m.k.intra. prepare()'s whole-frame intra (:164) keeps 4: it already has 2040 workgroups at 1080p, and every share recomputes dcv.
- 1080p gets k = 32: 960 workgroups a row instead of 120 (the 5090 has 170 SMs; the B50 fits about 128 such workgroups at once).
- 4K 16, 8K 8, 16K stays 4.
- Sweep 4/8/16/32/64 at 1080p and 4/8 at 16K with the env var, and sweep again after the shader items below.

**src/vk/shaders/common.slang:38-53 (hadamard4x4), :106-116 (satd8_q), :277-291 (block_compound_satd), cost.slang:20-28 (sub_satd), satd.slang:25**

Remove every loop-indexed private array. In cost.spv these are d[16] (%158), its by-value copy into hadamard (%159), t[16] (%151) and v[4] (%152). The 16 OpLoopMerge in cost.spv show the loops are not unrolled, and sub_satd's body inlines intra_pred, so the driver does not unroll them.

New form, exact:
  int4 hrow(int4 d) { int a0 = d.x+d.y, a1 = d.x-d.y, a2 = d.z+d.w, a3 = d.z-d.w; return int4(a0+a2, a1+a3, a0-a2, a1-a3); }
  public int hadamard4x4(int4 d0, int4 d1, int4 d2, int4 d3) { int4 t0 = hrow(d0), t1 = hrow(d1), t2 = hrow(d2), t3 = hrow(d3); int4 a0 = t0+t1, a1 = t0-t1, a2 = t2+t3, a3 = t2-t3; int4 v = abs(a0+a2) + abs(a1+a3) + abs(a0-a2) + abs(a1-a3); return v.x+v.y+v.z+v.w; }

Each caller builds its 4 residual rows with 4 explicit calls of a helper int4 res_row(i), with no d[16].

**src/vk/shaders/common.slang:58-74 (kSubpelTaps, subpel_tap, subpel_px), :212/:226 (kAngles), :139 (me_cell ox/oy)**

Remove the data-indexed arrays.
- subpel_tap(q, k) becomes a select chain on q (q==1 ? row1[k] : q==2 ? row2[k] : q==3 ? row3[k] : row0[k]), with [ForceUnroll] on subpel_px's i and j loops so k is a constant. This removes cost's two int[4][8] copies (%153, %155) and their int[8] rows (%154, %156), which the SPIR-V refills with OpStore at every call site, and finalize's four copies.
- kAngles[mode-7] becomes mode==7?45 : mode==8?67 : mode==9?113 : mode==10?135 : mode==11?157 : 203 (%157).
- me_cell's ox[5]/oy[5] (%95, %96 in me.spv) become me_wide's centre() ternaries.

Check with CELV2_VK_STATS on the B50: the target is scratch 0 for cost, finalize, me and me_wide (today 18304 / 3840 / 2816 B). If spills remain, make intra_part a specialization constant (VkSpecializationInfo in Context::kernel_words, vk_context.cpp:415), so the inter pipeline carries no intra_pred.

**src/vk/shaders/cost.slang:73-79, :87**

Split repeated_candidate() over the slot's lanes. Today every active lane of a slot repeats the full scan j < k (common.slang:264-271), each step with a candidate_mv_index containing an integer % and / by cw. My estimate: about 346k lane-iterations per superblock and reference, roughly 19M instructions against about 26M for the SATDs.

New code:
  mx/my = field[candidate_mv_index(kk)] as now;
  uint rep = 0;
  if (inter && active) for (int j = q0; j < kk; j += lanes) { int ij = candidate_mv_index(x, y, w, h, j, cols, rows); rep |= (mvx[r][ij] == mx && mvy[r][ij] == my) ? 1u : 0u; }
  for (off = lanes/2; off > 0; off >>= 1) rep |= WaveReadLaneAt(rep, min(lane+off, 31));   // as :87, executed by all lanes
  repeated = WaveReadLaneAt(rep, slot*lanes) != 0;

The 64x64 block's 71 steps become 3 a lane. Follow-up J1: a host table inter_cell[kInterSlots] with the candidate cell as (dcx, dcy) relative to (8*sbx, 8*sby), built in make_tables (vk_kernels.cpp:85-113) next to inter_k. The shader then clamps sbx*8+dcx and sby*8+dcy, with no division.

**src/vk/shaders/finalize.slang:73-108**

Remove rc/rmx/rmy (int64 rc[7], int rmx[7], rmy[7], indexed by r and by a/second) with a one-pass top-2 inside the r loop:
  if (a < 0 || rcr < c0) { second = a; c1 = c0; sx = ax; sy = ay; a = r; c0 = rcr; ax = mx; ay = my; }
  else if (second < 0 || rcr < c1) { second = r; c1 = rcr; sx = mx; sy = my; }
Write c_rcost/c_rmvx/c_rmvy[r][b] inside the loop. A second loop for r in [nrefs, 7) writes 0xffffffff / 0 / 0.

**src/vk/shaders/finalize.slang:156-185**

Replace stack_level/qx/qy[16] (indexed by sp) with a stackless pre-order walk. Track (level, m), m the Morton index. Square = (qx from m's even bits, qy from its odd bits). On split: emit 3, level++, m *= 4. Otherwise emit the leaves, then: while (level > 0 && (m & 3) == 3) { m >>= 2; level--; } if (level == 0) break; m++.

**src/vk/shaders/finalize.slang:92-96, :129-154; vk_kernels.cpp TableData; common.slang Tables**

Spread the compound SATD over the workgroup. Today warp 0's lane 0 does the 64x64 block's 4096 samples alone (b = 0 at :68 and :93); this is my reading of the 5090's 0.8 ms a row on 30 workgroups, refs 2 and 7 alike.
- Phase 1 (per block, as now): store a, second (or -1 when !compound_pair_ok), both MVs, ci and the intra c/m in groupshared; g_csum[b] = 0. Barrier.
- Phase 2: for g = tid; g < 3072; g += 256, over a host table ctile[3072] = b | tx << 9 | ty << 13 covering every 4x4 tile of the 425 blocks: if the pair is set, compute the tile's Hadamard of cur - (r0 + r1 + 1) >> 1 and InterlockedAdd(g_csum[b], h). Barrier.
- Phase 3 (per block): cc = int64_t(g_csum[b]) * 256 + lam * compound_bits(a, s); if (cc < ci) {...}; then the SbOut writes and best_* as now.
- Then the DP (:130-154) runs one level at a time over all threads: for i = tid; i < n*n; i += 256, then a barrier. Thread 0 keeps only the emission.
Groupshared grows from about 16 KB to about 25 KB.

**src/vk/shaders/me_wide.slang:25-58, :114-130; vk_kernels.cpp:199-208**

Five steps, each exact:
(a) least() with a packed key and one barrier. Search: key = cost << 9 | idx, cost <= 64*255 + 4*16 = 16384. Refine: key = cost << 4 | t, cost <= 4*65280 + 16, and threads >= 9 use 0xffffffff. m = WaveActiveMin(key); the first lane of each subgroup does InterlockedMin(g_min[n%3], m); barrier; read; thread 0 resets g_min[(n+1)%3] before that barrier. Three rotating slots make one barrier safe; today's version uses 8 (:25-43).
(b) Per search, load the reference window into groupshared int g_win[24*24], with the same clamped pc.ref.at(by*8+py-8+k/24, bx*8+px-8+k%24), and the cur block into g_cur[64]. The 289-position SAD loop then reads g_win[(dy+ii)*24 + dx+jj] with constant offsets: no clamps, no global loads.
(c) Refine split over (position, 4x4 tile): 36 tasks, InterlockedAdd into g_rs[9], then least over 9.
(d) Per-level names for CELV2_VK_TIMES ("me l0" ... "me wide l4", level passed from vk_engine.cpp:236).
(e) Re-sweep CELV2_VK_ME_WIDE_MAX over 32640 / 129600 / 518400 / inf at 1080p, 4K, 8K and 16K on untimed wall time. Timed runs put barriers on both sides of each dispatch (vk_context.cpp:498-514), which serialises the 7 references' searches that V3 overlaps when untimed (vk_engine.cpp:229-241), so stamps overstate the one-thread kernel's underfill.
Several blocks a workgroup sharing one window: not worth it. A block's window is 576 loads against 18,496 SAD reads (3%).
If a gap remains at mid sizes, use T threads per block (1/4/16/64) as a specialization constant chosen per level.

**src/vk/vk_kernels.cpp:96-104 (intra chunks), cost.slang:57-63**

Make intra chunks mode-major: a warp runs one mode across 'per' blocks of the same size, instead of 'per' modes of one block. Today an 8x8 chunk carries 8 modes, so intra_pred diverges up to 8 ways per sample.
- Table entry: (first index into a host block list intra_blist grouped by nsub, mode, count, lanes).
- Shader: b = intra_blist[first + slot], task = mode.
- The output slots b*13 + mode stay the same.

**src/vk/shaders/cost.slang (later)**

Load the superblock's source into groupshared once, as packed bytes. Rows oy-1 .. oy+95 and columns ox-1 .. ox+95 (an intra block of at most 32 has w+h-1 <= 63), with s.at()'s clamps: about 9.7 KB. Inter needs only the 64x64 part. cur and the intra edges then come from groupshared.

**src/vk/vk_engine.cpp:248, :260-268 (gated)**

Several rows a batch for small frames. R = max(1, 128 / sb_cols) contiguous rows (R = 1 when row_streams > 1, pipeline.cpp:83). Size m.costs for R rows and call k.costs/k.finalize with nrows = R: the kernels already take nrows (cost.slang:34/48, finalize sby = row0 + gid.y). Keep it only if the whole-encode fps gains on both GPUs, since collect() then gets rows R at a time.
Not recommended: merging finalize into cost with a last-workgroup counter. It needs device-scope release/acquire between workgroups on BDA pointers, and m.costs is reused every row, so stale L1 lines are a real risk.

**tests/test_vk_kernels.cpp**

Add costs_any_shares: Kernels::costs on a 3-ref, 202x122 frame for k in {1, 4, 32} must give byte-equal buffers.
Extend me_matches_ref (it already runs both me kernels via CELV2_VK_ME_WIDE_MAX, working tree :127-128) with a predictor field of +-40 so windows clamp at the frame edges.
Keep intra_costs_match_ref (k = 2) and satd/subpel as they are.

### Exactness

Every kernel here computes integers only (int, int64), so the float rules (operation order, no FMA) are not involved. Integer sums stay exact in any order within these bounds: a 4x4 Hadamard is at most 16*4080 = 65280; a 64x64 SATD at most 256*65280 < 2^24, which fits the int32 groupshared atomics.
- Hadamard as int4: the same row and column adds and subtracts as common.slang:40-52, and the same |v| terms.
- Taps and angles: the same constants. intra_pred calls dir_pred only for modes 7..12 (kNumModes 13), and subpel_px keeps its ty==0 skip.
- Split repeated scan: OR over j<k of (MV_j == MV_k) is exactly repeated_candidate's boolean. lanes is a power of two (min(nsub, 32)), the shuffle tree runs in uniform flow like :87, an inactive segment's head reads 0, and slot*lanes <= lane <= 31.
- J1: ox = 64*sbx, so x/8 = 8*sbx + xr/8 and cw does not depend on ox; the clamped cell is the same.
- kShares: the unit loop u = share*8 + warp + j*8k (cost.slang:54) gives each unit to exactly one warp, which writes its own slot (vk_engine.cpp:18 says the same).
- Finalize top-2: by induction the pair stays (first argmin of rc[0..r], first argmin of the rest). On v < c0 the old first becomes second; on v >= c0, strict < keeps the earlier index on ties. This equals the two loops at :76-90, including when all rc are kMax.
- Stackless walk: quadrant q -> (2qx + (q&1), 2qy + (q>>1)) is Morton digit order, the same as the stack's reverse push. Level-3 squares never split, since level 3 always takes the size branch at :137, so the depth is at most 3.
- Parallel compound: the same tiles summed, cc formed as int64 * 256 + lam * bits, and 'cc < ci' strict as at :95.
- Parallel DP: each square reads only the level below, which is done after the barrier.
- me_wide key: lexicographic (cost, idx) equals the min of cost << s | idx, given 0 <= idx < 2^s and cost < 2^(32-s) (bounds above), so ties still go to the first position in raster order, as in me_search/me_refine.
- Window: g_win holds the clamped samples, and |dx|, |dy| <= 8 keeps every read inside 24x24. least()'s closing barrier (me_wide.slang:41) protects g_win before the second search reloads it.
- Refine split: satd8_q is the sum of its 4 tiles.
- Mode-major intra: the same (block, mode) set, each written once.
- Multi-row: each superblock's costs and decisions do not depend on the grouping.

### Tests and measurement

Per item, without exception: test_vk_kernels (me_matches_ref, all 6 modes in the working tree; intra_costs_match_ref; satd; subpel), test_gpu_intra (cpu_engine_matches_gpu, multi_ref_engines_match: 3 refs, which exercises the compound pairs and same_ref's ref_cost/ref_mv/ref2/mv2), test_gpu_me, test_keyframe_gpu, and test_cli's CELV2_VK_PHASES A/B equality (test_cli.cpp:1668), on both GPUs.

New tests: costs_any_shares (k in {1, 4, 32}), and me_matches_ref with a +-40 predictor field.

CELV2_VK_STATS on the B50 after the scratch items: scratch per thread for cost, finalize, me and me_wide (today 18304 / 3840 / 2816 / not given); the target is 0. Once all are scratch-free, rerun probes_on_queues and kernels_on_queues, as tried-not-kept.md:134's revisit condition asks; no gain expected (the queues take turns).

Measurement on a quiet machine (no other GPU process; check the per-PID GPU counters), alternating old and new in one binary through env switches where possible, at least 3 rounds:
- engine_bench g33_am30 1080p, 30 frames, refs 7, and a16k 9 frames, refs 4.
- Primary: untimed wall ms/frame per GPU. Secondary: CELV2_VK_TIMES per kernel, totals divided by launches (frames - refs), not by frames.
- Whole-encode fps on S1 and a16k, per spec step 5 (lines 112-113).
- md5 identical on S1-S4 and 吉村 for --device cpu, the RTX 5090 and the B50.

Keep a change only if neither GPU is slower in engine time and fps. If the B50 proves power-limited, judge it by its engine wall time and encode fps, not by the per-kernel stamps.

### Expected gain (estimates unless measured)

Measured today (per launch at 1080p, 7 refs): B50 me 233 ms, cost inter 294 ms, cost intra 53 ms, finalize about 30 ms (step 0's figure). 5090: me 9 ms; cost inter about 24 ms, cost intra 4 ms and finalize 14 ms (step 0's figures).

Estimates below, neither measured nor additive.
- Scratch-free arrays: B50 cost inter -30 to -60% (scratch is about 3/4 of the SATD path's memory messages, by my count), cost intra -20 to -40%; 5090 0 to -10%.
- Split repeated scan (+J1 table): cost inter -20 to -40% on both GPUs; it dominates the large-block chunks' latency.
- kShares by frame size: 5090 1080p cost inter -30 to -60%; B50 0 to -30% (one wave today); 16K unchanged.
- Finalize compound in parallel (+arrays, DP): 5090 finalize -50 to -80% at 1080p and 16K (267 ms a frame at 16K in step 0); B50 -20 to -50%.
- me_wide window + one-barrier reduction: me_wide -30 to -50%, which is B50 -70 to -115 ms a launch.
- Mode-major intra: cost intra -30 to -60%.
- Groupshared source: -10 to -20%. Multi-row batches: 5090 1080p engine -5 to -15%.

Overall guess: B50 1080p engine 615 -> 350-450 ms/frame; 5090 57 -> 30-40 ms/frame; 16K B50 cost and finalize -30 to -50%. This is in line with the spec's -20 to -45% for the B50. With the per-level me rule already in the working tree, 16K me should be back at or below the one-thread kernel's 1.63 s (5090) and 43.6 s (B50), still to be measured.

### Risks

- The diagnosis is inference. Power versus thermal versus contention is decided only by the telemetry run. If the B50 is power-limited, its per-kernel times are coupled, a faster kernel can slow the others, and stamps will not add up.
- The scratch accounting (138 Function-array dwords x 32 lanes, against 18304 B) is my arithmetic from the SPIR-V. Which arrays the driver promotes is not visible without CELV2_VK_STATS on the B50.
- [ForceUnroll] / int4 forms grow code. sub_satd's intra path inlines intra_pred 16 times, which costs instruction cache and compile time; the specialization-constant split is the fallback.
- Changing common.slang's hadamard touches satd, me, me_wide and finalize at once, so all tests must run.
- Larger kShares duplicates the dcv init per workgroup for intra; keep prepare's 4.
- Finalize's groupshared grows to about 25 KB, fewer workgroups per SM, acceptable because it is underfilled.
- The 1-barrier key reduction needs the 3-slot rotation; the 2-barrier form is the safe fallback.
- Multi-row batches delay collect()'s first row and raise VRAM per pixel at small sizes; they are excluded when row_streams > 1.
- The working tree changed while I read it (me.slang reverted, me_wide.slang and the 480x270 rule added), and me_wide.slang:1 and vk_kernels.cpp:1 now start with a UTF-8 BOM; strip it before committing.
- Not checked: ISA, registers or occupancy on either GPU (no runs allowed); NVIDIA local-memory use; whether the 1080p versus 16K per-superblock difference comes from fill or from content (more repeated MVs at 16K).

### Order

0. Fix the reading first: engine_bench totals are per launch (frames - refs), and use 30 frames. Then the ABAB telemetry run for (1): A = CELV2_VK_ME_WIDE_MAX=0, B = default. Add the gauge, gap and repeat-dispatch switches only if the clock did not move. This changes no bytes.
1. kShares rule with the CELV2_VK_SHARES sweep (one line); keep it if both GPUs gain.
2. Scratch-free common.slang (hadamard int4, taps, angles, ox/oy) and its callers in cost, satd and finalize; check CELV2_VK_STATS for 0 scratch.
3. Split repeated scan in cost.slang, then the J1 cell table.
4. Finalize: one-pass top-2 and the stackless walk, then the parallel compound with the tile table, then the parallel DP.
5. me: per-level names and an untimed sweep of the working tree's limit; then me_wide's one-barrier key reduction, then its groupshared window, then the refine split; re-sweep the limit after each.
6. Mode-major intra chunks.
7. Groupshared superblock source.
8. Several rows a batch, gated on whole-encode fps.
9. The (MV, 4x4) dedupe as its own spec section.

Re-sweep kShares after steps 2-4, since they change its optimum. Each step is kept only with the same md5 everywhere and neither GPU slower in engine wall time and encode fps.

## Review

Would change bytes or results:
- me_wide (a), one-barrier least(): nothing initialises g_min[0] before the first call at the top level. The only barrier before search() is me_wide.slang:82, and it sits inside `if (pc.pred_x != nullptr)` (:76). At the top level, thread 0's reset of slot 0 races with the other subgroups' InterlockedMin, so the top-level MVs can be wrong. That changes the whole field, and with it the bytes. Fix: initialise all three slots, then one unconditional barrier at entry.
- me_wide (a): the keys and the 0xffffffff sentinel must be uint, with WaveActiveMin/InterlockedMin lowering to UMin. Today's reduction storage is int (me_wide.slang:20-21). If the new g_min copies that, the sentinel is -1 and always wins, so the refine picks a thread >= 9 and decodes a garbage step from it (me_wide.slang:125-126).
- Finalize parallel compound and parallel DP: finalize.slang:129 returns every thread except 0 right after the barrier at :128, and the DP comes after that. The new phase-2/phase-3 barriers and the per-level DP barriers would then sit in non-uniform control flow. That is undefined behaviour: a hang or partial g_csum sums on some drivers. The early return has to go, with only the emission gated on tid == 0. The design does not mention this.
- kShares override: `e ? atoi(e)` passes 0 (or garbage) straight through. pc.k = 0 then divides by zero at cost.slang:34 and makes a zero-size dispatch (vk_kernels.cpp:166, :173). m.costs is left stale from the previous row, so finalize decides from old costs and the bytes differ, with no error. Clamp the value to >= 1.
- costs_any_shares as specified, byte-equal whole buffers: make_tables writes only 4296 inter tasks into the 4400 slots per reference (vk_kernels.cpp:89-96; my count with a script replicating make_tables). cost.slang:89 never writes slots 4296..4399, so a whole-buffer comparison reads uninitialised device memory. With fresh buffers it fails falsely; with a reused buffer it passes vacuously. Pre-fill the buffer, or compare only tasks below inter_first[425].
- Split repeated scan: the OR tree and the final WaveReadLaneAt(rep, slot*lanes) must run outside `if (inter && active)`, and rep must be 0 in every lane. Slang lowers WaveReadLaneAt to OpGroupNonUniformShuffle (cost.spv's only group op, from cost.slang:87), and a shuffle from an inactive lane is undefined. The design says 'executed by all lanes'. Keep that as a hard rule, because a scan-local placement looks natural and would corrupt `repeated` for chunks where count < per.
- me_wide (b): the design never states the barrier needed after the g_win/g_cur loads, before the 289-position loop. It also cites 'least()'s closing barrier (me_wide.slang:41)' as what protects g_win before the second search reloads it. That barrier is at :40, and step (a) removes it. Safety then rests on (a)'s single barrier, which does lie between every thread's SAD reads and the reload. That is correct, but write it that way, so that a later change to least() does not silently reopen the race.
- Finalize top-2: c0 must start at kMax, and ax/ay/sx/sy and the bx/by defaults must stay 0, so that nrefs == 0 still yields ci = kMax with MV 0 (finalize.slang:69-70, :91). As written, `c1 = c0` reads an uninitialised c0 when a < 0. It is harmless only because c1 is not read until second >= 0. Initialise everything explicitly. The induction argument itself is correct, including ties and all-kMax rows.

Other problems:
- The diagnosis premise is not established. The raw step-0 log (celv2-bench/step0/engine_bench_times.txt, the 10-frame refs-7 run: 51 cost calls = 3 launches) gives the B50, with today's committed kernels (Vulkan code unchanged since 0824399), these figures per launch: me 1295.9/3 = 432 ms, cost inter 780.4/3 = 260 ms, intra 47 ms, finalize 29 ms. The design's 'today A' is me 533 / cost inter 237 for the same kernels. That is +23% / -9% A against A: either a different source or a 10-25% run-to-run spread, the same size as the +24% cost-inter shift being diagnosed. The task notes another measurement is running on this machine now, so calling a power or thermal clock drop 'likeliest' is premature. Contention or noise is at least as likely. The ABAB run is the right first step; the narrative should say the effect is unproven.
- The design's correction of step0 lines 86-88 is right, and the raw log confirms it. 5090: 211.9 ms total / 3 launches = 70.6 ms (wall 70.1). B50: 2345.9 / 3 = 782 ms (wall 758); the report's 21 and 234 were totals / 10. Caveat: the totals include the warm-up launch, which has 7 reference uploads plus 7 subpel calls (about 29 ms on the B50) and 17 per-row intra calls, while the wall average excludes it. Over 30 frames that biases about 1/23 of the totals. Say so, or subtract a short run.
- The scratch accounting (138 dwords x 32 lanes, about 18304 B, so 'very likely all in scratch') is contradicted by existing measurements. The B50 note (side-session-notes/local_92609e6e.../b50-queues-2026-11-08.md:44) gives cost at 6144 B with SIMD16, which is 96 dwords a lane, fewer than 138. At the forced SIMD32 (vk_kernels.cpp:129-130) it is 143 dwords a lane. So scratch depends on the register budget, and at least 47 dwords a lane at SIMD32 are register-pressure spill, not dynamic indexing. Also, me.spv carries 274 dwords of the same arrays (%79-%96: two Hadamard sets, four taps copies, ox/oy), yet me's scratch is 2816 B (22-44 dwords a lane), so the driver does promote or constant-fold most of them. Turning arrays into int4 SSA values keeps the same live values and can become spill at SIMD32, so 'scratch 0 for cost' is not a safe target.
- The taps arrays (%153-%156, and finalize's four copies) execute only on at_q's on-demand path (common.slang:25-31), that is layouts A and B. vk_phase_layout chooses B from 8192x8192 pixels up (vk_engine.cpp:131-135). At 1080p (layout '1') they never run, so their removal cannot help B50 1080p time. It can matter a lot at 16K. The gains section mixes the two.
- The taps select chain and the ox/oy ternaries look already tried. The session scratchpad's me1.py (2026-11-09 02:55) makes exactly these edits to common.slang, commented 'went to scratch memory on the Arc Pro B50 (me: 2816 bytes a thread)'. The tree does not carry the change, and tried-not-kept.md has no row for it. Find its outcome and record it before repeating it.
- '346k lane-iterations' checks out as a worst case: 345,984 by my replication of make_tables, with no early exit, but repeated_candidate returns at the first match (common.slang:268). SIMT time is serial warp iterations: about 13,500 per superblock and reference, against 2336 warp-serial sub_satd calls. The split cuts these to about 2265 iterations plus 4392 shuffle steps (my count), about 3-4x, not 32x. The -20 to -40% estimate is plausible; 'every lane repeats' is not where the time goes.
- The 5090's 1080p gains (kShares -30 to -60%, multi-row -5 to -15%, engine 57 -> 30-40 ms) do not move fps. step0 report:83-85: GPU time is about 61 ms a frame against 228 ms of encode a frame per worker, so the 5090 is not the limit at 1080p. Only its 16K numbers count.
- Telemetry plan: PresentMon writes a row per Present, and engine_bench presents nothing, so --track_gpu_telemetry will likely log nothing for it. GPU-Z may not expose Intel throttle reasons. Get-Counter on '*engtype_Compute' alone misses contention on the 3D and copy engines, which share the B50's EUs.
- The power hypothesis needs a long averaging window: cost is unchanged, so with an instantaneous limit it would run at the same clock in A and B. The gap test's 'a power average recovers within about 1 s' is an unsupported assumption.
- The cold-cache gauge ('cost inter again' in row 0) can explain at most row 0, which is 1 of 17 rows, so it cannot account for a +24% total. Drop it.
- kShares: the design keeps 4 for prepare's whole-frame intra because each share recomputes dcv (cost.slang:36-42), but then applies k = 32 to the per-row m.k.intra (vk_engine.cpp:262), which duplicates dcv 8x more. That is inconsistent with its own argument.
- 'B50 fits about 128 workgroups' assumes the normal GRF mode. If IGC uses large-GRF mode for the heavy SIMD32 cost kernel, occupancy halves, and today's 120 workgroups are already two waves.
- Finalize groupshared 'about 25 KB' holds only if best_cost/best_ref/best_ref2/best_mv/best_mv2 are reused for a/second/ci/MVs. With separate arrays it is about 37 KB (12 ints x 425 on top of 16.3 KB). State the reuse.
- The parallel DP (85 squares, finalize.slang:131-154) is a few microseconds of serial work. Four extra barriers are likely neutral or a loss, so drop it or keep it last.
- Factual slips: 1080p level 0 is 240x135 = 32400 blocks, not 32640 (frames are padded to a multiple of 8, y4m.cpp:19-20). The closing barrier is me_wide.slang:40, not :41. The '16K me 1.63 s (5090) / 43.6 s (B50)' figures are not in step 0, whose raw a16k logs give 2.35 s / 57.1 s over 9 frames (262 / 6339 ms a frame).
- me_wide's scratch size is 'not given'. Read it with CELV2_VK_STATS as part of step 0 of the diagnosis; it is free.
- Test gaps: the finalize rewrites (top-2, stackless walk, parallel compound) are covered only through the engine-match tests, which use 3 refs (test_gpu_intra.cpp:174). Ties across 7 refs, all-kMax refs and repeated-candidate rows are not exercised directly.

Fixes:
- me_wide (a): use uint keys with UMin. Thread 0 sets g_min[0..2] = 0xffffffff, followed by one unconditional barrier at kernel entry, before the pred_x branch. Keep the 3-slot rotation, which is race-free as argued once initialised.
- finalize: remove `if (tid != 0) return;` (finalize.slang:129). Let all threads reach phase 2, phase 3 and any DP barriers, and gate only the tree emission on tid == 0.
- cost_shares(): `std::max(1, atoi(e))`, with the default clamp as proposed. Keep the per-row m.k.intra at 4, or sweep it separately from m.k.costs.
- costs_any_shares: fill the costs buffer with a sentinel before each k, and compare tasks [0, inter_first[425]) per ref, or compare whole buffers after a fill.
- Split scan: rep = 0 in every lane; the scan inside `if (inter && active)`; the OR tree and the head broadcast after that block, in warp-uniform flow. Then add the J1 inter_cell table, and use it in finalize.slang:83 too.
- Diagnosis order: (0) read me_wide's scratch with CELV2_VK_STATS. (1) Run A,A,B,B,A,B on a verified quiet machine, logging all GPU engines per PID for the B50's LUID, with clock, power and throttle reasons from IGCL (ctlFrequencyGetState throttleReasons) or Level Zero Sysman (zesFrequencyGetState / GetThrottleTime) instead of PresentMon. (2) Treat a B-vs-A difference as real only if it exceeds the A-vs-A spread. Drop the cold-cache dispatch.
- Scratch work: split the cost pipeline into inter and intra variants (one specialization constant, vk_context.cpp:415) first, not as a fallback, so the inter pipeline carries no intra_pred or dcv and the intra pipeline no at_q/subpel. Then do the Hadamard rewrite, and check scratch at both SIMD32 and SIMD16 (CELV2_VK_COST_SUBGROUP) to separate spill from arrays. If spill remains at SIMD32, an exact option is int16 butterflies: rows stay within +-1020 and columns within +-4080, and only the final |v| sum (<= 65280) needs 32 bits; shaderInt16 is already enabled (vk_context.cpp:190).
- Move the taps select chain to the 16K item list (layout B), and look up me1.py's result first. Add a tried-not-kept row for whatever that attempt showed.
- Finalize groupshared: reuse best_* for a/second/MVs/ci, so the extra is about 16 B a block (intra c, m, g_csum). Keep the DP serial.
- Write the expected gains per GPU and per size: B50 1080p engine and fps, 5090 16K. Drop the 5090 1080p claims as fps gains.
- Add a finalize unit test against the CPU reference with 7 refs and crafted costs: ties across refs, all-kMax refs, kMax from repeated candidates, compound pairs at the limit, and every partition choice pattern for the walk.

Gain assessment: Measured, from the step-0 raw logs: the B50 per launch at 1080p with 7 refs, one-thread me: me 432, cost inter 260, intra 47, finalize 29 ms. The 5090: me 21, cost inter 29, intra 4, finalize 15 ms. Finalize is about 0.8-0.9 ms a row on the 5090 at both 2 and 7 refs, and 2.0 ms a row at 16K. Only 2x separates the GPUs there, against 9x for cost, which supports the serial 64x64 compound reading. Everything below is my estimate.

1. Split scan with the J1 table: credible. The serial scan drops about 13,500 -> 2,265 warp iterations per superblock and reference (worst case), comparable to the SATD work. -15 to -35% of cost inter on both GPUs is reasonable.
2. Parallel compound: credible for the 5090 at 16K, where finalize is 267 ms a frame. Small in absolute terms on the B50 (29 ms a launch).
3. kShares: large kernel-time gains on the 5090, but no fps at 1080p. Uncertain on the B50.
4. Scratch-free arrays: overstated. Part of cost's scratch is SIMD32 spill, the taps never run at 1080p, and the same arrays in me mostly stay out of scratch. B50 1080p cost inter could gain anywhere from about 0 to the claimed -30 to -60%. Separate spill from arrays at SIMD16/32 first.
5. me_wide window plus one barrier: plausible, unmeasured.

The overall 'B50 615 -> 350-450 ms' rests mostly on item 4, so 450-550 is a safer planning figure until the stats are read. The diagnosis in (1) may be chasing run-to-run spread or contention rather than a clock limit.
