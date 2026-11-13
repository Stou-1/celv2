# An exact faster TCQ Viterbi (speed-first step 3): plan and review

2026-11-09. Written by a design agent and checked by a second one trying to refute it (read only, nothing built or run). Verdict: **build with fixes**. Fold in the review before building; measure every step.

## Design

Two changes carry most of the gain. First, the scalar tcq_cands() (quant.cpp:114-130), with its data-dependent branches at :122 and :126, and the nine _mm256_setr_pd builds (:212-217) are replaced by a 4-wide pre-pass. It writes each scan index's candidates per quantizer as nonzero-even (NE), zero (Z) and odd (O). These are the same doubles in today's try order, and the Viterbi builds its vectors from them with broadcast+blend loads. Second, the 8 path costs are held as U = the lower state of each pair and V = the upper one, instead of lo = states 0-3 and hi = 4-7. With that split today's rule (lower state wins a tie, quant.cpp:183-185, the per-lane `second` mask at :191/:229) is the same in every lane. So every value becomes one _mm256_min_pd, and every decision is a separate strict cmp that stays off the dependency chain. The state permutation becomes a 1-cycle unpack on even scan indices and shufpd+vperm2f128 on odd ones, instead of vpermpd+vperm2f128 every step (:234-236). The --rd-zero fresh start is merged into the operand that is ready first, so it adds no latency. Decisions shrink to 18 bits per scan index (4 KB on the stack). That replaces the 64 KB `static thread_local int64_t step[1024][8]` (:187), and the backtrack walks two small per-phase tables. One kernel templated on <kPass, kTrack> serves both tcq_levels and tcq_cost (:331-370). The rest of the parent's list: the trellis update itself can never be skipped exactly. Only the even-level choice can be skipped, where a quantizer has a single even candidate or one provably dominates (optional phase 3). The thread_local issue costs very little CPU by itself (estimate), and the rewrite removes it anyway. Everything is judged first on blocks dumped from real encodes by tools/tcq_bench. It compares qcoeff, dqcoeff, eob and the cost bitwise against a frozen copy of today's code.

### Changes

**src/core/quant.cpp (candidates: replaces TcqCands / tcq_cands, :104-130)**

Reduction of today's list {b, b+1, 0} (clamped, skips at :122). For each quantizer q (0 = Q0, 1 = Q1): b = q ? (int)(e + 0.5) : (int)e, and M = max level (max_dc at pos 0, else max_ac).
  lo = min(b, M); has1 = b < M; odd = lo & 1
  NE = lo + odd,  exists iff (odd ? has1 : lo != 0)
  O  = lo | 1,    exists iff (odd || has1)
  Z  always, except at c == eob-1 when !pass (the last rule)
Today's even[q][0] is NE if it exists, else Z. even[q][1] is Z when NE exists. odd[q] is O.

constexpr int kChunk = 64;
struct TcqSoa {  // i = c - base, base a multiple of 64
  alignas(32) double ned[2][kChunk], nwb[2][kChunk];  // NE: d2 (inf if none), weight*level_bits
  alignas(32) double od[2][kChunk], owb[2][kChunk];   // O: the same
  alignas(32) double dz[kChunk];                      // Z's d2 (== dropped()); inf at eob-1 when !pass
  alignas(32) double fne[kChunk], fo[kChunk];         // pass only: ned[0], od[0] where e >= tcq_prep's threshold, else inf
};
__forceinline void tcq_cands4(e_of, scan, c /*multiple of 4*/, i, max_dc, max_ac, psy, first, first_ac, weight, TcqSoa& t, int32_t (*nel)[1024], int32_t (*ol)[1024]) {
  pos = _mm_cvtepi16_epi32(_mm_loadl_epi64(scan + c)); e = _mm256_i32gather_pd(e_of, pos, 8);
  dc = _mm_cmpeq_epi32(pos, 0); dcd = castpd(_mm256_cvtepi32_epi64(dc));
  ps = _mm256_andnot_pd(dcd, set1(psy));            // pos ? psy : 0, as :198
  M = _mm_blendv_epi8(set1(max_ac), set1(max_dc), dc);
  d2(rec): x = sub(e, rec); return add(mul(x, x), mul(ps, andnot(-0.0, x)));   // :125, the same ops in the same order, no FMA
  bits(l): g = _mm256_i32gather_pd(kLevelBits.data(), _mm_min_epi32(l, 255), 8);
           if (movemask(cmpgt(l, 255))) per lane g[k] = level_bits(l[k]);   // rare: level_bits_of() with log2
           return mul(set1(weight), g);
  for q in 0, 1:
    b = _mm256_cvttpd_epi32(q ? add(e, 0.5) : e);  // == static_cast<int>, :117
    lo = _mm_min_epi32(b, M); has1 = _mm_cmpgt_epi32(M, b); odd = lo & 1; oddm = cmpeq(odd, 1);
    ne = lo + odd; o = lo | 1;
    ne_ok = _mm_blendv_epi8(not(cmpeq(lo, 0)), has1, oddm); o_ok = oddm | has1;
    rec(l) = q ? sub(cvtepi32_pd(l), 0.5) : cvtepi32_pd(l);   // :124
    t.ned[q][i..] = blendv(inf, d2(rec(ne)), widen(ne_ok)); t.nwb[q][i..] = bits(ne);
    t.od[q][i..]  = blendv(inf, d2(rec(o)), widen(o_ok));   t.owb[q][i..] = bits(o);
    nel[q][c..] = ne & ne_ok (0 where none); ol[q][c..] = o;   // int32: the backtrack casts through int16 as :264
  t.dz[i..] = d2(0.0);   // e - 0.0 == e, so it equals dropped(e, pos, psy) bit for bit
  elig = cmp(e, blendv(first_ac, first, dcd), _CMP_GE_OQ);   // :241
  t.fne[i..] = blendv(inf, ned[0], elig); t.fo[i..] = blendv(inf, od[0], elig);
}
Delete tcq_cands, TcqCands and tcq_fresh (:135-151). The frozen copies stay in tools/tcq_ref.cpp.

**src/core/quant.cpp (one Viterbi step: replaces :196-243 and :339-369)**

Layouts, with pair k = states {2k, 2k+1}, Q1 iff k odd:
  phase 0 (even c): U = P[0 2 4 6], V = P[1 3 5 7], lane quantizers Q0 Q1 Q0 Q1
  phase 1 (odd c):  U = P[0 4 6 2], V = P[1 5 7 3], lane quantizers Q0 Q0 Q1 Q1
Next states per lane: X = next[2k][0] (lower even / upper odd), Y = next[2k][1] (lower odd / upper even).
  phase 0: X = D[0 1 6 7], Y = D[4 5 2 3]
  phase 1: X = D[0 6 7 1], Y = D[4 2 3 5]
State 0 is X lane 0 and state 4 is Y lane 0 in both phases.

template <int Ph, bool kPass, bool kTrack>
__forceinline void tcq_step(const TcqSoa& t, int i, int c, __m256d& U, __m256d& V, double& tail, double weight, uint32_t* dec) {
  constexpr int q1 = Ph ? 0b1100 : 0b1010;   // the lanes on Q1
  auto lanes = [&](const double (&a)[2][kChunk]) { return _mm256_blend_pd(_mm256_broadcast_sd(&a[0][i]), _mm256_broadcast_sd(&a[1][i]), q1); };
  const __m256d ned = lanes(t.ned), nwb = lanes(t.nwb), od = lanes(t.od), owb = lanes(t.owb);
  const __m256d dz = _mm256_broadcast_sd(&t.dz[i]), wz = set1(weight * level_bits(0));
  auto j = [](__m256d p, __m256d d, __m256d w) { return add(add(p, d), w); };   // (P + d2) + bits, as :222-223
  nu = j(U, ned, nwb); zu = j(U, dz, wz); ou = j(U, od, owb);
  nv = j(V, ned, nwb); zv = j(V, dz, wz); ov = j(V, od, owb);
  eu = _mm256_min_pd(zu, nu); ev = _mm256_min_pd(zv, nv);   // zero only if strictly less (:224's c1 < c0)
  ox = ov; oy = ou;
  if constexpr (kPass) {   // fresh start (:240-242): merged into the odd sums, which are ready one min earlier
    s = tail + weight * end_bits(c + 1);
    f = _mm_add_pd(_mm_add_pd(_mm_set1_pd(s), _mm_setr_pd(t.fne[i], t.fo[i])), _mm_setr_pd(t.nwb[0][i], t.owb[0][i]));   // (s + d2) + wb, as :138/:145
    fx = _mm256_blend_pd(inf, cast256(f), 1); fy = _mm256_blend_pd(inf, cast256(_mm_unpackhi_pd(f, f)), 1);
    ox = _mm256_min_pd(ov, fx); oy = _mm256_min_pd(ou, fy);
  }
  x = _mm256_min_pd(ox, eu); y = _mm256_min_pd(ev, oy);   // value: the least of each next state's candidates
  if constexpr (kTrack) {
    d = movemask(cmp(zu, nu, LT)) | movemask(cmp(zv, nv, LT)) << 4      // zero chosen, U / V lanes
      | movemask(cmp(ov, eu, LT)) << 8 | movemask(cmp(ev, ou, LT)) << 12; // upper state taken, X / Y lanes
    if constexpr (kPass) d |= (movemask(cmp(fx, _mm256_min_pd(ov, eu), LT)) & 1) << 16
                            | (movemask(cmp(fy, _mm256_min_pd(ev, ou), LT)) & 1) << 17;   // fresh start won, states 0 / 4
    dec[c] = d;
  }
  if constexpr (kPass) tail += t.dz[i];   // == dropped(e, pos, psy), in today's order
  if constexpr (Ph == 0) { U = _mm256_unpacklo_pd(x, y); V = _mm256_unpackhi_pd(x, y); }                      // -> phase 1 layout
  else { a = _mm256_shuffle_pd(x, y, 0b0110); b = _mm256_shuffle_pd(y, x, 0b0110);
         U = _mm256_permute2f128_pd(a, b, 0x20); V = _mm256_permute2f128_pd(a, b, 0x31); }                     // -> phase 0 layout
}
Hand-checked: phase 0 gives U = D[0 4 6 2], V = D[1 5 7 3]. Phase 1 gives a = D[0 2 1 3], b = D[4 6 5 7], so U = D[0 2 4 6], V = D[1 3 5 7].

**src/core/quant.cpp (driver, readout, backtrack in tcq_levels: replaces :181-271; same driver in tcq_cost)**

template <bool kPass, bool kTrack> void tcq_viterbi(...) {
  U = setr(kPass ? inf : 0, inf, inf, inf); V = set1(inf);   // state 0 is U lane 0 in both layouts (:188)
  TcqSoa t;
  for (int top = eob - 1; top >= 0; top = base - 1) {
    base = top / kChunk * kChunk;
    for (c = base; c <= top; c += 4) tcq_cands4(..., c, c - base, ...);   // c + 3 < n: n is a multiple of 16
    if (!kPass && top == eob - 1) t.dz[eob - 1 - base] = inf;              // the last coefficient keeps a level
    c = top; if (!(c & 1)) tcq_step<0>(t, c - base, c, ...), c--;
    for (; c > base; c -= 2) { tcq_step<1>(t, c - base, c, ...); tcq_step<0>(t, c - 1 - base, c - 1, ...); }
  }
  // c = 0 was a phase-0 step, so the layout is phase 1's
  store u = U, v = V; path = {u0, v0, u3, v3, u1, v1, u2, v2};
}
The caller keeps :249-258 unchanged: the first-least loop, none/zero and the cost loop.

Backtrack (dec[1024] and nel/ol[2][1024] on the stack, no thread_local):
  static constexpr uint8_t kTake[2][8] = {{0,1,6,7,4,5,2,3}, {0,3,5,6,4,7,1,2}};   // state -> bit (+8), indexed by c & 1
  static constexpr uint8_t kLane[2][4] = {{0,1,2,3}, {0,3,1,2}};                    // pair -> lane
  for (c = 0; c < eob; c++) {
    ph = c & 1; k = s & 3; q = s & 1; d = dec[c];
    if (pass && k == 0 && (d >> (16 + (s >> 2)) & 1)) { level = s ? ol[0][c] : nel[0][c]; write; end = c + 1; break; }   // as :266-270
    take = d >> (8 + kTake[ph][s]) & 1;
    even = ((0xC3 >> s) & 1) ^ take;   // X states 0 1 6 7
    level = !even ? ol[q][c] : ((d >> (kLane[ph][k] + 4 * take)) & 1) ? 0 : nel[q][c];
    l = static_cast<int16_t>(level); qcoeff[pos] = coeff[pos] < 0 ? -l : l;   // :264-265
    s = 2 * k + take;
  }
Keep the dequant loop :272-287 as it is. Optional, measured: fuse it, using `s` as the state of position c, because the walked source states are the decoder's states (the trellis edges are next[][]).

tcq_cost (:338-370) calls tcq_viterbi<pass, false> and keeps :371-378.

**src/core/quant.cpp (CELV2_TCQ_DUMP, in quantize() :459 and quantize_cost() :399, TCQ branches)**

Enabled once from getenv (as profile.cpp:244-250): CELV2_TCQ_DUMP=path[:every], default every 997. A global atomic call counter, and one mutex-guarded FILE* in append mode.
Record, little-endian: 'TCQ1'; kind (0 quantize, 1 quantize_cost); tx, tx_type, bd, g_rd_zero, cost!=nullptr, dqcoeff!=nullptr, eob_out!=nullptr; int16 qindex; double psy; uint16 n; int32 coeff[n].
Then this build's outputs: int32 eob/return; double cost (the 8 bytes); for kind 0, int32 qcoeff[n] and dqcoeff[n].
Off: one predicted branch per call, the same bytes.

**tools/tcq_ref.cpp, tools/tcq_ref.h (new)**

A verbatim copy of bd3fad8's TCQ path in namespace tcq_ref: level_bits, kEndBits, first_level, dropped, tcq_prep, TcqCands, tcq_cands, tcq_fresh, tcq_levels, tcq_cost, plus the TCQ branches of quantize / quantize_cost / tcq_end_cost (its RdTimer lines kept, for parity). It reads the real g_rd_zero, kTcqBitWeight and kQDepth from quant.h. It is frozen: the oracle and the timing baseline.

**tools/tcq_bench.cpp (new) and CMakeLists.txt**

Modes:
  self [seed]: synthetic blocks, new against ref (see tests).
  check dump...: every record through new and ref (quantize / quantize_cost with the record's tx, tx_type, qindex, bd, psy, g_rd_zero, null/non-null outputs). Compares eob, memcmp of the cost's 8 bytes, qcoeff[n] and dqcoeff[n], both against each other and against the recorded outputs. It also re-runs each record with g_rd_zero flipped and with psy 0 and 0.25, new vs ref only. Prints mismatch counts and the first mismatch (record, tx, qindex, psy, rd_zero, first differing index).
  time dump... [--rounds 7] [--core N]: SetThreadAffinityMask to one core, all records in memory, coeff copied to an aligned hot buffer before each call. Rounds alternate ref/new, each call between lfence;rdtsc. Reports medians of ticks per call (all, and eob>0), ticks per Viterbi step (sum of eob), by tx size. Also the eob histogram, the share of zero-region pairs (both quantizers base 0) and the share of positions above the threshold.
CMake:
  add_executable(tcq_bench tools/tcq_bench.cpp tools/tcq_ref.cpp)
  target_link_libraries(tcq_bench PRIVATE celv2_core)
  add_test(NAME tcq_bench_self COMMAND tcq_bench self)

**src/core/quant.cpp (phase 3, optional: even-choice skipped where it is decided)**

A second step body without the NE/Z min. Per lane, E = (P + selD) + selW, with the z bits taken from the pre-pass. The branch is per (odd c, c-1) pair, on a flag the pre-pass sets.
v1, trivially exact: both quantizers have base 0 and the pair excludes eob-1, so NE is missing, Z is the only even, and no fresh start exists (e >= first > 0.5 implies base_q1 >= 1).
v2: Z or NE dominates for every P <= B. B is the sum of each position's largest finite candidate. The test is |(dz+wz) - (ned+nwb)| > 8*2^-53*(2B + both sums), plus the rounding of the sums. Rounding is monotone and the two-add error is at most 2u(P+d+w). Undecided pairs take the general body.
The chain drops from add,add,min,min to add,add,min on !pass. Under pass, an eligible position needs the fresh min in series again, so it mainly helps 吉村/film.

**docs/design/speed-budget.md, docs/design/tried-not-kept.md, docs/reports/ (new report)**

A Saved row with commit, S1-S4/吉村 CPU a frame and fps, and CELV2_RD_TIMES Viterbi/prep/back before and after. A tried-not-kept row for every variant dropped (fold vs plain fresh min, chunk vs fused, dequant fusion, phase 3 v1/v2), with its bench numbers.

### Exactness

1. Candidate values: each lane runs today's scalar operations on the same e. e is gathered unchanged from tcq_prep's e_of. cvttpd2dq equals static_cast<int> for 0 <= e < 2^31 (:117). Q1 rounds e + 0.5 as :117. rec is (double)l, or minus 0.5 for Q1 (:124). d2 = (e-rec)*(e-rec) + psy*|e-rec| uses separate mul and add, as :125 under /fp:precise without contraction (CMakeLists.txt:7-8). The DC lane gets psy 0 and max_dc, as :198. weight*level_bits(l) reads the same kLevelBits doubles, and level_bits_of() (std::log2) for l >= 256. Z's d2 is e*e + psy*e bit for bit, because e - 0.0 == e and |e| == e for e >= +0; that is dropped() (:69). DC's + 0*e equals + 0.0.

2. Candidate sets, case by case (b = base, M = max level >= 1):
- b = 0: today even = [0], none if last; odd = [1] (k=2 is skipped at :122). New: NE none, Z, O = 1.
- 1 <= b < M: today even = [whichever of b, b+1 is even; 0 unless last], odd = [the other]. New: NE, Z, O.
- b >= M, M even: today even = [M; 0 unless last], and b+1 clamps to M, a repeat, so no odd. New: NE = M, Z, no O.
- b >= M, M odd: today odd = [M], even = [0 unless last]. New: no NE, Z, O = M.
The zero is always tried after the nonzero even, so today's even tie (c1 < c0, :224) is 'zero only if strictly less'. That equals cmp(gZ, gNE, LT), and the value equals min_pd(gZ, gNE). When NE is missing, today's (P+Z) against (P+inf)+0 gives the same value as min_pd(gZ, inf). A missing candidate is inf: (P + inf) + w = inf for any finite w, as today's (P+inf)+0.

3. Sums: (P + d2) + w per state and candidate, in the order of :222-223 and :138/:145.

4. Selections. Values: a next state's value is the least double in its candidate set. No NaN can arise (no inf - inf, no 0*inf), and no -0 (P starts at +0 or inf, and terms are >= +0), so the least is one bit pattern whatever operand order min_pd sees. Decisions come only from separate strict LT_OQ compares in today's order:
- even: zero iff gZ < gNE (:224);
- next state (lower even, upper odd): upper iff O_upper < E_lower (:229, first lane);
- next state (lower odd, upper even): upper iff E_upper < O_lower. Today's second lane takes the lower odd iff O <= E, i.e. !(E < O).
Holding lower states in U and upper states in V turns today's per-lane LT/LE mask (:191, :229) into the uniform 'lower wins ties'.

5. --rd-zero fresh start (:240-242, :135-151): s = tail + weight*end_bits(c+1), j = (s + d2) + w, eligible iff e >= first (pos 0) or first_ac. Under pass last is false, so even[0][0].l != 0 means 'Q0's NE exists', and no[0] means 'Q0's O exists'. Today's j replaces the trellis value only if strictly less. New: the value min(EU, min(OV, F)) is the same least double, and the decision is F < min_pd(OV, EU), strict. The tail adds the same dz values in the same order (eob-1 down to 0), after s is formed.

6. Lane moves (unpack, shufpd, vperm2f128) copy bits. The readout maps to path[0..7] in canonical order before today's first-least loop (:249-251), none/zero (:252-253) and the cost loop (:254-258). tcq_cost keeps :374-378.

7. Backtrack: decisions equal today's on every state with a finite cost. Inf states can differ (for example which empty even slot), but no finite path passes through one, since a finite next state's chosen source is finite. The chosen final state is finite: without pass, state 0 starts at 0. With pass, eob-1 is at or above tcq_prep's threshold, so Q0 has NE or O and a fresh start. Levels are the same values, cast through int16 as :264. A fresh start ends at c+1 as :267-270. The dequant is unchanged; the optional fusion uses the walked source states, which are the decoder's by construction.

8. force_eob (tcq_end_cost) is the !pass path, with the last rule at force_eob-1. The pre-pass covers [0, eob) wherever eob lies; n is a multiple of 16, so the groups of 4 stay inside n.

9. Only one build exists for this code: AVX2 (CMakeLists.txt:8). CELV2_SIMD is specified (2026-11-04-speed-budget-design.md:58-63) but has no hits in src/tools/tests. The code is CPU-only, so --device cpu and Vulkan get the same values.

10. Phase 3 v1 is exact because min_pd(gZ, inf) = gZ. v2 depends on the stated error bound and falls back to the general body when undecided.

### Tests and measurement

Correctness:
- tcq_bench self, registered as CTest tcq_bench_self: new against tools/tcq_ref.cpp on synthetic blocks.
  - Every TX_SIZES_ALL. qindex 1/40/130/220/255, plus a sweep of 32 qindex so max_level has both parities.
  - psy 0/0.25/2; g_rd_zero 0/1.
  - quantize with the cost null and given; quantize_cost with eob null and given (this reaches tcq_levels with dq null).
  - tcq_end_cost at every end on 8x8/16x16 and 16 random ends on larger sizes.
  - Distributions: all zero; mostly under one step (as test_transform.cpp:131); a few steps; |coeff| >= max_level*step (to hit b >= M); >= 256 steps (level_bits_of); exact multiples and half multiples of the step; a lone coefficient at each of the first 64 scan indices; all positions significant.
  - Compares eob, memcmp of the cost, qcoeff[n] and dqcoeff[n].
- tcq_bench check on dumps (33 frames each, q130, --refs 7): AM --tune animation (pass), 吉村 --tune film (!pass), Dragon Maid and AIR animation, AM --rd-zero 0, AM --psy-trellis 0.25 (psy > 0). Records are checked against the ref and against the recorded outputs, so the ref copy is proven to match the encoder build. Required: 0 mismatches.
- Existing suite unchanged, especially test_transform.cpp:119-182 (the rd-zero oracle), :184-201 (cost-only equals quantize) and test_psy.

Measurement:
- tcq_bench time on AM and 吉村 dumps, quiet machine, one pinned core, 7 alternating rounds, medians: ticks per Viterbi call and per step, by tx size. Run first on the old code alone, to get its cycles per step and the eob histogram.
- Encoder, per the speed-first spec's step 3:
  - same md5 on S1-S4 and 吉村 at --workers auto --refs 7 --qindex 130, plus one 10-bit run, --device cpu and Vulkan (RTX 5090, B50); md5 spot checks with --psy-trellis 0.25 and --rd-zero 0/1;
  - CELV2_RD_TIMES Viterbi/prep/back on S1, before and after;
  - CPU a frame and fps, old and new exe alternating, 3 rounds, medians. Kept if S1's CPU a frame drops by more than the 1% noise.

### Expected gain (estimates unless measured)

Measured (given, S1, CELV2_RD_TIMES): Viterbi 320 ms a frame (3.8 us x 50 M calls); tcq_prep + fills 67 ms; backtrack + dequant 50 ms; quantize 461 ms a frame over 193 M calls. S1 runs at 2.129 s a frame CPU after e1782ee (speed-budget.md:134).

Everything below is an estimate; nothing was compiled, disassembled or run here.
- Today's step: about 27-31 cycles on its dependency chain (add, add, cmp, blendv, cmp, blendv-of-masks, blendv, vpermpd, vperm2f128, at Golden/Redwood Cove-class latencies). On top of that come about 200 uops a step and data-dependent branches in tcq_cands (:122, :126) and tcq_fresh (:139, :146). Likely 40-60 cycles a step.
- New step: chain of 13-16.5 cycles (add, add, min, min, then 1 or 4 cycles of shuffle, 2.5 on average). The fresh start adds no latency. About 60-75 uops including the pre-pass, so about 15-22 cycles a step.
- Viterbi 2-3x faster: 320 to about 110-160 ms a frame, saving about 120-210 ms a frame, i.e. 6-10% of S1's CPU a frame.
- If the bench shows today's step already under about 30 cycles, the saving is about 80 ms (4%).
- The coder's tcq_cost gets the same kernel; its share is not measured.
- Backtrack: within about ±10% of its 50 ms.
- Phase 3: about 20-30% more off the new Viterbi time on !pass sources such as 吉村, little on S1.
- The thread_local step buffer: 24-44 RD threads a frame per worker each set up a 64 KB+ TLS block (pipeline.cpp:250-256), under 1 ms a frame, i.e. under 0.05% of CPU. Its real cost is footprint: 64 B a step overflows a 48 KB L1D past an eob of about 700. The new decisions take 4 B a step.

### Risks

- The gain figures are estimates. If today's loop is already near its chain bound, the gain shrinks to about 1.5x. Step 0 of the order measures this before anything is built.
- Exactness assumes the scalar d2 at :125 is not FMA-contracted by MSVC. If it were, the vector form would differ only when psy > 0 (with psy 0, fma(x,x,0) equals x*x). The psy dumps and the self test catch it.
- Exactness also assumes no NaN and no -0 in path costs; the argument is in the exactness field, and a debug assert can check it.
- Assumes max_level >= 1: 0xffffff/(2*step) with AV2 steps far below 2^23. Add an assert.
- The int16 truncation at :264 must be kept, not 'fixed'. Saturating packs (_mm_packs_epi32) must not be used for the levels.
- Register pressure: 16 ymm under <kPass, kTrack> may spill. Check in the bench.
- vgatherdpd is slow on Intel CPUs with the Gather Data Sampling microcode fix (pre-Golden-Cove). If the bench on such a CPU shows it, use scalar loads.
- Code size grows: 2 phases x 4 template combinations, plus phase 3.
- Internal decisions for inf-cost states differ from today's step[] contents. Harmless, but any debug code that reads step[] for inf states would see a difference; the bench compares outputs only.
- AVX-512 is not proposed: the tried-not-kept row (AVX-512 instead of AVX2) and no CELV2_SIMD dispatch exists. One zmm holding all 8 states is a later per-kernel revisit.
- The dump hook adds one predicted branch per call when off.
- Not checked: the actual average eob and cycles per step, the MSVC code generated for today's loop (whether tcq_fresh is inlined or lo/hi go through memory), and the L1D size of the Xeon 658X.

### Order

0. CELV2_TCQ_DUMP, tools/tcq_ref.cpp and tools/tcq_bench (self/check/time). No byte change. Dump AM, 吉村, DM, AIR, plus the psy and rd-zero-0 variants. Time the old code: cycles per step, eob histogram, zero-region and eligible shares. If the old step is already under about 25 cycles, stop and record it in tried-not-kept.
1. Core rewrite as one unit, for tcq_levels and tcq_cost: pre-pass (chunks of 64), tcq_step with the two layouts and min_pd, the fresh start merged into the odd sums, bit decisions on the stack (thread_local removed), table backtrack, dequant unchanged. Gate: bench check 0 mismatches and self passes; bench time faster on both AM and 吉村 dumps; then the encoder A/B (same md5 everywhere; S1 CPU a frame better than 1%).
2. Bench-only variants of step 1, each kept only if it gains on both dumps: (a) the merged fresh start against the plain min_pd(fx, trellis); (b) pre-pass in chunks of 64 against the whole block against fused per group of 4; (c) dequant fused into the backtrack. Ledger row after the encoder A/B of the kept combination.
3. Phase 3: decided even choice. v1 (zero-region pairs) first, then v2 (error-bound dominance). Kept only if 吉村's dump gains and AM's does not lose; then its own encoder A/B and ledger row.
4. Report, ledger rows and tried-not-kept rows for every dropped variant.

## Review

Would change bytes or results:
- Core rewrite: I could not refute it. I checked these by hand against quant.cpp. The candidate sets in all four base/M cases (:117-128). The NE/Z order, so zero is chosen only if Z < NE (:224). Lower state wins a tie in both lanes, which is the LT/LE mask at :229 under :191. Both layouts and both shuffles (unpack, and shufpd+vperm2f128 0x20/0x31). State 0 is X lane 0 and state 4 is Y lane 0 in both phases. The readout order and the table backtrack: k = s&3 is the source pair, q = s&1 its quantizer, plus the 0xC3 parity rule. The fresh start: merged-min value, strict F < min(OV,EU), s and tail in :241-242 order. dz equals dropped() (:69) bit for bit. No NaN and no -0 can occur, since psy is clamped to 0..2 at main.cpp:528. Path values are bitwise today's, so the first-least state (:249-251) and the finite-path decisions match.
- Phase 3 v2's bound is too small under pass. A path cost P includes the fresh start's s = tail + weight*end_bits(c+1) (quant.cpp:241). weight*end_bits is not any position's candidate and is worth up to 0.05*kEndBits[1024] = 0.85 (quant.cpp:74). B = 'sum of each position's largest finite candidate' can therefore sit below the real maximum P. For small blocks the 8u margin then fails to cover 2u(2P+S1+S2), and a near-tie gets the wrong dominance, which changes the value and the bytes.
- Phase 3 v1's flag must come from the computed bases (the cvttpd results, quant.cpp:117), not from e < 0.5. For e = 0.5 - 2^-54, e + 0.5 rounds to 1.0, so base_q1 = 1 and Q1's NE = 2 exists (lo = 1 is odd and has1 holds). A flag that tests e < 0.5 would drop the NE/Z min at that position and change the value.
- Not a defect, but unspecified: tcq_viterbi must also return tail. tcq_levels' none (:252) and tcq_cost's min (:375) need the final tail. The design only says 'store u = U, v = V; path = ...'.
- Test gap for an exactness rule the design itself relies on: the self test lists no levels above 32767. Those exercise the int16 cast at :264 and the dequant reading the truncated level back at :277/:285. qindex 1 with |coeff| of 32768 steps or more reaches it, since max_dc = 0xffffff/(2*dc_q) is far above 32767 at low qindex.

Other problems:
- The footprint is understated. The backtrack needs nel/ol int32[2][1024] (16 KB) plus TcqSoa (about 5.6 KB) on top of dec[1024] (4 KB). That is about 26 KB more stack per tcq_levels call and 20 B a step, not the '4 KB' (summary) or '4 B a step' (expected_gain).
- The backtrack chain gets longer. Today's recurrence is one s-dependent load plus a shift (quant.cpp:264-266), about 6-7 cycles a step. The new one puts the kTake[ph][s] load on the s->s chain (load, add, shrx, and, lea), about 8-9 cycles. So 'backtrack within about ±10% of its 50 ms' is optimistic: the walk can lose roughly 5-15 ms a frame.
- Chunks of 64 serialize the throughput-bound pre-pass with the latency-bound chain inside one thread. A ~512-entry ROB holds only about 6-10 steps of a 64-step chunk. The '15-22 cycles a step' estimate assumes an overlap that this structure mostly does not give: about 6-7 cycles of pre-pass plus about 15 of chain, roughly 21-22.
- The uop count is low. By my count the pre-pass is about 37-40 uops a position: 5 gathers per group of 4, and per quantizer cvtt, ten integer ops, two cvtdq2pd, two d2 (5 ops each), two widen+blend, two bits() with gather, movemask, branch and mul, plus 6 stores. The step is about 50 uops (!pass, kTrack) or about 78 (pass). That totals about 87-117 uops a step, not 60-75.
- SMT is not modelled. The RD pass runs on both hardware threads of each core (pipeline.cpp:251-254). A bench pinned to one core with an idle sibling rewards chain-latency cuts that a loaded core partly hides. Under full SMT load, the aggregate gain follows uops and mispredicts.
- Gathers on AMD are not considered. CelV2 is public (v0.2.0 on GitHub). The design names only Intel's GDS fix, but vgatherdpd is slow on Zen 2/3. With 5 gathers per 4 positions, the pre-pass could lose to today's scalar tcq_cands there.
- Phase 3 v1's expected gain is mis-attributed. By the design's own argument, v1 pairs have no eligible position (e < 0.5 < first = 0.55, quant.cpp:81). So under pass the fresh min is also absent there, and v1 helps S1 as much as 吉村. It is v2 that gains nothing at eligible positions under pass. Phase 3's per-pair branch is data-dependent, and a mispredict at each zero/nonzero transition can eat the 4 cycles it saves.
- The measured 320 ms covers tcq_levels only: the RdTimer emplaces are at quant.cpp:166/194/245, and tcq_cost (:307-379) has none. The coder's tcq_cost share is unmeasured. The sampler ranks it above tcq_levels (speed-step0.md:121), while the profile puts all of the coder's transforms and quantization at 5-9%.
- A smaller variant is missing (shortest-diff check). In today's lo/hi layout, values could be computed by min_pd, with the LT/LE take mask and the step[] blendv kept off the chain. That removes the three blendv from the chain (about 18 cycles against the design's 13-17) in roughly 20 lines, leaving step[] and the backtrack untouched. Without it in the bench, the two-phase rewrite's own share is unknown.
- The dump hook uses a global atomic counter hit on every call by 44+ RD threads, which causes cache-line contention while dumping. It only applies when the dump is on, so it is minor.
- Deleting the scalar tcq_cands/tcq_fresh from src leaves no scalar path. That matters if CELV2_SIMD=scalar (speed-budget-design.md:58-63) is ever built; today none exists, so it is not a current problem.
- Citation slips. The 'mostly under one step' distribution is test_transform.cpp:132-135 (line 134), not :131. The 'nine _mm256_setr_pd builds' at :214-217 are 6 setr_pd plus 3 setr_epi64x.

Fixes:
- Phase 3 v2: set B = (sum of each position's largest finite candidate, rounded up) + weight*kEndBits[eob] under pass. Or bound it by the largest finite path value of the previous step, which the trellis already holds.
- Phase 3 v1: build the per-pair flag from the pre-pass's integer bases (b0 == 0 && b1 == 0 for both positions), never from a comparison on e.
- Store nel/ol as int16_t. This is exact, because :264 casts through int16 anyway and the parity is kept. It brings the stack and footprint to 8 B a step plus dec. Or skip storing levels and recompute lo/NE/O for the visited c from e_of in the backtrack.
- Canonicalize decisions when recording, off the forward chain. Map the 8 take bits (X lanes, then Y lanes) to state order with a 256-entry byte LUT per phase; pdep(x,0xC3) and pdep(rot2(y),0x3C) also work on Intel. Map the zero bits to source-state order the same way. The walk then becomes take = (d >> (8+s)) & 1 and s = 2*(s&3) + take, about 3-4 cycles a step, faster than today's 6-7.
- Make fused-per-group-of-4 the primary build and chunks of 64 the variant, or present both as equals in step 2(b). Re-derive the expected cycles at about 90-115 uops a step.
- Add an SMT-pair mode to tcq_bench time: two threads on the two logical processors of one core, running the same records, reporting per-thread and aggregate ticks. Base step 0's stop rule (old step under about 25 cycles) on the aggregate number.
- Replace the kLevelBits gathers. NE and O are {lo, lo+1}, and Q0's and Q1's bases differ by at most 1, so one unaligned 32-byte load of kLevelBits[b0..b0+3] per position covers both quantizers: pad the table to 259 entries and keep the l > 255 slow path. Load e with 4 scalar loads. Time on one Zen machine before a release.
- Add bench variant 0 to step 2: today's layout with min_pd values and the LT/LE decisions off the chain, step[] and backtrack unchanged. Keep the two-phase rewrite only if it beats variant 0 on both dumps.
- In tcq_cands4, skip the nel/ol stores when !kTrack (tcq_cost). Return tail from tcq_viterbi.
- Self test: add levels above 32767 (qindex 1, |coeff| of 32768 steps or more) and e = 0.5 - 2^-54 exactly (the Q1 base rounding edge).
- Dump: use a thread_local call counter per thread instead of a global atomic.

Gain assessment: Measured (bd3fad8's commit message, CELV2_RD_TIMES, S1): Viterbi 320 ms a frame over 50 M calls, about 3.8 us a call, plus prep 67 ms and backtrack 50 ms. S1 runs at 2.129 s a frame (speed-budget.md:134). These cover tcq_levels only (quant.cpp:166/194/245); tcq_cost has no timer. Everything else is my estimate: nothing was compiled or run. The core rewrite is sound for exactness. On speed, today's step is probably uop- and mispredict-bound, about 200-250 uops plus branches at :122/:126/:139/:146, more than it is chain-bound. The new step is about 90-115 uops with a chain of about 15 cycles. Under the encoder's SMT load (pipeline.cpp:251-254) the aggregate gain follows uops, so about 2-2.5x on the Viterbi, if today's step really costs 40 cycles or more, which nobody has measured. That saves about 130-190 ms a frame, 6-9% of S1. If today's step is nearer 30 cycles, the saving drops to about 80 ms (4%). With the kTake load on the s chain, the backtrack can give back about 5-15 ms unless the take bits are canonicalized. Phase 3's extra gain is speculative and branch-limited. The design's 6-10% is plausible but optimistic: its uop count and pre-pass overlap are both too favourable. Step 0 (the dump and the bench, with an SMT-pair mode added) should decide before the rewrite.
