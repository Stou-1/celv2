# Real-cost trellis Q1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `--real-trellis 0|1` (default 0 under every tune): the coder's final TCQ levels by the coder's own bits, Q0's row Qc.

**Architecture:** The coder keeps its decisions (types, IST, CCTX) as now; with the option on, the chosen coding's coefficients
are quantized again by a Viterbi over TCQ's 8 states whose rates are the symbols' bits on the superblock's starting CDFs (Qc),
with the contexts BlockWriter writes with. Fixed point as C17 (`rd_int.h`): bits in 1/512, J in int64.

**Tech Stack:** C++17, AVM's entropy tables and context helpers (txb_common.h), the existing ctest suite.

**Spec:** docs/superpowers/specs/2026-11-06-real-cost-trellis-design.md (Q1, Speed, Tests). The user chose Q1 at Qc on
2026-11-10 ("A") after Q0c fell short on Dragon Maid (docs/reports/2026-11-04-av1-gap-breakdown.md, Q0c).

## Global Constraints

- Default 0 under every tune; `--real-trellis 0` gives the same bytes as before.
- On: AVM decodes equal to `--recon`, 8 and 10 bits; `--threads 1` and `--workers 1`, `2` give the same bytes.
- Rates in 1/512 bit from integer tables; J = 512 x e^2 + lam x bits512, e the coefficient's error in coefficient units
  (exact integers), lam = llround(weight x dc_step^2), weight Qa's (84/32 x plane_mult / 4 / 128 of a squared DC step).
- Speed: S1, S2 against off, bound +2% CPU a frame; past it, an offset first (speed-offset rule).
- No new GPU code.

## Review Focus

- A lone intra DC with a non-DCT type, and IST's end-of-block limits, after the levels change: the type must fall back as the
  oracle's code() does, or the decoder reads another type.
- CCTX: U losing its levels makes the type uncoded; V's flag context depends on U's end of block.
- Forward skip coding (inter IDTX) and lossless: not TCQ, untouched.
- Levels past the Golomb range (`high_range`) and 64-size transforms' coded 32x32.
- 10-bit: kQDepth tables, the clamp of dq.

---

### Task 1: real-cost Viterbi in src/core

**Files:**
- Create: `src/core/real_trellis.h`, `src/core/real_trellis.cpp` (from tools/oracle/real_trellis.inc, fixed point)
- Modify: `tools/oracle/real_trellis.inc` (becomes a thin use of the core one), `CMakeLists.txt`
- Test: `tests/test_real_trellis.cpp`

**Interfaces:**
- Produces: `namespace celv2::real { struct Block {const FRAME_CONTEXT* fc; TX_SIZE tx; TX_TYPE type; int plane, pl_ctx,
  dc_sign_ctx, qindex, bd; bool tcq; int32_t flag[2]; /* 1/512 bit */ int64_t lam; };
  int levels(const Block&, const tran_low_t* coeff, tran_low_t* q, tran_low_t* dq, int64_t* bits512); }`
- Per state, only the levels the contexts can still read (a window by scan index), not a whole array copied per step.

- [ ] Test `rate_matches_symbols`: on fixed CDFs, `*bits512` equals the writer's symbols' bits x 512 (rounded per symbol as
  the table) for random blocks of all 25 sizes, TCQ, all three planes, and the same coefficients x64 (Golomb).
- [ ] Test `same_as_double`: levels equal the oracle's double version on 20000 random blocks where no J tie is within 1e-9.
- [ ] Implement; both pass.

### Task 2: --real-trellis in the coder

**Files:** `src/core/encoder.cpp` (residual_code, residual_code_uv, the superblock loop's CDF copy), `src/core/config.h`
or where `--psy-trellis` lives, `src/main.cpp` (option), GUI none.

- [ ] Test `real_trellis_off_same_bytes` (test_cli): a clip at q130, default and `--real-trellis 0` byte-equal to HEAD's
  checked-in hash.
- [ ] Test `real_trellis_decodes`: on, 8 and 10 bits, AVM decode equals `--recon`; `--workers 1` and `2` equal.
- [ ] After the type and IST choice, the chosen coefficients again (fwd transform, IST), `real::levels()` per TCQ plane;
  lone intra DC and IST limits fall back as the oracle's code(); chroma U then V with V's flag context from U.

### Task 3: measure

- [ ] 33 frames, eight sources, q100-190, against off: PSNR-Y and YUV 6:1:1 (`tools/c17/bd_run.py`, `NEW_ENV`).
- [ ] Speed: `tools/c17/speed_ab.ps1 -NewEnv` or the option, S1 S2 S4.
- [ ] Report docs/reports/2026-11-1x-real-trellis-q1.md; Q2 (120 frames, SSIMULACRA2, XPSNR) next per the spec.
