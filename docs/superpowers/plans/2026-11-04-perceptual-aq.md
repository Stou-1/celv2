# Perceptual AQ (`--paq`, stage A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `celv2enc --paq S` moves each 64x64 superblock's qindex by a 16x16 perceptual importance map (variance, edge coherence, darkness) on every coded frame, and the bench decides by SSIMULACRA2 and XPSNR whether it is kept.

**Architecture:** A new `core/perceptual_aq` module builds the 16x16 map from the source luma and folds it into the existing `FrameSpec::sb_dq` grid (64x64). ChunkCoder adds it to every FrameSpec (TPL offsets summed when `--deltaq` is on). Pipeline snaps the 64x64 offsets to the coded 128x128 superblocks before the RD pass, so the RD pass and the coder use the same qindex.

**Tech Stack:** C++17 (MSVC, Ninja), AVM v1.0.0 as `avm.lib`, CTest executables with `tests/check.h`.

**Spec:** `docs/superpowers/specs/2026-11-04-perceptual-aq-design.md`

## Global Constraints

- Build from the PowerShell tool: `. scripts\env.ps1; cmake --build build`. After any header change, `ninja -t deps` in `build/` must list no `#deps 0` object (else touch those sources and rebuild).
- Tests: `scripts\test.ps1` (all), single ones as `build\test_<name>.exe`.
- `--paq 0` (the default) and `--deltaq 0`: the same bytes as before this plan, on every input.
- Offsets within +-35 (`9 * kDeltaQRes - 1`, `kDeltaQRes = 4` in `state/codec_state.h`); the map's mean per frame 0 before clamping.
- `--paq` range 0..2; values outside refused with exit != 0.
- The map reads `Frame420::y` (8-bit planes) at any bit depth; the visible area only (`width` x `height`).
- Starting constants (spec): `kEnergy = 4`, `kCoherence = 1`, `kDark = 4`, `kDarkLevel = 64`; each later change carries its measured evidence in a comment.
- Never include `av2/encoder` headers in `src/` (test `no_encoder_includes`).
- Commit with the repo's git config, no identity overrides; trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. No push.
- Downloads (the SSIMULACRA2 tool) only after the user approves file name, source and size.

## Review Focus

1. Frame sizes not a multiple of 16 or 64 (1080 lines: the last 16x16 row is 8 high): partial blocks use their visible samples; the map and `aq_sb64()` cover every 64x64 of the sb_dq grid. Pinned in Task 1 (`edge_sizes_cover_the_grid`, 202x122).
2. A frame of one flat colour (all-black fades, title cards): every offset 0, nothing NaN. Pinned in Task 1 (`flat_frame_is_zero`).
3. A tiny frame (one 64x64 or less): no division by zero, offsets 0. Pinned in Task 1 (`edge_sizes_cover_the_grid`, 32x32).
4. qindex near the ends with a strong AQ (`--qindex 250 --paq 2`, `--qindex 5 --paq 2`): per-superblock qindex stays 1..255 and AVM decodes equal to `--recon`. Pinned in Task 4 (`cli_paq`).
5. 10-bit input: the map equals the one from the 8-bit rounding; AVM decodes equal to `--recon`. Pinned in Task 1 (`ten_bit_reads_the_8bit_planes`) and Task 4.

---

### Task 1: The importance map

**Files:**
- Create: `src/core/perceptual_aq.h`, `src/core/perceptual_aq.cpp`
- Create: `tests/test_perceptual_aq.cpp`
- Modify: `CMakeLists.txt` (add the source to `celv2_core`, `celv2_test(test_perceptual_aq)`)

**Interfaces:**
- Produces (namespace `celv2`):
  - `struct AqMap { int cols = 0, rows = 0; std::vector<float> dq, log_energy, coherence, periodicity; };` — per 16x16, raster, `cols = ceil(width / 16)`, `rows = ceil(height / 16)`: the offset in qindex, log2(E + 1), C and P (0 where not computed). psy-rd reads the last three.
  - `AqMap aq_map(const Frame420& f, double strength);`
  - `std::vector<int> aq_sb64(const AqMap& m, int width, int height);` — one int per 64x64, raster, `ceil(width / 64) * ceil(height / 64)` (the `FrameSpec::sb_dq` grid).
  - `struct AqSpread { double within = 0, between = 0; };` and `AqSpread aq_spread(const AqMap& m, int coded_sb_px);` — the mean variance of the 16x16 offsets inside each coded superblock, and the variance of the superblocks' means.

- [ ] **Step 1: Write the failing tests** in `tests/test_perceptual_aq.cpp` (synthetic frames built with `make_frame()` / `pad_frame()`; `RUN` / `CHECK` from `check.h`):
  - `flat_frame_is_zero`: 256x128 frame, every luma 128, `aq_map(f, 1)`: every `dq` exactly 0 and finite.
  - `flat_finer_noise_coarser`: 256x128, left half luma 128, right half 128 + uniform noise in [-20, 20] (fixed LCG seed): mean of left-half `dq` < -1, mean of right half > 1, mean of all `dq` within 0.01 of 0.
  - `line_art_kept`: 256x128 flat 200 with 4-pixel black vertical lines every 16 pixels in the right half, against the same frame with noise of the same per-block variance in the right half: the line blocks' mean `dq` < the noise blocks' mean `dq` - 2.
  - `dark_finer`: 256x128, left half luma 30, right half 200, both with the same noise in [-4, 4]: mean `dq` left < mean right.
  - `halftone_kept`: 256x128, left half round dots of radius 2 (luma 40) on a 6-pixel 45-degree grid over 200, right half random noise of the same per-block variance: every left-half block's `periodicity` > 0.7, every right-half block's < 0.3, the left half's mean `dq` < the right half's - 2; on `flat_frame_is_zero`'s and `line_art_kept`'s line blocks `periodicity` is 0.
  - `offsets_clamped`: a frame of half flat, half noise in [-120, 120], strength 2: every `dq` in [-35, 35]; every `aq_sb64` value in [-35, 35].
  - `sb64_leans_sensitive`: 256x128 noise everywhere except one flat 16x16 inside the first 64x64: `aq_sb64()[0]` < the plain mean of that 64x64's 16 `dq` values.
  - `edge_sizes_cover_the_grid`: 202x122 and 32x32 frames: `cols`/`rows` as above, `aq_sb64().size() == ceil(w/64) * ceil(h/64)`, all finite; the 32x32 flat-plus-noise frame does not throw.
  - `ten_bit_reads_the_8bit_planes`: a 10-bit `make_frame(256, 128, 10)` whose `hy` is 4x an 8-bit frame's `y`, after `round_to_8bit()`: `aq_map` equal element by element to the 8-bit frame's.
  - `spread_separates`: a frame whose 16x16 offsets alternate inside every 128x128 (checkerboard of flat and noise 16x16 blocks): `within > between`; a frame flat in its left 128 columns and noisy in the rest: `between > within`.

- [ ] **Step 2: Run to verify they fail**

Run: `. scripts\env.ps1; cmake --build build --target test_perceptual_aq` then `build\test_perceptual_aq.exe`
Expected: build fails (`core/perceptual_aq.h` not found).

- [ ] **Step 3: Implement the module** per the spec's "The importance map": per 16x16 block over the visible samples, E = variance, C = structure-tensor coherence ((l1 - l2) / (l1 + l2))^2 from central differences (0 when l1 + l2 = 0), L = mean; `d = S * kEnergy * (log2(E + 1) - frame mean of log2(E + 1))`; a positive d times `(1 - kCoherence * C)`; for blocks with a positive d and C < 0.5, P = the largest normalized autocorrelation of the zero-mean block over lags 2..6 pixels away (half-plane: dy >= 0, dx > 0 when dy = 0; over the overlap), and a positive d times `(1 - kPeriodic * clamp((P - 0.3) / 0.4, 0, 1))` (`kPeriodic = 1`); `d -= S * kDark * max(0, (kDarkLevel - L) / kDarkLevel)`; subtract the frame mean of d; clamp to +-35. `aq_sb64`: per 64x64, the mean of (25th percentile of its blocks' dq, mean of its blocks' dq), rounded to the nearest int, clamped to +-35.

- [ ] **Step 4: Run to verify they pass**

Run: `build\test_perceptual_aq.exe`
Expected: exit 0, no output.

- [ ] **Step 5: Commit**

```bash
git add src/core/perceptual_aq.h src/core/perceptual_aq.cpp tests/test_perceptual_aq.cpp CMakeLists.txt
git commit -m "feat: core/perceptual_aq: a 16x16 perceptual importance map (variance, edge coherence, darkness) and its 64x64 superblock offsets"
```

### Task 2: The RD pass and the coder see the same superblock qindex

The RD pass (`Pipeline::collect`, `pipeline.cpp:189-194`) quantizes each 64x64 at `qindex + sb_dq[i]`, while CodecState (`codec_state.cpp:592-616`) codes each 128x128 at the mean of its four offsets rounded to `kDeltaQRes`. With AQ the four differ, so the RD pass decides at a qindex that is not coded.

**Files:**
- Modify: `src/state/codec_state.h`, `src/state/codec_state.cpp` (the free function below; `begin_frame` keeps its own averaging, now an identity on snapped input)
- Modify: `src/core/encoder.h` (accessor), `src/pipeline.cpp` (`Pipeline::code`, `Pipeline::code_group`: snap `spec.sb_dq` before `launch`/`collect`)
- Test: `tests/test_perceptual_aq.cpp` (a new case)

**Interfaces:**
- Produces:
  - `void snap_sb_dq(std::vector<int>& sb_dq, int width, int height, int coded_sb_px);` (codec_state.h) — every 64x64 offset replaced by its coded superblock's: the mean of the group's offsets rounded to the nearest multiple of `kDeltaQRes` (the same rule `begin_frame` uses, without the base clamp). Empty input stays empty.
  - `int Encoder::sb_px() const;` — `block_size_wide[cm->seq_params.sb_size]` (128 today).

- [ ] **Step 1: Write the failing test** `snap_matches_coded_superblocks`: width 300, height 200 (5x4 grid of 64x64, 3x2 coded 128x128); input offsets `0..19`; after `snap_sb_dq(v, 300, 200, 128)` each 64x64 in a group holds that group's mean rounded to a multiple of 4 (assert all 20 values against the hand-computed list), and the right/bottom partial groups average only the 64x64s they have.
- [ ] **Step 2: Run it**: fails to compile (`snap_sb_dq` undeclared).
- [ ] **Step 3: Implement** `snap_sb_dq`, `Encoder::sb_px`, and call `snap_sb_dq(spec.sb_dq, cfg_.width, cfg_.height, enc_.sb_px())` in `Pipeline::code` and for each spec in `Pipeline::code_group`, before the engine launch.
- [ ] **Step 4: Run** `build\test_perceptual_aq.exe` and `build\test_cli.exe`: exit 0. (`--deltaq 1` streams may change bytes; default runs must not: `cli_workers` and `cli_tiles_deterministic` compare bytes.)
- [ ] **Step 5: Audit the coder's lambdas**: in `src/core/encoder.cpp` and `src/bitstream/block_writer.cpp`, every lambda or quantizer derived from `cm->quant_params.base_qindex` or `cfg.qindex` inside the block walk uses the superblock's `enc_.st_->sb_qindex(mi_row, mi_col)` instead (the intra lambda already does, `encoder.cpp:1105-1106`). Record each site found and the decision in the commit message; if none needs a change, say so there.
- [ ] **Step 6: Commit**

```bash
git commit -am "fix: the RD pass decides each 64x64 at its coded 128x128 superblock's qindex (sb_dq snapped once in Pipeline)"
```

### Task 3: `--paq` on every coded frame

**Files:**
- Modify: `src/chunk_coder.h` (`LoopConfig`), `src/chunk_coder.cpp`, `src/main.cpp`

**Interfaces:**
- Consumes: `aq_map`, `aq_sb64`, `aq_spread` (Task 1); `snap_sb_dq` runs in Pipeline (Task 2).
- Produces: `LoopConfig::paq` (`double`, 0 = off); CLI `--paq S`; env `CELV2_AQ_STATS=1` prints one stderr line per coded frame: `celv2enc: aq frame <display>: within <w> between <b>` (`aq_spread(map, 128)`, two decimals).

- [ ] **Step 1: Write the failing CLI test** `cli_paq_args` in `tests/test_cli.cpp`: `--paq 2.5` and `--paq -1` exit != 0 with "--paq" in stderr; `--paq 1` on `device_input()` exits 0.
- [ ] **Step 2: Run it**: fails (unknown option).
- [ ] **Step 3: Implement**:
  - `main.cpp`: parse `--paq` with `strtod` (0..2, else `std::invalid_argument("--paq must be 0..2: " + v)`), `lc.paq = a.paq`; usage text: one entry after `--deltaq`, "`--paq S`: perceptual AQ strength (0 off, 1 nominal, 2 at most): each 64x64's qindex by a 16x16 importance map of the source luma (docs/superpowers/specs/2026-11-04-perceptual-aq-design.md)".
  - `chunk_coder.cpp`: one helper used at every FrameSpec site (lines 313, 328, 347, 399, 425, 460 today): with `lc_.paq > 0`, `aq_sb64(aq_map(src, lc_.paq), cfg.width, cfg.height)` from that frame's own source (`q(i).f`; the keyframe's unfiltered source under `--key-filter`), added element-wise to `sb_dq` when it is not empty, else assigned; each value clamped to +-35. With `CELV2_AQ_STATS` set, the stderr line above.
- [ ] **Step 4: Run** `build\test_cli.exe`: exit 0.
- [ ] **Step 5: Commit** (`feat: --paq S: perceptual AQ offsets on every coded frame, summed with TPL's under --deltaq`).

### Task 4: End-to-end checks

**Files:**
- Modify: `tests/test_cli.cpp`

- [ ] **Step 1: Write `cli_paq`** (11 frames 256x128 of `scene_frame` with a flat half and `--bframes 3`, so anchors, mids and non-reference B frames all carry offsets):
  - `--paq 1 --recon R`: AVM decodes equal to R (`decode_frames` + `write_y4m_frame_bytes`, as `cli_bframes`), and the IVF differs from a `--paq 0` run's;
  - `--paq 0`: the IVF equal byte for byte to a run without `--paq`;
  - `--paq 1 --workers 1` and `--workers 2` on 120 frames with a scene cut at 60 (as `cli_workers` builds them): equal IVFs;
  - `--qindex 250 --paq 2` and `--qindex 5 --paq 2`: decode equal to `--recon`;
  - the 10-bit variant of the first case (`C420p10`, as `cli_encodes_10bit`): decode equal to `--recon`.
- [ ] **Step 2: Run** `build\test_cli.exe`: exit 0 (the checks pass on Tasks 1-3; a failure here is a bug in them, fixed there with its own failing case).
- [ ] **Step 3: Run the whole suite**: `scripts\test.ps1`: every test passes (the GUI's 11 skips as before).
- [ ] **Step 4: Commit** (`test: --paq decodes as reconstructed on every frame type, 8 and 10 bits, at the qindex ends; the same bytes at any worker count`).

### Task 5: Measure, tune, report

**Files:**
- Create: `docs/reports/2026-11-0X-perceptual-aq.md` (the date of the run)
- Modify: `src/core/perceptual_aq.cpp` (constants, with evidence), `docs/design/tried-not-kept.md` (a row per dropped strength or term), `src/main.cpp` usage text (the measured numbers)

- [ ] **Step 1: Get SSIMULACRA2**: ask the user to approve one of libjxl's `ssimulacra2` (BSD-3) or `ssimulacra2_rs` (BSD-2), naming file, source and size. Unpack into its own new folder; run it with paths as arguments.
- [ ] **Step 2: Encode** the eight sources of the spec (120 frames, `--refs 7`, `--tune animation` on anime as the reports do) at qindex 100 / 130 / 160 / 190 with `--paq 0`, `0.5`, `1`, `1.5`; keep IVF, `--recon`, time, and the `CELV2_AQ_STATS` lines.
- [ ] **Step 3: Score**: XPSNR with ffmpeg `-lavfi "[0:v][1:v]xpsnr"` over all frames; SSIMULACRA2 on every 4th frame (frames 0, 4, ..., 116 as PNG), mean per point; global-MSE PSNR and VMAF as the earlier reports. BD-rate of each strength against `--paq 0` with `scripts/bdrate.py`.
- [ ] **Step 4: Decide** per the spec: kept when the mean over the sources is negative by both SSIMULACRA2 and XPSNR and no source is worse than +1% by either. If no strength passes, try the constants one at a time (kCoherence 0.5, kDark 0 / 8, kEnergy 2 / 6) on AIR OP, 吉村 and AM before a full rerun; every drop gets its tried-not-kept row.
- [ ] **Step 5: Stage B evidence**: from the AQ stats lines, per source the share of frames with `within > between`; state in the report whether stage B's condition holds.
- [ ] **Step 6: Viewing files** for the user: AIR OP and 吉村 at qindex 160, `--paq 0` and the chosen strength, decoded with AVM to y4m (paths in the report). Stop there: the user's viewing decides the default or tune.
- [ ] **Step 7: Commit** the report, the constants and the tried-not-kept rows.
