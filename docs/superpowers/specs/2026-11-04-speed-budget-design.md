# A speed budget: measure whole-video speed, make the CPU's heavy stages faster, move big blocks to the GPU (design)

2026-11-04. The source of the speed that new features spend. Today's designs (`--paq`, psy-rd, the
halftone fixes, `--screen`, the intra gap) each cost some time; their rule is that a slowdown past
their bound is offset by a gain elsewhere, or reported to the user for a decision. This builds those
gains, and the ledger that keeps the account.

## Decisions (agreed 2026-11-04)

- **The speed that counts is a whole video's**, many scenes coded at once with `--workers auto`, the CPU
  about full. There, what matters is the CPU work per frame. A single scene's speed (waiting between
  frames: on 2026-11-03 one chunk kept 25.6 of 48 threads busy, 92.9% of samples waiting) is recorded
  as secondary.
- **Staged:**
  - A: the CPU's heavy stages made faster without changing the bytes;
  - B: only the large blocks A cannot shrink enough move to the GPU (Vulkan), bit-exact.

## Measuring

**Bench runs** (`--workers auto --refs 7 --qindex 130`):

| Name | Source | Settings |
|---|---|---|
| S1 | AM, 600 frames 1080p (6 scenes) | `--tune animation` |
| S2 | the three 600-frame mixed stretches of the `--preset max` measurements (Angel Beats from 155 s, AIR from 54 s, Dragon Maid from 65 s) | `--tune animation` |
| S3 | AM 4K, 120 frames (`g_am30_4k`) | `--tune animation` |
| S4 | as S1 | `--preset max` |

吉村, 120 frames, is recorded as the single-scene live-action reference.

**Recorded:**
- wall time and fps;
- the process's total CPU time, per frame: the ledger's main number;
- the GPU's busy share;
- CPU time per stage.

**`CELV2_PROFILE=1`** (environment; committed, off by default; off, nothing is timed): per-stage CPU
time totals printed at the end. The stages: decision, RD pass, quantization (TCQ), transforms,
prediction, entropy coding, deblocking, CDEF, CCSO, loop restoration, TPL, reading, denoising. Finer
detail from `tools/sampler`.

**The ledger** (`docs/design/speed-budget.md`):
- the baseline: S1-S4's fps and CPU time per frame, with the commit and date;
- spent: each new feature's added seconds and percent;
- saved: each speed-up's seconds and percent;
- the balance against the baseline.

A feature's spec reads the balance to see whether an offset exists; without one, the user decides.

## Stage A: the CPU's heavy stages, the same bytes

**Order:** by CPU time per stage under the full load. Candidates from 2026-11-03: TCQ, the DCT, CDEF,
CCSO, loop restoration's sample filling, the RD pass's scoring, motion compensation's interpolation,
entropy coding.

Removing repeated work counts too. For example, the source transform cached per block, which psy-rd
needs as well, is shared.

**Instruction sets:**
- At start, CPUID picks AVX-512 (F, BW, VL), else AVX2, else plain C++, once.
- The AVX-512 code lives in its own files compiled with AVX-512 enabled, so one executable still runs
  on CPUs without it.
- `CELV2_SIMD=scalar|avx2|avx512` forces a level (tests).
- No AMX: its int8 / bf16 matrices lack the precision these computations need.

**The same bytes:**
- Every level gives exactly the scalar code's bytes.
- Integer kernels match by nature. Floating-point ones (the DCT) fix their operation order so every
  level computes the same values: exactly, not "about the same".
- A speed-up that changes the bytes is not stage A: it is a coding change and needs BD-rate.

**Per speed-up:**
1. S1-S4's CPU time per frame and fps measured.
2. The ledger's "saved" entry.
3. The same bytes checked at every level, 8 and 10 bits.

## Stage B: large blocks to the GPU

**A stage moves when all three hold:**
1. After stage A it still takes 10% or more of the CPU time under the full load.
2. It is data-parallel within a frame and can match the CPU exactly (integer or fixed point).
3. The GPU has room: busy below 60% in the profile.

**Candidates:**

| Stage | Fit |
|---|---|
| `--grain`'s denoising | Likely heavy. AVM's Wiener denoiser uses floating-point FFTs, so matching the CPU needs a fixed-point rewrite first. |
| TPL (and the full-scale TPL the halftone design may add) | Integer SAD / SATD; matches as the decision engine does. |
| Applying CDEF and loop restoration | Integer; matches. |
| Trial coding by the real cost (the AV1-gap design's W3) | Entropy coding is serial: not for the GPU. Out. |

**Rules:**
- Vulkan only, with a CPU reference giving identical results (as the decision engines).
- The GPU work overlaps the CPU's, so nothing waits on it.
- **Slow GPUs:** where the GPU is the slower side (the Arc Pro B50), a short timing at start keeps the
  stage on the CPU. Both give the same bytes, so this is only a choice of where to compute.
- Checked on the RTX 5090 and the Arc Pro B50.

**Transfers and VRAM:**

- **One round trip per frame.**
  - A frame goes up once. GPU stages chain on the resident copy: the denoised frame stays for the
    decision engine (③), and TPL reuses the engine's pyramid.
  - Only results come down: the denoised frame the CPU codes from, and small tables (TPL values).
  - Transfers run asynchronously, overlapped with computation, so the CPU never waits on them.
  - The volume is small: a 1080p frame about 3 MB, a 16K one about 200 MB (8-bit), about 4 ms on the
    RTX 5090 (about 50 GB/s) and 8 ms on the B50 (about 25 GB/s), against about 5.4 s a 16K frame.
    Waiting in lockstep is what costs, not the bytes.
- **A VRAM budget.**
  - Measured on a16k (9 frames, `--refs 7`, docs/reports/2026-11-04-vram-16k.md): 16.48 GB with 16 phase
    planes, 7.59 GB with the 4 of the layout below. The differences between the layouts put the phase
    planes at about 5.6 references' worth, and everything else at about 4.6 GB.
  - Estimates from that, not measured: all seven references held, about 19.5 GB with 16 planes and
    about 8.3 GB with 4.
  - A 16K frame (Y, U, V) is about 200 MB at 8 bits, 400 MB at 10. With 17 read-ahead frames resident at
    8 bits: about 11.7 GB with 4 planes, which fits the Arc Pro B50 (16 GB).
  - Rules:
    - At start, the free VRAM the GPU reports (Vulkan's memory budget extension), less the engine's
      share, sets how many preprocessed frames stay resident.
    - Frames beyond that come down to the CPU and go up again just before ③ (one more transfer, below
      0.2% of a 16K frame's time).
    - Short of VRAM, the encode never fails: at worst no frame stays resident.
- **How a reference's phases are held (decided 2026-11-05).**
  - Measured at 16K, the same bytes in all five (docs/reports/2026-11-04-vram-16k.md):

    | Layout | Peak VRAM, RTX 5090 / B50 | Engine on the B50, 2 / 4 references | Whole encode on the B50 (under load) |
    |---|---|---|---|
    | 16 planes (today) | 16.48 / 14.55 GB | 8211 / 12848 ms | 24.8 s/frame |
    | A: none, every phase interpolated on read | 5.36 / 5.39 GB | +30 / +32% | 23.8 s/frame |
    | B: the 4 half-pel planes, the other 12 interpolated on read | 7.59 / 7.61 GB | +12 / +12% | 20.9 s/frame |
    | C: 16 planes for the nearest 2 references | 9.32 / 9.35 GB | +5 / +21% | 21.8 s/frame |
    | D: a band of planes around the rows in flight | 5.61 / 5.63 GB | +52 / +67% | 26.9 s/frame |

    The RTX 5090's engine took 1172 ms with 16 planes and 1086 ms with B in the morning's run, within
    the run-to-run spread of about 8%.
  - The B50's 16-plane row reads lower than the RTX 5090's only there; the other four agree across the
    GPUs. It is likely at its limit and spilling to shared memory (not measured), which would explain
    why its whole encode gains with B while its engine alone loses.
  - **B becomes the default on Vulkan, whatever the free VRAM.** This replaces the earlier proposal (16
    planes where VRAM allows).
  - **First, one check, because all of this is 16K:** S1 and S3 on both GPUs, fps and engine time, on a
    quiet machine.
    - B stays everywhere if the whole encode is at most 1% slower.
    - If it is slower, 16 planes stay for frames below a size set by those runs, chosen by frame size:
      no rule per GPU.
  - **A is the fallback.** At start, the free VRAM is compared with what the engine needs: B if it fits,
    else A, else the CPU engine. Same bytes in each, and the encode never fails.
  - C and D go: rows in docs/design/tried-not-kept.md, and their measurement code out.
    `CELV2_VK_PHASES` stays for tests. The CUDA engine keeps 16 planes until it is removed.
  - The B50's engine loss of 10-12% is a target for stage C.
  - **Still to measure:**
    - peak VRAM on a 16K clip long enough to hold seven references, on both GPUs (the 8.3 GB above);
    - what the other 4.6 GB is, buffer by buffer: under B it is most of the total;
    - the RTX 5090's engine with all layouts on a quiet machine.

    In the report and docs/design/pipeline-map.md.

## Stage C: the GPU decision engine, for every GPU

The engine matters to the whole encode only where it limits it: large frames (16K) and slow GPUs. At
16K the Arc Pro B50's engine took 9.68 s a frame against the CPU's 4.3 s; the RTX 5090's 1.21 s
overlapped the coding. The aim is speed on any GPU, at the least maintenance.

**Done (each in one shader version, for every GPU):**
1. **Overlap with the CPU:** the coarse motion search on the sources ahead, without waiting for the
   previous frame's reconstruction (the SVT review's item 2, kept 2026-11-03). On the B50 at 16K it
   could hide up to about 13% (the CPU's 4.3 s); it helps the slower GPUs most.
2. **Less memory traffic:** kernels run back to back (the cost and the final decision) merged, so
   intermediate results are not written out and read back. Bandwidth matters more on slower GPUs.
3. **Less work:**
   - the same computation repeated per reference done once;
   - lossless early exits, where a candidate's partial cost already passes the best known one: its
     remaining sums stop, the decision unchanged.

   On the GPU every block's cost is computed in parallel, so early exits pay mainly in sequential
   parts (the ME refinement); these are a minor lever, measured like the rest.

**Not done:**
- a kernel version per GPU vendor or architecture (NVIDIA warps of 32, Intel subgroups of 8-32, AMD
  wavefronts of 64);
- reliance on vendor extensions;
- predictive pruning that may change decisions (a neural or heuristic prune: a coding change, the
  neural design's).

**Rules:**
- One version of each shader.
- The same results as the CPU reference (the existing engine tests).
- Kept only if it is faster, or at least not slower, on both the RTX 5090 and the Arc Pro B50 (engine
  time and whole-encode fps, S1-S4 and a16k).

## Measure

- Each speed-up alone: S1-S4's CPU time per frame and fps, entered in the ledger.
- Stage B also: not slower on the B50.
- Stage C: engine time and fps on the RTX 5090 and the Arc Pro B50, at 1080p (S1) and 16K (a16k).
- The ledger's balance states how much of the new features' time is covered.

## Tests

- **Instruction sets:** `CELV2_SIMD=scalar`, `avx2`, `avx512` give the same bytes (8 and 10 bits, with
  B frames).
- **Moved stages:**
  - each stage's CPU and Vulkan versions give the same results (unit tests);
  - whole encodes give the same bytes with `--device cpu` and the GPU.
- **`CELV2_PROFILE`:** on, the per-stage times are printed; off, the output is unchanged.
- **The rest:** every existing test passes.

## Not in this

- Parallelism inside one scene (two mids at once: deferred 2026-11-03, about 7%).
- Speed-ups that change the bytes (each a coding change with its own BD-rate).
- AMX and neural networks.
