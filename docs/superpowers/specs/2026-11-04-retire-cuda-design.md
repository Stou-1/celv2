# Retiring the CUDA engine: Vulkan as the one GPU path (design)

2026-11-04. Since 2026-10-31 all new GPU code is Vulkan only; this removes the CUDA decision engine
that remains beside it.

## Why

- **Speed:** the same bytes either way, and the whole encode as fast on the RTX 5090. At 16K (a16k,
  9 frames, q130, `--refs 7`, 2026-11-03): the encode took 5.4 s a frame on CUDA and on Vulkan. The
  engines alone (engine_bench, 2 references) took 1.09 s on CUDA and 1.21 s on Vulkan, about 11%
  more; the engine overlaps the CPU's coding, so the encode does not see it. At 1080p the engine is a
  smaller share still.
- **Work:** every engine change is now kept bit-identical three times (CUDA, Vulkan, the CPU
  reference `kernels_ref.cpp`); new GPU work (`--screen` B2, the speed budget's stage B) is Vulkan only,
  so the CUDA engine would fall behind.
- **Building and shipping:** no CUDA toolkit needed to build (`project(... CUDA)`,
  `CMAKE_CUDA_ARCHITECTURES`); no CUDA runtime in the release; the same path on NVIDIA, Intel (the Arc
  Pro B50) and AMD.
- **Cost:** about 11% in the engine alone where it limits the encode (16K, or a slow CPU); the Vulkan
  kernels are tuned further if that shows.

## Steps

1. **Check Vulkan is complete and fast enough.**
   - Every test passes with the Vulkan engine as the GPU (`--device vulkan`).
   - The CLI device tests use the CPU engine and Vulkan as the pair that must match.
   - Speed, CUDA against Vulkan on the RTX 5090, with the speed budget's runs (S1-S4: 1080p 600-frame
     anime, the three mixed stretches, AM 4K, `--preset max`) and a16k: whole-encode fps and the
     engine's time from `CELV2_PROFILE`.
   - The stale CMake comment ("Vulkan ... not used by the encoder yet") is corrected.
   - If Vulkan is more than 5% slower in a whole encode anywhere, its kernels are tuned before step 2,
     or the user decides.
2. **Vulkan becomes the default GPU.**
   - `--device auto`: the first discrete Vulkan GPU, else the CPU.
   - `--device gpu` becomes the first Vulkan GPU (saved GUI settings and scripts keep working).
   - `--list-devices` lists the Vulkan GPUs and the CPU.
   - One release ships like this, with CUDA still built (selectable as `--device cuda`) for a direct
     comparison if a user reports a difference.
3. **The tests stop needing CUDA.**
   - `test_vk_kernels` compares the Vulkan kernels with the CPU reference (`kernels_ref.cpp`) instead of
     the CUDA kernels.
   - `engine_bench` compares Vulkan with the CPU engine.
   - test_cli's `CUDA_VISIBLE_DEVICES` cases become "no Vulkan GPU" cases (an environment switch that
     hides Vulkan devices, for tests).
4. **CUDA removed.**
   - `src/gpu/engine.cu`, `src/gpu/kernels.cu`, `kernels.h` and the CUDA engine class go.
   - `__CUDACC__` branches go from `inter_common.h` and `intra_common.h`; the integer primitives stay
     shared by the CPU reference and, by porting, the Slang shaders.
   - The CUDA language and toolkit go from CMakeLists.txt; the build scripts and the release zip lose
     their CUDA steps and files.
   - `--device cuda` goes; `gpu` stays as the Vulkan alias.

## GUI

- `DeviceList.cs`: the 「GPU（CUDA）」 entry becomes the Vulkan GPUs by name.
- `AdvancedDialog.cs`'s note 「自動 = CUDA の GPU、なければ Vulkan の GPU、なければ CPU」 becomes
  「自動 = GPU（Vulkan）、なければ CPU」.
- `MainForm.cs` reads the engine line ("decisions on Vulkan (...)" as GPU, not only "decisions on GPU").
- `EncodeSettings` keeps "gpu" as a valid saved value.
- GUI tests updated to match.

## Measure

- Step 1's table (CUDA against Vulkan: fps and engine time on S1-S4 and a16k, RTX 5090; the B50 on
  Vulkan for reference) in `docs/reports/<date>-vulkan-only.md` and the speed ledger
  (`docs/design/speed-budget.md`).
- The same bytes as before on every source and setting of the tests (the engines already agree).

## Tests

- **Step 1:** the whole suite with `--device vulkan`.
- **Steps 2-3:**
  - `--device auto` and `gpu` pick Vulkan where there is a GPU, the CPU where there is none;
  - `test_vk_kernels` matches the CPU reference;
  - GUI tests pass with the new device list and note.
- **Step 4:**
  - a clean build with no CUDA toolkit installed succeeds;
  - every test passes;
  - the release zip holds no CUDA file.
- On the RTX 5090 and the Arc Pro B50, `--device vulkan` and `vulkan2` give the CPU engine's bytes.

## Speed

Steps 2-4 must not slow a whole encode by more than 5% on S1-S4 (step 1 checks before the switch). A
slowdown past it is offset by a Vulkan kernel speed-up or another gain in the speed ledger; if none is
found, the user is told and decides.

## Not in this

- Tuning the Vulkan kernels beyond closing a gap found in step 1.
- Moving more stages to the GPU (the speed budget's stage B).
