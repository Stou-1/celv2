# CelV2

An experimental AV2 encoder. A GPU (CUDA or Vulkan) searches motion and scores candidate
partitions and modes; the CPU re-picks them by rate and distortion, codes the bitstream and
reconstructs with the AVM reference code. Every frame CelV2 writes decodes bit-exactly with the AVM
v1.0.0 decoder: the tests compare per-frame MD5s against the in-process AVM decoder.

CelV2 is an independent project. It is not affiliated with or endorsed by the Alliance for
Open Media.

## Status

- 4:2:0 at 8 or 10 bits (y4m `C420` / `C420p10`); 10-bit coding by default, 8-bit inputs too
  (`--bit-depth 8` codes 8 bits). Fixed qindex per frame type.
- Keyframes at scene cuts; 8- or 16-frame mini-GOPs with a hidden anchor, hidden pyramid mid frames and
  B frames from both sides (the leaf B frames coded in parallel); up to 7 reference frames.
- Intra: directional, smooth and DC modes, intra prediction from further reference lines (MRLS).
- Inter: NEARMV / NEWMV with MV refinement, adaptive and joint MVDs, compound prediction, global
  motion, BAWP for fades, local warp (`--warp`, off by default), skip mode.
- Residual: DCT / ADST / identity transforms with transform partitions, trellis coded
  quantization (TCQ).
- In-loop filters: deblocking, CDEF (one strength per frame), CCSO. No loop restoration.
- Decision engines: CUDA (NVIDIA), Vulkan (tested on an RTX 5090 and an Arc Pro B50) and a CPU
  fallback, all giving the same bytes.

On the project's 1080p test clips it uses fewer bits than NVENC AV1 (p7) on most of them and still
more than SVT-AV1 (preset 6), by 4% to 63%. 1080p codes at about 6 fps on an RTX 5090 and a
24-core Xeon. See docs/design for the measurements behind each choice. (The clips named there,
"anime A" to "anime E" and "live action", are private and not distributed; the Xiph clips
crowd_run, park_joy and ducks_take_off are public.)

## Requirements (Windows)

- Visual Studio 2026 with the C++ x64 tools (MSVC), LLVM (clang-cl, for AVM), NASM, CMake, Ninja
  (`scripts\env.ps1` finds Visual Studio with vswhere; `CELV2_VS` and `CELV2_NASM` override it)
- CUDA toolkit 13.x (the build needs it even when only the Vulkan engine is used)
- Optional, for the Vulkan engine: the Vulkan headers and `slangc`, from the LunarG SDK
  (`VULKAN_SDK`), the PATH, or unpacked under `%LOCALAPPDATA%\bin` (`Vulkan-Headers-<version>`,
  `slang-<version>`). Without them the Vulkan engine is left out.
- A CPU with AVX2 (the build uses `/arch:AVX2`)
- The AVM source tree at tag `v1.0.0` (commit `966a7d7cd6fcf60360caf5dc413b2aeeb65e144d`), from
  <https://gitlab.com/AOMediaCodec/avm>
- Paths without non-ASCII characters (nvcc)

## Build and test

```powershell
$env:AVM_ROOT = 'C:\path\to\avm'   # optional; defaults to ~\avm
powershell -File scripts\build_avm_lib.ps1   # once: builds AVM as a static library
powershell -File scripts\build.ps1
powershell -File scripts\test.ps1
```

## Usage

```powershell
build\celv2enc.exe -i in.y4m -o out.ivf [--qindex 120] [--frames N] [--keyint 0] [--recon recon.yuv]
                    [--device auto|gpu|cpu|vulkan|vulkan2] [--tile-columns L] [--tile-rows L] [--threads N]
                    [--refs N] [--scenecut 0|1] [--bframes N] [--bq D] [--aq D] [--pyramid 0|1] [--psnr]
```

`-i -` reads the y4m from standard input (`ffmpeg ... -f yuv4mpegpipe - | celv2enc -i - ...`). The
full option list, with what each one does and its default, is at the top of
[src/main.cpp](src/main.cpp).

`--psnr` prints `psnr: P` at the end: the mean luma PSNR of the shown frames against the source,
at the coded depth.

`--bit-depth 8|10` (default 10) sets the coded depth. 8-bit sources code better at 10 bits (on the
1080p test clips 4% to 20% fewer bits for the same luma PSNR, at the same speed); `--qindex` keeps
the 8-bit scale at either depth.

`--colour 709|pq|hlg` (with `--range limited|full`) sends the colour description with every
keyframe, and `--mdcv "G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min)"` (x265's `--master-display` units) and
`--cll MaxCLL,MaxFALL` send HDR10 static metadata. For an HDR source, convert it to 10-bit 4:2:0 y4m
(`ffmpeg -i in -pix_fmt yuv420p10le -strict -1 out.y4m`) and pass these from its metadata.

`--keyint N` puts at most N frames between keyframes (0: no limit). `--scenecut 1` (the default) also
makes the first frame of each new shot a keyframe; one-frame flashes and black frames between shots
are not cuts. `--refs N` (1..7) sets how many past frames an inter frame may predict from.

`--bframes N` (0..15) codes up to N frames after the frame that follows them: that one is coded
hidden, the N frames predict from both sides, and the hidden one is then shown (default 7 with
`--refs 5` or more and `--qindex 100` or more, else 3). Every temporal unit still outputs one
frame, in display order. `--bq D` and `--aq D` offset the qindex of those B frames and of the
hidden frames. With `--pyramid 1` (the default) the middle of 3 or more B frames is coded next,
hidden, and the others predict from it too.

Tiles are coded on separate CPU threads. Without `--tile-columns` / `--tile-rows` (log2 counts),
celv2enc halves the width until it is under 1280 and the height until it is under 2560, up to 16
tiles: 2 at 1080p, 4 at 4K, 16 at 8K and 16K. Each tile costs a few bits (4K: 2, 4 and 8 tiles ran
at 2.4, 2.7 and 3.1 fps for +0.6% and +1.6% bytes).

`--device auto` (the default) decides on a CUDA GPU (Turing or newer, driver r580 or newer) if
there is one, else on the first discrete Vulkan GPU, else on the CPU, which is much slower. All
give the same output. `--device gpu|cpu|vulkan|vulkan2` picks one (`vulkan2`: the second discrete
Vulkan GPU). On the CPU path `--threads` limits only the tile threads; the decision threads take
the other cores.

### Settings by content, effort and size

`--tune animation|film|still` sets several options for the kind of video (as x264's `--tune`):
longer mini-GOPs where the picture stays still, and keyframe and anchor boosts for anime; delta-q
for live action; 16-frame mini-GOPs throughout for slides. Options given on the command line win.

`--preset max` ignores speed for quality and size: a 16-frame mini-GOP is coded both as one and as
two of 8 and the better kept (`--gop-trial`), and live action gets a per-frame deblocking search
(`--deblock-search`). See [docs/design/2026-10-31-preset-max.md](docs/design/2026-10-31-preset-max.md).

A target size takes two runs over the same input:

```powershell
celv2enc -i in.y4m -o pass1.ivf --pass 1 --stats stats.txt [options]
celv2enc -i in.y4m -o out.ivf --pass 2 --stats stats.txt --target-size 10000000 [the same options]
```

The second run keeps the IVF file at or under the given bytes (one qindex for the whole video,
moved by at most 2 a mini-GOP); if it still ends over, it says so and exits with 3. See
[docs/design/2026-10-31-target-size.md](docs/design/2026-10-31-target-size.md).

`--workers N|auto` codes scenes (a keyframe to the next, 48 frames at least) at once on N pipelines;
without a target size the output is the same bytes as one worker. With a target size `auto` is
one worker: sharing the size control between chunks came out further from the target (see
[docs/design/2026-10-31-chunk-parallel.md](docs/design/2026-10-31-chunk-parallel.md)). `--memory-limit MB` (default 80%
of the installed RAM) caps the memory: celv2enc then takes fewer workers, then turns off parallel B
frames (the same bytes), then the GOP trial, each with a warning.

To compare against AVM, run `scripts\bench_vs_avm.ps1 -Input clip.y4m -AvmBin <folder with
avmenc.exe and avmdec.exe>`, then `python scripts\bdrate.py docs\reports\bench.csv`. The
bench also needs ffmpeg, and bdrate.py needs numpy.

## GUI (Windows)

`gui\` holds a small WinForms front end (C# / .NET Framework 4.8, Japanese UI) that converts videos to AV2 with `celv2enc`:
drop a video, pick a quality preset, press the button. It uses ffmpeg to read the input and writes an `.mp4` (video with
av2_mux, audio added with MP4Box) or, without MP4Box, an `.ivf` plus a separate audio file.

```powershell
$env:AV2MUX_DIR = 'C:\path\to\folder with the patched av2_mux.exe'
powershell -File gui\build.ps1      # builds gui\bin (needs build\celv2enc.exe; copies av2_mux.exe from AV2MUX_DIR)
powershell -File gui\test.ps1       # unit and integration tests
powershell -File gui\package.ps1    # dist\celv2-gui-<date>.zip
```

The zip holds the app, `celv2enc.exe`, `av2_mux.exe`, the VC++ runtime DLLs and the licenses. It does not include ffmpeg or
MP4Box. The GUI can also code the audio as HE-AACv2 with `fdkaac.exe` (nu774/fdkaac built with
libfdk-aac), which is not included either: libfdk-aac comes under its own license
(Fraunhofer FDK AAC Codec Library license) and grants no patent rights. See [gui/README_ja.txt](gui/README_ja.txt) (Japanese) for how to use it. `av2_mux.exe` is av2-tools commit ffbcb39
with its MP4 writer replaced so it no longer uses MPEG's libisomediafile; see
[gui/third_party/av2-tools/](gui/third_party/av2-tools/).

## License

CelV2 is licensed under the BSD 3-Clause Clear License; see [LICENSE](LICENSE).

Some parts are derived from the AVM decoder. These files say so in their headers, and those
parts remain under AVM's license: Copyright (c) 2021, Alliance for Open Media, BSD 3-Clause
Clear License.

CelV2 also links the AVM library and compiles AVM's `common/md5_utils.c`. AVM's license and
the Alliance for Open Media Patent License 1.0 are in [third_party/avm/](third_party/avm/).
Include both files when you redistribute CelV2 in source or binary form.

Neither license grants patent rights beyond what those files state.

`tests/data/avm_testsrc2.ivf` is ffmpeg's synthetic `testsrc2` pattern encoded with AVM's
`avmenc`.
