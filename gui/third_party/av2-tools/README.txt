av2_mux.exe (bundled with CelV2 GUI)

Source:   https://github.com/AOMediaCodec/av2-tools
Commit:   ffbcb39 ("adding my personal test server termporarily")
Build:    MSVC x64, Release, with one local modification (below).
License:  BSD 3-Clause Clear License (LICENSE) and the Alliance for Open Media
          Patent License 1.0 (PATENTS). Both files are in this folder.

Local modification: av2-tools-celv2.patch (apply with "git apply" in a checkout of ffbcb39)
  Replaces apps/av2_mux/mp4_writer.cpp/.h with a writer that builds the MP4
  boxes itself (ISO/IEC 14496-12), so av2_mux no longer uses or links MPEG's
  libisomediafile. The CMake files build av2_mux without it (av2_demux still
  needs it and stays behind BUILD_CONTAINER_TOOLS). Same samples, sample tables
  and av2C/colr as before; the differences: the durations in mvhd/tkhd/mdhd are
  filled in (libisomedia wrote 0), the creation/modification times are 0, there
  is no iods box, the tkhd size is the picture size (libisomedia left 320x240),
  and cslg's compositionEndTime includes the last sample's duration.
  Build: cmake -S . -B _build -G Ninja -DCMAKE_BUILD_TYPE=Release
         cmake --build _build --target av2_mux
  Then set AV2MUX_DIR to _build\apps\av2_mux for gui\build.ps1 and gui\package.ps1.

Packaging modification: application manifest (gui/utf8-manifest.ps1)
  gui/build.ps1 and gui/package.ps1 merge src/utf8.manifest into the copy of
  av2_mux.exe they place in gui/bin and in the zip (mt.exe from the Windows
  SDK). It adds <activeCodePage>UTF-8</activeCodePage>, so the ANSI
  main(argc, argv) receives file names outside the system code page (e, u
  with accents, emoji) intact on Windows 10 1903 and later. The code is not
  changed; the original manifest (asInvoker) is kept.
