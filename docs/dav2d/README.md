# Playing CelV2 streams with dav2d

[dav2d](https://code.videolan.org/videolan/dav2d) is VideoLAN's AV2 decoder. It is still in
development, and the October 2026 versions (for example 3812a7a) cannot decode CelV2 v0.3 streams:
on all eight test streams they crashed, failed to parse a frame header, or gave almost every frame
different from the encoder's `--recon`.

## Why older dav2d fails

Until late October 2026, dav2d had bugs or missing parts in tools that CelV2 uses: B-frame motion
vectors (joint MVD, the distance to far references), global motion, chroma loop restoration (the
frame's non-separable Wiener filter), deblocking at low q in 10-bit, per-segment quantizers, and
BAWP. Upstream main has fixed all of them by 2026-11-12, most between 2026-10-25 and 2026-11-02.

## What works

- AVM v1.0.0's reference decoder `avmdec` decodes every CelV2 stream; use it to check a decode:
  `avmdec --rawvideo -o avm.yuv input.ivf`
- Upstream dav2d main 9b5cef2 (2026-11-12), unmodified, decodes the v0.3 test streams (8 clips x 4
  qindex, and mini-GOPs with 0, 7 and 15 B-frames) byte for byte the same as `--recon` and
  `avmdec`.

Build it (meson, ninja and nasm; on Windows, from a Visual Studio developer prompt):

```
git clone https://code.videolan.org/videolan/dav2d.git
cd dav2d
git checkout 9b5cef2
meson setup build --buildtype=release --default-library=static
ninja -C build
```

Check a decode: `build/tools/dav2d -q -i input.ivf -o dav2d.yuv --muxer yuv`, then compare its hash
with avmdec's output (`certutil -hashfile dav2d.yuv MD5` or `md5sum`). If a newer main does not
match, go back to 9b5cef2.

## Windows: `-o -` (standard output)

On Windows, dav2d writes standard output in text mode, so `-o -` turns every 0x0A byte into
0x0D 0x0A and the piped YUV or Y4M is broken. Upstream has not fixed this yet. Either write to a
file, or apply [dav2d-win-stdout.patch](dav2d-win-stdout.patch) before building:

```
git apply path/to/dav2d-win-stdout.patch
```

It switches standard output to binary mode in the `yuv` and `yuv4mpeg2` muxers and gives it a
4 MB buffer (the default 4 KB writes slow the pipe). It applies to 3812a7a and 9b5cef2 and changes
nothing on other systems.

## License

dav2d is under the BSD 2-Clause license ([COPYING](COPYING), copied from dav2d). The patch is a
change to dav2d and is distributed under the same license. CelV2 itself does not include or link
dav2d.
