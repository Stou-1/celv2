# `--grain N`: denoise, code the clean picture, let the decoder add the grain back (design)

2026-11-03. Perceptual plan item 2. Opt-in.

## Why

Film grain and sensor noise are random: coding them costs many bits and still looks smeared at
anything but high rates. AV2 can carry a grain model in each frame header (film_grain_params: an
autoregressive pattern and a piecewise intensity curve per plane) that the decoder synthesises on the
output picture only, never on the references. The encoder then codes the denoised picture.

## How

1. **Model and denoise.** For each source frame, AVM's `avm_denoise_and_model_run()` (avm_dsp/
   noise_model.c, outside av2/encoder) on a YV12 wrapper of the Frame420 (8 or 10 bits): it finds flat
   blocks, fits the noise model, Wiener-denoises the frame in place and fills an `avm_film_grain_t`.
   `N / 10` is its noise level (aomenc's `--denoise-noise-level N`), 32x32 blocks. Runs in the
   reader thread, frame by frame in display order (the model carries across frames), before scene
   cuts and chunking.
2. **Signal.** Unlike AV1, AV2 keeps the model out of the frame header: a film grain model OBU
   (`OBU_FILM_GRAIN_MODEL`, obu_fgm.c `read_fgm_obu()`) stores up to 8 models (scaling points, AR lag and
   coefficients, shifts, chroma multipliers, overlap / clipping / block size flags) in the decoder, and
   a shown frame's header (`setup_film_grain()`) carries only apply_grain, the model id (3 bits) and a
   16-bit seed. Sequence header `film_grain_params_present = 1`.
   - One model id, 0. A model OBU goes before the first coded frame OBU of a temporal unit (AVM's OBU
     order check rejects one after a coded frame), so also before hidden frames prepended to a unit,
     at each keyframe and whenever a shown frame's model differs from the last one sent, in stream
     order (ChunkCoder's output: the parallel B frames are coded out of display order). Every chunk
     starts with a keyframe, so chunks coded on different workers need nothing from each other.
   - The model of a frame rides on its frame buffer (`RefCntBuffer::film_grain_params`, as AVM's
     encoder keeps it): a hidden frame's show-existing OBU writes its grain fields from the slot.
   - Seed: 7391 + 3381 x the display index (libaom's step; never 0).
3. **Code** the denoised frames as now. `--recon` and `--psnr` see the coded picture without grain
   (what the references hold); a decoder's output has the grain. The in-process verify decoder skips
   the grain (`AV2D_SET_SKIP_FILM_GRAIN`) unless asked.

## Interface

- `--grain N` (1..50, 0 off; default 0). `--tune film` may turn it on later, after measuring.
- GUI: a 「フィルムグレイン」 setting in 詳細設定, after the CLI is measured.

## Measure

- Sources: the live-action 吉村 clips and crowd_run / park_joy (grain and noise); anime with grain or
  noise if any (most of the anime sources are clean: there it must not change the bytes by much).
- Bytes at the same qindex with and without `--grain` (the bits the noise took).
- PSNR / VMAF of the decoded output (grain synthesised) against the source fall by construction: report
  them, but the deciding check is the user's viewing of the decoded files side by side.
- Decoding: AVM (avmdec) applies the grain; the stream must decode equal to `--recon` with
  `--skip-film-grain` (or the grain off in the AVM decoder in tests).

Measured: docs/reports/2026-11-04-film-grain.md (live action at `--grain 25`: q110 -14 to -28%,
q150 -6 to -12%, q190 -1.4 to +0.3% bytes at the same qindex, about the same time).

## Not in this

- A grain model per scene or per region; denoising with a better filter than the noise model's Wiener
  filter; the GUI setting.
