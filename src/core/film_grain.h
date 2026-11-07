// --grain: AVM's noise model (avm_dsp/noise_model.c) denoises each source frame and fits the film
// grain it took out; the frames are coded denoised and the decoder adds the grain back on output
// (docs/superpowers/specs/2026-11-03-film-grain-design.md).
#pragma once
#include <cstdint>
#include <memory>

#include "io/y4m.h"

extern "C" {
#include "avm_dsp/grain_synthesis.h"
}

struct GrainParams {
  avm_film_grain_t p;  // apply_grain set; the seed is the frame's (grain_seed())
};

// avm_denoise_and_model_run() in two stages: the Wiener denoising (a flat noise spectrum fixed by the
// level, so frames are independent: any thread, any order) and the model update (display order).
class GrainModel {
 public:
  // level: --grain N (1..50); AVM's noise level is N / 10 (aomenc's --denoise-noise-level).
  GrainModel(int bit_depth, int level);
  ~GrainModel();
  GrainModel(const GrainModel&) = delete;
  GrainModel& operator=(const GrainModel&) = delete;

  struct Denoised;  // a frame's 16-bit planes, denoised planes and flat blocks
  // Stage 1 (thread-safe).
  std::shared_ptr<Denoised> denoise(const Frame420& f) const;
  // Stage 2, in display order: updates the model with d (from denoise(f)), then replaces f's planes by
  // the denoised ones and sets f.grain to the grain fitted so far (f unchanged while there is none).
  void finish(Frame420& f, Denoised& d);
  void run(Frame420& f) { finish(f, *denoise(f)); }

 private:
  struct Model;
  std::unique_ptr<Model> m_;
  int bit_depth_;
  float level_;
};

// The same model in a film grain model OBU (av2_check_grain_params_equiv: the seed aside).
bool same_grain_model(const avm_film_grain_t& a, const avm_film_grain_t& b);
// The grain's random seed for a frame shown at this display index (never 0).
uint16_t grain_seed(int display);
