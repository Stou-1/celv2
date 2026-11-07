// --psy-rd: a psychovisual term in the RD decisions, keeping the source's AC energy (texture, line
// art) that squared error lets a smooth candidate drop (docs/superpowers/specs/2026-11-04-psy-rd-design.md).
#pragma once
#include <cstdint>

#include "core/perceptual_aq.h"

namespace celv2 {

// Which content the term guards: texture (the default; live action) or drawn lines (--tune animation).
enum class PsyRule { Texture, Line };

// The term's scale against squared error in steps (the spec's starting value; set by measurement).
constexpr double kPsyScale = 1;

// For one transform block, in quantizer steps: S the source's coefficients, c the residual's (source
// minus prediction), dq their dequantized levels, n coefficients (index 0 the DC, left out): the
// reconstruction's are S - (c - dq), and the term is | sum |S| - sum |S - (c - dq)| | over the AC.
// Blurring (AC lost) and added ringing or grain (AC gained) both cost.
double psy_term(const int32_t* S, const int32_t* c, const int32_t* dq, int n, double step);

// The block's weight (0..1) from the --paq map over the 16x16s the w x h block at (x, y) covers:
// t(E) rising from 0 at log2(variance + 1) <= 4 to 1 at >= 8, times 1 - C (texture rule) or C
// (line rule), C the edge coherence.
double psy_weight(const AqMap& m, int x, int y, int w, int h, PsyRule rule);

}  // namespace celv2
