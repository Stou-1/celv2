// CELV2_WIDE=<list>, the measurement switches of docs/superpowers/specs/2026-11-04-av1-gap-breakdown-design.md
// (W1-W6): an environment variable, not an option. Off, the bytes are unchanged.
// 1: each reference's best MV tried on every block of the RD pass (the pass already compares every
// partition shape CelV2 codes). 2: a full search per reference in the pass. 3: real cost (trial_64).
// 4: chroma in the pass. 5: every transform type by quantize() cost at every size, and --rd-intra.
// 6: every pair of references in the pass.
#pragma once
#include <cstdlib>
#include <cstring>

namespace celv2 {

// Whether the list holds n (one digit each, as "34").
inline bool wide(char n) {
  const char* const e = std::getenv("CELV2_WIDE");
  return e && std::strchr(e, n);
}

}  // namespace celv2
