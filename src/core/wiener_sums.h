// The normal equations' sums of the chroma Wiener (WIENER_NONSEP) fit, in integers.
#pragma once
#include <cstdint>

namespace celv2 {

constexpr int kWienerFeatures = 18;  // wienerns_filter_uv's taps

// A pixel's features: f[px[k]] = o[dx[k]] + o[-dx[k]] - 2 o[0] (the reconstruction, offsets in its stride),
// f[pl[k]] = l[dl[k]] - l[0] (the downsampled luma, offsets in its stride), f[kWienerFeatures] = target - o[0].
// Features no tap writes are 0.
struct WienerTaps {
  int nx = 0, dx[kWienerFeatures] = {}, px[kWienerFeatures] = {};
  int nl = 0, dl[kWienerFeatures] = {}, pl[kWienerFeatures] = {};
};

constexpr int kWienerSums = kWienerFeatures + 1;

// sums[i][j] (i <= j) = sum of f_i f_j over rows [2, h - 2) and columns [2, w - 2) of a w x h plane x (stride st),
// with its luma (stride lw) and target (s16 if not null, else s8; stride sst). Exact for samples up to 12 bits.
// On `bands` threads of the pool (any split gives the same sums).
void wiener_sums(const uint16_t* x, int st, const int16_t* luma, int lw, const uint16_t* s16, const uint8_t* s8,
                 int sst, int w, int h, const WienerTaps& t, int64_t sums[kWienerSums][kWienerSums], int bands);
// The pairs i <= j of the same sums, in order (kWienerPairs of them), over rows [y0, y1) and columns [x0, x1) only (the
// caller keeps them 2 or more from the plane's edges). One thread.
constexpr int kWienerPairs = kWienerSums * (kWienerSums + 1) / 2;
void wiener_rect_sums(const uint16_t* x, int st, const int16_t* luma, int lw, const uint16_t* s16, const uint8_t* s8,
                      int sst, int x0, int x1, int y0, int y1, const WienerTaps& t, int64_t* out);
// The same, one pixel at a time (the tests' reference).
void wiener_sums_ref(const uint16_t* x, int st, const int16_t* luma, int lw, const uint16_t* s16, const uint8_t* s8,
                     int sst, int w, int h, const WienerTaps& t, int64_t sums[kWienerSums][kWienerSums]);

// The luma fit's sums per class (no luma features: t.nl is 0). Feature k < t.nx is o[dx[k]] + o[-dx[k]] - 2 o[0] (in t's
// order, not px), feature t.nx the target less o[0]; sums[c * P + q], P = (t.nx + 1)(t.nx + 2) / 2, q running over the
// pairs i <= j in order, of the samples of [x0, x1) x rows y0, y0 + ystep, ... below y1 (x0 a multiple of 4) whose 4x4 block's class is c:
// sub[cls[(y / 4) * cst + x / 4]] (< classes <= 16). Exact for samples up to 12 bits. One thread. Reads x up to 15
// samples past x1 + the taps' reach.
void wiener_class_sums(const uint16_t* x, int st, const uint16_t* s16, const uint8_t* s8, int sst, int x0, int x1, int y0,
                       int y1, int ystep, const WienerTaps& t, const uint8_t* cls, int cst, const uint8_t* sub, int classes,
                       int64_t* sums);
void wiener_class_sums_ref(const uint16_t* x, int st, const uint16_t* s16, const uint8_t* s8, int sst, int x0, int x1,
                           int y0, int y1, int ystep, const WienerTaps& t, const uint8_t* cls, int cst, const uint8_t* sub,
                           int classes, int64_t* sums);

}  // namespace celv2
