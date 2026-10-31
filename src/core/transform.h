// Forward transform (encoder side, not normative): separable orthonormal DCT-II in float (Lee's
// recursive butterflies, vectorised across rows/columns), scaled to the coefficient scale of AVM's
// inverse transform (see Encoder's calibration).
#pragma once

extern "C" {
#include "av2/common/av2_common_int.h"
}

// residual: tx_w x tx_h, row-major with `stride`. out: AVM layout (pos = row * min(w,32) + col,
// row = vertical frequency), only the top-left min(w,32) x min(h,32) frequencies.
void fwd_dct(const int16_t* residual, int stride, TX_SIZE tx, double scale, tran_low_t* out);
// Any transform type of a tx at most 32x32 (ADST and FLIPADST at most 16): AVM's integer kernels
// (av2/common tables: columns, then rows, with fwd_tx_shift and the sqrt(2) step of 2:1 sizes),
// so the coefficients are on the scale AVM's inverse expects. Same layout as fwd_dct.
void fwd_transform(const int16_t* residual, int stride, TX_SIZE tx, TX_TYPE type, tran_low_t* out);
// Of types[0..n) (tx at most 16x16), the one whose fwd_transform has the least sum of coefficient
// magnitudes; out gets its coefficients. Each vertical 1D pass is made once for all types sharing it.
TX_TYPE most_compact(const int16_t* residual, int stride, TX_SIZE tx, const TX_TYPE* types, int n, tran_low_t* out);
