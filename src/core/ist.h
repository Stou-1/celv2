// Intra secondary transform (IST), encoder side: AVM's forward kernels (the transpose of the
// normative inverse, av2_inv_stxfm()) on the low frequencies of a DCT_DCT or ADST_ADST block.
#pragma once

extern "C" {
#include "av2/common/av2_common_int.h"
}

// The IST size class of a tx and primary type (block_signals_sec_tx_type()'s st_size_class):
// 0 = 4xN / Nx4, 1 = 8x8 DCT, 2 = larger DCT, 3 = 8x8 and larger ADST_ADST.
int ist_size_class(TX_SIZE tx, TX_TYPE primary);
// How many coefficients IST codes for that class (the first ones in the primary scan order).
int ist_out_count(int size_class);
// In place on coeff (AVM layout of tx, the primary coefficients of `primary`): the secondary
// coefficients of kernel set `set` (0..6, +7 for ADST_ADST, as get_secondary_tx_set()) and type
// stx (1..3), at the first ist_out_count() positions of the primary scan, the rest zero. mode: the
// block's luma intra mode, DC_PRED for an inter block (the decoder transposes for some modes).
// Returns the energy the IST dropped (sum of squares before less after, in coefficient units).
double fwd_ist(tran_low_t* coeff, TX_SIZE tx, TX_TYPE primary, PREDICTION_MODE mode, int set, int stx, int bd);
