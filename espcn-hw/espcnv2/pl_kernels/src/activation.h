#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include <stdlib.h>
#include "ap_fixed.h"

#define L1_IM_H 16
#define L1_IM_W 16
#define L1_IN_C 1
#define L1_OUT_C 64
#define L1_K 5
#define L1_S 1
#define L1_P 2
#define L1_OUT_H 64
#define L1_OUT_W 256
#define L1_OUT_SIZE L1_OUT_H*L1_OUT_W
#define L1_OUT_TILE_H 16
#define L1_OUT_TILE_W 64
#define L1_NUM_TILES 16
#define L1_OUT_TILE_SIZE L1_OUT_TILE_H*L1_OUT_TILE_W


#define L2_IM_H 16
#define L2_IM_W 16
#define L2_IN_C L1_OUT_C
#define L2_IM_SIZE L2_IM_H*L2_IM_W*L2_IN_C
#define L2_OUT_C 32
#define L2_K 3
#define L2_S 1
#define L2_P 1
#define L2_COL_H 576
#define L2_COL_W 256
#define L2_COL_TILE_H 576
#define L2_COL_TILE_W 8
#define L2_COL_SIZE L2_COL_H*L2_COL_W
#define L2_IN_TILE_SIZE L2_COL_TILE_H*L2_COL_TILE_W
#define L2_OUT_H 32
#define L2_OUT_W 256
#define L2_OUT_SIZE L2_OUT_H*L2_OUT_W
#define L2_OUT_TILE_H 8
#define L2_OUT_TILE_W 8
#define L2_NUM_TILES 128
#define L2_OUT_TILE_SIZE L2_OUT_TILE_H*L2_OUT_TILE_W

#define L3_IM_H 16
#define L3_IM_W 16
#define L3_IN_C L2_OUT_C
#define L3_IM_SIZE L3_IM_H*L3_IM_W*L3_IN_C
#define L3_OUT_C 4
#define L3_K 3
#define L3_S 1
#define L3_P 1
#define L3_COL_H 288
#define L3_COL_W 256
#define L3_COL_TILE_H 288
#define L3_COL_TILE_W 16
#define L3_COL_SIZE L2_COL_H*L2_COL_W
#define L3_IN_TILE_SIZE L3_COL_TILE_H*L3_COL_TILE_W
#define L3_OUT_H 4
#define L3_OUT_W 256
#define L3_OUT_SIZE L3_OUT_H*L3_OUT_W
#define L3_OUT_TILE_H 4
#define L3_OUT_TILE_W 16
#define L3_NUM_TILES 16
#define L3_OUT_TILE_SIZE L2_OUT_TILE_H*L2_OUT_TILE_W


#define BITS 8		// set bitwidth of multipliers
#define BITS_EXP 1024 //must be set to 2^(BITS+2). Should match tanh_vals size


//typedef ap_fixed<BITS+2,2,AP_RND> quantized_type;		// multipliers
typedef ap_fixed<BITS+4,4,AP_RND> l_quantized_type;	// intermediate results

#endif
