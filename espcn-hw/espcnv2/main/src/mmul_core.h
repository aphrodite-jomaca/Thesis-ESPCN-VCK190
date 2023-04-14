#ifndef MMUL_CORE_H_
#define MMUL_CORE_H_

#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "aie_api/utils.hpp"

#include <stdint.h>
#include <adf.h>

/*
 * LAYER 1
 */

#ifndef ROW_A1
#define ROW_A1 64
#endif

#ifndef COL_A1
#define COL_A1 32
#endif

#ifndef COL_B1
#define COL_B1 256
#endif

#ifndef ROW_A1_TILE
#define ROW_A1_TILE 16
#endif

#ifndef COL_A1_TILE
#define COL_A1_TILE 16
#endif

#ifndef COL_B1_TILE
#define COL_B1_TILE 64
#endif

#ifndef NUM_TILES1
#define NUM_TILES1 (COL_B1/COL_B1_TILE)*(ROW_A1/ROW_A1_TILE)
#endif

#ifndef ITERS1
#define ITERS1 COL_A1/COL_A1_TILE
#endif

#ifndef WSIZE1
#define WSIZE1 ROW_A1_TILE*COL_A1
#endif


#ifndef SIZE_OUT1
#define SIZE_OUT1 ROW_A1_TILE*COL_B1_TILE
#endif

#ifndef WIN_SIZE_IN1
#define WIN_SIZE_IN1 COL_B1_TILE*COL_A1*4
#endif

#ifndef WIN_SIZE_OUT1
#define WIN_SIZE_OUT1 SIZE_OUT1*4
#endif

#ifndef ROW_OFF1
#define ROW_OFF1 0
#endif

#ifndef COL_OFF1
#define COL_OFF1 6
#endif

/*
 * LAYER 2
 */

#ifndef ROW_A2
#define ROW_A2 32
#endif

#ifndef COL_A2
#define COL_A2 576
#endif

#ifndef COL_B2
#define COL_B2 256
#endif

#ifndef ROW_A2_TILE
#define ROW_A2_TILE 8
#endif

#ifndef COL_A2_TILE
#define COL_A2_TILE 576
#endif

#ifndef COL_B2_TILE
#define COL_B2_TILE 8
#endif

#ifndef NUM_TILES2
#define NUM_TILES2 (COL_B2/COL_B2_TILE)*(ROW_A2/ROW_A2_TILE)
#endif

#ifndef ITERS2
#define ITERS2 COL_A2/COL_A2_TILE
#endif

#ifndef WSIZE2
#define WSIZE2 COL_A2*ROW_A2_TILE
#endif


#ifndef SIZE_OUT2
#define SIZE_OUT2 ROW_A2_TILE*COL_B2_TILE
#endif

#ifndef WIN_SIZE_IN2
#define WIN_SIZE_IN2 COL_B2_TILE*COL_A2*4
#endif

#ifndef WIN_SIZE_OUT2
#define WIN_SIZE_OUT2 SIZE_OUT2*4
#endif

#ifndef ROW_OFF2
#define ROW_OFF2 0
#endif

#ifndef COL_OFF2
#define COL_OFF2 1+NUM_TILES1
#endif

/*
 * LAYER 3
 */

#ifndef ROW_A3
#define ROW_A3 4
#endif

#ifndef COL_A3
#define COL_A3 288
#endif

#ifndef COL_B3
#define COL_B3 256
#endif

#ifndef ROW_A3_TILE
#define ROW_A3_TILE 4
#endif

#ifndef COL_A3_TILE
#define COL_A3_TILE 288
#endif

#ifndef COL_B3_TILE
#define COL_B3_TILE 16
#endif

#ifndef NUM_TILES3
#define NUM_TILES3 (COL_B3/COL_B3_TILE)*(ROW_A3/ROW_A3_TILE)
#endif

#ifndef ITERS3
#define ITERS3 COL_A3/COL_A3_TILE
#endif

#ifndef WSIZE3
#define WSIZE3 COL_A3*ROW_A3_TILE
#endif


#ifndef SIZE_OUT3
#define SIZE_OUT3 ROW_A3_TILE*COL_B3_TILE
#endif

#ifndef WIN_SIZE_IN3
#define WIN_SIZE_IN3 COL_B3_TILE*COL_A3*4
#endif

#ifndef WIN_SIZE_OUT3
#define WIN_SIZE_OUT3 SIZE_OUT3*4
#endif

#ifndef ROW_OFF3
#define ROW_OFF3 0
#endif

#ifndef COL_OFF3
#define COL_OFF3 1+NUM_TILES1+NUM_TILES2
#endif


//#ifndef INLINE
//#ifndef INLINE_DECL
//#define INLINE_DECL
//#endif

#define __AIE_API_TYPES__HPP__

#pragma once

class MMUL_T_1
{
private:
	float (&wgts)[WSIZE1];
	float (&b)[SIZE_OUT1];
	float (&intrmdtRes)[SIZE_OUT1];
//	float intrmdtRes[SIZE_OUT1];

public:
	MMUL_T_1(float(&weights)[WSIZE1], float(&zeros)[SIZE_OUT1], float(&bias)[SIZE_OUT1]);//, float(&zeros)[SIZE_OUT1]);

	void mmul1(const int RowA_tile, const int ColA_tile, const int ColB_tile, float* A_in, float* C_out, int tile, int shift);

	void mmul1_top(input_window_float* in, output_window_float* out);

	static void registerKernelClass()
	{
		REGISTER_FUNCTION(MMUL_T_1::mmul1_top);
		REGISTER_PARAMETER(wgts);
		REGISTER_PARAMETER(b);
		REGISTER_PARAMETER(intrmdtRes);
	}
};

class MMUL_T_2
{
private:
	float (&wgts)[WSIZE2];
	float (&b)[SIZE_OUT2];
//	float intrmdtRes[SIZE_OUT2];

public:
	MMUL_T_2(float(&weights)[WSIZE2], float(&bias)[SIZE_OUT2]);//);

	void mmul2(const int RowA_tile, const int ColA_tile, const int ColB_tile, float* A_in, float* C_out, int tile, int shift);

	void mmul2_top(input_window_float* in, output_window_float* out);

	static void registerKernelClass()
	{
		REGISTER_FUNCTION(MMUL_T_2::mmul2_top);
		REGISTER_PARAMETER(wgts);
		REGISTER_PARAMETER(b);
	}
};

class MMUL_T_3
{
private:
	float (&wgts)[WSIZE3];
	float (&b)[SIZE_OUT3];
//	float intrmdtRes[SIZE_OUT3];

public:
	MMUL_T_3(float(&weights)[WSIZE3], float(&bias)[SIZE_OUT3]);//);

	void mmul3(const int RowA_tile, const int ColA_tile, const int ColB_tile, float* A_in, float* C_out, int tile, int shift);

	void mmul3_top(input_window_float* in, output_window_float* out);

	static void registerKernelClass()
	{
		REGISTER_FUNCTION(MMUL_T_3::mmul3_top);
		REGISTER_PARAMETER(wgts);
		REGISTER_PARAMETER(b);
	}
};


//#else
//#  ifndef INLINE_DECL
//#  ifdef __llvm__
//#    define INLINE_DECL inline __attribute__((always_inline))
//#  else
//#  define INLINE_DECL inline
//#  endif
//#  endif
//#  undef INLINE
//#  include "mmul_core1.cc"
//#  include "mmul_core2.cc"
//#  define INLINE
//# endif


#endif
