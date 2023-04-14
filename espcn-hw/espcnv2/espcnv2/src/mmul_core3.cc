#include "mmul_core.h"

MMUL_T_3::MMUL_T_3(float(&weights)[WSIZE3], float(&bias)[SIZE_OUT3])
: wgts(weights), b(bias)
{ }


void MMUL_T_3::mmul3(
        const int RowA_tile,//1
        const int ColA_tile,//144
        const int ColB_tile,//4
        float* B_in,
        float* C_out,
		int tile,
	int shift
) {

	//********** Matrix dimensions********/
	constexpr size_t sizeTileA = 4 * 2;
	constexpr size_t sizeTileB = 2 * 4;
	constexpr size_t sizeTileC = 4 * 4;


	//********** Mul Intrinsic********/
	using MMUL = aie::mmul<4, 2, 4, float, float>;

    unsigned int i,j,z;


	for (z=0; z<RowA_tile; z+=1) //1time
//		chess_loop_range(1,)
	{
		//********** Output vector ********/
		float * __restrict pC1 = C_out + (      z * ColB_tile +       0) * sizeTileC;
		float * __restrict pBias = b   + (      z * ColB_tile +       0) * sizeTileC;


		for (j=0; j<ColB_tile; j+=2) //2times
//			chess_loop_range(1,)
		{
			const float * __restrict pA1 = wgts + (      0 * ColA_tile +       0) * sizeTileA + tile*sizeTileA*RowA_tile*ColA_tile;
          	const float * __restrict pB1 = B_in + (      0 * ColB_tile +       j) * sizeTileB + tile*sizeTileB*ColA_tile*ColB_tile;
          	const float * __restrict pB2 = B_in + (      0 * ColB_tile + (j + 1)) * sizeTileB + tile*sizeTileB*ColA_tile*ColB_tile;


          	aie::vector<float, sizeTileA> A0 = aie::load_v<sizeTileA>(pA1); pA1 += sizeTileA;
          	aie::vector<float, sizeTileB> B0 = aie::load_v<sizeTileB>(pB1); pB1 += sizeTileB * ColB_tile;
      		aie::vector<float, sizeTileB> B1 = aie::load_v<sizeTileB>(pB2); pB2 += sizeTileB * ColB_tile;


          	MMUL C00; C00.mul(A0, B0);
          	MMUL C01; C01.mul(A0, B1);


          	for (i = 1; i < ColA_tile; ++i)
//          	chess_prepare_for_pipelining
//          	chess_loop_range(143,)
          	{
          	    A0 = aie::load_v<sizeTileA>(pA1); pA1 += sizeTileA;
          	    B0 = aie::load_v<sizeTileB>(pB1); pB1 += sizeTileB * ColB_tile;
          	    B1 = aie::load_v<sizeTileB>(pB2); pB2 += sizeTileB * ColB_tile;


          	    C00.mac(A0, B0);
          	    C01.mac(A0, B1);
          	}

          	aie::vector<float, sizeTileC> C00b = aie::load_v<sizeTileC>(pBias);
          	C00b = aie::add(C00b, C00.template to_vector<float>(shift));
			aie::store_v(pC1, C00b); pC1 += sizeTileC; pBias +=sizeTileC;

			aie::vector<float, sizeTileC> C01b = aie::load_v<sizeTileC>(pBias);
			C01b = aie::add(C01b, C01.template to_vector<float>(shift));
			aie::store_v(pC1, C01b); pC1 += sizeTileC; pBias +=sizeTileC;


		}
	}

}


void MMUL_T_3::mmul3_top(input_window_float* in, output_window_float* out)
{
	int shift = 9;
	set_sat();
	set_rnd(rnd_sym_inf);


	mmul3(ROW_A3_TILE >> 2, COL_A3_TILE >> 1, COL_B3_TILE >> 2, (float *) in -> ptr, (float *) out -> ptr, 0, shift);

}
