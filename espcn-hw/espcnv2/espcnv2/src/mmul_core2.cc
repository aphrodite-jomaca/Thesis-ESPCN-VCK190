#include "mmul_core.h"

MMUL_T_2::MMUL_T_2(float(&weights)[WSIZE2], float(&bias)[SIZE_OUT2])
: wgts(weights), b(bias)
{ }


void MMUL_T_2::mmul2(
        const int RowA_tile,
        const int ColA_tile,
        const int ColB_tile,
        float* B_in,
        float* C_out,
		int tile,
	int shift
) {

	//********** Matrix dimensions********/
	constexpr size_t sizeTileA = 4 * 2;
	constexpr size_t sizeTileB = 2 * 4;		//8 * 4;
	constexpr size_t sizeTileC = 4 * 4;		//4 * 4;


	//********** Mul Intrinsic********/
	using MMUL = aie::mmul<4, 2, 4, float, float>;

    unsigned int i,j,z;

	for (z=0; z<RowA_tile; z+=2) //1times
//		chess_loop_range(1,)
	{
		//********** Output vector ********/
		float * __restrict pC1 = C_out + (      z * ColB_tile +       0) * sizeTileC;
		float * __restrict pC2 = C_out + ((z + 1) * ColB_tile +       0) * sizeTileC;
		float * __restrict pBias1 = b  + (      z * ColB_tile +       0) * sizeTileC;
		float * __restrict pBias2 = b  + ((z + 1) * ColB_tile +       0) * sizeTileC;


		for (j=0; j<ColB_tile; j+=2) //1time
//			chess_loop_range(1,)
		{

			const float * __restrict pA1 = wgts + (      z * ColA_tile +       0) * sizeTileA + tile*sizeTileA*RowA_tile*ColA_tile;
            const float * __restrict pA2 = wgts + ((z + 1) * ColA_tile +       0) * sizeTileA + tile*sizeTileA*RowA_tile*ColA_tile;
          	const float * __restrict pB1 = B_in +    (      0 * ColB_tile +       j) * sizeTileB + tile*sizeTileB*ColA_tile*ColB_tile;
          	const float * __restrict pB2 = B_in +    (      0 * ColB_tile + (j + 1)) * sizeTileB + tile*sizeTileB*ColA_tile*ColB_tile;

          	aie::vector<float, sizeTileA> A0 = aie::load_v<sizeTileA>(pA1); pA1 += sizeTileA;
          	aie::vector<float, sizeTileA> A1 = aie::load_v<sizeTileA>(pA2); pA2 += sizeTileA;
          	aie::vector<float, sizeTileB> B0 = aie::load_v<sizeTileB>(pB1); pB1 += sizeTileB * ColB_tile;
      		aie::vector<float, sizeTileB> B1 = aie::load_v<sizeTileB>(pB2); pB2 += sizeTileB * ColB_tile;

          	MMUL C00; C00.mul(A0, B0);
          	MMUL C01; C01.mul(A0, B1);
          	MMUL C10; C10.mul(A1, B0);
          	MMUL C11; C11.mul(A1, B1);

          	for (i = 1; i < ColA_tile; ++i)
//          	chess_prepare_for_pipelining
//          	chess_loop_range(277,)
          	{
          	    A0 = aie::load_v<sizeTileA>(pA1); pA1 += sizeTileA;
          	    A1 = aie::load_v<sizeTileA>(pA2); pA2 += sizeTileA;
          	    B0 = aie::load_v<sizeTileB>(pB1); pB1 += sizeTileB * ColB_tile;
          	    B1 = aie::load_v<sizeTileB>(pB2); pB2 += sizeTileB * ColB_tile;


          	    C00.mac(A0, B0);
          	    C01.mac(A0, B1);
          	    C10.mac(A1, B0);
          	    C11.mac(A1, B1);
          	}

			aie::vector<float, sizeTileC> C00b = aie::load_v<sizeTileC>(pBias1);
			C00b = aie::add(C00b, C00.template to_vector<float>(shift));
			aie::store_v(pC1, C00b); pC1 += sizeTileC; pBias1 +=sizeTileC;

			aie::vector<float, sizeTileC> C01b = aie::load_v<sizeTileC>(pBias1);
			C01b = aie::add(C01b, C01.template to_vector<float>(shift));
			aie::store_v(pC1, C01b); pC1 += sizeTileC; pBias1 +=sizeTileC;

			aie::vector<float, sizeTileC> C10b = aie::load_v<sizeTileC>(pBias2);
			C10b = aie::add(C10b, C10.template to_vector<float>(shift));
			aie::store_v(pC2, C10b); pC2 += sizeTileC; pBias2 +=sizeTileC;

			aie::vector<float, sizeTileC> C11b = aie::load_v<sizeTileC>(pBias2);
			C11b = aie::add(C11b, C11.template to_vector<float>(shift));
			aie::store_v(pC2, C11b); pC2 += sizeTileC; pBias2 +=sizeTileC;


		}
	}

}


void MMUL_T_2::mmul2_top(input_window_float* in, output_window_float* out)
{
	int shift = 9;
	set_sat();
	set_rnd(rnd_sym_inf);

	mmul2(ROW_A2_TILE >> 2, COL_A2_TILE >> 1, COL_B2_TILE >> 2, (float *) in -> ptr, (float *) out -> ptr, 0, shift); //,tile,

}
