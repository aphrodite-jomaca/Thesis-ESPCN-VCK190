#include "mmul_core.h"

MMUL_T_1::MMUL_T_1(float(&weights)[WSIZE1], float(&zeros)[SIZE_OUT1], float(&bias)[SIZE_OUT1])
: wgts(weights), intrmdtRes(zeros), b(bias)
{ }


void MMUL_T_1::mmul1(
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


	for (z=0; z<RowA_tile; z+=2)
//		chess_loop_range(1,)
	{
		//********** Output vector ********/
		float * __restrict pC1 = C_out + (      z * ColB_tile +       0) * sizeTileC;
		float * __restrict pC2 = C_out + ((z + 1) * ColB_tile +       0) * sizeTileC;


		for (j=0; j<ColB_tile; j+=2)
//			chess_loop_range(1,)
		{
			const float * __restrict pA1 = wgts + (      z * ColA_tile +       0) * sizeTileA + tile*sizeTileA*RowA_tile*ColA_tile;
            const float * __restrict pA2 = wgts + ((z + 1) * ColA_tile +       0) * sizeTileA + tile*sizeTileA*RowA_tile*ColA_tile;
          	const float * __restrict pB1 = B_in + (      0 * ColB_tile +       j) * sizeTileB + tile*sizeTileB*ColA_tile*ColB_tile;
          	const float * __restrict pB2 = B_in + (      0 * ColB_tile + (j + 1)) * sizeTileB + tile*sizeTileB*ColA_tile*ColB_tile;


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
//          	chess_loop_range(7,)
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
          	if (tile == 0){
				aie::store_v(pC1, C00.template to_vector<float>(shift)); pC1 += sizeTileC;
				aie::store_v(pC1, C01.template to_vector<float>(shift)); pC1 += sizeTileC;
				aie::store_v(pC2, C10.template to_vector<float>(shift)); pC2 += sizeTileC;
				aie::store_v(pC2, C11.template to_vector<float>(shift)); pC2 += sizeTileC;
          	}
          	else{

				//LOAD
				aie::vector<float, sizeTileC> C_00 = aie::load_v<sizeTileC>(pC1);
				//ADD
				C_00 = aie::add(C_00, C00.template to_vector<float>(shift));
				//STORE & INCR pC1_4
				aie::store_v(pC1, C_00); pC1 += sizeTileC;
				aie::vector<float, sizeTileC> C_01 = aie::load_v<sizeTileC>(pC1);
				C_01 = aie::add(C_01, C01.template to_vector<float>(shift));
				aie::store_v(pC1, C_01); pC1 += sizeTileC;

				aie::vector<float, sizeTileC> C_10 = aie::load_v<sizeTileC>(pC2);
				C_10 = aie::add(C_10, C10.template to_vector<float>(shift));
				aie::store_v(pC2, C_10); pC2 += sizeTileC;
				aie::vector<float, sizeTileC> C_11 = aie::load_v<sizeTileC>(pC2);
				C_11 = aie::add(C_11, C11.template to_vector<float>(shift));
				aie::store_v(pC2, C_11); pC2 += sizeTileC;
          	}


		}
	}

}


void MMUL_T_1::mmul1_top(input_window_float* in, output_window_float* out)
{
	int shift = 9;
	set_sat();
	set_rnd(rnd_sym_inf);
	unsigned int i, times=SIZE_OUT1/32;

	for (i = 0; i < ITERS1; i++)
	{

		mmul1(ROW_A1_TILE >> 2, COL_A1_TILE >> 1, COL_B1_TILE >> 2, (float *) in -> ptr, (float *) intrmdtRes, i, shift); //,tile,

	}

	float * ptr = intrmdtRes;
	float * bias_p = b;
	float * out_ptr = (float *) out -> ptr;
	aie::vector<float, 32> res = aie::load_v<32>(ptr);
	aie::vector<float, 32> bias_v = aie::load_v<32>(bias_p);
	aie::vector<float, 32> out_v = aie::add(res,bias_v);
	aie::store_v(out_ptr, out_v);


	for (i = 1; i < times; i++)
	{
		ptr += 32;
		bias_p += 32;
		out_ptr += 32;
		res = aie::load_v<32>(ptr);
		bias_v = aie::load_v<32>(bias_p);
		out_v = aie::add(res,bias_v);
		aie::store_v(out_ptr, out_v);
	}
}
