#include "pixelshuffle.h"
#include <hls_stream.h>
#include <stdint.h>


void untile3(float* input, int in_rows, int in_cols, int t_rows, int t_cols, float* output)
{
	unsigned int i, j, k, l;

	untiling_loop:
    for (i = 0; i < (in_rows/t_rows); i++){
    	for (j = 0; j < t_rows; j++){
	        for (k = 0; k < (in_cols/t_cols); k++){
	        	for (l = 0; l < t_cols; l++){
		              output[i*in_cols*t_rows+j*in_cols+k*t_cols+l] = input[i*in_cols*t_rows+j*t_cols+k*t_rows*t_cols+l];
	        	}
	        }
    	}
    }

}

void pxlshfl(float* input, float* output)
{
	for(int i = 0; i < L3_IM_H; i++){
		for(int j = 0; j < L3_IM_W; j++){
			output[2*OUT_W*i+2*j] = input[L3_IM_W*i+j];
			output[2*OUT_W*i+2*j+1] = input[L3_IM_W*i+j+L3_OUT_W];
			output[2*OUT_W*i+2*j+OUT_W] = input[L3_IM_W*i+j+2*L3_OUT_W];
			output[2*OUT_W*i+2*j+OUT_W+1] = input[L3_IM_W*i+j+3*L3_OUT_W];
		}
	}
}


void read_input3(hls::stream<float>& x0, hls::stream<float>& x1, hls::stream<float>& x2, hls::stream<float>& x3,
				hls::stream<float>& x4, hls::stream<float>& x5, hls::stream<float>& x6, hls::stream<float>& x7,
				hls::stream<float>& x8, hls::stream<float>& x9, hls::stream<float>& x10, hls::stream<float>& x11,
				hls::stream<float>& x12, hls::stream<float>& x13, hls::stream<float>& x14, hls::stream<float>& x15, float* l3_out){

	for (int i = 0; i < L3_OUT_TILE_SIZE; i++)
	{
			#pragma HLS PIPELINE II=1
			l3_out[i] = x0.read();
			l3_out[i+L3_OUT_TILE_SIZE] = x1.read();
			l3_out[i+2*L3_OUT_TILE_SIZE] = x2.read();
			l3_out[i+3*L3_OUT_TILE_SIZE] = x3.read();
			l3_out[i+4*L3_OUT_TILE_SIZE] = x4.read();
			l3_out[i+5*L3_OUT_TILE_SIZE] = x5.read();
			l3_out[i+6*L3_OUT_TILE_SIZE] = x6.read();
			l3_out[i+7*L3_OUT_TILE_SIZE] = x7.read();
			l3_out[i+8*L3_OUT_TILE_SIZE] = x8.read();
			l3_out[i+9*L3_OUT_TILE_SIZE] = x9.read();
			l3_out[i+10*L3_OUT_TILE_SIZE] = x10.read();
			l3_out[i+11*L3_OUT_TILE_SIZE] = x11.read();
			l3_out[i+12*L3_OUT_TILE_SIZE] = x12.read();
			l3_out[i+13*L3_OUT_TILE_SIZE] = x13.read();
			l3_out[i+14*L3_OUT_TILE_SIZE] = x14.read();
			l3_out[i+15*L3_OUT_TILE_SIZE] = x15.read();
	}
}

void write_output3(hls::stream<float>& y0, float* out){

	for (int i=0; i<L3_OUT_SIZE; i++)
	{
		#pragma HLS PIPELINE II=1
		y0.write(out[i]);
	}
}

extern "C" {
void pixelshuffle(hls::stream<float>& x0, hls::stream<float>& x1, hls::stream<float>& x2, hls::stream<float>& x3,
				hls::stream<float>& x4, hls::stream<float>& x5, hls::stream<float>& x6, hls::stream<float>& x7,
				hls::stream<float>& x8, hls::stream<float>& x9, hls::stream<float>& x10, hls::stream<float>& x11,
				hls::stream<float>& x12, hls::stream<float>& x13, hls::stream<float>& x14, hls::stream<float>& x15,
				hls::stream<float>& y0)
{

	#pragma HLS INTERFACE axis port = x0
    #pragma HLS INTERFACE axis port = x1
	#pragma HLS INTERFACE axis port = x2
	#pragma HLS INTERFACE axis port = x3
	#pragma HLS INTERFACE axis port = x4
	#pragma HLS INTERFACE axis port = x5
	#pragma HLS INTERFACE axis port = x6
	#pragma HLS INTERFACE axis port = x7
	#pragma HLS INTERFACE axis port = x8
	#pragma HLS INTERFACE axis port = x9
    #pragma HLS INTERFACE axis port = x10
	#pragma HLS INTERFACE axis port = x11
	#pragma HLS INTERFACE axis port = x12
	#pragma HLS INTERFACE axis port = x13
	#pragma HLS INTERFACE axis port = x14
	#pragma HLS INTERFACE axis port = x15

	#pragma HLS INTERFACE axis port = y0

    #pragma HLS INTERFACE ap_ctrl_none port=return


	float conv3_out[L3_OUT_SIZE];
	float unt1_l3[L3_OUT_SIZE];
	float unt2_l3[L3_OUT_SIZE];
	float out[L3_OUT_SIZE];


	//partition memory to create parallel read/write ports
	#pragma HLS array_partition variable=conv3_out block factor=16 dim=1
	#pragma HLS array_partition variable=out block factor=16 dim=1


	l3_read:
	read_input3(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, conv3_out);	//input4x256_t4x16rb_t4x4rb

	l3_untiling4x4:
	untile3(conv3_out, L3_OUT_H*(L3_OUT_W/L3_OUT_TILE_W), L3_OUT_TILE_W, 4, 4, unt1_l3);	// => input4x256_t4x16rb

	l3_untiling4x16:
	untile3(unt1_l3, L3_OUT_H, L3_OUT_W, L3_OUT_TILE_H, L3_OUT_TILE_W, unt2_l3);	// => input4x256

	l3_pxlshfl:
	pxlshfl(unt2_l3, out);	// => input32x256_t8x8rb


	l3_write:
	write_output3(y0, out);


}
}
