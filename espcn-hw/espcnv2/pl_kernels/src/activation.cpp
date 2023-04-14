#include "activation.h"
#include "tanh.h"
#include <hls_stream.h>
#include <stdint.h>
#include <cmath>


float im2col_get_pixel(float *im, int height, int width,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

void im2col(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col)
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    im2col_loop:
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

void tile(float* input, int in_rows, int in_cols, int t_rows, int t_cols, float* output)
{
	unsigned int i, j, k, l;

	tiling_loop:
    for (i=0; i<(in_rows/t_rows); i++){
    	for (j=0; j<(in_cols/t_cols); j++){
	        for (k=0; k<t_rows; k++){
	        	for (l=0; l<t_cols; l++){
		              output[i*in_cols*t_rows+j*t_rows*t_cols+k*t_cols+l] = input[i*in_cols*t_rows+j*t_cols+k*in_cols+l];
	        	}
	        }
    	}
    }

}

void untile(float* input, int in_rows, int in_cols, int t_rows, int t_cols, float* output)
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

int float2fix(float n, int sft)
{
	return round(n * pow(2.0, sft));
}


void tanh(float* in, float* tanh_l1)
{
	for(int i=0; i<L1_OUT_SIZE; i++)
	{
		#pragma HLS PIPELINE II=1
		if (in[i] >= 4)
			tanh_l1[i] = 1.0;
		else if (in[i] < -4)
			tanh_l1[i] = -1.0;
		else
		{
			tanh_l1[i] = tanh_lut4[(BITS_EXP/2) + float2fix(in[i], 7)];
		}
	}
}


void read_input(hls::stream<float>& x0,  hls::stream<float>& x1,  hls::stream<float>& x2,  hls::stream<float>& x3,
		hls::stream<float>& x4,  hls::stream<float>& x5,  hls::stream<float>& x6,  hls::stream<float>& x7,
		hls::stream<float>& x8,  hls::stream<float>& x9,  hls::stream<float>& x10, hls::stream<float>& x11,
		hls::stream<float>& x12, hls::stream<float>& x13, hls::stream<float>& x14, hls::stream<float>& x15, float* l1_out){

	for (int i = 0; i < L1_OUT_TILE_SIZE; i++)
	{
			#pragma HLS PIPELINE II=1
			l1_out[i] = x0.read();
			l1_out[i+L1_OUT_TILE_SIZE] = x1.read();
			l1_out[i+2*L1_OUT_TILE_SIZE] = x2.read();
			l1_out[i+3*L1_OUT_TILE_SIZE] = x3.read();
			l1_out[i+4*L1_OUT_TILE_SIZE] = x4.read();
			l1_out[i+5*L1_OUT_TILE_SIZE] = x5.read();
			l1_out[i+6*L1_OUT_TILE_SIZE] = x6.read();
			l1_out[i+7*L1_OUT_TILE_SIZE] = x7.read();
			l1_out[i+8*L1_OUT_TILE_SIZE] = x8.read();
			l1_out[i+9*L1_OUT_TILE_SIZE] = x9.read();
			l1_out[i+10*L1_OUT_TILE_SIZE] = x10.read();
			l1_out[i+11*L1_OUT_TILE_SIZE] = x11.read();
			l1_out[i+12*L1_OUT_TILE_SIZE] = x12.read();
			l1_out[i+13*L1_OUT_TILE_SIZE] = x13.read();
			l1_out[i+14*L1_OUT_TILE_SIZE] = x14.read();
			l1_out[i+15*L1_OUT_TILE_SIZE] = x15.read();
	}
}

void write_output(hls::stream<float>& y0,  hls::stream<float>& y1, hls::stream<float>& y2,  hls::stream<float>& y3,
		hls::stream<float>& y4,  hls::stream<float>& y5,  hls::stream<float>& y6,  hls::stream<float>& y7,
		hls::stream<float>& y8,  hls::stream<float>& y9,  hls::stream<float>& y10, hls::stream<float>& y11,
		hls::stream<float>& y12, hls::stream<float>& y13, hls::stream<float>& y14, hls::stream<float>& y15,
		hls::stream<float>& y16, hls::stream<float>& y17, hls::stream<float>& y18, hls::stream<float>& y19,
		hls::stream<float>& y20, hls::stream<float>& y21, hls::stream<float>& y22, hls::stream<float>& y23,
		hls::stream<float>& y24, hls::stream<float>& y25, hls::stream<float>& y26, hls::stream<float>& y27,
		hls::stream<float>& y28, hls::stream<float>& y29, hls::stream<float>& y30, hls::stream<float>& y31, float* l2_in){

	for (int i=0; i<L2_IN_TILE_SIZE; i++)
	{
		#pragma HLS PIPELINE II=1
		y0.write(l2_in[i]);
		y1.write(l2_in[i+L2_IN_TILE_SIZE]);
		y2.write(l2_in[i+2*L2_IN_TILE_SIZE]);
		y3.write(l2_in[i+3*L2_IN_TILE_SIZE]);
		y4.write(l2_in[i+4*L2_IN_TILE_SIZE]);
		y5.write(l2_in[i+5*L2_IN_TILE_SIZE]);
		y6.write(l2_in[i+6*L2_IN_TILE_SIZE]);
		y7.write(l2_in[i+7*L2_IN_TILE_SIZE]);
		y8.write(l2_in[i+8*L2_IN_TILE_SIZE]);
		y9.write(l2_in[i+9*L2_IN_TILE_SIZE]);
		y10.write(l2_in[i+10*L2_IN_TILE_SIZE]);
		y11.write(l2_in[i+11*L2_IN_TILE_SIZE]);
		y12.write(l2_in[i+12*L2_IN_TILE_SIZE]);
		y13.write(l2_in[i+13*L2_IN_TILE_SIZE]);
		y14.write(l2_in[i+14*L2_IN_TILE_SIZE]);
		y15.write(l2_in[i+15*L2_IN_TILE_SIZE]);
		y16.write(l2_in[i+16*L2_IN_TILE_SIZE]);
		y17.write(l2_in[i+17*L2_IN_TILE_SIZE]);
		y18.write(l2_in[i+18*L2_IN_TILE_SIZE]);
		y19.write(l2_in[i+19*L2_IN_TILE_SIZE]);
		y20.write(l2_in[i+20*L2_IN_TILE_SIZE]);
		y21.write(l2_in[i+21*L2_IN_TILE_SIZE]);
		y22.write(l2_in[i+22*L2_IN_TILE_SIZE]);
		y23.write(l2_in[i+23*L2_IN_TILE_SIZE]);
		y24.write(l2_in[i+24*L2_IN_TILE_SIZE]);
		y25.write(l2_in[i+25*L2_IN_TILE_SIZE]);
		y26.write(l2_in[i+26*L2_IN_TILE_SIZE]);
		y27.write(l2_in[i+27*L2_IN_TILE_SIZE]);
		y28.write(l2_in[i+28*L2_IN_TILE_SIZE]);
		y29.write(l2_in[i+29*L2_IN_TILE_SIZE]);
		y30.write(l2_in[i+30*L2_IN_TILE_SIZE]);
		y31.write(l2_in[i+31*L2_IN_TILE_SIZE]);
	}
}

extern "C" {
void activation(hls::stream<float>& x0,  hls::stream<float>& x1,  hls::stream<float>& x2,  hls::stream<float>& x3,  hls::stream<float>& x4,  hls::stream<float>& x5,  hls::stream<float>& x6,  hls::stream<float>& x7,
				hls::stream<float>& x8,  hls::stream<float>& x9,  hls::stream<float>& x10, hls::stream<float>& x11, hls::stream<float>& x12, hls::stream<float>& x13, hls::stream<float>& x14, hls::stream<float>& x15,
				hls::stream<float>& y0,  hls::stream<float>& y1,  hls::stream<float>& y2,  hls::stream<float>& y3,  hls::stream<float>& y4,  hls::stream<float>& y5,  hls::stream<float>& y6,  hls::stream<float>& y7,
				hls::stream<float>& y8,  hls::stream<float>& y9,  hls::stream<float>& y10, hls::stream<float>& y11, hls::stream<float>& y12, hls::stream<float>& y13, hls::stream<float>& y14, hls::stream<float>& y15,
				hls::stream<float>& y16, hls::stream<float>& y17, hls::stream<float>& y18, hls::stream<float>& y19, hls::stream<float>& y20, hls::stream<float>& y21, hls::stream<float>& y22, hls::stream<float>& y23,
				hls::stream<float>& y24, hls::stream<float>& y25, hls::stream<float>& y26, hls::stream<float>& y27, hls::stream<float>& y28, hls::stream<float>& y29, hls::stream<float>& y30, hls::stream<float>& y31)
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
	#pragma HLS INTERFACE axis port = y1
	#pragma HLS INTERFACE axis port = y2
	#pragma HLS INTERFACE axis port = y3
	#pragma HLS INTERFACE axis port = y4
	#pragma HLS INTERFACE axis port = y5
	#pragma HLS INTERFACE axis port = y6
	#pragma HLS INTERFACE axis port = y7
	#pragma HLS INTERFACE axis port = y8
	#pragma HLS INTERFACE axis port = y9
	#pragma HLS INTERFACE axis port = y10
	#pragma HLS INTERFACE axis port = y11
	#pragma HLS INTERFACE axis port = y12
	#pragma HLS INTERFACE axis port = y13
	#pragma HLS INTERFACE axis port = y14
	#pragma HLS INTERFACE axis port = y15
	#pragma HLS INTERFACE axis port = y16
	#pragma HLS INTERFACE axis port = y17
	#pragma HLS INTERFACE axis port = y18
	#pragma HLS INTERFACE axis port = y19
	#pragma HLS INTERFACE axis port = y20
	#pragma HLS INTERFACE axis port = y21
	#pragma HLS INTERFACE axis port = y22
	#pragma HLS INTERFACE axis port = y23
	#pragma HLS INTERFACE axis port = y24
	#pragma HLS INTERFACE axis port = y25
	#pragma HLS INTERFACE axis port = y26
	#pragma HLS INTERFACE axis port = y27
	#pragma HLS INTERFACE axis port = y28
	#pragma HLS INTERFACE axis port = y29
	#pragma HLS INTERFACE axis port = y30
	#pragma HLS INTERFACE axis port = y31

    #pragma HLS INTERFACE ap_ctrl_none port=return


	float conv1_out[L1_OUT_SIZE];
	float unt1_l1[L1_OUT_SIZE];
	float unt2_l1[L1_OUT_SIZE];
	float tanh_l1[L1_OUT_SIZE];
	float im2col_l2[L2_COL_SIZE];
	float im2col_l2_in[L2_COL_SIZE];
	float im2col_l2_t[L2_COL_SIZE];


	//partition memory to create parallel read/write ports
	#pragma HLS array_partition variable=conv1_out block factor=16 dim=1
	#pragma HLS array_partition variable=im2col_l2_in block factor=16 dim=1

	l1_read:
	read_input(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, conv1_out);	//input64x256_t64x128rb_t4x4rb

	l1_untiling4x4:
	untile(conv1_out, L1_OUT_H*(L1_OUT_W/L1_OUT_TILE_W), L1_OUT_TILE_W, 4, 4, unt1_l1);	//input64x256_t64x128rb

	l1_untiling64x16:
	untile(unt1_l1, L1_OUT_H, L1_OUT_W, L1_OUT_TILE_H, L1_OUT_TILE_W, unt2_l1);	//input64x256

	l1_tanh:
	tanh(unt2_l1, tanh_l1);	//Max diff: 0.666018, Out: -0.007812, Golden: -0.003909

	l1_im2col:
	im2col(tanh_l1, L2_IN_C, L2_IM_H, L2_IM_W, L2_K, L2_S, L2_P, im2col_l2);	//im2col576x256

	l1_tiling576x8:
	tile(im2col_l2, L2_COL_H, L2_COL_W, L2_COL_TILE_H, L2_COL_TILE_W, im2col_l2_t);	//im2col576x256_t576x8rb

	l1_tiling2x4:
	tile(im2col_l2_t, L2_COL_H*(L2_COL_W/L2_COL_TILE_W), L2_COL_TILE_W, 2, 4, im2col_l2_in);	//im2col576x256_t576x8rb_t2x4rb

	layer_1_write:
	write_output(y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15,
				y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, im2col_l2_in);


}
}
