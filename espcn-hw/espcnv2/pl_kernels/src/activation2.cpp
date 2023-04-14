#include "activation.h"
#include "tanh.h"
#include <hls_stream.h>
#include <stdint.h>
#include <cmath>


float im2col_get_pixel2(float *im, int height, int width,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

void im2col2(float* data_im,
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
                data_col[col_index] = im2col_get_pixel2(data_im, height, width,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

void tile2(float* input, int in_rows, int in_cols, int t_rows, int t_cols, float* output)
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

void untile2(float* input, int in_rows, int in_cols, int t_rows, int t_cols, float* output)
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

int float2fix2(float n, int sft)
{
	return round(n * pow(2.0, sft));
}


void tanh2(float* in, float* tanh_l2)
{
	for(int i=0; i<L2_OUT_SIZE; i++)
	{
		#pragma HLS PIPELINE II=1
		if (in[i] >= 4)
			tanh_l2[i] = 1.0;
		else if (in[i] < -4)
			tanh_l2[i] = -1.0;
		else
		{
			tanh_l2[i] = tanh_lut4[(BITS_EXP/2) + float2fix2(in[i], 7)];
		}
	}
}


void read_input2(hls::stream<float>& x0, hls::stream<float>& x1, hls::stream<float>& x2, hls::stream<float>& x3, hls::stream<float>& x4, hls::stream<float>& x5, hls::stream<float>& x6, hls::stream<float>& x7, hls::stream<float>& x8, hls::stream<float>& x9, hls::stream<float>& x10, hls::stream<float>& x11, hls::stream<float>& x12, hls::stream<float>& x13, hls::stream<float>& x14, hls::stream<float>& x15, hls::stream<float>& x16, hls::stream<float>& x17, hls::stream<float>& x18, hls::stream<float>& x19, hls::stream<float>& x20, hls::stream<float>& x21, hls::stream<float>& x22, hls::stream<float>& x23, hls::stream<float>& x24, hls::stream<float>& x25, hls::stream<float>& x26, hls::stream<float>& x27, hls::stream<float>& x28, hls::stream<float>& x29, hls::stream<float>& x30, hls::stream<float>& x31,
		hls::stream<float>& x32, hls::stream<float>& x33, hls::stream<float>& x34, hls::stream<float>& x35, hls::stream<float>& x36, hls::stream<float>& x37, hls::stream<float>& x38, hls::stream<float>& x39, hls::stream<float>& x40, hls::stream<float>& x41, hls::stream<float>& x42, hls::stream<float>& x43, hls::stream<float>& x44, hls::stream<float>& x45, hls::stream<float>& x46, hls::stream<float>& x47, hls::stream<float>& x48, hls::stream<float>& x49, hls::stream<float>& x50, hls::stream<float>& x51, hls::stream<float>& x52, hls::stream<float>& x53, hls::stream<float>& x54, hls::stream<float>& x55, hls::stream<float>& x56, hls::stream<float>& x57, hls::stream<float>& x58, hls::stream<float>& x59, hls::stream<float>& x60, hls::stream<float>& x61, hls::stream<float>& x62, hls::stream<float>& x63,
		hls::stream<float>& x64, hls::stream<float>& x65, hls::stream<float>& x66, hls::stream<float>& x67, hls::stream<float>& x68, hls::stream<float>& x69, hls::stream<float>& x70, hls::stream<float>& x71, hls::stream<float>& x72, hls::stream<float>& x73, hls::stream<float>& x74, hls::stream<float>& x75, hls::stream<float>& x76, hls::stream<float>& x77, hls::stream<float>& x78, hls::stream<float>& x79, hls::stream<float>& x80, hls::stream<float>& x81, hls::stream<float>& x82, hls::stream<float>& x83, hls::stream<float>& x84, hls::stream<float>& x85, hls::stream<float>& x86, hls::stream<float>& x87, hls::stream<float>& x88, hls::stream<float>& x89, hls::stream<float>& x90, hls::stream<float>& x91, hls::stream<float>& x92, hls::stream<float>& x93, hls::stream<float>& x94, hls::stream<float>& x95,
		hls::stream<float>& x96, hls::stream<float>& x97, hls::stream<float>& x98, hls::stream<float>& x99, hls::stream<float>& x100, hls::stream<float>& x101, hls::stream<float>& x102, hls::stream<float>& x103, hls::stream<float>& x104, hls::stream<float>& x105, hls::stream<float>& x106, hls::stream<float>& x107, hls::stream<float>& x108, hls::stream<float>& x109, hls::stream<float>& x110, hls::stream<float>& x111, hls::stream<float>& x112, hls::stream<float>& x113, hls::stream<float>& x114, hls::stream<float>& x115, hls::stream<float>& x116, hls::stream<float>& x117, hls::stream<float>& x118, hls::stream<float>& x119, hls::stream<float>& x120, hls::stream<float>& x121, hls::stream<float>& x122, hls::stream<float>& x123, hls::stream<float>& x124, hls::stream<float>& x125, hls::stream<float>& x126, hls::stream<float>& x127, float* l2_out){

	for (int i = 0; i < L2_OUT_TILE_SIZE; i++)
	{
			#pragma HLS PIPELINE II=1
			l2_out[i] = x0.read();
			l2_out[i+L2_OUT_TILE_SIZE] = x1.read();
			l2_out[i+2*L2_OUT_TILE_SIZE] = x2.read();
			l2_out[i+3*L2_OUT_TILE_SIZE] = x3.read();
			l2_out[i+4*L2_OUT_TILE_SIZE] = x4.read();
			l2_out[i+5*L2_OUT_TILE_SIZE] = x5.read();
			l2_out[i+6*L2_OUT_TILE_SIZE] = x6.read();
			l2_out[i+7*L2_OUT_TILE_SIZE] = x7.read();
			l2_out[i+8*L2_OUT_TILE_SIZE] = x8.read();
			l2_out[i+9*L2_OUT_TILE_SIZE] = x9.read();
			l2_out[i+10*L2_OUT_TILE_SIZE] = x10.read();
			l2_out[i+11*L2_OUT_TILE_SIZE] = x11.read();
			l2_out[i+12*L2_OUT_TILE_SIZE] = x12.read();
			l2_out[i+13*L2_OUT_TILE_SIZE] = x13.read();
			l2_out[i+14*L2_OUT_TILE_SIZE] = x14.read();
			l2_out[i+15*L2_OUT_TILE_SIZE] = x15.read();
			l2_out[i+16*L2_OUT_TILE_SIZE] = x16.read();
			l2_out[i+17*L2_OUT_TILE_SIZE] = x17.read();
			l2_out[i+18*L2_OUT_TILE_SIZE] = x18.read();
			l2_out[i+19*L2_OUT_TILE_SIZE] = x19.read();
			l2_out[i+20*L2_OUT_TILE_SIZE] = x20.read();
			l2_out[i+21*L2_OUT_TILE_SIZE] = x21.read();
			l2_out[i+22*L2_OUT_TILE_SIZE] = x22.read();
			l2_out[i+23*L2_OUT_TILE_SIZE] = x23.read();
			l2_out[i+24*L2_OUT_TILE_SIZE] = x24.read();
			l2_out[i+25*L2_OUT_TILE_SIZE] = x25.read();
			l2_out[i+26*L2_OUT_TILE_SIZE] = x26.read();
			l2_out[i+27*L2_OUT_TILE_SIZE] = x27.read();
			l2_out[i+28*L2_OUT_TILE_SIZE] = x28.read();
			l2_out[i+29*L2_OUT_TILE_SIZE] = x29.read();
			l2_out[i+30*L2_OUT_TILE_SIZE] = x30.read();
			l2_out[i+31*L2_OUT_TILE_SIZE] = x31.read();
			l2_out[i+32*L2_OUT_TILE_SIZE] = x32.read();
			l2_out[i+33*L2_OUT_TILE_SIZE] = x33.read();
			l2_out[i+34*L2_OUT_TILE_SIZE] = x34.read();
			l2_out[i+35*L2_OUT_TILE_SIZE] = x35.read();
			l2_out[i+36*L2_OUT_TILE_SIZE] = x36.read();
			l2_out[i+37*L2_OUT_TILE_SIZE] = x37.read();
			l2_out[i+38*L2_OUT_TILE_SIZE] = x38.read();
			l2_out[i+39*L2_OUT_TILE_SIZE] = x39.read();
			l2_out[i+40*L2_OUT_TILE_SIZE] = x40.read();
			l2_out[i+41*L2_OUT_TILE_SIZE] = x41.read();
			l2_out[i+42*L2_OUT_TILE_SIZE] = x42.read();
			l2_out[i+43*L2_OUT_TILE_SIZE] = x43.read();
			l2_out[i+44*L2_OUT_TILE_SIZE] = x44.read();
			l2_out[i+45*L2_OUT_TILE_SIZE] = x45.read();
			l2_out[i+46*L2_OUT_TILE_SIZE] = x46.read();
			l2_out[i+47*L2_OUT_TILE_SIZE] = x47.read();
			l2_out[i+48*L2_OUT_TILE_SIZE] = x48.read();
			l2_out[i+49*L2_OUT_TILE_SIZE] = x49.read();
			l2_out[i+50*L2_OUT_TILE_SIZE] = x50.read();
			l2_out[i+51*L2_OUT_TILE_SIZE] = x51.read();
			l2_out[i+52*L2_OUT_TILE_SIZE] = x52.read();
			l2_out[i+53*L2_OUT_TILE_SIZE] = x53.read();
			l2_out[i+54*L2_OUT_TILE_SIZE] = x54.read();
			l2_out[i+55*L2_OUT_TILE_SIZE] = x55.read();
			l2_out[i+56*L2_OUT_TILE_SIZE] = x56.read();
			l2_out[i+57*L2_OUT_TILE_SIZE] = x57.read();
			l2_out[i+58*L2_OUT_TILE_SIZE] = x58.read();
			l2_out[i+59*L2_OUT_TILE_SIZE] = x59.read();
			l2_out[i+60*L2_OUT_TILE_SIZE] = x60.read();
			l2_out[i+61*L2_OUT_TILE_SIZE] = x61.read();
			l2_out[i+62*L2_OUT_TILE_SIZE] = x62.read();
			l2_out[i+63*L2_OUT_TILE_SIZE] = x63.read();
			l2_out[i+64*L2_OUT_TILE_SIZE] = x64.read();
			l2_out[i+65*L2_OUT_TILE_SIZE] = x65.read();
			l2_out[i+66*L2_OUT_TILE_SIZE] = x66.read();
			l2_out[i+67*L2_OUT_TILE_SIZE] = x67.read();
			l2_out[i+68*L2_OUT_TILE_SIZE] = x68.read();
			l2_out[i+69*L2_OUT_TILE_SIZE] = x69.read();
			l2_out[i+70*L2_OUT_TILE_SIZE] = x70.read();
			l2_out[i+71*L2_OUT_TILE_SIZE] = x71.read();
			l2_out[i+72*L2_OUT_TILE_SIZE] = x72.read();
			l2_out[i+73*L2_OUT_TILE_SIZE] = x73.read();
			l2_out[i+74*L2_OUT_TILE_SIZE] = x74.read();
			l2_out[i+75*L2_OUT_TILE_SIZE] = x75.read();
			l2_out[i+76*L2_OUT_TILE_SIZE] = x76.read();
			l2_out[i+77*L2_OUT_TILE_SIZE] = x77.read();
			l2_out[i+78*L2_OUT_TILE_SIZE] = x78.read();
			l2_out[i+79*L2_OUT_TILE_SIZE] = x79.read();
			l2_out[i+80*L2_OUT_TILE_SIZE] = x80.read();
			l2_out[i+81*L2_OUT_TILE_SIZE] = x81.read();
			l2_out[i+82*L2_OUT_TILE_SIZE] = x82.read();
			l2_out[i+83*L2_OUT_TILE_SIZE] = x83.read();
			l2_out[i+84*L2_OUT_TILE_SIZE] = x84.read();
			l2_out[i+85*L2_OUT_TILE_SIZE] = x85.read();
			l2_out[i+86*L2_OUT_TILE_SIZE] = x86.read();
			l2_out[i+87*L2_OUT_TILE_SIZE] = x87.read();
			l2_out[i+88*L2_OUT_TILE_SIZE] = x88.read();
			l2_out[i+89*L2_OUT_TILE_SIZE] = x89.read();
			l2_out[i+90*L2_OUT_TILE_SIZE] = x90.read();
			l2_out[i+91*L2_OUT_TILE_SIZE] = x91.read();
			l2_out[i+92*L2_OUT_TILE_SIZE] = x92.read();
			l2_out[i+93*L2_OUT_TILE_SIZE] = x93.read();
			l2_out[i+94*L2_OUT_TILE_SIZE] = x94.read();
			l2_out[i+95*L2_OUT_TILE_SIZE] = x95.read();
			l2_out[i+96*L2_OUT_TILE_SIZE] = x96.read();
			l2_out[i+97*L2_OUT_TILE_SIZE] = x97.read();
			l2_out[i+98*L2_OUT_TILE_SIZE] = x98.read();
			l2_out[i+99*L2_OUT_TILE_SIZE] = x99.read();
			l2_out[i+100*L2_OUT_TILE_SIZE] = x100.read();
			l2_out[i+101*L2_OUT_TILE_SIZE] = x101.read();
			l2_out[i+102*L2_OUT_TILE_SIZE] = x102.read();
			l2_out[i+103*L2_OUT_TILE_SIZE] = x103.read();
			l2_out[i+104*L2_OUT_TILE_SIZE] = x104.read();
			l2_out[i+105*L2_OUT_TILE_SIZE] = x105.read();
			l2_out[i+106*L2_OUT_TILE_SIZE] = x106.read();
			l2_out[i+107*L2_OUT_TILE_SIZE] = x107.read();
			l2_out[i+108*L2_OUT_TILE_SIZE] = x108.read();
			l2_out[i+109*L2_OUT_TILE_SIZE] = x109.read();
			l2_out[i+110*L2_OUT_TILE_SIZE] = x110.read();
			l2_out[i+111*L2_OUT_TILE_SIZE] = x111.read();
			l2_out[i+112*L2_OUT_TILE_SIZE] = x112.read();
			l2_out[i+113*L2_OUT_TILE_SIZE] = x113.read();
			l2_out[i+114*L2_OUT_TILE_SIZE] = x114.read();
			l2_out[i+115*L2_OUT_TILE_SIZE] = x115.read();
			l2_out[i+116*L2_OUT_TILE_SIZE] = x116.read();
			l2_out[i+117*L2_OUT_TILE_SIZE] = x117.read();
			l2_out[i+118*L2_OUT_TILE_SIZE] = x118.read();
			l2_out[i+119*L2_OUT_TILE_SIZE] = x119.read();
			l2_out[i+120*L2_OUT_TILE_SIZE] = x120.read();
			l2_out[i+121*L2_OUT_TILE_SIZE] = x121.read();
			l2_out[i+122*L2_OUT_TILE_SIZE] = x122.read();
			l2_out[i+123*L2_OUT_TILE_SIZE] = x123.read();
			l2_out[i+124*L2_OUT_TILE_SIZE] = x124.read();
			l2_out[i+125*L2_OUT_TILE_SIZE] = x125.read();
			l2_out[i+126*L2_OUT_TILE_SIZE] = x126.read();
			l2_out[i+127*L2_OUT_TILE_SIZE] = x127.read();
	}
}

void write_output2(hls::stream<float>& y0,  hls::stream<float>& y1,  hls::stream<float>& y2,  hls::stream<float>& y3, hls::stream<float>& y4,  hls::stream<float>& y5,  hls::stream<float>& y6,  hls::stream<float>& y7,
		hls::stream<float>& y8,  hls::stream<float>& y9,  hls::stream<float>& y10,  hls::stream<float>& y11, hls::stream<float>& y12,  hls::stream<float>& y13,  hls::stream<float>& y14,  hls::stream<float>& y15, float* l3_in){

	for (int i=0; i<L3_IN_TILE_SIZE; i++)
	{
		#pragma HLS PIPELINE II=1
		y0.write(l3_in[i]);
		y1.write(l3_in[i+L3_IN_TILE_SIZE]);
		y2.write(l3_in[i+2*L3_IN_TILE_SIZE]);
		y3.write(l3_in[i+3*L3_IN_TILE_SIZE]);
		y4.write(l3_in[i+4*L3_IN_TILE_SIZE]);
		y5.write(l3_in[i+5*L3_IN_TILE_SIZE]);
		y6.write(l3_in[i+6*L3_IN_TILE_SIZE]);
		y7.write(l3_in[i+7*L3_IN_TILE_SIZE]);
		y8.write(l3_in[i+8*L3_IN_TILE_SIZE]);
		y9.write(l3_in[i+9*L3_IN_TILE_SIZE]);
		y10.write(l3_in[i+10*L3_IN_TILE_SIZE]);
		y11.write(l3_in[i+11*L3_IN_TILE_SIZE]);
		y12.write(l3_in[i+12*L3_IN_TILE_SIZE]);
		y13.write(l3_in[i+13*L3_IN_TILE_SIZE]);
		y14.write(l3_in[i+14*L3_IN_TILE_SIZE]);
		y15.write(l3_in[i+15*L3_IN_TILE_SIZE]);
	}
}

extern "C" {
void activation2(hls::stream<float>& x0, hls::stream<float>& x1, hls::stream<float>& x2, hls::stream<float>& x3, hls::stream<float>& x4, hls::stream<float>& x5, hls::stream<float>& x6, hls::stream<float>& x7, hls::stream<float>& x8, hls::stream<float>& x9, hls::stream<float>& x10, hls::stream<float>& x11, hls::stream<float>& x12, hls::stream<float>& x13, hls::stream<float>& x14, hls::stream<float>& x15, hls::stream<float>& x16, hls::stream<float>& x17, hls::stream<float>& x18, hls::stream<float>& x19, hls::stream<float>& x20, hls::stream<float>& x21, hls::stream<float>& x22, hls::stream<float>& x23, hls::stream<float>& x24, hls::stream<float>& x25, hls::stream<float>& x26, hls::stream<float>& x27, hls::stream<float>& x28, hls::stream<float>& x29, hls::stream<float>& x30, hls::stream<float>& x31,
		hls::stream<float>& x32, hls::stream<float>& x33, hls::stream<float>& x34, hls::stream<float>& x35, hls::stream<float>& x36, hls::stream<float>& x37, hls::stream<float>& x38, hls::stream<float>& x39, hls::stream<float>& x40, hls::stream<float>& x41, hls::stream<float>& x42, hls::stream<float>& x43, hls::stream<float>& x44, hls::stream<float>& x45, hls::stream<float>& x46, hls::stream<float>& x47, hls::stream<float>& x48, hls::stream<float>& x49, hls::stream<float>& x50, hls::stream<float>& x51, hls::stream<float>& x52, hls::stream<float>& x53, hls::stream<float>& x54, hls::stream<float>& x55, hls::stream<float>& x56, hls::stream<float>& x57, hls::stream<float>& x58, hls::stream<float>& x59, hls::stream<float>& x60, hls::stream<float>& x61, hls::stream<float>& x62, hls::stream<float>& x63,
		hls::stream<float>& x64, hls::stream<float>& x65, hls::stream<float>& x66, hls::stream<float>& x67, hls::stream<float>& x68, hls::stream<float>& x69, hls::stream<float>& x70, hls::stream<float>& x71, hls::stream<float>& x72, hls::stream<float>& x73, hls::stream<float>& x74, hls::stream<float>& x75, hls::stream<float>& x76, hls::stream<float>& x77, hls::stream<float>& x78, hls::stream<float>& x79, hls::stream<float>& x80, hls::stream<float>& x81, hls::stream<float>& x82, hls::stream<float>& x83, hls::stream<float>& x84, hls::stream<float>& x85, hls::stream<float>& x86, hls::stream<float>& x87, hls::stream<float>& x88, hls::stream<float>& x89, hls::stream<float>& x90, hls::stream<float>& x91, hls::stream<float>& x92, hls::stream<float>& x93, hls::stream<float>& x94, hls::stream<float>& x95,
		hls::stream<float>& x96, hls::stream<float>& x97, hls::stream<float>& x98, hls::stream<float>& x99, hls::stream<float>& x100, hls::stream<float>& x101, hls::stream<float>& x102, hls::stream<float>& x103, hls::stream<float>& x104, hls::stream<float>& x105, hls::stream<float>& x106, hls::stream<float>& x107, hls::stream<float>& x108, hls::stream<float>& x109, hls::stream<float>& x110, hls::stream<float>& x111, hls::stream<float>& x112, hls::stream<float>& x113, hls::stream<float>& x114, hls::stream<float>& x115, hls::stream<float>& x116, hls::stream<float>& x117, hls::stream<float>& x118, hls::stream<float>& x119, hls::stream<float>& x120, hls::stream<float>& x121, hls::stream<float>& x122, hls::stream<float>& x123, hls::stream<float>& x124, hls::stream<float>& x125, hls::stream<float>& x126, hls::stream<float>& x127,
		hls::stream<float>& y0,  hls::stream<float>& y1,  hls::stream<float>& y2,  hls::stream<float>& y3, hls::stream<float>& y4,  hls::stream<float>& y5,  hls::stream<float>& y6,  hls::stream<float>& y7,
		hls::stream<float>& y8,  hls::stream<float>& y9,  hls::stream<float>& y10,  hls::stream<float>& y11, hls::stream<float>& y12,  hls::stream<float>& y13,  hls::stream<float>& y14,  hls::stream<float>& y15)
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
	#pragma HLS INTERFACE axis port = x16
	#pragma HLS INTERFACE axis port = x17
	#pragma HLS INTERFACE axis port = x18
	#pragma HLS INTERFACE axis port = x19
	#pragma HLS INTERFACE axis port = x20
	#pragma HLS INTERFACE axis port = x21
	#pragma HLS INTERFACE axis port = x22
	#pragma HLS INTERFACE axis port = x23
	#pragma HLS INTERFACE axis port = x24
	#pragma HLS INTERFACE axis port = x25
	#pragma HLS INTERFACE axis port = x26
	#pragma HLS INTERFACE axis port = x27
	#pragma HLS INTERFACE axis port = x28
	#pragma HLS INTERFACE axis port = x29
	#pragma HLS INTERFACE axis port = x30
	#pragma HLS INTERFACE axis port = x31
	#pragma HLS INTERFACE axis port = x32
	#pragma HLS INTERFACE axis port = x33
	#pragma HLS INTERFACE axis port = x34
	#pragma HLS INTERFACE axis port = x35
	#pragma HLS INTERFACE axis port = x36
	#pragma HLS INTERFACE axis port = x37
	#pragma HLS INTERFACE axis port = x38
	#pragma HLS INTERFACE axis port = x39
	#pragma HLS INTERFACE axis port = x40
	#pragma HLS INTERFACE axis port = x41
	#pragma HLS INTERFACE axis port = x42
	#pragma HLS INTERFACE axis port = x43
	#pragma HLS INTERFACE axis port = x44
	#pragma HLS INTERFACE axis port = x45
	#pragma HLS INTERFACE axis port = x46
	#pragma HLS INTERFACE axis port = x47
	#pragma HLS INTERFACE axis port = x48
	#pragma HLS INTERFACE axis port = x49
	#pragma HLS INTERFACE axis port = x50
	#pragma HLS INTERFACE axis port = x51
	#pragma HLS INTERFACE axis port = x52
	#pragma HLS INTERFACE axis port = x53
	#pragma HLS INTERFACE axis port = x54
	#pragma HLS INTERFACE axis port = x55
	#pragma HLS INTERFACE axis port = x56
	#pragma HLS INTERFACE axis port = x57
	#pragma HLS INTERFACE axis port = x58
	#pragma HLS INTERFACE axis port = x59
	#pragma HLS INTERFACE axis port = x60
	#pragma HLS INTERFACE axis port = x61
	#pragma HLS INTERFACE axis port = x62
	#pragma HLS INTERFACE axis port = x63
	#pragma HLS INTERFACE axis port = x64
	#pragma HLS INTERFACE axis port = x65
	#pragma HLS INTERFACE axis port = x66
	#pragma HLS INTERFACE axis port = x67
	#pragma HLS INTERFACE axis port = x68
	#pragma HLS INTERFACE axis port = x69
	#pragma HLS INTERFACE axis port = x70
	#pragma HLS INTERFACE axis port = x71
	#pragma HLS INTERFACE axis port = x72
	#pragma HLS INTERFACE axis port = x73
	#pragma HLS INTERFACE axis port = x74
	#pragma HLS INTERFACE axis port = x75
	#pragma HLS INTERFACE axis port = x76
	#pragma HLS INTERFACE axis port = x77
	#pragma HLS INTERFACE axis port = x78
	#pragma HLS INTERFACE axis port = x79
	#pragma HLS INTERFACE axis port = x80
	#pragma HLS INTERFACE axis port = x81
	#pragma HLS INTERFACE axis port = x82
	#pragma HLS INTERFACE axis port = x83
	#pragma HLS INTERFACE axis port = x84
	#pragma HLS INTERFACE axis port = x85
	#pragma HLS INTERFACE axis port = x86
	#pragma HLS INTERFACE axis port = x87
	#pragma HLS INTERFACE axis port = x88
	#pragma HLS INTERFACE axis port = x89
	#pragma HLS INTERFACE axis port = x90
	#pragma HLS INTERFACE axis port = x91
	#pragma HLS INTERFACE axis port = x92
	#pragma HLS INTERFACE axis port = x93
	#pragma HLS INTERFACE axis port = x94
	#pragma HLS INTERFACE axis port = x95
	#pragma HLS INTERFACE axis port = x96
	#pragma HLS INTERFACE axis port = x97
	#pragma HLS INTERFACE axis port = x98
	#pragma HLS INTERFACE axis port = x99
	#pragma HLS INTERFACE axis port = x100
	#pragma HLS INTERFACE axis port = x101
	#pragma HLS INTERFACE axis port = x102
	#pragma HLS INTERFACE axis port = x103
	#pragma HLS INTERFACE axis port = x104
	#pragma HLS INTERFACE axis port = x105
	#pragma HLS INTERFACE axis port = x106
	#pragma HLS INTERFACE axis port = x107
	#pragma HLS INTERFACE axis port = x108
	#pragma HLS INTERFACE axis port = x109
	#pragma HLS INTERFACE axis port = x110
	#pragma HLS INTERFACE axis port = x111
	#pragma HLS INTERFACE axis port = x112
	#pragma HLS INTERFACE axis port = x113
	#pragma HLS INTERFACE axis port = x114
	#pragma HLS INTERFACE axis port = x115
	#pragma HLS INTERFACE axis port = x116
	#pragma HLS INTERFACE axis port = x117
	#pragma HLS INTERFACE axis port = x118
	#pragma HLS INTERFACE axis port = x119
	#pragma HLS INTERFACE axis port = x120
	#pragma HLS INTERFACE axis port = x121
	#pragma HLS INTERFACE axis port = x122
	#pragma HLS INTERFACE axis port = x123
	#pragma HLS INTERFACE axis port = x124
	#pragma HLS INTERFACE axis port = x125
	#pragma HLS INTERFACE axis port = x126
	#pragma HLS INTERFACE axis port = x127

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


    #pragma HLS INTERFACE ap_ctrl_none port=return


	float conv2_out[L2_OUT_SIZE];
	float unt1_l2[L2_OUT_SIZE];
	float unt2_l2[L2_OUT_SIZE];
	float tanh_l2[L2_OUT_SIZE];
	float im2col_l3[L3_COL_SIZE];
	float im2col_l3_in[L3_COL_SIZE];
	float im2col_l3_t[L3_COL_SIZE];


	//partition memory to create parallel read/write ports
	#pragma HLS array_partition variable=conv2_out block factor=16 dim=1
	#pragma HLS array_partition variable=im2col_l3_in block factor=16 dim=1


	l2_read:
	read_input2(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31,
			x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63,
			x64, x65, x66, x67, x68, x69, x70, x71, x72, x73, x74, x75, x76, x77, x78, x79, x80, x81, x82, x83, x84, x85, x86, x87, x88, x89, x90, x91, x92, x93, x94, x95,
			x96, x97, x98, x99, x100, x101, x102, x103, x104, x105, x106, x107, x108, x109, x110, x111, x112, x113, x114, x115, x116, x117, x118, x119, x120, x121, x122, x123, x124, x125, x126, x127, conv2_out);	//input64x256_t16x64rb_t4x4rb

	l2_untiling4x4:
	untile2(conv2_out, L2_OUT_H*(L2_OUT_W/L2_OUT_TILE_W), L2_OUT_TILE_W, 4, 4, unt1_l2);	// => input32x256_t8x8rb

	l2_untiling64x16:
	untile2(unt1_l2, L2_OUT_H, L2_OUT_W, L2_OUT_TILE_H, L2_OUT_TILE_W, unt2_l2);	// => input32x256

	l1_tanh:
	tanh2(unt2_l2, tanh_l2);	//Max diff: 0.666018, Out: -0.007812, Golden: -0.003909

	l1_im2col:
	im2col2(tanh_l2, L3_IN_C, L3_IM_H, L3_IM_W, L3_K, L3_S, L3_P, im2col_l3);	// => im2col288x256

	l1_tiling576x8:
	tile2(im2col_l3, L3_COL_H, L3_COL_W, L3_COL_TILE_H, L3_COL_TILE_W, im2col_l3_t);	// => im2col288x256_t288x16rb

	l1_tiling2x4:
	tile2(im2col_l3_t, L3_COL_H*(L3_COL_W/L3_COL_TILE_W), L3_COL_TILE_W, 2, 4, im2col_l3_in);	//im2col288x256_t288x16rb_t2x4rb

	layer_1_write:
	write_output2(y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, im2col_l3_in);


}
}
