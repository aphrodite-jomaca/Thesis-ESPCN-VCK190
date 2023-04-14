#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <stdint.h>
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>

extern "C" {

void mm2s(float* mem0, hls::stream<float>& s0, hls::stream<float>& s1,
						hls::stream<float>& s2, hls::stream<float>& s3, int size, int iter)
{

#pragma HLS INTERFACE m_axi port=mem0 offset=slave bundle=gmem

#pragma HLS INTERFACE s_axilite port=mem0 bundle=control
#pragma HLS INTERFACE s_axilite port=size bundle=control
#pragma HLS INTERFACE s_axilite port=iter bundle=control
#pragma HLS interface s_axilite port=return bundle=control

	float x0,x1,x2,x3;

	for(int j = 0; j < iter; j++){
		for(int i = 0; i < size; i++) {
	#pragma HLS PIPELINE II=1
			x0 = mem0[i+j*4*size];
			x1 = mem0[i+size+j*4*size];
			x2 = mem0[i+2*size+j*4*size];
			x3 = mem0[i+3*size+j*4*size];
			s0.write(x0);
			s1.write(x1);
			s2.write(x2);
			s3.write(x3);
		}
	}

}

}
