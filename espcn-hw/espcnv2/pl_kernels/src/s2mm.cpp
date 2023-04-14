#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <stdint.h>
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h>


extern "C" {

void s2mm(float* mem, hls::stream<float>& s0, int size)
{


#pragma HLS INTERFACE m_axi port=mem offset=slave bundle=gmem


#pragma HLS INTERFACE s_axilite port=mem bundle=control
#pragma HLS INTERFACE s_axilite port=size bundle=control
#pragma HLS interface s_axilite port=return bundle=control

	float x0;

	for(int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
			x0 = s0.read();
			mem[i] = x0;

		}

}

}
