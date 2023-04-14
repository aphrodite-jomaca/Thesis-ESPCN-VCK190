#ifndef __SUBSYS_H__
#define __SUBSYS_H__

#include <adf.h>
#include "mmul_core.h"
#include "weights_lut.h"
#include "bias_lut.h"

using namespace adf;

template <int COL_OFFSET, int ROW_OFFSET>
class tiledMmulSysGraph : public adf::graph {
private:
  kernel layer1[NUM_TILES1];//16
  kernel layer2[NUM_TILES2];//128
  kernel layer3[NUM_TILES3];//16

public:
  input_port in1[NUM_TILES1/4];
  input_port in2[NUM_TILES2/4];
  input_port in3[NUM_TILES3];

  output_port out1[NUM_TILES1];
  output_port out2[NUM_TILES2];
  output_port out3[NUM_TILES3];

  tiledMmulSysGraph() {

	  /*
	   * LAYER 1
	   */

	  for (unsigned int j=0; j<NUM_TILES1; j++)
	  {
		  if (j < 4 ){
			  layer1[j] = kernel::create_object<MMUL_T_1>(W1_1, zeros1, bias1_1);
		  	  single_buffer(in1[j]);
//		  	  location<kernel>(layer1[j]) = tile(COL_OFFSET,ROW_OFFSET+(2*j));
		  	  location<kernel>(layer1[j]) = tile(COL_OFFSET,ROW_OFFSET+(2*j));
		  }
		  else if (j < 8){
			  layer1[j] = kernel::create_object<MMUL_T_1>(W1_2, zeros1, bias1_2);
//			  location<kernel>(layer1[j]) = tile(COL_OFFSET,ROW_OFFSET+(2*(j&3))+1);
			  location<kernel>(layer1[j]) = tile(COL_OFFSET+1,ROW_OFFSET+(2*(j&3))+1);
		  }
		  else if (j < 12){
			  layer1[j] = kernel::create_object<MMUL_T_1>(W1_3, zeros1, bias1_3);
//			  location<kernel>(layer1[j]) = tile(COL_OFFSET+1,ROW_OFFSET+(2*(j&3)));
			  location<kernel>(layer1[j]) = tile(COL_OFFSET+2,ROW_OFFSET+(2*(j&3)));

		  }
		  else{
			  layer1[j] = kernel::create_object<MMUL_T_1>(W1_4, zeros1, bias1_4);
//			  location<kernel>(layer1[j]) = tile(COL_OFFSET+1,ROW_OFFSET+(2*(j&3))+1);
			  location<kernel>(layer1[j]) = tile(COL_OFFSET+3,ROW_OFFSET+(2*(j&3))+1);
		  }
		source(layer1[j]) = "mmul_core1.cc";
		runtime<ratio>(layer1[j]) = 0.8;
		location<buffer>(layer1[j].out[0]) = location<kernel>(layer1[j]);
		location<buffer>(layer1[j].in[0]) = location<kernel>(layer1[j]);
		location<parameter>(layer1[j].param[0]) = location<kernel>(layer1[j]);
		location<parameter>(layer1[j].param[1]) = location<kernel>(layer1[j]);
		location<parameter>(layer1[j].param[2]) = location<kernel>(layer1[j]);


		connect< window <WIN_SIZE_IN1> >(in1[j&3], layer1[j].in[0]);
		connect< window <WIN_SIZE_OUT1> >(layer1[j].out[0], out1[j]);

		single_buffer(layer1[j].in[0]);
		single_buffer(layer1[j].out[0]);
		single_buffer(out1[j]);
	  }

	  /*
	   * LAYER 2
	   */

	  for (unsigned int j=0; j<NUM_TILES2; j++)
	  {
		  if (j < 32){
			  layer2[j] = kernel::create_object<MMUL_T_2>(W2_1, bias2_1);
		  	  single_buffer(in2[j]);
		  }
		  else if (j < 64){
			  layer2[j] = kernel::create_object<MMUL_T_2>(W2_2, bias2_2);
		  }
		  else if (j < 96){
			  layer2[j] = kernel::create_object<MMUL_T_2>(W2_3, bias2_3);
		  }
		  else{
			  layer2[j] = kernel::create_object<MMUL_T_2>(W2_4, bias2_4);
		  }
		source(layer2[j]) = "mmul_core2.cc";
		runtime<ratio>(layer2[j]) = 0.8;


		connect< window <WIN_SIZE_IN2> >(in2[j&31], layer2[j].in[0]);
		connect< window <WIN_SIZE_OUT2> >(layer2[j].out[0], out2[j]);

		single_buffer(layer2[j].in[0]);
		single_buffer(layer2[j].out[0]);
		single_buffer(out2[j]);
	  }

	  for (unsigned int i=0; i<32; i++){ //32
		  for (unsigned int j=0; j<4; j++){ //4
			if(i%2 ==0){
			  location<kernel>(layer2[j+i*4]) = tile(COL_OFFSET+4+i,ROW_OFFSET+1+(2*j));
			}
			else{
			  location<kernel>(layer2[j+i*4]) = tile(COL_OFFSET+4+i,ROW_OFFSET+(2*j));
			}
			location<buffer>(layer2[j+i*4].out[0]) = location<kernel>(layer2[j+i*4]);
			location<buffer>(layer2[j+i*4].in[0]) = location<kernel>(layer2[j+i*4]);
			location<parameter>(layer2[j+i*4].param[1]) = location<kernel>(layer2[j+i*4]);
		  }
	  }

	  /*
	   * LAYER 3
	   */

	  for (unsigned int j=0; j<NUM_TILES3; j++)
	  {
		  if (j < 4){
			  layer3[j] = kernel::create_object<MMUL_T_3>(W3, bias3);
//			  location<kernel>(layer3[j]) = tile(COL_OFFSET+36,ROW_OFFSET+(2*j));
		  	  location<kernel>(layer3[j]) = tile(COL_OFFSET+36,ROW_OFFSET+1+(2*j));
		  }
		  else if (j < 8){
			  layer3[j] = kernel::create_object<MMUL_T_3>(W3, bias3);
//			  location<kernel>(layer3[j]) = tile(COL_OFFSET+36,ROW_OFFSET+(2*(j&3))+1);
			  location<kernel>(layer3[j]) = tile(COL_OFFSET+37,ROW_OFFSET+(2*(j&3)));
		  }
		  else if (j < 12){
			  layer3[j] = kernel::create_object<MMUL_T_3>(W3, bias3);
//			  location<kernel>(layer3[j]) = tile(COL_OFFSET+37,ROW_OFFSET+(2*(j&3)));
			  location<kernel>(layer3[j]) = tile(COL_OFFSET+38,ROW_OFFSET+(2*(j&3))+1);
		  }
		  else{
			  layer3[j] = kernel::create_object<MMUL_T_3>(W3, bias3);
//			  location<kernel>(layer3[j]) = tile(COL_OFFSET+37,ROW_OFFSET+(2*(j&3))+1);
			  location<kernel>(layer3[j]) = tile(COL_OFFSET+39,ROW_OFFSET+(2*(j&3)));
		  }
		source(layer3[j]) = "mmul_core3.cc";
		runtime<ratio>(layer3[j]) = 0.8;
		location<buffer>(layer3[j].out[0]) = location<kernel>(layer3[j]);
		location<buffer>(layer3[j].in[0]) = location<kernel>(layer3[j]);
		location<parameter>(layer3[j].param[0]) = location<kernel>(layer3[j]);
		location<parameter>(layer3[j].param[1]) = location<kernel>(layer3[j]);

		connect< window <WIN_SIZE_IN3> >(in3[j], layer3[j].in[0]);
		connect< window <WIN_SIZE_OUT3> >(layer3[j].out[0], out3[j]);

		single_buffer(layer3[j].in[0]);
		single_buffer(layer3[j].out[0]);
		single_buffer(in3[j]);
		single_buffer(out3[j]);
	  }


	  location<graph>(*this) = bounding_box(COL_OFFSET,ROW_OFFSET,COL_OFFSET+39,ROW_OFFSET+7); //+NUM_TILES2+2

  }

};

#endif //__SUBSYS_H__
