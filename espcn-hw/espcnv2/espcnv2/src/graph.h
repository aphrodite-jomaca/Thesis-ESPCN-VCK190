#include <adf.h>
#include "subgraph.h"
#include "mmul_core.h"

using namespace adf;

template<int COL0>
class TEST_GRAPH: public graph{

public:
    input_plio  in1[NUM_TILES1/4];
    input_plio  in2[NUM_TILES2/4];
    input_plio  in3[NUM_TILES3];

    output_plio out1[NUM_TILES1];
    output_plio out2[NUM_TILES2];
    output_plio out3[NUM_TILES3];


    tiledMmulSysGraph<COL_OFF1, ROW_OFF1> mygraph;

    TEST_GRAPH(){

        // Layer1
        for(unsigned k=0; k<NUM_TILES1; k++) {
        	if (k<4){
        		in1[k]=input_plio::create("DataInL1_"+std::to_string(k), plio_128_bits, "data/l1_64plio_tiles/i32x256_coltile"+std::to_string(k)+"_64plio.txt");
        		connect<>(in1[k].out[0], mygraph.in1[k]);
        	}

            out1[k]=output_plio::create("DataOutL1_"+std::to_string(k), plio_128_bits, "data/l1_64plio_out/tile"+std::to_string(k)+".txt");
            connect<>(mygraph.out1[k], out1[k].in[0]);
        }

        //	Layer2
        for(unsigned k=0; k<NUM_TILES2; k++) {
        	if(k<32){
				in2[k]=input_plio::create("DataInL2_"+std::to_string(k), plio_64_bits, "data/l2_64plio_tiles/i576x256_coltile"+std::to_string(k)+"_64plio.txt");
				connect<>(in2[k].out[0], mygraph.in2[k]);
        	}
            out2[k]=output_plio::create("DataOutL2_"+std::to_string(k), plio_64_bits, "data/l2_64plio_out/tile"+std::to_string(k)+".txt");
            connect<>(mygraph.out2[k], out2[k].in[0]);
        }

        //	Layer3
        for(unsigned k=0; k<NUM_TILES3; k++) {
			in3[k]=input_plio::create("DataInL3_"+std::to_string(k), plio_64_bits, "data/l3_64plio_tiles/i288x256_coltile"+std::to_string(k)+"_64plio.txt");
			connect<>(in3[k].out[0], mygraph.in3[k]);

            out3[k]=output_plio::create("DataOutL3_"+std::to_string(k), plio_64_bits, "data/l3_64plio_out/tile"+std::to_string(k)+".txt");
            connect<>(mygraph.out3[k], out3[k].in[0]);
        }

    };

}; // end of class
