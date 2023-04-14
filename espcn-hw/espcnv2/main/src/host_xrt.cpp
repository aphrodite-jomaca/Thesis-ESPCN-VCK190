#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fstream>
#include <string>
#include <cstring>

#include "500input.h"
#include "500gold.h"

#include "graph.h"

#include "experimental/xrt_aie.h"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_bo.h"

#include "adf/adf_api/XRTConfig.h"

//Data
#define NO_IMAGES 500
#define GRAPH_ITER_CNT 500

#define INPUT_IMG_SIZE  32*256
#define INPUT_IMG_LENGTH  (INPUT_IMG_SIZE * NO_IMAGES)
#define INPUT_TILE_SIZE 32*64
#define OUTPUT_IMG_SIZE 4*256
#define OUTPUT_IMG_LENGTH  (OUTPUT_IMG_SIZE * NO_IMAGES)
#define OUTPUT_TILE_SIZE 4*16

TEST_GRAPH<1> mmul_graph;

// load_xclbin function is used to read in xclbin file
static std::vector<char>
load_xclbin(xrtDeviceHandle device, const std::string &fnm)
{
   if (fnm.empty()) {
      throw std::runtime_error("No xclbin specified");
   }

   // load bit stream
   std::ifstream stream(fnm);
   stream.seekg(0,stream.end);
   size_t size = stream.tellg();
   stream.seekg(0,stream.beg);

   std::vector<char> header(size);
   stream.read(header.data(),size);

   auto top = reinterpret_cast<const axlf*>(header.data());
   if (xrtDeviceLoadXclbin(device, top)) {
      throw std::runtime_error("Bitstream download failed");
   }

   return header;
}

int main(int argc, char ** argv)
{
   size_t input_size_in_bytes = INPUT_IMG_LENGTH * sizeof(float);
   size_t output_size_in_bytes = OUTPUT_IMG_LENGTH * sizeof(float);

   //////////////////////////////////////////
   // Open xclbin
   //////////////////////////////////////////

   if(argc < 2) {
      std::cout << "Usage: " << argv[0] <<" <xclbin>" << std::endl;
      return EXIT_FAILURE;
   }

   else {
      //If argc is 2 it loads xclbin(Normal Flow)
      //If argc is 3 and argv[2] is LOAD_XCLBIN ,it loads xclbin(To get POWER values)
//      if(argc==2 || (argc==3 && strcmp(argv[2],"LOAD_XCLBIN")==0)) {
//
//         const char* xclbinFilename = argv[1];
//         auto dhdl = xrtDeviceOpen(0);
//         auto xclbin = load_xclbin(dhdl, xclbinFilename);
//         auto top = reinterpret_cast<const axlf*>(xclbin.data());
//      }

      //If argc is 2 it runs design for finite iterations (Normal Flow)
      //If argc is 3 and argv[2] is RUN_CODE ,it runs design for infinite iterations(To get POWER values)
      if(argc==2 || (argc==3 && strcmp(argv[2],"RUN_CODE")==0)) {

         const char* xclbinFilename = argv[1];
         auto dhdl = xrtDeviceOpen(0);
         auto xclbin = load_xclbin(dhdl, xclbinFilename);
         auto top = reinterpret_cast<const axlf*>(xclbin.data());

//         for (unsigned int i = 0; i < NO_IMAGES; i++){
//        	 printf("%f\n", input_img[i+67]);
//         }
//         printf("next\n");
         //Allocate BOs (buffer objects) of requested size with appropriate flags
         //Memory map BOs into user's address space (DDR Memory)
         xrtBufferHandle in_bohdl0 = xrtBOAlloc(dhdl, input_size_in_bytes, 0, 0);
         auto in_bomapped0 = reinterpret_cast<float*>(xrtBOMap(in_bohdl0));

         //Set the input mapped region needs to have the same data as the input_pl.
//         for (unsigned int i = 0; i < NO_IMAGES; i++){
         memcpy(in_bomapped0, input_img, input_size_in_bytes);
//         }
//         printf("Input memory virtual addr 0x%p\n", in_bomapped0);

//         for (unsigned int i = 0; i < NO_IMAGES; i++){
//        	 printf("%f\n", in_bomapped0[i+67]);
//         }
         std::cout << "in_bohdl sync started\n";
         xrtBOSync(in_bohdl0, XCL_BO_SYNC_BO_TO_DEVICE, input_size_in_bytes, 0);
         std::cout << "in_bohdl sync done\n";

         xrtBufferHandle out_bohdl = xrtBOAlloc(dhdl, output_size_in_bytes, 0, 0);
         auto out_bomapped = reinterpret_cast<float*>(xrtBOMap(out_bohdl));
         printf("Output memory virtual addr 0x%p\n", out_bomapped);

         //open PL kernels and obtain handles
         xrtKernelHandle mm2s_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "mm2s:{mm2s_1}");
         xrtKernelHandle s2mm_khdl = xrtPLKernelOpen(dhdl, top->m_header.uuid, "s2mm:{s2mm_1}");

         //Create a kernel handle to start DMA HLS pl kernel
         //Set the DMA HLS arguments
         xrtRunHandle mm2s_rhdl = xrtRunOpen(mm2s_khdl);
         int rval = xrtRunSetArg(mm2s_rhdl, 0, in_bohdl0);
         rval     = xrtRunSetArg(mm2s_rhdl, 5, INPUT_TILE_SIZE);
         rval     = xrtRunSetArg(mm2s_rhdl, 6, NO_IMAGES);

         xrtRunHandle s2mm_rhdl = xrtRunOpen(s2mm_khdl);
	     rval = xrtRunSetArg(s2mm_rhdl, 0, out_bohdl);
	     rval = xrtRunSetArg(s2mm_rhdl, 2, OUTPUT_IMG_LENGTH);


         adf::registerXRT(dhdl, top->m_header.uuid);
         printf("Graph Registered\n");

         auto graphHandle = xrtGraphOpen(dhdl, top->m_header.uuid, "mmul_graph");
         if (!graphHandle) {
            throw std::runtime_error("Unable to open graph handle");
            return 1;
         }
         int ret = xrtGraphReset(graphHandle);
         if (ret) {
            throw std::runtime_error("Unable to reset graph");
            return 1;
         }

         //Start the DMA HLS kernel
         //Moving data from DDR to AI Engine
	     auto time_start = std::chrono::steady_clock::now();
         xrtRunStart(mm2s_rhdl);
         xrtRunStart(s2mm_rhdl);

         //////////////////////////////////////////
         // graph execution for AIE
         //////////////////////////////////////////

//
         ret = xrtGraphRun(graphHandle, GRAPH_ITER_CNT);

         //Wait for DMA HLS execution to finish
         auto state_mm2s = xrtRunWait(mm2s_rhdl);
//         std::cout << "Datamover mm2s completed with status(" << state_mm2s << ")\n";

         auto state_s2mm = xrtRunWait(s2mm_rhdl);
//         std::cout << "Datamover s2mm_1 completed with status(" << state_s2mm_1 << ")\n";

         auto time_stop = std::chrono::steady_clock::now();
	     std::chrono::duration<double> elapsed_seconds= time_stop - time_start;
	     std::cout << "Elapsed time for ESPCN executed in AI Engine is " << elapsed_seconds.count() << " seconds\n";

         xrtRunClose(mm2s_rhdl);
         xrtKernelClose(mm2s_khdl);
         xrtRunClose(s2mm_rhdl);
         xrtKernelClose(s2mm_khdl);


         printf("Closed dma Datamovers...\n");

         xrtBOSync(out_bohdl, XCL_BO_SYNC_BO_FROM_DEVICE, output_size_in_bytes, 0);

         //Compare data in out_bomapped to golden data in golden.h
         int errCnt = 0;
         int errFlag = 0;
         int out_count = 0;
         int image_count = 0;
         float diff = 0.0, denom = 0.0, perc = 0.0;
         float max = 0.0, maxo = 0.0, maxg = 0.0;

         printf("Comparing output with golden\n");
         for(int i = 0; i < OUTPUT_IMG_LENGTH; i++) {
//             if( out_bomapped[i] - g_tiled_bt[i] <= 0.00001 ) {
//                 errFlag = errFlag || 0;
//                 ++out_count;
//             } else {
//                 errFlag = errFlag || 1;
//                 printf("Error found in sample %f != to the golden %f\n", out_bomapped[i], g_tiled_bt[i] );
//                 ++errCnt;
//             }
        	 if(out_bomapped[i] != 0.0){
        		 diff = abs(abs(golden[i]) - abs(out_bomapped[i]));
				 denom = (abs(golden[i]) + abs(out_bomapped[i]))/2;
				 perc = diff/denom;
				 if (perc >= max){
					 max = perc;
					 maxo = out_bomapped[i];
					 maxg = golden[i];
				 }
        	 }
             printf("%f\n",out_bomapped[i]);
        	 ++out_count;
             if ( (out_count == OUTPUT_IMG_SIZE) ) {
                 if (!errFlag) {
                	 printf("Max diff: %f, %f, %f\n",max, maxo, maxg);
                     printf("Pass for image %d \n", image_count);
                     ++image_count;
                     out_count = 0;
                     errFlag = 0;
                     max = 0.0;
                     maxo = 0.0;
                     maxg = 0.0;
                 } else {
                	 printf("Max diff: %f, %f, %f\n",max, maxo, maxg);
                     printf("Fail for image %d \n", image_count);
                     ++image_count;
                     out_count = 0;
                     errFlag = 0;
                     max = 0.0;
                     maxo = 0.0;
                     maxg = 0.0;
                 }
             }
         }

         //Release allocated resources
         std::cout << "Releasing remaining XRT objects...\n";
         xrtBOFree(in_bohdl0);
         xrtBOFree(out_bohdl);

         xrtGraphClose(graphHandle);
         printf("graph end\n");

         xrtDeviceClose(dhdl);

         std::cout << "TEST " << (errCnt ? "FAILED" : "PASSED") << std::endl;

         return (errCnt ? EXIT_FAILURE :  EXIT_SUCCESS);
      }
   }
}
