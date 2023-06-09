/********************************************* Disclaimer *********************************************/
/* This file is generated by aiecompiler 2022.1. */
/* Changes to this file may cause incorrect behavior and will be lost if aiecompiler is invoked again.*/

#include <iostream>
#include "adf/adf_api/AIEControlConfig.h"
#include "/home/atzomaka/VitisProjects/espcnv2/espcnv2/src/mmul_core.h"


/************************** Graph Configurations  *****************************/

  adf::GraphConfig GraphConfigurations[] = {
  //{id, name, graphLoadElfFunc, graphInitFunc, graphDebugHalt, coreColumns, coreRows, iterMemColumns, iterMemRows, iterMemAddrs, triggered, plKernelInstanceNames, plAxiLiteModes, plDriverStartFuncs, plDriverCheckIPDoneFuncs}
    {0, "mmul_graph", nullptr, nullptr, nullptr, {6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 31, 32, 32, 32, 32, 33, 33, 33, 33, 34, 34, 34, 34, 35, 35, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37, 38, 38, 38, 38, 39, 39, 39, 39, 40, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 42, 43, 43, 43, 43, 44, 44, 44, 44, 45, 45, 45, 45}, {0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6}, {6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 10, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 31, 32, 32, 32, 32, 33, 33, 33, 33, 34, 34, 34, 34, 35, 35, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37, 38, 38, 38, 38, 39, 39, 39, 39, 40, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 43, 43, 43, 43, 43, 44, 44, 44, 44, 45, 45, 45, 45}, {0, 2, 5, 7, 0, 2, 4, 6, 0, 1, 3, 7, 0, 3, 4, 7, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 2, 4, 7, 0, 2, 4, 6, 1, 2, 4, 6, 1, 3, 3, 6}, {0x2804, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x4, 0x6004, 0x4, 0x4, 0x3c4, 0x5a64, 0x3c4, 0x5a64, 0x7964, 0x4804, 0x3c4, 0x3c4, 0x7964, 0x3c4, 0x3c4, 0x7a64, 0x3c4, 0x4804, 0x3c4, 0x3c4, 0x3c4, 0x3c4, 0x3c4, 0x4804, 0x3c4, 0x7a64, 0x3c4, 0x3c4, 0x7964, 0x3c4, 0x7964, 0x4804, 0x3c4, 0x5a64, 0x3c4, 0x3c4, 0x3c4, 0x7964, 0x7964, 0x204, 0x5a64, 0x204, 0x3c4, 0x3c4, 0x3c4, 0x3c4, 0x3c4, 0x5a64, 0x104, 0x204, 0x3c4, 0x3c4, 0x5a64, 0x3c4, 0x3c4, 0x5a64, 0x204, 0x204, 0x3c4, 0x3c4, 0x5a64, 0x3c4, 0x3c4, 0x4804, 0x5a64, 0x204, 0x3c4, 0x3c4, 0x204, 0x3c4, 0x3c4, 0x4804, 0x3c4, 0x1964, 0x3c4, 0x4804, 0x204, 0x3c4, 0x3c4, 0x5a64, 0x3c4, 0x4804, 0x3c4, 0x104, 0x1964, 0x3c4, 0x3c4, 0x4804, 0x3c4, 0x5a64, 0x3c4, 0x5a64, 0x204, 0x3c4, 0x3c4, 0x5a64, 0x3c4, 0x5a64, 0x3c4, 0x204, 0x204, 0x3c4, 0x3c4, 0x4804, 0x3c4, 0x3c4, 0x3c4, 0x104, 0x5a64, 0x3c4, 0x3c4, 0x5a64, 0x3c4, 0x3c4, 0x3c4, 0x204, 0x5a64, 0x3c4, 0x3c4, 0x5a64, 0x3c4, 0x3c4, 0x3c4, 0x5a64, 0x204, 0x3c4, 0x3c4, 0x5a64, 0x3c4, 0x3c4, 0x3c4, 0x4804, 0x1a64, 0x4, 0x4, 0x4, 0x5a64, 0x5a64, 0x1a64, 0x1a64, 0x1a64, 0x4, 0x4, 0x4, 0x4, 0x4, 0x2004, 0x5a64}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {}, {}, {}, {},  }, 
  };
  const int NUM_GRAPH = 1;

/************************** PLIO Configurations  *****************************/

  adf::PLIOConfig PLIOConfigurations[] = {
  //{id, name, loginal_name, shim_column, slaveOrMaster, streamId}
    {0, "mmul_graph.in1[0]", "DataInL1_0", 7, 0, 4},
    {1, "mmul_graph.out1[0]", "DataOutL1_0", 6, 1, 0},
    {2, "mmul_graph.in1[1]", "DataInL1_1", 8, 0, 0},
    {3, "mmul_graph.out1[1]", "DataOutL1_1", 6, 1, 2},
    {4, "mmul_graph.in1[2]", "DataInL1_2", 7, 0, 0},
    {5, "mmul_graph.out1[2]", "DataOutL1_2", 6, 1, 4},
    {6, "mmul_graph.in1[3]", "DataInL1_3", 8, 0, 4},
    {7, "mmul_graph.out1[3]", "DataOutL1_3", 7, 1, 0},
    {8, "mmul_graph.out1[4]", "DataOutL1_4", 8, 1, 0},
    {9, "mmul_graph.out1[5]", "DataOutL1_5", 7, 1, 2},
    {10, "mmul_graph.out1[6]", "DataOutL1_6", 7, 1, 4},
    {11, "mmul_graph.out1[7]", "DataOutL1_7", 8, 1, 2},
    {12, "mmul_graph.out1[8]", "DataOutL1_8", 9, 1, 4},
    {13, "mmul_graph.out1[9]", "DataOutL1_9", 8, 1, 4},
    {14, "mmul_graph.out1[10]", "DataOutL1_10", 9, 1, 0},
    {15, "mmul_graph.out1[11]", "DataOutL1_11", 11, 1, 2},
    {16, "mmul_graph.out1[12]", "DataOutL1_12", 12, 1, 2},
    {17, "mmul_graph.out1[13]", "DataOutL1_13", 13, 1, 2},
    {18, "mmul_graph.out1[14]", "DataOutL1_14", 9, 1, 2},
    {19, "mmul_graph.out1[15]", "DataOutL1_15", 10, 1, 2},
    {20, "mmul_graph.in2[0]", "DataInL2_0", 19, 0, 6},
    {21, "mmul_graph.out2[0]", "DataOutL2_0", 10, 1, 0},
    {22, "mmul_graph.in2[1]", "DataInL2_1", 18, 0, 1},
    {23, "mmul_graph.out2[1]", "DataOutL2_1", 10, 1, 5},
    {24, "mmul_graph.in2[2]", "DataInL2_2", 18, 0, 4},
    {25, "mmul_graph.out2[2]", "DataOutL2_2", 10, 1, 1},
    {26, "mmul_graph.in2[3]", "DataInL2_3", 25, 0, 4},
    {27, "mmul_graph.out2[3]", "DataOutL2_3", 10, 1, 4},
    {28, "mmul_graph.in2[4]", "DataInL2_4", 24, 0, 4},
    {29, "mmul_graph.out2[4]", "DataOutL2_4", 11, 1, 0},
    {30, "mmul_graph.in2[5]", "DataInL2_5", 19, 0, 1},
    {31, "mmul_graph.out2[5]", "DataOutL2_5", 11, 1, 5},
    {32, "mmul_graph.in2[6]", "DataInL2_6", 19, 0, 4},
    {33, "mmul_graph.out2[6]", "DataOutL2_6", 11, 1, 1},
    {34, "mmul_graph.in2[7]", "DataInL2_7", 20, 0, 1},
    {35, "mmul_graph.out2[7]", "DataOutL2_7", 11, 1, 4},
    {36, "mmul_graph.in2[8]", "DataInL2_8", 20, 0, 4},
    {37, "mmul_graph.out2[8]", "DataOutL2_8", 12, 1, 1},
    {38, "mmul_graph.in2[9]", "DataInL2_9", 27, 0, 0},
    {39, "mmul_graph.out2[9]", "DataOutL2_9", 12, 1, 4},
    {40, "mmul_graph.in2[10]", "DataInL2_10", 24, 0, 0},
    {41, "mmul_graph.out2[10]", "DataOutL2_10", 12, 1, 0},
    {42, "mmul_graph.in2[11]", "DataInL2_11", 21, 0, 1},
    {43, "mmul_graph.out2[11]", "DataOutL2_11", 12, 1, 5},
    {44, "mmul_graph.in2[12]", "DataInL2_12", 25, 0, 1},
    {45, "mmul_graph.out2[12]", "DataOutL2_12", 13, 1, 0},
    {46, "mmul_graph.in2[13]", "DataInL2_13", 23, 0, 0},
    {47, "mmul_graph.out2[13]", "DataOutL2_13", 13, 1, 5},
    {48, "mmul_graph.in2[14]", "DataInL2_14", 21, 0, 4},
    {49, "mmul_graph.out2[14]", "DataOutL2_14", 13, 1, 1},
    {50, "mmul_graph.in2[15]", "DataInL2_15", 29, 0, 0},
    {51, "mmul_graph.out2[15]", "DataOutL2_15", 13, 1, 4},
    {52, "mmul_graph.in2[16]", "DataInL2_16", 28, 0, 1},
    {53, "mmul_graph.out2[16]", "DataOutL2_16", 14, 1, 0},
    {54, "mmul_graph.in2[17]", "DataInL2_17", 22, 0, 1},
    {55, "mmul_graph.out2[17]", "DataOutL2_17", 14, 1, 5},
    {56, "mmul_graph.in2[18]", "DataInL2_18", 22, 0, 4},
    {57, "mmul_graph.out2[18]", "DataOutL2_18", 14, 1, 1},
    {58, "mmul_graph.in2[19]", "DataInL2_19", 30, 0, 0},
    {59, "mmul_graph.out2[19]", "DataOutL2_19", 14, 1, 3},
    {60, "mmul_graph.in2[20]", "DataInL2_20", 31, 0, 0},
    {61, "mmul_graph.out2[20]", "DataOutL2_20", 15, 1, 0},
    {62, "mmul_graph.in2[21]", "DataInL2_21", 23, 0, 6},
    {63, "mmul_graph.out2[21]", "DataOutL2_21", 15, 1, 5},
    {64, "mmul_graph.in2[22]", "DataInL2_22", 23, 0, 1},
    {65, "mmul_graph.out2[22]", "DataOutL2_22", 15, 1, 1},
    {66, "mmul_graph.in2[23]", "DataInL2_23", 23, 0, 4},
    {67, "mmul_graph.out2[23]", "DataOutL2_23", 15, 1, 3},
    {68, "mmul_graph.in2[24]", "DataInL2_24", 32, 0, 1},
    {69, "mmul_graph.out2[24]", "DataOutL2_24", 16, 1, 0},
    {70, "mmul_graph.in2[25]", "DataInL2_25", 24, 0, 6},
    {71, "mmul_graph.out2[25]", "DataOutL2_25", 16, 1, 5},
    {72, "mmul_graph.in2[26]", "DataInL2_26", 24, 0, 1},
    {73, "mmul_graph.out2[26]", "DataOutL2_26", 16, 1, 1},
    {74, "mmul_graph.in2[27]", "DataInL2_27", 28, 0, 4},
    {75, "mmul_graph.out2[27]", "DataOutL2_27", 16, 1, 3},
    {76, "mmul_graph.in2[28]", "DataInL2_28", 32, 0, 4},
    {77, "mmul_graph.out2[28]", "DataOutL2_28", 17, 1, 0},
    {78, "mmul_graph.in2[29]", "DataInL2_29", 26, 0, 1},
    {79, "mmul_graph.out2[29]", "DataOutL2_29", 17, 1, 5},
    {80, "mmul_graph.in2[30]", "DataInL2_30", 26, 0, 4},
    {81, "mmul_graph.out2[30]", "DataOutL2_30", 17, 1, 1},
    {82, "mmul_graph.in2[31]", "DataInL2_31", 33, 0, 0},
    {83, "mmul_graph.out2[31]", "DataOutL2_31", 17, 1, 3},
    {84, "mmul_graph.out2[32]", "DataOutL2_32", 18, 1, 0},
    {85, "mmul_graph.out2[33]", "DataOutL2_33", 18, 1, 5},
    {86, "mmul_graph.out2[34]", "DataOutL2_34", 18, 1, 1},
    {87, "mmul_graph.out2[35]", "DataOutL2_35", 18, 1, 3},
    {88, "mmul_graph.out2[36]", "DataOutL2_36", 19, 1, 0},
    {89, "mmul_graph.out2[37]", "DataOutL2_37", 19, 1, 5},
    {90, "mmul_graph.out2[38]", "DataOutL2_38", 19, 1, 1},
    {91, "mmul_graph.out2[39]", "DataOutL2_39", 19, 1, 3},
    {92, "mmul_graph.out2[40]", "DataOutL2_40", 20, 1, 0},
    {93, "mmul_graph.out2[41]", "DataOutL2_41", 20, 1, 5},
    {94, "mmul_graph.out2[42]", "DataOutL2_42", 20, 1, 1},
    {95, "mmul_graph.out2[43]", "DataOutL2_43", 20, 1, 3},
    {96, "mmul_graph.out2[44]", "DataOutL2_44", 21, 1, 0},
    {97, "mmul_graph.out2[45]", "DataOutL2_45", 21, 1, 5},
    {98, "mmul_graph.out2[46]", "DataOutL2_46", 21, 1, 1},
    {99, "mmul_graph.out2[47]", "DataOutL2_47", 21, 1, 3},
    {100, "mmul_graph.out2[48]", "DataOutL2_48", 22, 1, 0},
    {101, "mmul_graph.out2[49]", "DataOutL2_49", 22, 1, 5},
    {102, "mmul_graph.out2[50]", "DataOutL2_50", 22, 1, 1},
    {103, "mmul_graph.out2[51]", "DataOutL2_51", 22, 1, 3},
    {104, "mmul_graph.out2[52]", "DataOutL2_52", 23, 1, 0},
    {105, "mmul_graph.out2[53]", "DataOutL2_53", 23, 1, 5},
    {106, "mmul_graph.out2[54]", "DataOutL2_54", 23, 1, 1},
    {107, "mmul_graph.out2[55]", "DataOutL2_55", 23, 1, 3},
    {108, "mmul_graph.out2[56]", "DataOutL2_56", 24, 1, 0},
    {109, "mmul_graph.out2[57]", "DataOutL2_57", 24, 1, 5},
    {110, "mmul_graph.out2[58]", "DataOutL2_58", 24, 1, 1},
    {111, "mmul_graph.out2[59]", "DataOutL2_59", 24, 1, 3},
    {112, "mmul_graph.out2[60]", "DataOutL2_60", 25, 1, 0},
    {113, "mmul_graph.out2[61]", "DataOutL2_61", 25, 1, 5},
    {114, "mmul_graph.out2[62]", "DataOutL2_62", 25, 1, 1},
    {115, "mmul_graph.out2[63]", "DataOutL2_63", 25, 1, 3},
    {116, "mmul_graph.out2[64]", "DataOutL2_64", 26, 1, 0},
    {117, "mmul_graph.out2[65]", "DataOutL2_65", 26, 1, 5},
    {118, "mmul_graph.out2[66]", "DataOutL2_66", 26, 1, 1},
    {119, "mmul_graph.out2[67]", "DataOutL2_67", 26, 1, 3},
    {120, "mmul_graph.out2[68]", "DataOutL2_68", 27, 1, 0},
    {121, "mmul_graph.out2[69]", "DataOutL2_69", 27, 1, 5},
    {122, "mmul_graph.out2[70]", "DataOutL2_70", 27, 1, 1},
    {123, "mmul_graph.out2[71]", "DataOutL2_71", 27, 1, 3},
    {124, "mmul_graph.out2[72]", "DataOutL2_72", 28, 1, 0},
    {125, "mmul_graph.out2[73]", "DataOutL2_73", 28, 1, 5},
    {126, "mmul_graph.out2[74]", "DataOutL2_74", 28, 1, 1},
    {127, "mmul_graph.out2[75]", "DataOutL2_75", 28, 1, 3},
    {128, "mmul_graph.out2[76]", "DataOutL2_76", 29, 1, 0},
    {129, "mmul_graph.out2[77]", "DataOutL2_77", 29, 1, 5},
    {130, "mmul_graph.out2[78]", "DataOutL2_78", 29, 1, 1},
    {131, "mmul_graph.out2[79]", "DataOutL2_79", 29, 1, 3},
    {132, "mmul_graph.out2[80]", "DataOutL2_80", 30, 1, 0},
    {133, "mmul_graph.out2[81]", "DataOutL2_81", 30, 1, 5},
    {134, "mmul_graph.out2[82]", "DataOutL2_82", 30, 1, 1},
    {135, "mmul_graph.out2[83]", "DataOutL2_83", 30, 1, 3},
    {136, "mmul_graph.out2[84]", "DataOutL2_84", 31, 1, 0},
    {137, "mmul_graph.out2[85]", "DataOutL2_85", 31, 1, 5},
    {138, "mmul_graph.out2[86]", "DataOutL2_86", 31, 1, 1},
    {139, "mmul_graph.out2[87]", "DataOutL2_87", 31, 1, 3},
    {140, "mmul_graph.out2[88]", "DataOutL2_88", 32, 1, 0},
    {141, "mmul_graph.out2[89]", "DataOutL2_89", 32, 1, 5},
    {142, "mmul_graph.out2[90]", "DataOutL2_90", 32, 1, 1},
    {143, "mmul_graph.out2[91]", "DataOutL2_91", 32, 1, 3},
    {144, "mmul_graph.out2[92]", "DataOutL2_92", 33, 1, 0},
    {145, "mmul_graph.out2[93]", "DataOutL2_93", 33, 1, 5},
    {146, "mmul_graph.out2[94]", "DataOutL2_94", 33, 1, 1},
    {147, "mmul_graph.out2[95]", "DataOutL2_95", 33, 1, 3},
    {148, "mmul_graph.out2[96]", "DataOutL2_96", 34, 1, 0},
    {149, "mmul_graph.out2[97]", "DataOutL2_97", 34, 1, 5},
    {150, "mmul_graph.out2[98]", "DataOutL2_98", 34, 1, 1},
    {151, "mmul_graph.out2[99]", "DataOutL2_99", 34, 1, 3},
    {152, "mmul_graph.out2[100]", "DataOutL2_100", 35, 1, 0},
    {153, "mmul_graph.out2[101]", "DataOutL2_101", 35, 1, 5},
    {154, "mmul_graph.out2[102]", "DataOutL2_102", 35, 1, 1},
    {155, "mmul_graph.out2[103]", "DataOutL2_103", 35, 1, 3},
    {156, "mmul_graph.out2[104]", "DataOutL2_104", 36, 1, 0},
    {157, "mmul_graph.out2[105]", "DataOutL2_105", 36, 1, 5},
    {158, "mmul_graph.out2[106]", "DataOutL2_106", 36, 1, 1},
    {159, "mmul_graph.out2[107]", "DataOutL2_107", 36, 1, 3},
    {160, "mmul_graph.out2[108]", "DataOutL2_108", 37, 1, 0},
    {161, "mmul_graph.out2[109]", "DataOutL2_109", 37, 1, 5},
    {162, "mmul_graph.out2[110]", "DataOutL2_110", 37, 1, 1},
    {163, "mmul_graph.out2[111]", "DataOutL2_111", 37, 1, 3},
    {164, "mmul_graph.out2[112]", "DataOutL2_112", 38, 1, 0},
    {165, "mmul_graph.out2[113]", "DataOutL2_113", 38, 1, 5},
    {166, "mmul_graph.out2[114]", "DataOutL2_114", 38, 1, 1},
    {167, "mmul_graph.out2[115]", "DataOutL2_115", 38, 1, 3},
    {168, "mmul_graph.out2[116]", "DataOutL2_116", 39, 1, 0},
    {169, "mmul_graph.out2[117]", "DataOutL2_117", 39, 1, 5},
    {170, "mmul_graph.out2[118]", "DataOutL2_118", 39, 1, 1},
    {171, "mmul_graph.out2[119]", "DataOutL2_119", 39, 1, 3},
    {172, "mmul_graph.out2[120]", "DataOutL2_120", 40, 1, 0},
    {173, "mmul_graph.out2[121]", "DataOutL2_121", 40, 1, 5},
    {174, "mmul_graph.out2[122]", "DataOutL2_122", 40, 1, 1},
    {175, "mmul_graph.out2[123]", "DataOutL2_123", 40, 1, 3},
    {176, "mmul_graph.out2[124]", "DataOutL2_124", 41, 1, 0},
    {177, "mmul_graph.out2[125]", "DataOutL2_125", 41, 1, 5},
    {178, "mmul_graph.out2[126]", "DataOutL2_126", 41, 1, 1},
    {179, "mmul_graph.out2[127]", "DataOutL2_127", 41, 1, 3},
    {180, "mmul_graph.in3[0]", "DataInL3_0", 42, 0, 0},
    {181, "mmul_graph.out3[0]", "DataOutL3_0", 42, 1, 0},
    {182, "mmul_graph.in3[1]", "DataInL3_1", 42, 0, 6},
    {183, "mmul_graph.out3[1]", "DataOutL3_1", 42, 1, 5},
    {184, "mmul_graph.in3[2]", "DataInL3_2", 42, 0, 1},
    {185, "mmul_graph.out3[2]", "DataOutL3_2", 42, 1, 1},
    {186, "mmul_graph.in3[3]", "DataInL3_3", 42, 0, 4},
    {187, "mmul_graph.out3[3]", "DataOutL3_3", 42, 1, 3},
    {188, "mmul_graph.in3[4]", "DataInL3_4", 43, 0, 0},
    {189, "mmul_graph.out3[4]", "DataOutL3_4", 43, 1, 2},
    {190, "mmul_graph.in3[5]", "DataInL3_5", 43, 0, 6},
    {191, "mmul_graph.out3[5]", "DataOutL3_5", 43, 1, 0},
    {192, "mmul_graph.in3[6]", "DataInL3_6", 43, 0, 1},
    {193, "mmul_graph.out3[6]", "DataOutL3_6", 43, 1, 5},
    {194, "mmul_graph.in3[7]", "DataInL3_7", 43, 0, 4},
    {195, "mmul_graph.out3[7]", "DataOutL3_7", 43, 1, 1},
    {196, "mmul_graph.in3[8]", "DataInL3_8", 44, 0, 1},
    {197, "mmul_graph.out3[8]", "DataOutL3_8", 44, 1, 3},
    {198, "mmul_graph.in3[9]", "DataInL3_9", 44, 0, 4},
    {199, "mmul_graph.out3[9]", "DataOutL3_9", 43, 1, 3},
    {200, "mmul_graph.in3[10]", "DataInL3_10", 44, 0, 5},
    {201, "mmul_graph.out3[10]", "DataOutL3_10", 44, 1, 4},
    {202, "mmul_graph.in3[11]", "DataInL3_11", 43, 0, 5},
    {203, "mmul_graph.out3[11]", "DataOutL3_11", 44, 1, 2},
    {204, "mmul_graph.in3[12]", "DataInL3_12", 44, 0, 2},
    {205, "mmul_graph.out3[12]", "DataOutL3_12", 43, 1, 4},
    {206, "mmul_graph.in3[13]", "DataInL3_13", 44, 0, 0},
    {207, "mmul_graph.out3[13]", "DataOutL3_13", 44, 1, 0},
    {208, "mmul_graph.in3[14]", "DataInL3_14", 44, 0, 6},
    {209, "mmul_graph.out3[14]", "DataOutL3_14", 44, 1, 5},
    {210, "mmul_graph.in3[15]", "DataInL3_15", 43, 0, 2},
    {211, "mmul_graph.out3[15]", "DataOutL3_15", 44, 1, 1},
  };
  const int NUM_PLIO = 212;


/************************** ADF API initializer *****************************/

  class InitializeAIEControlXRT
  {
  public:
    InitializeAIEControlXRT()
    {
      std::cout<<"Initializing ADF API..."<<std::endl;
#ifdef __EXCLUDE_PL_CONTROL__
      bool exclude_pl_control = true;
#else
      bool exclude_pl_control = false;
#endif
      adf::initializeConfigurations(nullptr, 0, 0, 0,
                                    GraphConfigurations, NUM_GRAPH,
                                    nullptr, 0,
                                    nullptr, 0,
                                    nullptr, 0,
                                    nullptr, 0,
                                    nullptr, 0,
                                    nullptr, 0,
                                    PLIOConfigurations, NUM_PLIO,
                                    nullptr, 0, 0, nullptr,
                                    false, exclude_pl_control, false, nullptr,
                                    true, 2);

    }
  } initAIEControlXRT;



#if !defined(__CDO__)

// Kernel Stub Definition
  void MMUL_T_1::mmul1_top(input_window<float> *,output_window<float> *) { /* Stub */ } 
  void MMUL_T_2::mmul2_top(input_window<float> *,output_window<float> *) { /* Stub */ } 
  void MMUL_T_3::mmul3_top(input_window<float> *,output_window<float> *) { /* Stub */ } 
#endif
