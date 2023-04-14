TARGET=vck190

# analyze DPU/CPU subgraphs
xir    graph ./build/compiled_model/ESPCN_${TARGET}.xmodel 2>&1 | tee ./build/logs/espcn_graph_info.txt
xir subgraph ./build/compiled_model/ESPCN_${TARGET}.xmodel 2>&1 | tee ./build/logs/espcn_subgraph_tree.txt
xir dump_txt ./build/compiled_model/ESPCN_${TARGET}.xmodel            ./build/logs/espcn_dump_xmodel.txt
xir png      ./build/compiled_model/ESPCN_${TARGET}.xmodel            ./build/logs/espcn_xmodel.png
xir svg      ./build/compiled_model/ESPCN_${TARGET}.xmodel            ./build/logs/espcn_xmodel.svg
