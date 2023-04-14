#include "graph.h"
#include "mmul_core.h"


TEST_GRAPH<COL_OFF1> mmul_graph;

#if defined  (__AIESIM__) || defined(__X86SIM__)
int main(void) {
	mmul_graph.init();
	mmul_graph.run(1);
	mmul_graph.end();
    return 0;
}
#endif
