#ifndef __COMMON_H_
#define __COMMON_H_
#include "csr.h"

CsrGraph* createEmptyCSRGraphOnGPU(unsigned int num_nodes, unsigned int num_edges);

void freeCSRGraphOnGPU(CsrGraph* graph);

void DrayfusWagnerGPU(CsrGraph* graph, CsrGraph* graph_d, unsigned int numTerminals, unsigned int* terminals, unsigned int* DP, unsigned int* apsp);

void copyCSRGraphToGPU(CsrGraph* graph, CsrGraph* graph_d);


CsrGraph* readCSRgraph(const char* filename);

#endif
