#pragma once

#include "csr.h"

CsrGraph* createEmptyCSRGraphOnGPU(unsigned int num_nodes, unsigned int num_edges);

void freeCSRGraphOnGPU(CsrGraph* graph);

void DrayfusWagnerGPU(CsrGraph* graph, unsigned int numTerminals, unsigned int* terminals, unsigned int* DP);

void copyCSRGraphToGPU(CsrGraph* graph, CsrGraph* graph_d);