#ifndef __COMMON_H_
#define __COMMON_H_
#include "csr.h"
#include "Coo.h"
CsrGraph* createEmptyCSRGraphOnGPU(unsigned int num_nodes, unsigned int num_edges);

void freeCSRGraphOnGPU(CsrGraph* graph);

void DrayfusWagnerGPU(CsrGraph* graph, CsrGraph* graph_d, unsigned int numTerminals, unsigned int* terminals, unsigned int* DP, unsigned int* apsp);

void copyCSRGraphToGPU(CsrGraph* graph, CsrGraph* graph_d);


CsrGraph* readCSRgraph(const char* filename);

void DrayfusWagnerGPU_o1(CsrGraph* graph_cpu, CsrGraph* graph, unsigned int numTerminals, unsigned int* terminals, unsigned int* DP, unsigned int* apsp);

CooGraph* readCOOGraph(const char* filename);

void printDP(unsigned int* DP,unsigned int num_nodes, unsigned int num_subsets);

void printSubsetByIndex(unsigned int* allSubsets, unsigned int index, unsigned int num_nodes);

void generateCOOGraph(unsigned int num_nodes);

#endif
