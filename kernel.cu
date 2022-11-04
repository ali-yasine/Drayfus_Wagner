#include "Util.h"
#include "csr.h"
#define MAX_THREADS 1024

void handleSingletons(unsigned int* DP, unsigned int* apsp, unsigned int* allSubsets , unsigned int numTerminals, unsigned int num_nodes, unsigned int* terminals) {

    unsigned int totalSubsetCount = (1 << numTerminals) - 1;

    for(unsigned int vertex = 0; vertex < num_nodes; ++vertex) {
        for(unsigned int subset = 0; subset < numTerminals; ++subset) {
            
            //find index of 1 in subset
            unsigned int index = 0;
            for(unsigned int i = 0; i < numTerminals; ++i){
                if (allSubsets[subset * numTerminals + i]){
                    index = i;
                    break;
                }
            }
            
            DP[vertex * totalSubsetCount + subset] = apsp[vertex * num_nodes + terminals[index]];
        }
    }
}

__global__ void DW_kernel(CsrGraph* graph, unsigned int numTerminals, unsigned int* DP, unsigned int* apsp, unsigned int* allSubsets, unsigned int numSubsets, unsigned int coarseFactor) {
    unsigned 
}


void DrayfusWagnerGPU(CsrGraph* graph_d, unsigned int numTerminals, unsigned int* terminals, unsigned int* DP, unsigned int* apsp) {

    unsigned int* allSubsets = ;

    unsigned int* DP_d, *apsp_d, *allSubsets_d, *terminals_d;
    unsigned int numSubsets = (1 << numTerminals) - 1;

    handleSingletons(DP, apsp, allSubsets, numTerminals, graph_d->num_nodes, terminals);
    //allocate memory 
    cudaMalloc((void**) &DP_d, sizeof(unsigned int) * graph.num_nodes * ((1 << numTerminals) - 1));
    cudaMalloc((void**) &apsp_d, sizeof(unsigned int) * graph.num_nodes * graph.num_nodes);
    cudaMalloc((void**) &allSubsets_d, sizeof(unsigned int) * numSubsets);
    cudaMalloc((void**) &terminals_d, sizeof(unsigned int) * numTerminals);
    //copy data to device
    cudaMemcpy(DP_d, DP, sizeof(unsigned int) * graph.num_nodes * ((1 << numTerminals) - 1), cudaMemcpyHostToDevice);
    cudaMemcpy(apsp_d, apsp, sizeof(unsigned int) * graph.num_nodes * graph.num_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(allSubsets_d, allSubsets, sizeof(unsigned int) * numSubsets, cudaMemcpyHostToDevice);
    cudaMemcpy(terminals_d, terminals, sizeof(unsigned int) * numTerminals, cudaMemcpyHostToDevice);

    //launch kernel
    unsigned int coarseFactor = (MAX_THREADS + graph.num_nodes - 1) / graph.num_nodes;
    dim3 numBlocks (graph.num_nodes, numSubsets);

    for(unsigned int k = 2; k < numTerminals; k++) {

    }

    cudaFree(DP_d);
    cudaFree(apsp_d);
    cudaFree(allSubsets_d);
    cudaFree(terminals_d);
}
