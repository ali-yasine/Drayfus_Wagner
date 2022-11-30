#include "stdio.h"
#include "common.h"
#include "subsets.h"
#include "timer.h"
#include <stdlib.h>
#include <algorithm>
#define MAX_THREADS 1024
#define MAX_SHARED_MEM 12288 

bool compare(unsigned int a, unsigned int b){
    return __builtin_popcount(a) < __builtin_popcount(b);
}

static void handleSingletons(unsigned int* DP, unsigned int* apsp, unsigned int* allSubsets , unsigned int numTerminals, unsigned int num_nodes, unsigned int* terminals) {
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

static unsigned int* getSortedSubsetsDecimal(unsigned int numTerminals) {
    unsigned int totalSubsetCount = (1 << numTerminals) - 1;
    unsigned int* result = (unsigned int*) malloc(sizeof(unsigned int) * totalSubsetCount);

    for(unsigned int i = 1; i < totalSubsetCount; ++i) {
        result[i] = i;
    }
    std::sort(result, result + totalSubsetCount, compare);
    return result;
}



void DrayfusWagnerDecimalsGPU(CsrGraph* graph_cpu, CsrGraph* graph, unsigned int numTerminals, unsigned int* terminals, unsigned int* DP, unsigned int* apsp) {
    Timer timer; cudaError_t err;

    unsigned int* allSubsets = getSortedSubsetsDecimal(numTerminals);
    unsigned int *DP_d, *apsp_d, *allSubsets_d, *terminals_d;
    unsigned int numSubsets = (1 << numTerminals) - 1;

    for(unsigned int i = 0; i < graph_cpu->num_nodes * numSubsets; ++i)
      DP[i] = UINT_MAX;

    handleSingletons(DP, apsp, allSubsets, numTerminals, graph_cpu->num_nodes, terminals);

    startTime(&timer);


    //allocate memory 
    cudaMalloc((void**) &DP_d, sizeof(unsigned int) * graph_cpu->num_nodes * numSubsets);
    cudaMalloc((void**) &apsp_d, sizeof(unsigned int) * graph_cpu->num_nodes * graph_cpu->num_nodes);
    cudaMalloc((void**) &allSubsets_d, sizeof(unsigned int) * numSubsets);
    cudaMalloc((void**) &terminals_d, sizeof(unsigned int) * numTerminals);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Allocation Error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time: ");
    
    startTime(&timer);
    //copy data to device
    cudaMemcpy(DP_d, DP, sizeof(unsigned int) * graph_cpu->num_nodes * numSubsets, cudaMemcpyHostToDevice);
    cudaMemcpy(apsp_d, apsp, sizeof(unsigned int) * graph_cpu->num_nodes * graph_cpu->num_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(allSubsets_d, allSubsets, sizeof(unsigned int) * numSubsets * numTerminals, cudaMemcpyHostToDevice);
    cudaMemcpy(terminals_d, terminals, sizeof(unsigned int) * numTerminals, cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Copy Error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    stopTime(&timer);

    printElapsedTime(timer, "Copy to device time: ");

    unsigned int subsetsDoneSoFar = numTerminals;
    
    startTime(&timer);
    //launch kernel
    for(unsigned int k = 2; k <= numTerminals; k++) {

        unsigned int numThreads = MAX_THREADS;
        unsigned int coarseFactor = 1;
        unsigned int currSubsetNum = choose(numTerminals, k);
        unsigned int subSubetsNum = (1 << k) - 1;
        
        if (MAX_THREADS < subSubetsNum) 
            coarseFactor = (MAX_THREADS +  subSubetsNum - 1) / subSubetsNum;
        
        else 
            numThreads = (1 << k) - 1;

        unsigned int* subSubets;
        cudaMalloc((void**) &subSubets, ((1 << k) - 1) * currSubsetNum  * sizeof(unsigned int));
        
        err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("subSubets malloc Error: %s in launch num: %u and subsetsSoFar: %u\n", cudaGetErrorString(err), k - 1, subsetsDoneSoFar);
        
        cudaDeviceSynchronize();

        dim3 numBlocks (graph_cpu->num_nodes, currSubsetNum);

        DW_kernel<<<numBlocks, numThreads>>>(graph, numTerminals, terminals_d, DP_d, apsp_d, allSubsets_d, numSubsets, coarseFactor, k, subsetsDoneSoFar, subSubets);
        
        err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Kernel Error: %s in launch num: %u and subsetsSoFar: %u\n", cudaGetErrorString(err), k - 1, subsetsDoneSoFar);
            
        cudaFree(subSubets);
        cudaDeviceSynchronize();
        subsetsDoneSoFar += currSubsetNum;
    }
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time: ", GREEN);

    //copy data back to host
    startTime(&timer);
    
    cudaMemcpy(DP, DP_d, sizeof(unsigned int) * graph_cpu->num_nodes * numSubsets, cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Copy Back Error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    stopTime(&timer);

    printElapsedTime(timer, "Copy to host time: ");
    
    //free memory
    cudaFree(DP_d);
    cudaFree(apsp_d);
    cudaFree(allSubsets_d);
    cudaFree(terminals_d);
    cudaDeviceSynchronize();

    free(allSubsets);
    
}