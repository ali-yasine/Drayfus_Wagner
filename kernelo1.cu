#include "stdio.h"
#include "common.h"
#include "subsets.h"
#include "timer.h"
#include <stdlib.h>
#define MAX_THREADS 1024


void handleSingletons_o1(unsigned int* DP, unsigned int* apsp, unsigned int* allSubsets , unsigned int numTerminals, unsigned int num_nodes, unsigned int* terminals) {

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
            DP[subset * num_nodes + vertex] = apsp[vertex * num_nodes + terminals[index]];
        }
    }
}

__global__ void DW_kernel_o1(CsrGraph* graph, unsigned int numTerminals, unsigned int* terminals, unsigned int* DP, unsigned int* apsp, unsigned int* allSubsets, unsigned int numSubsets, unsigned int coarseFactor, unsigned int k, unsigned int subsetsDoneSoFar) {

    unsigned int root = blockIdx.x;
    unsigned int* subset = allSubsets + ((subsetsDoneSoFar + blockIdx.y ) * numTerminals);
    unsigned int num_sub_subsets = (1 << k) - 1;

    //TODO: try to do on shared memory
    unsigned int* subSubets = generateSubsetsGPU(subset, numTerminals);
    
    if (root < graph->num_nodes && (blockIdx.y + subsetsDoneSoFar < numSubsets) && threadIdx.x < num_sub_subsets) {
        for(unsigned int sub_sub_set = threadIdx.x * coarseFactor; sub_sub_set < threadIdx.x * coarseFactor + coarseFactor ; ++sub_sub_set) {
            if (sub_sub_set < num_sub_subsets) {
                unsigned int* subSubset = subSubets + (sub_sub_set * numTerminals);
                if (!equals(subset, subSubset, numTerminals)) {

                    unsigned int ss_index = getSubsetIndex(subSubset, numTerminals, allSubsets);
                    
                    unsigned int* sMinusSS = setDifferenceGPU(subset, subSubset, numTerminals);
                    unsigned int sMinusSS_index = getSubsetIndex(sMinusSS, numTerminals, allSubsets);

                    for(unsigned int vertex = 0; vertex < graph->num_nodes; ++vertex) {

                        unsigned int v_to_sub_Subset = DP[ss_index * graph->num_nodes + vertex];
                        unsigned int v_S_minusSS = DP[sMinusSS_index * graph->num_nodes + vertex]; 
                        unsigned int root_to_v = apsp[root * graph->num_nodes + vertex]; 
                        

                        if (v_to_sub_Subset != UINT_MAX && v_S_minusSS != UINT_MAX && root_to_v != UINT_MAX) {

                            unsigned int sum = v_to_sub_Subset + v_S_minusSS + root_to_v;
                            atomicMin(& DP[( blockIdx.y + subsetsDoneSoFar) * graph->num_nodes + root], sum);
                            
                        }   
                    }
                }
            }
        }
    }
    cudaFree(subSubets);
}


void DrayfusWagnerGPU_o1(CsrGraph* graph_cpu, CsrGraph* graph, unsigned int numTerminals, unsigned int* terminals, unsigned int* DP, unsigned int* apsp) {

    Timer timer;

    unsigned int* allSubsets = getSortedSubsets(numTerminals);;
    unsigned int *DP_d, *apsp_d, *allSubsets_d, *terminals_d;
    unsigned int numSubsets = (1 << numTerminals) - 1;

    for(unsigned int i = 0; i < graph_cpu->num_nodes * numSubsets; ++i)
      DP[i] = UINT_MAX;

    handleSingletons_o1(DP, apsp, allSubsets, numTerminals, graph_cpu->num_nodes, terminals);
    startTime(&timer);

    
    //allocate memory 
    cudaMalloc((void**) &DP_d, sizeof(unsigned int) * graph_cpu->num_nodes * numSubsets);
    cudaMalloc((void**) &apsp_d, sizeof(unsigned int) * graph_cpu->num_nodes * graph_cpu->num_nodes);
    cudaMalloc((void**) &allSubsets_d, sizeof(unsigned int) * numSubsets * numTerminals);
    cudaMalloc((void**) &terminals_d, sizeof(unsigned int) * numTerminals);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time: ");
    
    startTime(&timer);
    //copy data to device
    cudaMemcpy(DP_d, DP, sizeof(unsigned int) * graph_cpu->num_nodes * numSubsets, cudaMemcpyHostToDevice);
    cudaMemcpy(apsp_d, apsp, sizeof(unsigned int) * graph_cpu->num_nodes * graph_cpu->num_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(allSubsets_d, allSubsets, sizeof(unsigned int) * numSubsets * numTerminals, cudaMemcpyHostToDevice);
    cudaMemcpy(terminals_d, terminals, sizeof(unsigned int) * numTerminals, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    stopTime(&timer);

    printElapsedTime(timer, "Copy to device time: ");

    unsigned int subsetsDoneSoFar = numTerminals;

    startTime(&timer);
    //launch kernel
    for(unsigned int k = 2; k <= numTerminals; k++) {
        unsigned int numThreads = MAX_THREADS;
        unsigned int coarseFactor;
        unsigned int currSubsetNum = choose(numTerminals, k);
        if (MAX_THREADS < currSubsetNum)
            coarseFactor = (MAX_THREADS +  currSubsetNum - 1) / currSubsetNum;
        else {
            numThreads = (1 << k) - 1;
            // numThreads = currSubsetNum;
            coarseFactor = 1;
        }
        dim3 numBlocks (graph_cpu->num_nodes, currSubsetNum);
        DW_kernel_o1<<<numBlocks, numThreads>>>(graph, numTerminals, terminals_d, DP_d, apsp_d, allSubsets_d, numSubsets, coarseFactor, k, subsetsDoneSoFar);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));

        cudaDeviceSynchronize();
        
        subsetsDoneSoFar += currSubsetNum;
    }
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time: ", GREEN);

    //copy data back to host
    startTime(&timer);
    cudaMemcpy(DP, DP_d, sizeof(unsigned int) * graph_cpu->num_nodes * ((1 << numTerminals) - 1), cudaMemcpyDeviceToHost);
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