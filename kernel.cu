//#include "Util.nqdsh"
//#include "csr.h"
#include "stdio.h"
#include "common.h"
#include "subsets.h"
#include <stdlib.h>
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

__global__ void DW_kernel(CsrGraph* graph, unsigned int numTerminals, unsigned int* terminals, unsigned int* DP, unsigned int* apsp, unsigned int* allSubsets, unsigned int numSubsets, unsigned int coarseFactor, unsigned int k, unsigned int subsetsDoneSoFar) {

    unsigned int root = blockIdx.x;
    unsigned int* subset = allSubsets + ((subsetsDoneSoFar + blockIdx.y ) * numTerminals);
    
    unsigned int num_sub_subsets = (1 << k) - 1;

    //TODO: try to do on shared memory
    unsigned int* subSubets = generateSubsetsGPU(subset, numTerminals);
    if (k == 3 && blockIdx.y == 0 && blockIdx.x == 0 && threadIdx.x == 0) {
        for (unsigned int i = 0; i < num_sub_subsets; ++i) {
            printf("subset %u: ",i);
            for(unsigned int j = 0; j < numTerminals; ++j) {
                printf("%d ", subSubets[i * numTerminals + j]);
            }
            printf("\n");
        }
    }
    //try removing this
    __syncthreads();    
    if (root < graph->num_nodes && (blockIdx.y + subsetsDoneSoFar < numSubsets) && threadIdx.x < num_sub_subsets) {
        for(unsigned int sub_sub_set = threadIdx.x * coarseFactor; sub_sub_set < threadIdx.x * coarseFactor + coarseFactor ; ++sub_sub_set) {
            if (sub_sub_set < num_sub_subsets) {
                unsigned int* subSubset = subSubets + (sub_sub_set * numTerminals);
                if (!equals(subset, subSubset, numTerminals)) {

                    unsigned int ss_index = getSubsetIndex(subSubset, numTerminals, allSubsets);
                    
                    unsigned int* sMinusSS = setDifferenceGPU(subset, subSubset, numTerminals);
                    unsigned int sMinusSS_index = getSubsetIndex(sMinusSS, numTerminals, allSubsets);

                    for(unsigned int vertex = 0; vertex < graph->num_nodes; ++vertex) {

                        unsigned int v_to_sub_Subset = DP[vertex * numSubsets + ss_index];
                        unsigned int v_S_minusSS = DP[vertex * numSubsets + sMinusSS_index]; 
                        unsigned int root_to_v = apsp[root * graph->num_nodes + vertex]; 
                        if (k == 3 && root == 0 && blockIdx.y == 0 ) {
                            printf("vertex: %u, threadIdx.x: %u, subset: %u, subsubset: %u ss_index: %u v_to_sub_Subset: %u v_S_minusSS: %u,sMinusss_index: %u, root_to_v: %u. sum: %u\n",vertex, threadIdx.x, blockIdx.y, sub_sub_set, ss_index, v_to_sub_Subset, v_S_minusSS, sMinusSS_index, root_to_v, v_to_sub_Subset + v_S_minusSS + root_to_v);
                        }

                        if (v_to_sub_Subset != UINT_MAX && v_S_minusSS != UINT_MAX && root_to_v != UINT_MAX) {

                            unsigned int sum = v_to_sub_Subset + v_S_minusSS + root_to_v;
                            atomicMin(& DP[root * numSubsets + blockIdx.y + subsetsDoneSoFar], sum);

                        }   
                    }
                }
            }
        }
    }
    __syncthreads();
    cudaFree(subSubets);
}


void DrayfusWagnerGPU(CsrGraph* graph_cpu, CsrGraph* graph, unsigned int numTerminals, unsigned int* terminals, unsigned int* DP, unsigned int* apsp) {

    unsigned int* allSubsets = getSortedSubsets(numTerminals);;

    unsigned int *DP_d, *apsp_d, *allSubsets_d, *terminals_d;
    //printf("Done Terminals%u\n", numTerminals);
    unsigned int numSubsets = (1 << numTerminals) - 1;
    for(unsigned int subset = 0; subset < numSubsets; ++subset) {
        printf("subset %u: ", subset);
        for(unsigned int t = 0; t < numTerminals; ++t) {
           printf("%u ", allSubsets[subset * numTerminals + t]);
        }
        printf("\n");
    }
    for(unsigned int i = 0; i < graph_cpu->num_nodes * numSubsets; ++i)
      DP[i] = UINT_MAX;

    handleSingletons(DP, apsp, allSubsets, numTerminals, graph_cpu->num_nodes, terminals);
    
    //allocate memory 
    cudaMalloc((void**) &DP_d, sizeof(unsigned int) * graph_cpu->num_nodes * numSubsets);
    cudaMalloc((void**) &apsp_d, sizeof(unsigned int) * graph_cpu->num_nodes * graph_cpu->num_nodes);
    cudaMalloc((void**) &allSubsets_d, sizeof(unsigned int) * numSubsets * numTerminals);
    cudaMalloc((void**) &terminals_d, sizeof(unsigned int) * numTerminals);
    
    cudaDeviceSynchronize();
    
    //copy data to device
    cudaMemcpy(DP_d, DP, sizeof(unsigned int) * graph_cpu->num_nodes * numSubsets, cudaMemcpyHostToDevice);
    cudaMemcpy(apsp_d, apsp, sizeof(unsigned int) * graph_cpu->num_nodes * graph_cpu->num_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(allSubsets_d, allSubsets, sizeof(unsigned int) * numSubsets * numTerminals, cudaMemcpyHostToDevice);
    cudaMemcpy(terminals_d, terminals, sizeof(unsigned int) * numTerminals, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    
    
    unsigned int subsetsDoneSoFar = numTerminals;
    //launch kernel
    for(unsigned int k = 2; k <= numTerminals; k++) {
        unsigned int numThreads = MAX_THREADS;
        unsigned int coarseFactor;
        unsigned int currSubsetNum = choose(numTerminals, k);
        if (MAX_THREADS < currSubsetNum)
            coarseFactor = (MAX_THREADS +  currSubsetNum - 1) / currSubsetNum;
        else {
            numThreads = (k == numTerminals) ?  ((1 << k) - 1) : currSubsetNum;
            // numThreads = currSubsetNum;
            coarseFactor = 1;
        }
        dim3 numBlocks (graph_cpu->num_nodes, currSubsetNum);
        printf("k: %u, numthreads: %u\n", k, numThreads);
        DW_kernel<<<numBlocks, numThreads>>>(graph, numTerminals, terminals_d, DP_d, apsp_d, allSubsets_d, numSubsets, coarseFactor, k, subsetsDoneSoFar);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error: %s\n", cudaGetErrorString(err));
        }
        subsetsDoneSoFar += currSubsetNum;
    }
    cudaDeviceSynchronize();
    //copy data back to host
    cudaMemcpy(DP, DP_d, sizeof(unsigned int) * graph_cpu->num_nodes * ((1 << numTerminals) - 1), cudaMemcpyDeviceToHost);


    cudaDeviceSynchronize();
    for(unsigned int i = 0; i < graph_cpu->num_nodes; ++i) {
      for(unsigned int j = 0; j < ((1 << numTerminals) - 1); ++j) 
        printf("%u\t", DP[i * ((1 << numTerminals) - 1) + j]);
      printf("\n");
    }
    //free memory
    cudaFree(DP_d);
    cudaFree(apsp_d);
    cudaFree(allSubsets_d);
    cudaFree(terminals_d);

    cudaDeviceSynchronize();

    free(allSubsets);
}
