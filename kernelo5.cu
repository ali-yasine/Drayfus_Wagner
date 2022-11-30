#include "stdio.h"
#include "common.h"
#include "subsets.h"
#include "timer.h"
#include <stdlib.h>
#include <algorithm>
#define MAX_THREADS 1024
#define MAX_SHARED_MEM 12288 
#define MAX_TERMINALS 25

static bool compare(unsigned int a, unsigned int b){
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

static __device__ unsigned int getSubsetIndexDecimal(unsigned int subset, unsigned int* allSubsets) {
    for(unsigned int i = 0; i < (1 << subset); ++i) {
        if (allSubsets[i] == subset) {
            return i;
        }
    }
    return UINT_MAX;
}

static  unsigned int* getSortedSubsetsDecimal(unsigned int numTerminals) {
    unsigned int totalSubsetCount = (1 << numTerminals) - 1;
    unsigned int* result = (unsigned int*) malloc(sizeof(unsigned int) * totalSubsetCount);

    for(unsigned int i = 1; i < totalSubsetCount; ++i) {
        result[i] = i;
    }
    std::sort(result, result + totalSubsetCount, compare);
    return result;
}

static __device__ void generateDecimalSubsets(unsigned int subset, unsigned int* result) {
    unsigned int count = 0;

    for(unsigned int set = subset; set; set = (set - 1) & subset) {
        result[count++] = set;
    }
}

__global__ void DW_Decimal_kernel(CsrGraph* graph, unsigned int* DP, unsigned int* apsp, unsigned int* allSubsets, unsigned int numSubsets, unsigned int coarseFactor, unsigned int k,  unsigned int subsetsDoneSoFar) {
    
    extern __shared__ unsigned int subSubSets_s[];
    unsigned int root = blockIdx.x;
    unsigned int subset = allSubsets[blockIdx.y + subsetsDoneSoFar];
    unsigned int threadMin = UINT_MAX;

    unsigned int num_sub_subsets = (1 << k) - 1;

    if (threadIdx.x == 0) {
        generateDecimalSubsets(subset, subSubSets_s);
    }
    __syncthreads();

    if (root < graph->num_nodes && (blockIdx.y + subsetsDoneSoFar) * coarseFactor < numSubsets) {

        for(unsigned int sub_sub_set = threadIdx.x * coarseFactor; sub_sub_set < threadIdx.x * coarseFactor + coarseFactor ; ++sub_sub_set) {
          
            if (sub_sub_set < num_sub_subsets) {

                unsigned int subSubset = subSubSets_s[sub_sub_set];
                
                if (!(subset == subSubset)) {

                    unsigned int ss_index = getSubsetIndexDecimal(subSubset, allSubsets);
                    unsigned int sMinusSS = subset  & (~subSubset);
                    unsigned int sMinusSS_index = getSubsetIndexDecimal(sMinusSS, allSubsets);

                    for(unsigned int vertex = 0; vertex < graph->num_nodes; ++vertex) {

                        unsigned int v_to_sub_Subset = DP[vertex * numSubsets + ss_index];
                        unsigned int v_S_minusSS = DP[vertex * numSubsets + sMinusSS_index]; 
                        unsigned int root_to_v = apsp[root * graph->num_nodes + vertex]; 
                       
                        if (v_to_sub_Subset != UINT_MAX && v_S_minusSS != UINT_MAX && root_to_v != UINT_MAX) {

                            unsigned int sum = v_to_sub_Subset + v_S_minusSS + root_to_v;
                           
                            if (sum < threadMin) {
                                threadMin = sum;
                            }
                        }   
                    }
                }
            }
        }
    }
    atomicMin( &DP[root * numSubsets + blockIdx.y + subsetsDoneSoFar], threadMin);
}

void DrayfusWagnerDecimalsGPU(CsrGraph* graph_cpu, CsrGraph* graph, unsigned int numTerminals, unsigned int* terminals, unsigned int* DP, unsigned int* apsp) {
    Timer timer; cudaError_t err;

    unsigned int* allSubsets = getSortedSubsetsDecimal(numTerminals);
    unsigned int *DP_d, *apsp_d, *allSubsets_d;
    unsigned int numSubsets = (1 << numTerminals) - 1;

    for(unsigned int i = 0; i < graph_cpu->num_nodes * numSubsets; ++i)
      DP[i] = UINT_MAX;

    handleSingletons(DP, apsp, allSubsets, numTerminals, graph_cpu->num_nodes, terminals);

    startTime(&timer);


    //allocate memory 
    cudaMalloc((void**) &DP_d, sizeof(unsigned int) * graph_cpu->num_nodes * numSubsets);
    cudaMalloc((void**) &apsp_d, sizeof(unsigned int) * graph_cpu->num_nodes * graph_cpu->num_nodes);
    cudaMalloc((void**) &allSubsets_d, sizeof(unsigned int) * numSubsets);
    
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
    cudaMemcpy(allSubsets_d, allSubsets, sizeof(unsigned int) * numSubsets , cudaMemcpyHostToDevice);

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

        dim3 numBlocks (graph_cpu->num_nodes, currSubsetNum);
        
        unsigned int sharedMemPerBlock = ((1 << k) - 1) * sizeof(unsigned int);

        DW_Decimal_kernel<<<numBlocks, numThreads, sharedMemPerBlock>>> (graph, DP_d, apsp_d, allSubsets_d, numSubsets, coarseFactor, k, subsetsDoneSoFar);
        
        err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Kernel Error: %s in launch num: %u and subsetsSoFar: %u\n", cudaGetErrorString(err), k - 1, subsetsDoneSoFar);
            
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
    free(allSubsets);
    cudaFree(DP_d);
    cudaFree(apsp_d);
    cudaFree(allSubsets_d);
    cudaDeviceSynchronize();

    
}