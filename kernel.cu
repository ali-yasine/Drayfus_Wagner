#include "Util.h"
#include "csr.h"
#include "common.h"
#include "subsets.h"
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

__global__ void DW_kernel(CsrGraph* graph, unsigned int numTerminals, unsigned int* terminals, unsigned int* DP, unsigned int* apsp, unsigned int* allSubsets, unsigned int numSubsets, unsigned int coarseFactor, unsigned int k) {

    unsigned int root = blockIdx.x;
    unsigned int* subset = allSubsets + (blockIdx.y * numTerminals);
    unsigned int num_sub_subsets = (1 << k) - 1;

    //TODO: try to do on shared memory
    unsigned int* subSubets = generateSubsetsGPU(subset, num_sub_subsets);

    //try removing later
    __syncthreads();

    for(unsigned int sub_sub_set = threadIdx.x * coarseFactor; sub_sub_set < threadIdx.x * coarseFactor + coarseFactor; ++sub_sub_set) {
        if (sub_sub_set <= num_sub_subsets) {
            unsigned int* subSubset = subSubets + (sub_sub_set * numTerminals);
            if (!equals(subset, subSubset, numTerminals)) {

                unsigned int ss_index = getSubsetIndex(subSubset, numTerminals, allSubsets);
                
                unsigned int* sMinusSS = setDifferenceGPU(subset, subSubset, numTerminals);
                unsigned int sMinusSS_index = getSubsetIndex(sMinusSS, numTerminals, allSubsets);

                for(unsigned int vertex = 0; vertex < graph->num_nodes; ++vertex) {

                    unsigned int v_to_sub_Subset = DP[vertex * numSubsets + ss_index];
                    unsigned int v_S_minusSS = DP[vertex * numSubsets + sMinusSS_index]; 
                    unsigned int root_to_v = apsp[root * graph->num_nodes + vertex]; 

                    if (v_to_sub_Subset != UINT_MAX && v_S_minusSS != UINT_MAX && root_to_v != UINT_MAX) {

                        unsigned int sum = v_to_sub_Subset + v_S_minusSS + root_to_v;
                        
                        atomicMin(& DP[root * numSubsets + blockIdx.y], sum);
                    }   
                }
            }
        }
    }
    __syncthreads();
    cudaFree(subSubets);
}


void DrayfusWagnerGPU(CsrGraph* graph, unsigned int numTerminals, unsigned int* terminals, unsigned int* DP, unsigned int* apsp) {

    unsigned int* allSubsets = getSortedSubsets(numTerminals);;

    unsigned int* DP_d, *apsp_d, *allSubsets_d, *terminals_d;
    unsigned int numSubsets = (1 << numTerminals) - 1;

    handleSingletons(DP, apsp, allSubsets, numTerminals, graph->num_nodes, terminals);

    //allocate memory 
    cudaMalloc((void**) &DP_d, sizeof(unsigned int) * graph->num_nodes * numSubsets);
    cudaMalloc((void**) &apsp_d, sizeof(unsigned int) * graph->num_nodes * graph->num_nodes);
    cudaMalloc((void**) &allSubsets_d, sizeof(unsigned int) * numSubsets);
    cudaMalloc((void**) &terminals_d, sizeof(unsigned int) * numTerminals);
    
    cudaDeviceSynchronize();

    //copy data to device
    cudaMemcpy(DP_d, DP, sizeof(unsigned int) * graph->num_nodes * numSubsets, cudaMemcpyHostToDevice);
    cudaMemcpy(apsp_d, apsp, sizeof(unsigned int) * graph->num_nodes * graph->num_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(allSubsets_d, allSubsets, sizeof(unsigned int) * numSubsets, cudaMemcpyHostToDevice);
    cudaMemcpy(terminals_d, terminals, sizeof(unsigned int) * numTerminals, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    //launch kernel
    for(unsigned int k = 2; k < numTerminals; k++) {

        unsigned int currSubsetNum = choose(numTerminals, k);
        unsigned int coarseFactor = (MAX_THREADS +  currSubsetNum - 1) / currSubsetNum;
        dim3 numBlocks (graph->num_nodes, currSubsetNum);

        DW_kernel<<<numBlocks, MAX_THREADS>>>(graph, numTerminals, terminals_d, DP_d, apsp_d, allSubsets_d, numSubsets, coarseFactor, k);
        cudaDeviceSynchronize();
    }

    //copy data back to host
    cudaMemcpy(DP, DP_d, sizeof(unsigned int) * graph->num_nodes * ((1 << numTerminals) - 1), cudaMemcpyDeviceToHost);


    cudaDeviceSynchronize();

    //free memory
    cudaFree(DP_d);
    cudaFree(apsp_d);
    cudaFree(allSubsets_d);
    cudaFree(terminals_d);

    cudaDeviceSynchronize();

    free(allSubsets);
}
