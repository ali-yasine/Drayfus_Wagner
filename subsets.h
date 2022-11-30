#ifndef __SUBSETS_H_
#define __SUBSETS_H_
#include <stdlib.h>
#include "Util.h"
#ifndef MAX_TERMINALS
#define MAX_TERMINALS 25
#endif


__host__ static  unsigned int* generateSubsets(unsigned int* terminals, unsigned int size) {
    unsigned int num_ones = 0;
    unsigned int* result, *decimalSubsets;

    for(unsigned int i = 0; i < size; ++i){
        if (terminals[i])
            num_ones++;
    }
    
    unsigned int numSubsets = (1 << num_ones) - 1;
    result = (unsigned int*) calloc(numSubsets * size, sizeof(unsigned int));
    decimalSubsets = (unsigned int*) calloc(numSubsets, sizeof(unsigned int));

    unsigned int decimalVal = binaryToDecimal(terminals, size);
    unsigned int currSubset = 0;
    for(unsigned int s = decimalVal; s; s = (s - 1) & decimalVal){
        unsigned int* subset = decimalToBinary(s, size);
        for(unsigned int i = 0; i < size; ++i){
            result[currSubset * size + i] = subset[i];
        }
        currSubset++;

        free(subset);
    }

    free(decimalSubsets);
    return result;
}

__device__ static void generateSubsetsGPU(unsigned int* terminals, unsigned int size, unsigned int* result) {
    unsigned int num_ones = 0;
    unsigned int* decimalSubsets;

    for(unsigned int i = 0; i < size; ++i){
        if (terminals[i])
            num_ones++;
    }
    
    unsigned int numSubsets = (1 << num_ones) - 1;

    cudaMalloc(&decimalSubsets, numSubsets * sizeof(unsigned int));

    unsigned int decimalVal = binaryToDecimal(terminals, size);
    unsigned int currSubset = 0;
    for(unsigned int s = decimalVal; s; s = (s - 1) & decimalVal){
        unsigned int* subset = decimalToBinaryGPU(s, size);
        for(unsigned int i = 0; i < size; ++i){
            result[currSubset * size + i] 
            = subset[i];
        }
        currSubset++;
        cudaFree(subset);
        
    }
    cudaFree(decimalSubsets);
}


__host__ static  unsigned int* subsetK(unsigned int* set, unsigned int k, unsigned int size) {
    unsigned int* allSubsets = generateSubsets(set, size);
    unsigned int* result;

    unsigned int num_ones = 0;
    unsigned int currSubset = 0;
    for(unsigned int i = 0; i < size; ++i){
        if (set[i])
            num_ones++;
    }
    
    unsigned int numSubsets = choose(num_ones, k);
    
    result = (unsigned int*) malloc(numSubsets * size * sizeof(unsigned int));


    for(unsigned int i = 0; i < (1 << num_ones) - 1; ++i){
        
        unsigned int curr_num_ones = 0;
        for (unsigned int j = 0; j < size; ++j)
            if (allSubsets[i * size + j])
                curr_num_ones++;
            
        if (curr_num_ones == k) {
            for (unsigned int j = 0; j < size; ++j)
                result[currSubset * size + j] = allSubsets[i * size + j];
            currSubset++;
        }
    }
    
    free(allSubsets);
    return result;
}

__device__ static   unsigned int* subsetKGPU(unsigned int* set, unsigned int k, unsigned int size) {

    unsigned int* allSubsets;
    cudaMalloc( (void**) &allSubsets, ((1 << k) - 1) * size * sizeof(unsigned int));
    generateSubsetsGPU(set, size, allSubsets);
    unsigned int* result;

    unsigned int num_ones = 0;
    unsigned int currSubset = 0;
    for(unsigned int i = 0; i < size; ++i){
        if (set[i])
            num_ones++;
    }
    
    unsigned int numSubsets = choose(num_ones, k);
    cudaMalloc(&result, numSubsets * size * sizeof(unsigned int));

    for(unsigned int i = 0; i < (1 << num_ones) - 1; ++i){
        
        unsigned int curr_num_ones = 0;
        for (unsigned int j = 0; j < size; ++j)
            if (allSubsets[i * size + j])
                curr_num_ones++;
            
        if (curr_num_ones == k) {
            for (unsigned int j = 0; j < size; ++j)
                result[currSubset * size + j] = allSubsets[i * size + j];
            currSubset++;
        }
    }

    cudaFree(allSubsets);
    return result;
}


__host__ static  unsigned int* getSortedSubsets(unsigned int size) {
    
    
    unsigned int* terminals, *result;
    
    terminals = (unsigned int*) calloc(size, sizeof(unsigned int));
    result = (unsigned int*) calloc( ( (1 << size) - 1) * size, sizeof(unsigned int));
    

    for (unsigned int i = 0; i < size; ++i) {
        terminals[i] = 1;
    }
    unsigned int currSubset = 0;
    for(unsigned int k = 1; k <= size; ++k) {
        unsigned int subsetNum = choose(size, k);
        unsigned int* subsets = subsetK(terminals, k, size);
        
        for(unsigned int i = 0; i < (subsetNum * size); ++i) {
            result[currSubset * size + i] = subsets[i];
        }
        currSubset += subsetNum;
        free(subsets);

    }

    free(terminals);
    return result;
}
__device__ static unsigned int* getSortedSubsetsGPU(unsigned int size) {
    
    
    unsigned int* terminals, *result;
    
    cudaMalloc(&terminals, size * sizeof(unsigned int));
    cudaMalloc(&result, ((1 << size ) - 1) * size * sizeof(unsigned int));
    
    for (unsigned int i = 0; i < size; ++i) {
        terminals[i] = 1;
    }
    unsigned int currSubset = 0;
    for(unsigned int k = 1; k <= size; ++k) {
        unsigned int subsetNum = choose(size, k);
        unsigned int* subsets = subsetKGPU(terminals, k, size);
        
        for(unsigned int i = 0; i < (subsetNum * size); ++i) {
            result[currSubset * size + i] = subsets[i];
        }
        currSubset += subsetNum;
        cudaFree(subsets);
    }

    cudaFree(terminals);
    return result;
}
__device__ __host__ static  unsigned int getSubsetIndex(unsigned int* set, unsigned int size, unsigned int* allSubsets){

    unsigned int num_ones = 0;

    for(unsigned int i = 0; i < size; ++i){
        if (set[i])
            num_ones++;
    }

    unsigned int num_subsets = (1 << size) - 1;
    unsigned int start = 0;
    
    for(unsigned int i = 0; i < num_ones; ++i){
        start += choose(size, i);
    }

    for(unsigned int i = start - 1; i < num_subsets; ++i){
        for(unsigned int j = 0; j < size; ++j){

            if (set[j] != allSubsets[i * size + j])
                break;
            if (j == size - 1)
                return i;
        }
    }
    return UINT_MAX;
}

__device__ static void generateSubsetsGPUO3(unsigned int* terminals, unsigned int size, unsigned int* result) {
    unsigned int num_ones = 0;
    unsigned int* decimalSubsets;

    for(unsigned int i = 0; i < size; ++i){
        if (terminals[i])
            num_ones++;
    }
    
    unsigned int numSubsets = (1 << num_ones) - 1;

    cudaMalloc(&decimalSubsets, numSubsets * sizeof(unsigned int));

    unsigned int decimalVal = binaryToDecimal(terminals, size);
    unsigned int currSubset = 0;
    unsigned int subset[MAX_TERMINALS];
    for(unsigned int s = decimalVal; s; s = (s - 1) & decimalVal){
        decimalToBinaryGPUO3(s, size, subset);
        for(unsigned int i = 0; i < size; ++i){
            result[currSubset * size + i] = subset[i];
        }
        currSubset++;
    }
    cudaFree(decimalSubsets);
}

__device__ static unsigned int* getSortedSubsetsGPUO3(unsigned int size) {
    
    
    unsigned int terminals[MAX_TERMINALS];
    unsigned int* result;
    
    cudaMalloc(&result, ((1 << size ) - 1) * size * sizeof(unsigned int));
    
    for (unsigned int i = 0; i < size; ++i) {
        terminals[i] = 1;
    }
    unsigned int currSubset = 0;
    for(unsigned int k = 1; k <= size; ++k) {
        unsigned int subsetNum = choose(size, k);
        unsigned int* subsets = subsetKGPU(terminals, k, size);
        
        for(unsigned int i = 0; i < (subsetNum * size); ++i) {
            result[currSubset * size + i] = subsets[i];
        }
        currSubset += subsetNum;
        cudaFree(subsets);
    }

    return result;
}

#endif
