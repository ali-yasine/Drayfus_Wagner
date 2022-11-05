#pragma once
#include <stdlib.h>
#include "Util.h"



__device__ __host__ unsigned int* generateSubsets(unsigned int* terminals, unsigned int size) {
    unsigned int num_ones = 0;
    for(unsigned int i = 0; i < size; ++i){
        if (terminals[i])
            num_ones++;
    }
    
    const unsigned int numSubsets = (1 << num_ones) - 1;
    unsigned int* result = (unsigned int*) calloc(numSubsets * size, sizeof(unsigned int));

    unsigned int decimalVal = binaryToDecimal(terminals, size);
    unsigned int decimalSubsets = (unsigned int*) calloc(numSubsets, sizeof(unsigned int));
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

__device__ __host__ unsigned int* subsetK(unsigned int* set, unsigned int k, unsigned int size) {
    unsigned int* allSubsets = generateSubsets(set, size);
    
    unsigned int num_ones = 0;
    unsigned int currSubset = 0;
    for(unsigned int i = 0; i < size; ++i){
        if (set[i])
            num_ones++;
    }
    
    unsigned int numSubsets = choose(num_ones, k);
    unsigned int* result = (unsigned int*) malloc(numSubsets * size * sizeof(unsigned int));
    
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



__device__ __host__ unsigned int* getSortedSubsets(unsigned int size) {
    unsigned int terminals = (unsigned int*) calloc(size, sizeof(unsigned int));

    unsigned int* result = (unsigned int*) calloc( ( (1 << size) - 1) * size, sizeof(unsigned int));
    
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

__device__ __host__ unsigned int getSubsetIndex(unsigned int* set, unsigned int size, unsigned int* allSubsets){

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
    return -1;
}
