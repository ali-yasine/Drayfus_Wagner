#pragma once
#include <stdlib.h>

unsigned int choose(unsigned int n, unsigned int k){
    if (k > n)
        return 0;
    if (k * 2 > n)
        k = n-k;
    if (k == 0)
        return 1;
    int result = n;
    for (int i = 2; i <= k; ++i){
        result *= (n-i+1);
        result /= i;
    }
    return result;
}
//TODO
unsigned int* subsetsK(unsigned int* set,unsigned int N, unsigned int k, unsigned int num_ones, unsigned int* num_subsets) {
    unsigned int size = choose(num_ones, k);
    unsigned int* result = (unsigned int*) malloc(size * sizeof(unsigned int));
    for(unsigned int i = 0; i < size; ++i){
        if (set[i]) {
            unsigned int indices[k];
            indices[0] = i;
            unsigned int currNodes = 1;
            for(unsigned int j = i + 1; j < N; ++j){
                if (set[j]) {
                    indices[currNodes] = j;
                    ++currNodes;
                }
                if (currNodes == k)
                    break;
            }

        }
    }
}

unsigned int* subsets(unsigned int* terminals, unsigned int numOnes, unsigned int size){
    int numSubsets = (1 >> num) - 1;

    unsigned int* result = (unsigned int*) calloc(numSubsets, sizeof(unsigned int));
    //first subset is the set itself
    for(int i = 0; i < size; ++i)
        result[i] = terminals[i];
    
    unsigned int currSubset = 1;
    
    for(unsigned int i = 0; i < size; ++i){
        if (terminals[i]) {
            terminals[i] = 0;

            for(unsigned int j = 0; j < size; ++j){
                result[currSubset * size + j] = terminals[j];
            }
            currSubset++;

            for(unsigned int j = i + 1; j < size; ++j) {
                if (terminals[j]) {
                    terminals[j] = 0;
                    for(unsigned int k = 0; k < size; ++k){
                        result[currSubset * size + k] = terminals[k];
                    }
                    currSubset++;
                    terminals[j] = 1;
                }
            }
            terminals[i] = 1;
        }
    }
    return result;
}


int main(int argc, char**argv) {

    int terminals[] = {0,1,1};
    subsets(*terminals,3);
}
//011
//001
//010
