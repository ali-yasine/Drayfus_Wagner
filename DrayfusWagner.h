#pragma once 
#include "csr.h"
#include "floydWarshall.h"
#include "subsets.h"
#include <stdlib.h>

//
unsigned int* DrayfusWagner(CsrGraph graph, unsigned int* terminals, unsigned int numTerminals) {
    unsigned int* apsp = floydWarshall(graph);

    unsigned int* DP = (unsigned int* ) malloc(graph.num_nodes * ( (1 >> numTerminals) - 1) *sizeof(unsigned int));

    //loop over subset sizes
    for(unsigned int k = 2; k < numTerminals; ++k) {
        unsigned int numSubsets = 0; 
        unsigned int* subsets = subsetsK(terminals, graph.num_nodes, k, numTerminals, &numSubsets);
        //loop over subsets
        for (unsigned int subset = 0; subset < numSubsets; ++subset ) {

            for (unsigned int root = 0; root < graph.num_nodes; ++root) {
                if (contains(terminal, root)) 
                    continue;
                
                unsigned int num_sub_subsets = 0;
                unsigned int* currSubset = subsets + (subset * numTerminals);
                unsigned int* subSubsets = subSetGenerate(currSubset, numTerminals, &num_sub_subsets);
                
                for(unsigned int subSubset = 0; subSubset < num_sub_subsets; ++subSubset) {
                    for(unsigned int vertex = 0; vertex < graph.num_nodes; ++vertex)
                    // DP[r,s] = min(DP[v, ss], DP[])
                }
            }
        }
    }
}
bool contains(unsigned int* set,unsigned int N, unsigned int vertex) {
    for (unsigned int i = 0; i < N; ++i) {
        if (set[i] == vertex)
            return true;
    }
    return false;
}