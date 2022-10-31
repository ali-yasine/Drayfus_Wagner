#pragma once 
#include "csr.h"
#include "floydWarshall.h"
#include "subsets.h"
#include "Util.h"
#include <stdlib.h>

//TODO add frees to handle memory leaks
unsigned int* DrayfusWagner(CsrGraph graph, unsigned int* terminals, unsigned int numTerminals) {
    unsigned int* apsp = floydWarshall(graph);

    unsigned int* DP = (unsigned int* ) calloc(graph.num_nodes * ( (1 << numTerminals) - 1), sizeof(unsigned int));
    //handle singletons TODO

    unsigned int* allSubsets = getSortedSubsets(numTerminals);
    unsigned int totalSubsets = (1 << numTerminals) - 1;
    unsigned int curr_Subset = 0;
    
    //loop over subset sizes
    for(unsigned int k = 2; k < numTerminals; ++k) {
        unsigned int numSubsets = choose(numTerminals, k);
        
        //loop over subsets
        for (unsigned int subset = currSubset; subset < currSubset + numSubsets; ++subset ) {

            unsigned int* currSubset = allSubsets + (subset * numTerminals);

            unsigned int s_index = getSubsetIndex(currSubset, numTerminals, allSubsets);

            for (unsigned int root = 0; root < graph.num_nodes; ++root) {
                
                if (contains(terminals, root)) 
                    continue;

                unsigned int num_sub_subsets = (1 << k) - 1;

                unsigned int* subSubsets = generateSubsets(currSubset, numTerminals);
                
                for(unsigned int subSubset = 0; subSubset < num_sub_subsets; ++subSubset) {

                    unsigned int* curr_sub_subset = subSubsets + (subSubset * numTerminals);

                    unsigned int ss_index = getSubsetIndex(curr_sub_subset, numTerminals, allSubsets);

                    unsigned int* sMinusSS = setDifference(currSubset, curr_sub_subset, numTerminals);

                    unsigned int sMinusSS_index = getSubsetIndex(sMinusSS, numTerminals, allSubsets);

                    for(unsigned int vertex = 0; vertex < graph.num_nodes; ++vertex){
                        // DP[r, s] min= DP[v, ss] + DP[v, s / ss] + dist(r, v)
                        unsigned int sum = (DP[vertex * totalSubsets + ss_index] + DP[vertex * allSubsets + sMinusSS_index] + apsp[root * graph.num_nodes + vertex]; 

                        if (sum < DP[root * allSubsets + s_index])
                            DP[root * allSubsets + s_index] = sum;
                    }
                }
            }
        }
        currSubset += numSubsets;
    }
    free(apsp);
    free(allSubsets);
    return DP;
}
