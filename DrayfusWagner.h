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
    
    unsigned int* allSubsets = getSortedSubsets(numTerminals);
    unsigned int totalSubsets = (1 << numTerminals) - 1;
    unsigned int curr_Subset = 0;

    //handle singletons 
    for(unsigned int vertex = 0; vertex < graph.num_nodes; ++vertex){
        for(unsigned int* subset = allSubsets; subset < allSubsets numTerminals; ++subset){
            //find index of 1 in subset
            unsigned int index = 0;
            for(unsigned int i = 0; i < numTerminals; ++i){
                if (subset[i]){
                    index = i;
                    break;
                }
            }
            if (terminals[index] == vertex) {
                DP[vertex * totalSubsets + subset] = 0;
            } else {
                DP[vertex * totalSubsets + curr_Subset] = apsp[vertex * graph.num_nodes + terminals[index]];
            }
            curr_Subset++;
        }
    }
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

                    unsigned int cost = DP[root * totalSubsets + s_index];
                    for(unsigned int vertex = 0; vertex < graph.num_nodes; ++vertex){
                        // DP[r, s] min= DP[v, ss] + DP[v, s / ss] + dist(r, v)
                        unsigned int sum = (DP[vertex * totalSubsets + ss_index] + DP[vertex * totalSubsets + sMinusSS_index] + apsp[root * graph.num_nodes + vertex]; 

                        if (sum < cost){
                            DP[root * totalSubsets + s_index] = sum;
                            cost = sum;
                        }
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