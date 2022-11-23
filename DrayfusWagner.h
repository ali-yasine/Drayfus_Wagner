#ifndef __DRAYFUSWAGNER_H_
#define __DRAYFUSWAGNER_H_

#include "floydWarshall.h"
#include "subsets.h"
#include <iostream>
#include <limits.h>

static unsigned int* DrayfusWagner_cpu(CsrGraph graph, unsigned int numTerminals, unsigned int* terminalMap, unsigned int* apsp) {
    
    unsigned int total_subset_count = (1 << numTerminals) - 1;

    unsigned int* DP = (unsigned int* ) calloc(graph.num_nodes * total_subset_count, sizeof(unsigned int));
    
    unsigned int* allSubsets = getSortedSubsets(numTerminals);
    
    //init DP to INF
    for(unsigned int i = 0; i < graph.num_nodes; ++i)
        for(unsigned int j = 0; j < total_subset_count; ++j)
            DP[i * total_subset_count + j] = UINT_MAX;
    
    //handle singletons 
    for(unsigned int vertex = 0; vertex < graph.num_nodes; ++vertex) {
        for(unsigned int subset = 0; subset < numTerminals; ++subset) {
            
            //find index of 1 in subset
            unsigned int index = 0;
            for(unsigned int i = 0; i < numTerminals; ++i){
                if (allSubsets[subset * numTerminals + i]){
                    index = i;
                    break;
                }
            }
            
            DP[vertex * total_subset_count + subset] = apsp[vertex * graph.num_nodes + terminalMap[index]];
        }
    }

    //start at numTerminals since we already handled singletons
    unsigned int curr_subset_index = numTerminals;

    //loop over subset sizes
    for(unsigned int k = 2; k <= numTerminals; ++k) {
        
        unsigned int numSubsets = choose(numTerminals, k);
        
        //loop over subsets
        for (unsigned int subset = curr_subset_index; subset < curr_subset_index + numSubsets; ++subset ) {
            
            unsigned int* currSubset = allSubsets + (subset * numTerminals);
            
            
            for (unsigned int root = 0; root < graph.num_nodes; ++root) {

                unsigned int num_sub_subsets = (1 << k) - 1;

                unsigned int* subSubsets = generateSubsets(currSubset, numTerminals);
                
                unsigned int curr_cost = DP[root * total_subset_count + subset];
                
                for(unsigned int subSubset = 0; subSubset < num_sub_subsets; ++subSubset) {

                    unsigned int* curr_sub_subset = subSubsets + (subSubset * numTerminals);

                    //if they are equal then S/ss is empty so we skip
                    if (!equals(curr_sub_subset, currSubset, numTerminals)) {

                        unsigned int ss_index = getSubsetIndex(curr_sub_subset, numTerminals, allSubsets);
                        unsigned int* sMinusSS = setDifference(currSubset, curr_sub_subset, numTerminals);        

                        unsigned int sMinusSS_index = getSubsetIndex(sMinusSS, numTerminals, allSubsets);

                        for(unsigned int vertex = 0; vertex < graph.num_nodes; ++vertex){
                            // DP[r, s] min= DP[v, ss] + DP[v, s / ss] + dist(r, v)

                            unsigned int v_to_sub_Subset = DP[vertex * total_subset_count + ss_index];
                            unsigned int v_S_minusSS = DP[vertex * total_subset_count + sMinusSS_index]; 
                            unsigned int root_to_V = apsp[root * graph.num_nodes + vertex]; 

                            if (v_to_sub_Subset != UINT_MAX && v_S_minusSS != UINT_MAX && root_to_V != UINT_MAX) {
                                unsigned int sum = v_to_sub_Subset + v_S_minusSS + root_to_V;
                                if (sum < curr_cost) {
                                    DP[root * total_subset_count + subset] = sum;
                                    curr_cost = sum;
                                }
                            }
                        }
                    }
                }
            }
        }
        curr_subset_index += numSubsets;
    }
    free(allSubsets);
    return DP;
}
#endif
