#pragma once 
#include "csr.h"
#include "floydWarshall.h"
#include "subsets.h"
#include "Util.h"
#include <iostream>

//TODO add frees to handle memory leaks
unsigned int* DrayfusWagner(CsrGraph graph, unsigned int* terminals, unsigned int numTerminals, unsigned int* terminalMap) {
    unsigned int* apsp = floydWarshall(graph);
    if (apsp == NULL) {
        printf("Error: floydWarshall returned NULL\n");
        return NULL;
    }
    unsigned int* DP = (unsigned int* ) calloc(graph.num_nodes * ( (1 << numTerminals) - 1), sizeof(unsigned int));
    if (DP == NULL) {
        printf("Error:DP calloc returned NULL\n");
        return NULL;
    }
    unsigned int* allSubsets = getSortedSubsets(numTerminals);
    if (allSubsets == NULL) {
        std::cout << "Error: allSubsets is NULL\n";
        return NULL;
    }

    unsigned int totalSubsets = (1 << numTerminals) - 1;
    unsigned int curr_Subset = 0;

    //init DP to INF
    for(unsigned int i = 0; i < graph.num_nodes; ++i){
        for(unsigned int j = 0; j < totalSubsets; ++j){
            DP[i * totalSubsets + j] = INF;
        }
    }

    //handle singletons 
    for(unsigned int vertex = 0; vertex < graph.num_nodes; ++vertex){
        for(unsigned int* subset = allSubsets; subset < allSubsets + numTerminals; ++subset){
            //find index of 1 in subset
            unsigned int index = 0;
            for(unsigned int i = 0; i < numTerminals; ++i){
                if (subset[i]){
                    index = i;
                    break;
                }
            }
            if (terminals[index] == vertex) {
                DP[vertex * totalSubsets + curr_Subset] = 0;
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
        for (unsigned int subset = curr_Subset; subset < curr_Subset + numSubsets; ++subset ) {
            std::cout << "subset: " << subset << "\n";
            unsigned int* currSubset = allSubsets + (subset * numTerminals);
            if (currSubset == NULL) {
                std::cout << "Error: currSubset is NULL\n";
                return NULL;
            }
            std::cout << "currSubset init  \n";
            //unsigned int s_index = getSubsetIndex(currSubset, numTerminals, allSubsets);
            unsigned int s_index = subset;
            if (s_index == -1) {
                std::cout << "Error: s_index is -1\n";
                return NULL;
            }
            std::cout << "s_index init  \n";
            for (unsigned int root = 0; root < graph.num_nodes; ++root) {
                std::cout << "root: " << root << "\n";
                if (! contains(terminalMap, numTerminals, root)) { 

                    unsigned int num_sub_subsets = (1 << k) - 1;
                    printf("num_sub_subsets: %d\n", num_sub_subsets);
                    unsigned int* subSubsets = generateSubsets(currSubset, numTerminals);
                    if (subSubsets == NULL) {
                        std::cout << "Error: subSubsets is NULL\n";
                        return NULL;
                    }
                    for(unsigned int subSubset = 0; subSubset < num_sub_subsets; ++subSubset) {
                        unsigned int* curr_sub_subset = subSubsets + (subSubset * numTerminals);
                        printf("getting ss_index: \n");
                        unsigned int ss_index = getSubsetIndex(curr_sub_subset, numTerminals, allSubsets);

                        unsigned int* sMinusSS = setDifference(currSubset, curr_sub_subset, numTerminals);
                        if (sMinusSS == NULL) {
                            std::cout << "Error: sMinusSS is NULL\n";
                            return NULL;
                        }
                        unsigned int sMinusSS_index = getSubsetIndex(sMinusSS, numTerminals, allSubsets);

                        unsigned int cost = DP[root * totalSubsets + s_index];
                        for(unsigned int vertex = 0; vertex < graph.num_nodes; ++vertex){
                            // DP[r, s] min= DP[v, ss] + DP[v, s / ss] + dist(r, v)
                            unsigned int sum = DP[vertex * totalSubsets + ss_index] + DP[vertex * totalSubsets + sMinusSS_index] + apsp[root * graph.num_nodes + vertex]; 

                            if (sum < cost){
                                DP[root * totalSubsets + s_index] = sum;
                                cost = sum;
                            }
                        }
                    }
                }
            }
        }
        curr_Subset += numSubsets;
    }
    free(apsp);
    free(allSubsets);
    return DP;
}