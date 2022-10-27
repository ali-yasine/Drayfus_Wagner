#pragma once
#include "csr.h"
#include <stdlib.h>

#define min(a,b) ((a) < (b) ? (a) : (b))

/*
    input: CsrGraph graph
    output: dp array where d[i * num_nodes + j] is the shortest path from i to j
*/

unsigned int* floydWarshall(CsrGraph graph){
    unsigned int* d = (unsigned int*) malloc(graph.num_nodes * graph.num_nodes * sizeof(unsigned int));

    for (int i = 0; i < graph.num_nodes; ++i){
        for (int j = 0; j < graph.num_nodes; ++j){
            d[i * graph.num_nodes + j] = getEdgeWeight(graph, i, j);
        }
    }

    for(unsigned int k = 0; k < graph.num_nodes; ++k){

        for(unsigned int i = 0; i < graph.num_nodes; ++i){

            for(unsigned int j = 0; j < graph.num_nodes; ++j) {

                unsigned int newerWeight = d[i * graph.num_nodes + k] + d[k * graph.num_nodes + j];
                if (newerWeight < d[i * graph.num_nodes + j]){
                    d[i * graph.num_nodes + j] = newerWeight;
                }
                
            }
        }
    }
    return d;
}
#undef min