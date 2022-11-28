#ifndef __FLOYDWARSHALL_H_
#define __FLOYDWARSHALL_H_
#include "csr.h"
#include <limits.h>
#include <iostream>

#define min(a,b) ((a) < (b) ? (a) : (b))

/*
    input: CsrGraph graph
    output: dp array where d[i * num_nodes + j] is the shortest path from i to j
*/

unsigned int* floydWarshall(CsrGraph* graph){
    unsigned int* d = (unsigned int*) malloc(graph->num_nodes * graph->num_nodes * sizeof(unsigned int));
    for (unsigned int i = 0; i < graph->num_nodes; ++i){
        for (unsigned int j = 0; j < graph->num_nodes; ++j){
            if (i == j)
                d[i * graph->num_nodes + j] = 0;
            else
                d[i * graph->num_nodes + j] = graph->getEdgeWeight(i, j);
        }
    }


    for(unsigned int k = 0; k < graph->num_nodes; ++k){

        for(unsigned int i = 0; i < graph->num_nodes; ++i){

            for(unsigned int j = 0; j < graph->num_nodes; ++j) {

                unsigned int dist_i_k = d[i * graph->num_nodes + k];
                unsigned int dist_k_j = d[k * graph->num_nodes + j];
                
                if (dist_i_k != UINT_MAX && dist_k_j != UINT_MAX && d[i * graph->num_nodes + j] > dist_i_k + dist_k_j ){
                    d[i * graph->num_nodes + j] = dist_i_k + dist_k_j;
                }
                
            }
        }
    }
    return d;
}
#undef min
#endif
