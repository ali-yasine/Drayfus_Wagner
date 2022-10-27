#include "floydWarshall.h"
#include "subsets.h"
#include "csr.h"
#include <iostream>
int main() {
    const unsigned int num_nodes = 3;
    const unsigned int num_edges = 2; 
    unsigned int rows[] {0, 1, 2};
    unsigned int cols[] {1, 2};
    unsigned int weights[] {1, 1};
    CsrGraph graph {
        num_nodes,
        num_edges,
        rows,
        cols,
        weights
    };
    
}