#include "DrayfusWagner.h"

#include <iostream>
int main() {
    int graph[36] {
        {0, 1, 2, 0, 0, 0, //A
        1, 0, 0, 5, 1, 0, //B
        2, 0, 0, 2, 3, 0, //C
        0, 5, 2, 0, 2, 2, //D
        0, 1, 3, 2, 0, 1, //E
        0, 0, 0, 2, 1, 0};
    int terminals[3] {1, 4, 6};
    
    CsrGraph csrGraph{
        .num_nodes = 6,
        .num_edges = 36,
        .row_offsets = new unsigned int[7],
        .col_indices = new unsigned int[36],
        .edge_weights = new unsigned int[36]
    };
    unsigned int* result = DrayfusWagner(graph, terminals, , 3);

}