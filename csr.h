#pragma once
#define INF 0xFFFFFFFF;
struct CsrGraph {
    unsigned int num_nodes;
    unsigned int num_edges;
    unsigned int* row_offsets;
    unsigned int* col_indices;
    unsigned int* edge_weights;
};

unsigned int getEdgeWeight(CsrGraph graph, int src, int dst) {
    if (src == dst) 
        return 0;
    
    for(unsigned int i = graph.row_offsets[src]; i < graph.row_offsets[src+1]; ++i){
        if(graph.col_indices[i] == dst)
            return graph.edge_weights[i];
    }
    return INF; 
}
