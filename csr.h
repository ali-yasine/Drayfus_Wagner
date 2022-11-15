#ifndef __CSR_H_
#define __CSR_H_
#include <limits.h>

struct CsrGraph {
    unsigned int num_nodes;
    unsigned int num_edges;
    unsigned int* row_offsets;
    unsigned int* col_indices;
    unsigned int* edge_weights;

    unsigned int getEdgeWeight(unsigned int src, unsigned int dst) {
        if (src == dst) 
            return 0;
        
        for(unsigned int i = row_offsets[src]; i < row_offsets[src + 1]; ++i)
            if(col_indices[i] == dst)
                return edge_weights[i];
                
        return UINT_MAX; 
    }
};

#endif
