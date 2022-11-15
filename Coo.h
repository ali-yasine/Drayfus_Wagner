#ifndef __COO_H__
#define __COO_H__
#include <limits.h>
#include "csr.h"

struct CooGraph {
    unsigned int num_nodes;
    unsigned int num_edges;
    unsigned int* row_indices;
    unsigned int* col_indices;
    unsigned int* edge_weights

}

CsrGraph* cooToCSR(CooGraph* coo);

CooGraph* readGraph(const char* filename);
