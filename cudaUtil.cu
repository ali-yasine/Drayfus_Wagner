#include "csr.h"

#include <cstdlib>
#include <stdio.h>
#include <assert.h>

CsrGraph* createEmptyCSRGraphOnGPU(unsigned int num_nodes, unsigned int num_edges) {

    CsrGraph graph_shadow;
    graph_shadow.num_nodes = num_nodes;
    graph_shadow.num_edges = num_edges;
    cudaMalloc((void**) &graph_shadow.row_offsets, sizeof(unsigned int) * (num_nodes + 1));
    cudaMalloc((void**) &graph_shadow.col_indices, sizeof(unsigned int) * num_edges);
    cudaMalloc((void**) &graph_shadow.edge_weights, sizeof(unsigned int) * num_edges);

    CsrGraph* graph;

    cudaMalloc((void**) &graph, sizeof(CsrGraph));
    cudaMemcpy(graph, &graph_shadow, sizeof(CsrGraph), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    return graph;
}

void freeCSRGraphOnGPU(CsrGraph* graph) {
    
    cudaFree(graph->row_offsets);
    cudaFree(graph->col_indices);
    cudaFree(graph->edge_weights);
    cudaFree(graph);
}
void copyCSRGraphToGPU(CsrGraph* graph, CsrGraph* graph_d) {
    CsrGraph graph_shadow;
    cudaMemcpy(&graph_shadow, graph_d, sizeof(CsrGraph), cudaMemcpyDeviceToHost);
    
    assert(graph_shadow.num_nodes == graph->num_nodes);
    assert(graph_shadow.num_edges == graph->num_edges);

    cudaMemcpy(graph_shadow.row_offsets, graph->row_offsets, (graph->num_nodes + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(graph_shadow.col_indices, graph->col_indices, graph->num_edges * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(graph_shadow.edge_weights, graph->edge_weights, graph->num_edges * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}