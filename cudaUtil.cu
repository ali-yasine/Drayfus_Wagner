#include "csr.h"
#include "Coo.h"
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

CsrGraph* cooToCSR(CooGraph* coo) {
    CsrGraph* graph = (CsrGraph*) malloc(sizeof(CsrGraph));
    
    // Initialize fields
    graph->num_nodes = coo->num_nodes;
    graph->num_edges = coo->num_edges;
    graph->row_offsets = (unsigned int*) calloc((coo->num_nodes + 1),sizeof(unsigned int));
    graph->col_indices = (unsigned int*) calloc(coo->num_edges, sizeof(unsigned int) );
    graph->edge_weights = (unsigned int*) calloc(coo->num_edges, sizeof(unsigned int));

    //histogram rows
    for(unsigned int i = 0; i < coo->num_edges; ++i) {
        graph->row_offsets[coo->row_indices[i]]++;
    }
    
    //prefix sum row offsets
    unsigned int sumBeforeNextRow = 0;
    for(unsigned int row = 0; row < graph->num_nodes; ++row) {
        unsigned int sumBeforeRow = sumBeforeNextRow;
        sumBeforeNextRow += graph->row_offsets[row];
        graph->row_offsets[row] = sumBeforeRow;
    }
    graph->row_offsets[graph->num_nodes] = sumBeforeNextRow;

    //Bin the edges
    for (unsigned int i = 0; i < coo->num_edges; ++i) {
        unsigned int row = coo->row_indices[i];
        unsigned int j = graph->row_offsets[row]++;
        graph->col_indices[j] = coo->col_indices[i];
        graph->edge_weights[j] = coo->edge_weights[i];
    }

    //Restore row offsets
    for (unsigned int row = graph->num_nodes; row > 0; --row) {
        graph->row_offsets[row] = graph->row_offsets[row - 1];
    }

    graph->row_offsets[0] = 0;

    return graph;
}

CooGraph* readCOOGraph(const char* filename) {
    CooGraph* graph = (CooGraph*) malloc(sizeof(CooGraph));
    FILE* fp = fopen(filename, "r");

    //Initialize fields
    int x = 1;
    x |= fscanf(fp, "%u", &graph->num_nodes);
    x |= fscanf(fp, "%u", &graph->num_edges);
    graph->row_indices = (unsigned int*) malloc(sizeof(unsigned int) * (graph->num_edges));
    graph->col_indices = (unsigned int*) malloc(sizeof(unsigned int) * graph->num_edges);
    graph->edge_weights = (unsigned int*) malloc(sizeof(unsigned int) * graph->num_edges);

    //Read the graph
    for(unsigned int i = 0; i < graph->num_edges; ++i) {
        x |= fscanf(fp, "%u", &graph->row_indices[i]);
        x |= fscanf(fp, "%u", &graph->col_indices[i]);
        x |= fscanf(fp, "%u", &graph->edge_weights[i]);
    }   

    fclose(fp);
    return graph;
}

CsrGraph* readCSRgraph(const char* filename) {
    CooGraph* coo = readCOOGraph(filename);
    CsrGraph* graph = cooToCSR(coo);
    free(coo->row_indices);
    free(coo->col_indices);
    free(coo->edge_weights);
    free(coo);
    return graph;
}

void generateCOOGraph(unsigned int num_nodes) {
    FILE* file = fopen(num_nodes + ".txt", "w");
    unsigned int num_edges = (num_nodes * num_nodes * 30) / 100;
    fprintf(file, "%u %u\n", num_nodes, num_edges);

    bool edges[num_nodes][num_nodes];
    memset(edges, 0, sizeof(bool) * num_nodes * num_nodes);

    for(unsigned int i = 0; i < (num_edges + 1)/ 2; ++i) {
        unsigned int src = rand() % num_nodes;
        unsigned int dst = rand() % num_nodes; 
        if (edges[src][dst]) {
            --i;
            continue;
        }
        unsigned int edgeval = (src == dst) ? 0 : rand() % 100;
        fprintf(file, "%u %u %u\n", src, dst, edgeval);
        fprintf(file, "%u %u %u\n", dst, src, edgeval);
        edges[src][dst] = true;
    }
    fclose(file);
}

void printDP(unsigned int* DP,unsigned int num_nodes, unsigned int num_subsets) {
    for(unsigned int v = 0; v < num_nodes; ++v) {
        for(unsigned int s = 0; s < num_subsets; ++s) {
            printf("(%u,%u): %u\t",v, s, DP[v * num_subsets + s]);
        }
        printf("\n");
    }
}

void printSubsetByIndex(unsigned int* allSubsets, unsigned int index, unsigned int num_nodes) {
    for(unsigned int i = 0; i < num_nodes; ++i) {
        printf("%u\t",allSubsets[index * num_nodes + i]);
    }
    printf("\n");
}