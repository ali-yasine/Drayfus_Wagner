#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "DrayfusWagner.h"
#include "csr.h"
#include "timer.h"
#include "common.h"
int main() {
    cudaDeviceSynchronize();
    setbuf(stdout, NULL);


    //Testing on a graph of 10 vertices
    unsigned int numberOfVertices = 7;   
    unsigned int values[] =  {4, 8, 4, 2, 3, 8, 5, 7, 2, 6, 3, 5, 1, 1, 7, 1, 6, 1};
    unsigned int col[] =     {1, 2, 0, 3, 4, 0, 4, 5, 1, 6, 1, 2, 5, 6, 2, 4, 3, 4};
    unsigned int rowPtr[] =  {0, 2, 5, 8, 10, 14, 16, 18};
    unsigned int numberOfTerminals = 3;
    unsigned int terminals[] {2, 3, 5};

    // Testing on a graph of 20 vertices
    // unsigned int numberOfVertices = 20;   
    // unsigned int values[] =  {29359, 16828, 2996, 14605, 12383, 5448, 11539, 17036, 28704, 4665, 12317, 1843, 30107, 12383, 6730, 15351, 3549, 19955, 13932, 22930, 13932, 2307, 22387, 6271, 15574, 16513, 13291, 4032, 18008, 29359, 5448, 27754, 14946, 6423, 2307, 18763, 27596, 11539, 6730, 22387, 30837, 11021, 24022, 19669, 8282, 15351, 4032, 27754, 26419, 18128, 24649, 17808, 14946, 16828, 18763, 26419, 30304, 17036, 3549, 6271, 30837, 32703, 20486, 14344, 2996, 28704, 22930, 11021, 29315, 12317, 15574, 24022, 18128, 19797, 15282, 19955, 16513, 6423, 29315, 20799, 1843, 13291, 19669, 32703, 20799, 23623, 14605, 4665, 30107, 8282, 24649, 30304, 20486, 19797, 23623, 6039, 18008, 27596, 17808, 14344, 15282, 6039};
    // unsigned int col[] =     {7, 12, 14, 18, 3, 7, 9, 13, 14, 18, 15, 17, 18, 1, 9, 10, 13, 16, 5, 14, 4, 8, 9, 13, 15, 16, 17, 10, 19, 0, 1, 10, 11, 16, 5, 12, 19, 1, 3, 5, 13, 14, 15, 17, 18, 3, 6, 7, 12, 15, 18, 19, 7, 0, 8, 10, 18, 1, 3, 5, 9, 17, 18, 19, 0, 1, 4, 9, 16, 2, 5, 9, 10, 18, 19, 3, 5, 7, 14, 17, 2, 5, 9, 13, 16, 18, 0, 1, 2, 9, 10, 12, 13, 15, 17, 19, 6, 8, 10, 13, 15, 18};
    // unsigned int rowPtr[] =  {0, 4, 10, 13, 18, 20, 27, 29, 34, 37, 45, 52, 53, 57, 64, 69, 75, 80, 86, 96, 102};
    // unsigned int numberOfTerminals = 5;
    // unsigned int terminals[] {3, 5, 7, 8, 9};
    // unsigned int bitTerminals[] {1,1,1,1, 1};

    CsrGraph graph {
        numberOfVertices,
        numberOfTerminals,
        rowPtr,
        col,
        values
    };
    //TODO 
    // printf("Reading graph from file: %s\n", matrixFile);
    // CsrGraph* graph = CSRfromFile(matrixFile);
    // printf("Graph read from file: %s\n", matrixFile);

    Timer timer;
    startTime(&timer);
    printf("Computing Floyd-Warshall...\n");
    unsigned int* apsp = floydWarshall(graph);
    stopTime(&timer);
    printElapsedTime(timer, "    Floyd-Warshall");

    printf("Running CPU version\n");
    unsigned int* cpuResult = DrayfusWagner_cpu(graph, numberOfTerminals, terminals, apsp);
    stopTime(&timer);
    printElapsedTime(timer, "    CPU time", CYAN);

    printf("Running GPU version\n");


    // Allocate GPU memory
    startTime(&timer);
    CsrGraph* graph_d = createEmptyCSRGraphOnGPU(graph.num_nodes, graph.num_edges);
    stopTime(&timer);
    printElapsedTime(timer, "  GPU allocation time");

    // Copy graph to GPU
    startTime(&timer);
    
    copyCSRGraphToGPU(&graph, graph_d);
    cudaDeviceSynchronize();

    stopTime(&timer);
    printElapsedTime(timer, "  GPU copy time");

    unsigned int* DP = (unsigned int*) malloc(sizeof(unsigned int) * graph.num_nodes *  ((1 << numberOfTerminals) - 1) );
    
    for(int i = 0; i < graph.num_nodes *  ((1 << numberOfTerminals) - 1); i++) {
        DP[i] = UINT_MAX;
    }
    
    startTime(&timer);
    DrayfusWagnerGPU(graph_d, numberOfTerminals, terminals, DP, apsp);
    stopTime(&timer);

    printElapsedTime(timer, "  GPU Kernel time", CYAN);
    
}
