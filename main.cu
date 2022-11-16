#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "DrayfusWagner.h"
#include "timer.h"
#include "common.h"
#include "Coo.h"
void verify(unsigned int * DP, unsigned int * DP_d, unsigned int num_nodes, unsigned int num_subsets, unsigned int* allSubsets, unsigned int numberOfTerminals){

  for(unsigned int v = 0; v < num_nodes; ++v) {
    for (unsigned int subset = 0; subset < num_subsets; ++subset) {
      if (DP[v * num_subsets + subset] != DP_d[v * num_subsets + subset]) {
        printf("mismatch at vertex v: %u\t,subset:  DP: %u\tDP_d: %u\n", v, DP[v * num_subsets + subset], DP_d[v * num_subsets + subset]);
      }
    }
  }
}

void verifyFlippedDP(unsigned int* DP, unsigned int* Dp_d, unsigned int num_nodes, unsigned int num_subsets) {
  for(unsigned int v = 0; v < num_nodes; ++v) {
    for (unsigned int subset = 0; subset < num_subsets; ++subset) {
      if (DP[v * num_subsets + subset] != Dp_d[subset * num_nodes + v]) {
        printf("mismatch at vertex v: %u\t, subset: %u \tDP: %u\tDP_d: %u\n", v, subset, DP[v * num_subsets + subset], Dp_d[subset * num_nodes + v]);
      }
    }
  } 
}

int main(int argc, char** argv) {
    cudaDeviceSynchronize();
    setbuf(stdout, NULL);
    unsigned int num_nodes = argc > 1 ;

    //Testing on a graph of 10 vertices
    const char* filename = "data/10.txt";

    CsrGraph* graph = readCSRgraph(filename);


    unsigned int numberOfTerminals = 5;
    unsigned int terminals[] = {2,3,5,7,9};

    // Testing on a graph of 20 vertices
    /*
    unsigned int numberOfVertices = 20;   
    unsigned int values[] =  {29359, 16828, 2996, 14605, 12383, 5448, 11539, 17036, 28704, 4665, 12317, 1843, 30107, 12383, 6730, 15351, 3549, 19955, 13932, 22930, 13932, 2307, 22387, 6271, 15574, 16513, 13291, 4032, 18008, 29359, 5448, 27754, 14946, 6423, 2307, 18763, 27596, 11539, 6730, 22387, 30837, 11021, 24022, 19669, 8282, 15351, 4032, 27754, 26419, 18128, 24649, 17808, 14946, 16828, 18763, 26419, 30304, 17036, 3549, 6271, 30837, 32703, 20486, 14344, 2996, 28704, 22930, 11021, 29315, 12317, 15574, 24022, 18128, 19797, 15282, 19955, 16513, 6423, 29315, 20799, 1843, 13291, 19669, 32703, 20799, 23623, 14605, 4665, 30107, 8282, 24649, 30304, 20486, 19797, 23623, 6039, 18008, 27596, 17808, 14344, 15282, 6039};
    unsigned int col[] =     {7, 12, 14, 18, 3, 7, 9, 13, 14, 18, 15, 17, 18, 1, 9, 10, 13, 16, 5, 14, 4, 8, 9, 13, 15, 16, 17, 10, 19, 0, 1, 10, 11, 16, 5, 12, 19, 1, 3, 5, 13, 14, 15, 17, 18, 3, 6, 7, 12, 15, 18, 19, 7, 0, 8, 10, 18, 1, 3, 5, 9, 17, 18, 19, 0, 1, 4, 9, 16, 2, 5, 9, 10, 18, 19, 3, 5, 7, 14, 17, 2, 5, 9, 13, 16, 18, 0, 1, 2, 9, 10, 12, 13, 15, 17, 19, 6, 8, 10, 13, 15, 18};
    unsigned int rowPtr[]4 =  {0, 4, 10, 13, 18, 20, 27, 29, 34, 37, 45, 52, 53, 57, 64, 69, 75, 80, 86, 96, 102};
    unsigned int numberOfTerminals = 5;
    unsigned int terminals[] = {3, 5, 7, 8, 9};
    unsigned int bitTerminals[] = {1,1,1,1, 1};
    */
    Timer timer;
    startTime(&timer);
    printf("Computing Floyd-Warshall...\n");
    unsigned int* apsp = floydWarshall(*graph);
    stopTime(&timer);
    printElapsedTime(timer, "Floyd-Warshall");
    
    
    printf("Running CPU version\n");

    unsigned int* cpuResult = DrayfusWagner_cpu(*graph, numberOfTerminals, terminals, apsp);
    
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", CYAN);
    
    printf("Running GPU version\n");

    // Allocate GPU memory
    CsrGraph* graph_d = createEmptyCSRGraphOnGPU(graph->num_nodes, graph->num_edges);

    // Copy graph to GPU

    copyCSRGraphToGPU(graph, graph_d);
    cudaDeviceSynchronize();

    unsigned int* DP = (unsigned int*) malloc(sizeof(unsigned int) * graph->num_nodes *  ((1 << numberOfTerminals) - 1) );
    
    
    startTime(&timer);
    DrayfusWagnerGPU(graph, graph_d, numberOfTerminals, terminals, DP, apsp);
    stopTime(&timer);

    printElapsedTime(timer, "GPU total time", CYAN);
    
    verify(cpuResult, DP, graph->num_nodes , ((1 << numberOfTerminals) - 1), getSortedSubsets(numberOfTerminals), numberOfTerminals);
    
    CsrGraph* graph_opt1_d = createEmptyCSRGraphOnGPU(graph->num_nodes, graph->num_edges);
    copyCSRGraphToGPU(graph, graph_opt1_d);
    cudaDeviceSynchronize();



    unsigned int* DP_opt1 = (unsigned int*) malloc(sizeof(unsigned int) * graph->num_nodes *  ((1 << numberOfTerminals) - 1) );
    startTime(&timer);
    DrayfusWagnerGPU_o1(graph, graph_opt1_d, numberOfTerminals, terminals, DP_opt1, apsp);
    stopTime(&timer);
    printElapsedTime(timer, "GPU opt1 time", CYAN);
    verifyFlippedDP(cpuResult, DP_opt1, graph->num_nodes, ((1 << numberOfTerminals) - 1));
    
}
