#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "DrayfusWagner.h"
#include "timer.h"
#include "common.h"
#include "Coo.h"
void verify(unsigned int * DP, unsigned int * DP_d, unsigned int num_nodes, unsigned int num_subsets, unsigned int* allSubsets, unsigned int numberOfTerminals){
  unsigned int num_mismatch = 0;
  for(unsigned int v = 0; v < num_nodes; ++v) {
    for (unsigned int subset = 0; subset < num_subsets; ++subset) {
      if (DP[v * num_subsets + subset] != DP_d[v * num_subsets + subset]) {
        num_mismatch++;
        if (num_mismatch < 10)
          printf("mismatch at vertex v: %u,subset: %u DP: %u\tDP_d: %u\n", v, subset, DP[v * num_subsets + subset], DP_d[v * num_subsets + subset]);
      }
    }
  }
  if( num_mismatch > 0) {
    printf("Number of mismatches: %u\n", num_mismatch);
  }
}

void verifyFlippedDP(unsigned int* DP, unsigned int* Dp_d, unsigned int num_nodes, unsigned int num_subsets) {
  unsigned int num_mismatch = 0;
  for(unsigned int v = 0; v < num_nodes; ++v) {
    for (unsigned int subset = 0; subset < num_subsets; ++subset) {
      if (DP[v * num_subsets + subset] != Dp_d[subset * num_nodes + v]) {
        num_mismatch++;
        if (num_mismatch < 10)
          printf("mismatch at vertex v: %u\t, subset: %u \tDP: %u\tDP_d: %u\n", v, subset, DP[v * num_subsets + subset], Dp_d[subset * num_nodes + v]);
      }
    }
  } 
  if( num_mismatch > 0) 
    printf("Number of mismatches: %u\n", num_mismatch);
}

int main(int argc, char** argv) {
  cudaDeviceSynchronize();
  setbuf(stdout, NULL);
  
  unsigned int num_nodes = (argc > 1) ?  atoi(argv[1]) : 300;
  unsigned int numberOfTerminals = (argc > 2) ?  atoi(argv[2]) : 8;
  
  unsigned int* terminals = (unsigned int*) malloc(sizeof(unsigned int) * numberOfTerminals);

  for (unsigned int i = 0; i < numberOfTerminals; ++i) {
    terminals[i] = rand() % num_nodes;
  }
  generateCOOGraph(num_nodes);

  
  char filename[100];
  sprintf(filename, "data/%u.txt", num_nodes);
  CsrGraph* graph = readCSRgraph(filename);
 
  Timer timer;
  
  startTime(&timer);
  printf("Computing Floyd-Warshall...\n");
  unsigned int* apsp = floydWarshall(*graph);
  stopTime(&timer);
  printElapsedTime(timer, "Floyd-Warshall");
  
  startTime(&timer);
  
  printf("Running CPU version\n");

  unsigned int* cpuResult = DrayfusWagner_cpu(*graph, numberOfTerminals, terminals, apsp);
  
  stopTime(&timer);
  printElapsedTime(timer, "CPU time", CYAN);
  
  printf("Running GPU version\n");
  // startTime(&timer);
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
  free(DP);

  CsrGraph* graph_opt1_d = createEmptyCSRGraphOnGPU(graph->num_nodes, graph->num_edges);
  copyCSRGraphToGPU(graph, graph_opt1_d);
  cudaDeviceSynchronize();



  unsigned int* DP_opt1 = (unsigned int*) malloc(sizeof(unsigned int) * graph->num_nodes *  ((1 << numberOfTerminals) - 1) );
  startTime(&timer);
  DrayfusWagnerGPU_o1(graph, graph_opt1_d, numberOfTerminals, terminals, DP_opt1, apsp);
  stopTime(&timer);
  printElapsedTime(timer, "GPU opt1 time", CYAN);
  verifyFlippedDP(cpuResult, DP_opt1, graph->num_nodes, ((1 << numberOfTerminals) - 1));
  free(DP_opt1);

  CsrGraph* graph_opt2_d = createEmptyCSRGraphOnGPU(graph->num_nodes, graph->num_edges);
  copyCSRGraphToGPU(graph, graph_opt2_d);
  cudaDeviceSynchronize(); 
  unsigned int* DP_opt2 = (unsigned int*) malloc(sizeof(unsigned int) * graph->num_nodes *  ((1 << numberOfTerminals) - 1) );
  
  startTime(&timer);
  DrayfusWagnerGPU_o2(graph, graph_opt2_d, numberOfTerminals, terminals, DP_opt2, apsp);
  stopTime(&timer);
  printElapsedTime(timer, "GPU opt2 time", CYAN);
  verifyFlippedDP(cpuResult, DP_opt2, graph->num_nodes, ((1 << numberOfTerminals) - 1));


  free(DP_opt2);
  free(apsp);
  free(cpuResult);
}
