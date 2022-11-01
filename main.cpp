#include "DrayfusWagner.h"

#include <iostream>
int main() {
    //Testing on a graph of 7 vertices
    unsigned int numberOfVertices = 7;   
    unsigned int numberOfTeriminals = 3;
    unsigned int values[] =  {4, 8, 4, 2, 3, 8, 5, 7, 2, 6, 3, 5, 1, 1, 7, 1, 6, 1};
    unsigned int col[] =     {1, 2, 0, 3, 4, 0, 4, 5, 1, 6, 1, 2, 5, 6, 2, 4, 3, 4};
    unsigned int rowPtr[] =  {0, 2, 5, 8, 10, 14, 16, 18};
    unsigned int numberOfTerminals = 3;
    unsigned int terminals[] {2, 3, 5};
    unsigned int bitTerminals[] {1,1,1};

    // Testing on a graph of 10 vertices
    // unsigned int numberOfVertices = 10;   
    // unsigned int numberOfTeriminals = 3;
    // unsigned int values[] =  {4, 8, 4, 2, 3, 8, 5, 7, 2, 6, 3, 5, 1, 1, 3, 7, 1, 2, 4, 6, 1, 3, 3, 2, 4, 3};
    // unsigned int col[] =     {1, 2, 0, 3, 4, 0, 4, 5, 1, 6, 1, 2, 5, 6, 7, 2, 4, 7, 8, 3, 4, 9, 4, 5, 5, 6};
    // unsigned int rowPtr[] =  {0, 2, 5, 8, 10, 15, 19, 22, 24, 25, 26};
    // unsigned int terminals[] {7, 8, 9};
    // unsigned int bitTerminals[] {1,1,1};
    // unsigned int numberOfTerminals = 3;

    CsrGraph graph {
        numberOfVertices,
        numberOfTeriminals,
        rowPtr,
        col,
        values
    };


    std::cout << "starting\n";
    unsigned int* result = DrayfusWagner(graph, bitTerminals, numberOfTerminals, terminals);
    std::cout << "finished\n";
    // Print the result
    for(unsigned int vertex = 0; vertex < numberOfVertices; vertex++) {
        for(unsigned int subset = 0; subset < (1 << numberOfTerminals) -1; subset++) {
            std::cout << result[vertex * ((1 << numberOfTerminals )-1) + subset] << "\t";
        }
        std ::cout << std::endl;
    }
}
