#pragma once

unsigned int binaryToDecimal(unsigned int* arr, unsigned int size) {
    unsigned int result = 0;
    for (unsigned int i = 0; i < size; ++i) {
        result += arr[i] * (1 << i);
    }
    return result;
}
unsigned int* decimalToBinary(unsigned int decimal, unsigned int binarySize) {
    unsigned int* result = (unsigned int*) calloc(binarySize, sizeof(unsigned int));
    for (unsigned int i = 0; i < binarySize; ++i) {
        result[i] = (decimal >> i) & 1;
    }
    return result;
}
unsigned int choose(unsigned int n, unsigned int k){
    if (k > n)
        return 0;
    if (k * 2 > n)
        k = n-k;
    if (k == 0)
        return 1;
    int result = n;
    for (int i = 2; i <= k; ++i){
        result *= (n-i+1);
        result /= i;
    }
    return result;
}

unsigned int* setDifference(unsigned int* set1, unsigned int* set2, unsigned int size){
    unsigned int* set1MinusSet2 = (unsigned int*) malloc(size * sizeof(unsigned int));
    for (unsigned int i = 0; i < size; i++){
        if (set1[i] == 1 && set2[i] == 0){
            set1MinusSet2[i] = 1;
        }
        else {
            set1MinusSet2[i] = 0;
        }
    }
    return set1MinusSet2;
}

bool contains(unsigned int* set, unsigned int N, unsigned int vertex) {
    for (unsigned int i = 0; i < N; ++i) {
        if (set[i] == vertex)
            return true;
    }
    return false;
}

bool equals(unsigned int* set1, unsigned int* set2, unsigned int size) {
    for (unsigned int i = 0; i < size; ++i) {
        if (set1[i] != set2[i])
            return false;
    }
    return true;
}

void printMatrix(unsigned int* matrix, unsigned int rows, unsigned int cols) {
    for (unsigned int i = 0; i < rows; ++i) {
        for (unsigned int j = 0; j < cols; ++j) {
            printf("%u\t", matrix[i * cols + j]);
        }
        printf("\n");
    }
}