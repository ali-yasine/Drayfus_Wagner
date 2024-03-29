#ifndef __UTIL_H_
#define __UTIL_H_

__host__ __device__ static unsigned int binaryToDecimal(unsigned int* arr, unsigned int size) {
    unsigned int result = 0;
    for (unsigned int i = 0; i < size; ++i) {
        result += arr[i] * (1 << i);
    }
    return result;
}
__host__ static unsigned int* decimalToBinary(unsigned int decimal, unsigned int binarySize) {
    
    unsigned int* result = (unsigned int*) calloc(binarySize, sizeof(unsigned int));

    for (unsigned int i = 0; i < binarySize; ++i) {
        result[i] = (decimal >> i) & 1;
    }
    return result;
}
__device__ static unsigned int* decimalToBinaryGPU(unsigned int decimal, unsigned int binarySize) {
    
    unsigned int* result;

    cudaMalloc(&result, binarySize * sizeof(unsigned int));
    for (unsigned int i = 0; i < binarySize; ++i) {
        result[i] = (decimal >> i) & 1;
    }
    return result;
}
__device__ static void decimalToBinaryGPUO3(unsigned int decimal, unsigned int binarySize, unsigned int* result) {
    
    for (unsigned int i = 0; i < binarySize; ++i) {
        result[i] = (decimal >> i) & 1;
    }
}

__device__ __host__ static unsigned int choose(unsigned int n, unsigned int k){
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

__host__ __device__ static unsigned int* setDifference(unsigned int* set1, unsigned int* set2, unsigned int size){
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
__device__ static void setDifferenceGPU(unsigned int* set1, unsigned int* set2, unsigned int size, unsigned int* set1MinusSet2){

    for (unsigned int i = 0; i < size; i++){
        if (set1[i] == 1 && set2[i] == 0){
            set1MinusSet2[i] = 1;
        }
        else {
            set1MinusSet2[i] = 0;
        }
    }
}

__device__ __host__ static  bool contains(unsigned int* set, unsigned int N, unsigned int vertex) {
    for (unsigned int i = 0; i < N; ++i) {
        if (set[i] == vertex)
            return true;
    }
    return false;
}

__device__ __host__ static bool equals(unsigned int* set1, unsigned int* set2, unsigned int size) {
    for (unsigned int i = 0; i < size; ++i) {
        if (set1[i] != set2[i])
            return false;
    }
    return true;
}
#endif
