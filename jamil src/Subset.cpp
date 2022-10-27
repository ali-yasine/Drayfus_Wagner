#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

class Subset
{
    private:
    void subsetsUtil(vector<int> &A, vector<vector<int>> &res,
                     vector<int> &subset, int index)
    {
        if (subset.size() != 0) // We dont need the empty set in out application
            res.push_back(subset);
        // Loop to choose from different elements present
        // after the current index 'index'
        for (int i = index; i < A.size(); i++)
        {

            // include the A[i] in subset.
            subset.push_back(A[i]);

            // move onto the next element.
            subsetsUtil(A, res, subset, i + 1);

            // exclude the A[i] from subset and triggers
            // backtracking.
            subset.pop_back();
        }

        return;
    }

    // void subsetsUtilArray(int a[], int** res,
    //                  int* subset, int index)
    // {
    //     if (sizeof(subset)/sizeof(subset[0]) != 0) // We dont need the empty set in out application
    //         res.push_back(subset);
    //     // Loop to choose from different elements present
    //     // after the current index 'index'
    //     for (int i = index; i < a.size(); i++)
    //     {

    //         // include the A[i] in subset.
    //         subset.push_back(A[i]);

    //         // move onto the next element.
    //         subsetsUtil(A, res, subset, i + 1);

    //         // exclude the A[i] from subset and triggers
    //         // backtracking.
    //         subset.pop_back();
    //     }

    //     return;
    // }

    public:
    Subset(){}
    // below function returns the subsets of vector A.
    vector<vector<int>> subsets(vector<int> &A)
    {
        vector<int> subset;
        vector<vector<int>> res;

        // keeps track of current element in vector A
        // and the number of elements present in the array subset
        int index = 0;
        subsetsUtil(A, res, subset, index);

        return res;
    }

    int** subsetsArray(int a[])
    {
        int** response ;
        // vector<int> subset;
        // vector<vector<int>> res;

        // // keeps track of current element in vector A
        // // and the number of elements present in the array subset
        // int index = 0;
        // subsetsUtil(A, res, subset, index);

        return response;
    }

    void PrintAllSubsets(int *arr, int i, int n,int *subset, int j, int **result, int &k){    
    // checking if all elements of the array are traverse or not
    if(i==n){
        // print the subset array
        int idx = 0;
        cout << "**********************" << n<< endl;
        result[k] = (int*)calloc(j, sizeof(int));
        // for(int i = 0; i < j; i++){
        //     result[k][idx] = 0;
        // }
        cout << sizeof(int) * j << endl;
        cout << k << "\t"<< j <<   endl;
        cout << sizeof(result[k])/ sizeof(int) << endl;
        while(idx<j){
            // cout<<subset[idx]<<' ';
            if(j != 0)
                result[k][idx] = subset[idx];
            // cout<<result[k][idx]<<' ';
            ++idx;
        }
        cout << "**********************" << endl;
        cout << endl;
        
        // result[k] = subset;
        k = k +1;
        return ;
    }
    // for each index i, we have 2 options
    // case 1: i is not included in the subset
    // in this case simply increment i and move ahead
    PrintAllSubsets(arr,i+1,n,subset,j, result, k);
    // case 2: i is included in the subset
    // insert arr[i] at the end of subset
    // increment i and j
    subset[j] = arr[i];

    PrintAllSubsets(arr,i+1,n,subset,j+1, result , k);
        
}

    void print( vector<vector<int> > res){
        for (int i = 0; i < res.size(); i++) {
            for (int j = 0; j < res[i].size(); j++){
                std::cout << res[i][j] << " ";
            }
            cout << endl;
        }
    }
};


void generateSubsets(int *array, int size,int ** output, int *k){
    if (size==0){
        output[*k]=(int*) calloc(1,sizeof(int));    // allocating for the empty array 
        output[*k][0]=0;                            // adding the size of the empty set, size 0
        ++(*k);                                     // incrementing k
        cout<<"EmptySet"<<endl;
        return;
    }
    generateSubsets(array,size-1,output, k);        // recursion call
    int k1=*k-1;                                    // k1 is for iterating from k till 0
    while(k1>=0){
        int newSize=output[k1][0]+1;                // getting new Size for allocating for the array
        output[*k]=(int*)calloc(newSize+1,sizeof(int)); // allocating memory
        output[*k][0]=newSize;                          // inserting the size in the begining
        cout<<newSize<<": ";                     
        for(int i=1;i<newSize;++i){                     // inserting the elements 
            output[*k][i]=output[k1][i];
            cout<<output[*k][i]<<' ';
        }
        output[*k][newSize]=array[size-1];
        cout<<output[*k][newSize]<<endl;
        ++(*k);
        --k1;
    }
}

int* addTwoArrays(int arr1[], int arr2[], int size){
    int* result = (int *) calloc(size, sizeof(int));
    // cout << "START ADD\t";
    for(int i = 0; i < size; i++){
        
        if(arr1[i] + arr2[i] >= 1){
            result[i] = 1;
        }
        else{
            result[i] = 0;
        }
        // cout << arr1[i] << "\t"<< arr2[i] << "\t" << result[i] << "\t" << endl;
        
    }
    // cout << "END ADD"<<endl;
    return result;
}

void subsetBitVector(int *array, int size, int index, int ** output, int* k, int n, int resultSize){
    // cout << *k << endl;
    if(index == resultSize){
        cout << index << "\t result size:"<< resultSize<< endl;
        for(int x = 0; x < n; x++){
            output[x]=(int*) calloc(size,sizeof(int));
            // (*k)++;
        }
            
        for(int i = 0; i < size; i++){
            // output[*k]=(int*) calloc(size,sizeof(int));
            if(array[i] == 1)
            {
                for(int j = 0; j < size; j++){
                    if( i == j){
                        output[*k][j] = array[i];
                    }
                    else{
                        output[*k][j] = 0;
                    }
                    // cout<< output[*k][j] << " \t";
                }
                // cout<< endl;
                (*k)++;
            }
            cout<< *k << endl; 
        }
       
       return;
    }
    // cout << *k <<"\t\t"<< size<< endl;
    // (*k)++;
    subsetBitVector(array, size, index+1, output, k, n, resultSize);
    // ++(*k);

    int k1=*k-1;                                    // k1 is for iterating from k till 0
    int count = *k;
    while(k1>=0 && *k < resultSize){

        // cout<<"3n3"<<endl;
        for(int i=k1-1;i>=0;i--){ 
            // cout<< i << "\t" << k1<< endl;
            output[*k]=(int*)calloc(size,sizeof(int)); // allocating memory
            output[*k] = addTwoArrays(output[k1] , output[i], size);
            // for(int j = 0; j < size ; j++){
            //     cout << output[*k][j] << " ";
            // }
            // cout << endl;
            // ++(*k);
            cout << *k << endl;
            (*k)++;
        }
        // ++(*k);
        --k1;
    }

}

void ttt(int* array, int index, int n, int ** result, int* k, int * currentArray, int i){
    if( index == n){
        result[*k] = (int*) calloc(n,sizeof(int));
        result[*k] = currentArray;
        cout<< *k << endl;
        for(int j = 0; j < n ; j++){
                cout << currentArray[j] << " ";
        }
        cout<< endl;
        return;
    }
    int* arr1 = (int*) calloc(n,sizeof(int));
    // if(array[i] == 1){
    //     arr1[i] = array[i];
    // }
    arr1[i] = array[i];
    ttt(array, index + 1 , n, result, k, currentArray, i);
    int* arr2 = (int*) calloc(n,sizeof(int));
    for(int j = 0; j < i ; j++){
        arr2[j] = currentArray[j];
        // cout << currentArray[j] << " ";
    }
    (*k)++;
    arr2[i] = array[i];
    // array[i] = 0;

    ttt(array, index + 1, n, result, k, arr2, i+1);

}



// void subsetBitVector(int *array, int size, int index, int ** output, int* k, int n){

//     if(size == 0){
//         // for(int x = 0; x < n; x++){
            
//             for(int i = 0; i < n; i++){
//                 output[*k]=(int*) calloc(size,sizeof(int));
//                 for(int j = 0; j < n; j++){
//                     if( i == j){
//                         output[*k][j] = array[i];
//                     }
//                     else{
//                         output[*k][j] = 0;
//                     }
//                     // cout<< output[*k][j] << " \t";
//                 }
//                 // cout<< endl;
//                 // cout<< *k << endl; 
//                 (*k)++;
//             }
            
//         // }
       
//        return;
//     }
//     // cout << *k <<"\t\t"<< size<< endl;
//     // (*k)++;
//     subsetBitVector(array, size -1, index+1, output, k, n);
//     // ++(*k);

// }

// // Driver Code.
// int main()
// {
//     Subset s ;//= new Subset();
//     // find the subsets of below vector.
//     vector<int> array = { 1, 2, 3 };
 
//     // res will store all subsets.
//     // O(2 ^ (number of elements inside array))
//     // because total number of subsets possible
//     // are O(2^N)
//     // vector<vector<int> > res = s.subsets(array);
 
//     // s.print(res);
//     // Print result
//     // for (int i = 0; i < res.size(); i++) {
//     //     for (int j = 0; j < res[i].size(); j++){
//     //         std::cout << res[i][j] << " ";
//     //     }
//     //     cout << endl;
//     // }

//     int terminals[] = {7, 8, 9};//1, 2, 3, 4, 5, 6, 

//     int arr[] = {1,1,1,1};//{1,2,3,4}; // input array
//     int subset[8];       // temporary array to store subset
//     int numberOfTerminals = 4;
//     int numberOfterminalsSelected = 4;
    
//     // for(int i = 0; i < 8; i++)
//     //     res[i] = new int[8];
//     int size = 1;//pow(2, n); // loop 2 n times
//     for(int i = 0; i < numberOfTerminals ; i++){
//         size = size * 2;
//     }
//     cout<<size<<"size"<<endl;
//     int ** result = new int*[size-1];
//     int k = 0;
//     // s.PrintAllSubsets(terminals,0,n,subset,0, result, k);   

//     // cout << "**********************" << endl;
//     // cout<< result[1][0] << endl;
//     // cout << "**********************" << endl;
    
//     subsetBitVector(arr, numberOfTerminals, 2, result, &k, numberOfterminalsSelected, size);
//     int current[] = {0,0,0};
//     // ttt(arr, 0, numberOfTerminals, result, &k, current ,0);

//     for(int i = 0; i < size -1; i++){
//         for(int j = 0; j <numberOfTerminals; j++){
//             // cout<< sizeof(result[i]) << "\t";
//             // cout << i <<" ";
//             cout<< result[i][j] << "\t";
//         }
//         cout<< endl;
//     }
//     // cout << k;
//     // for(int i = 0; i < 8; i++){
//     //     for(int j = 0; j <= sizeof(result[i])/sizeof(int); j++){
//     //         // cout<< sizeof(result[i]) << "\t";
//     //         // cout << j <<" ";
//     //         cout<< result[i][j] << "\t";
//     //     }
//     // generateSubsets(terminals, n, result, &k);

//     // for(int i = 0; i < size; i++){
//     //     for(int j = 1; j <=result[i][0]; j++){
//     //         // cout<< sizeof(result[i]) << "\t";
//     //         // cout << j <<" ";
//     //         cout<< result[i][j] << "\t";
//     //     }
//     //     cout<< endl;
//     // }
//     //     cout << subset[i] << endl; 
//     return 0;
// }