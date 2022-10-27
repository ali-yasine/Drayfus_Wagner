using namespace std;
#include <iostream>
#define nV 10

int getValueFromCSR(int * values, int * col, int * rowPtr, int rowIndex, int colIndex){
    for(int z = rowPtr[rowIndex]; z < rowPtr[rowIndex+1]; z++){
        // cout << "j = " << j << " z = " << z << " col[z] = " << col[z] << endl;
        if(col[z] == colIndex){
            return values[z];
        // cout << "DONE: j = " << j << " z = " << z << " answer: " << answer <<endl;
        }
    }
    return 0;
}

int getIndexFromCSR(int * values, int * col, int * rowPtr, int rowIndex, int colIndex){
    for(int z = rowPtr[rowIndex]; z < rowPtr[rowIndex+1]; z++){
        // cout << "j = " << j << " z = " << z << " col[z] = " << col[z] << endl;
        if(col[z] == colIndex){
        return values[z];
        // cout << "DONE: j = " << j << " z = " << z << " answer: " << answer <<endl;
        }
    }
    return -1;
}

int **floydWarshallUsingCSR(int * values, int * col, int * rowPtr)
{
    // int matrix[nV][nV], 
    int i, j, k;
    int **d = new int *[nV];

    for (i = 0; i < nV; i++)
    {
        d[i] = new int[nV];
        for (j = 0; j < nV; j++)
        {
            int value = getValueFromCSR(values, col, rowPtr, i, j);
            // cout << getValueFromCSR(values, col, rowPtr, 4, 2) << endl;
            if(i == j){
                d[i][j] = 0;
            }
            else if(value != 0){
                d[i][j] = value;
                // cout << "true" << endl;
            }
            else{
                d[i][j] = 999;
            }
            // d[i] = new int[nV];
        }
    }
        

    // Adding vertices individually
    // for (k = 0; k < nV; k++)
    // {
    //     for (i = 0; i < nV; i++)
    //     {
    //         for (j = 0; j < nV; j++)
    //         {
    //             // if (matrix[i][k] + matrix[k][j] < matrix[i][j])
    //             //     matrix[i][j] = matrix[i][k] + matrix[k][j];
    //             // d[i][j] = matrix[i][j];
    //             // printf("%4d", i+k);
    //             if (values[i+rowPtr[k]] + values[k+rowPtr[j]] < values[i+rowPtr[j]])
    //                 d[i][j] = values[i+rowPtr[k]] + values[k+rowPtr[j]];
    //             //  = matrix[i][j];
    //         }
    //     }
    // }
    for (k = 0; k < nV; k++)
    {

        for (i = 0; i < nV; i++)
            {
                for (j = 0; j < i; j++)
                {

                    int pathIJ = getValueFromCSR(values, col, rowPtr, i, j);
                    if(d[i][k] + d[k][j] < d[i][j])
                    {
                        d[i][j] = d[i][k] + d[k][j] ;//+ values[k+rowPtr[j]];
                        d[j][i] = d[i][k] + d[k][j] ;
                        //d[i][j] = answerJ
                        // cout << "true" << endl;
                    }
                    // else{
                        // d[i][j] = pathIJ;
                        // cout << "false" << endl;
                    // }
//
                    // if(i == j){
                    //     cout << d[i][k] + d[k][j] << endl;
                    //     cout << d[i][j] << " done"<<endl;
                    // }
                    // if (matrix[i][k] + matrix[k][j] < matrix[i][j])
                    //     matrix[i][j] = matrix[i][k] + matrix[k][j];
                    // d[i][j] = matrix[i][j];
                    // printf("%4d", i+k);
                    // if (values[i+rowPtr[k]] + values[k+rowPtr[j]] < values[i+rowPtr[j]])
                    // int answerKJ = 999;
                    // // cout << "j = " << j << " rowPtr[i] = " << rowPtr[i] << " rowPtr[i+1] = " << rowPtr[i+1] << endl;
                    // for(int z = rowPtr[k]; z < rowPtr[k+1]; z++){
                    //     // cout << "j = " << j << " z = " << z << " col[z] = " << col[z] << endl;
                    //     if(col[z] == j){
                    //         answerKJ = values[z];
                    //         // cout << "DONE: j = " << j << " z = " << z << " answer: " << answer <<endl;
                    //     }
                    // }
                    // cout << "answerKJ\t" << answerKJ<< " \t";
                    // int answerIK = 999;
                    // // cout << "j = " << j << " rowPtr[i] = " << rowPtr[i] << " rowPtr[i+1] = " << rowPtr[i+1] << endl;
                    // for(int z = rowPtr[i]; z < rowPtr[i+1]; z++){
                    //     // cout << "j = " << j << " z = " << z << " col[z] = " << col[z] << endl;
                    //     if(col[z] == k){
                    //         answerIK = values[z];
                    //         // cout << "DONE: j = " << j << " z = " << z << " answer: " << answer <<endl;
                    //     }
                    // }
                    // 
                    // cout << "answerIK\t" << answerIK << " \t";
                    // int answerIJ = 999;
                    // // cout << "j = " << j << " rowPtr[i] = " << rowPtr[i] << " rowPtr[i+1] = " << rowPtr[i+1] << endl;
                    // for(int z = rowPtr[i]; z < rowPtr[i+1]; z++){
                    //     // cout << "j = " << j << " z = " << z << " col[z] = " << col[z] << endl;
                    //     if(col[z] == j){
                    //         answerIJ = values[z];
                    //         // cout << "DONE: j = " << j << " z = " << z << " answer: " << answer <<endl;
                    //     }
                    // }
                    // cout << "answerIJ\t" << answerIJ << endl;
// 
                    // cout << "k: " << k << " i: " << i << " j: " << j << endl;
// 
                    // int indexIJ = getValueFromCSR(values, col, rowPtr, i, j);
                    // int indexIK = getValueFromCSR(values, col, rowPtr, i, k);
                    // int indexKJ = getValueFromCSR(values, col, rowPtr, k, j);
                    // // if(i != j ){//|| j != k || k!= i){
                    //     // cout << "SAVED: " << answer << endl;
                    //     if(indexIJ != 0){
                    //         cout << "its not -1" << endl;
                    //         // cout << indexIK << "\t" << indexKJ << endl;
                            // if( indexIK != 0 && indexKJ != 0 && (d[i][k] + d[k][j] < d[i][j]))
                            // {
                            //         d[i][j] = indexIK+ indexKJ ;//+ values[k+rowPtr[j]];
                            //     //d[i][j] = answerJ
                            //     cout << "true" << endl;
                            // }
                            // else{
                            //     d[i][j] = indexIJ;
                            //     cout << "false" << endl;
                            // }
                    //     }
                    //     else{
                    //         d[i][j] = 999;
                    //     }
                    //     int indexIJ = getValueFromCSR(values, col, rowPtr, i, j);
                    // int indexIK = getValueFromCSR(values, col, rowPtr, i, k);
                    // int indexKJ = getValueFromCSR(values, col, rowPtr, k, j);
                    // // if(i != j ){//|| j != k || k!= i){
                    //     // cout << "SAVED: " << answer << endl;
                    //     if(indexIJ != -1){
                    //         cout << "its not -1" << endl;
                    //         cout << indexIK << "\t" << indexKJ << endl;
                    //         if(indexIK != -1 && indexKJ != -1 && (values[indexIK]+ values[indexIK] < values[indexIJ]))
                    //         {
                    //                 d[i][j] = values[indexIK]+ values[indexIK] ;//+ values[k+rowPtr[j]];
                    //             //d[i][j] = answerJ
                    //             cout << "true" << endl;
                    //         }
                    //         else{
                    //             d[i][j] = values[indexIJ];
                    //             cout << "false" << endl;
                    //         }
                    //     }
                    //     else{
                    //         d[i][j] = 999;
                    //     }
                    // }
                    // else{
                    //     d[i][j] = 0;
                    // }
                    // int answerKJ = 999;
                    // // cout << "j = " << j << " rowPtr[i] = " << rowPtr[i] << " rowPtr[i+1] = " << rowPtr[i+1] << endl;
                    // for(int z = rowPtr[k]; z < rowPtr[k+1]; z++){
                    //     // cout << "j = " << j << " z = " << z << " col[z] = " << col[z] << endl;
                    //     if(col[z] == j){
                    //         answerKJ = values[z];
                    //         // cout << "DONE: j = " << j << " z = " << z << " answer: " << answer <<endl;
                    //     }
                    // }
                    // // cout << "answerKJ\t" << answerKJ<< " \t";
                    // int answerIK = 999;
                    // // cout << "j = " << j << " rowPtr[i] = " << rowPtr[i] << " rowPtr[i+1] = " << rowPtr[i+1] << endl;
                    // for(int z = rowPtr[i]; z < rowPtr[i+1]; z++){
                    //     // cout << "j = " << j << " z = " << z << " col[z] = " << col[z] << endl;
                    //     if(col[z] == k){
                    //         answerIK = values[z];
                    //         // cout << "DONE: j = " << j << " z = " << z << " answer: " << answer <<endl;
                    //     }
                    // }
                    // 
                    // // cout << "answerIK\t" << answerIK << " \t";
                    // int answerIJ = 999;
                    // // cout << "j = " << j << " rowPtr[i] = " << rowPtr[i] << " rowPtr[i+1] = " << rowPtr[i+1] << endl;
                    // for(int z = rowPtr[j]; z < rowPtr[j+1]; z++){
                    //     // cout << "j = " << j << " z = " << z << " col[z] = " << col[z] << endl;
                    //     if(col[z] == i){
                    //         answerIJ = values[z];
                    //         // cout << "DONE: j = " << j << " z = " << z << " answer: " << answer <<endl;
                    //     }
                    // }
                    // cout << "answerIJ\t" << answerIJ << endl;
                    // if(i != j){
                    //     // cout << "SAVED: " << answer << endl;
                    //     if(answerKJ+ answerIK < answerIJ)
                    //     {
                    //             d[i][j] = answerKJ+ answerIK ;//+ values[k+rowPtr[j]];
                    //         //d[i][j] = answerJ
                    //     }
                    //     else{
                    //         d[i][j] = answerIJ;
                    //     }
                    // }
                    // else{
                    //     d[i][j] = 0;
                    // }
                    //  = matrix[i][j];
                }
            }
    }
    // printMatrix(matrix);

    return d;
}



int main(){
    int values[] =  {4, 8, 4, 2, 3, 8, 5, 7, 2, 6, 3, 5, 1, 1, 3, 7, 1, 2, 4, 6, 1, 3, 3, 2, 4, 3};//{0, 4, 8, 4, 0, 2, 3, 8, 0, 5, 7, 2, 0, 6, 3, 5, 0, 1, 1, 3, 7, 1, 0, 2, 4, 6, 1, 0, 3, 3, 2, 0, 4, 0, 3, 0};
    int col[] =     {1, 2, 0, 3, 4, 0, 4, 5, 1, 6, 1, 2, 5, 6, 7, 2, 4, 7, 8, 3, 4, 9, 4, 5, 5, 6};//{0, 1, 2, 0, 1, 3, 4, 0, 2, 4, 5, 1, 3, 6, 1, 2, 4, 5, 6, 7, 2, 4, 5, 7, 8, 3, 4, 6, 9, 4, 5, 7, 5, 8, 6, 9};
    int rowPtr[] =  {0, 2, 5, 8, 10, 15, 19, 22, 24, 25, 26};//{0, 3, 7, 11, 14, 20, 25, 29, 32, 34, 36};
    int ** d = floydWarshallUsingCSR(values, col, rowPtr);

    for (int i = 0; i < nV; i++)
    {
        for (int j = 0; j < nV; j++)
        {
            printf("%4d", d[i][j]);
        }
        printf("\n");
    }

    cout << getValueFromCSR(values, col, rowPtr, 4, 2) << endl;
    cout << getValueFromCSR(values, col, rowPtr, 2, 4) << endl;

    // for (int i = 0; i < nV; i++)
    // {
    //     for (int j = 0; j < nV; j++)
    //     {
    //         int x =getValueFromCSR(values, col, rowPtr, i, j);
    //         int y =getValueFromCSR(values, col, rowPtr, j, i);
    //         cout <<  i << "\t" <<  j << "\t" << x << "\t" << y << "\t" ;
    //         if(x == y){
    //             cout << "true" << endl;
    //         }
    //         else{
    //             return 0;
    //         }

    //     }
    //     // printf("\n");
    // }
}