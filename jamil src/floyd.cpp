// Floyd-Warshall Algorithm in C++

#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <iterator>
#include "C:\Users\JamilMahmoud\Desktop\Thesis\Subset.cpp"
using namespace std;

// defining the number of vertices
#define nV 10
// defining the number of terminals
#define nT 3

#define INF 999


void printMatrix(int matrix[][nV]);

// Implementing floyd warshall algorithm
int **floydWarshall(int graph[][nV], int src, int dest)
{
    int matrix[nV][nV], i, j, k;
    int **d = new int *[nV];

    for (i = 0; i < nV; i++)
        for (j = 0; j < nV; j++)
        {
            matrix[i][j] = graph[i][j];
            d[i] = new int[nV];
        }

    // Adding vertices individually
    for (k = 0; k < 2; k++)
    {
        for (i = 0; i < nV; i++)
        {
            for (j = 0; j < nV; j++)
            {
                if (matrix[i][k] + matrix[k][j] < matrix[i][j])
                    matrix[i][j] = matrix[i][k] + matrix[k][j];
                d[i][j] = matrix[i][j];
            }
        }
    }

    printMatrix(matrix);

    return d;
}

int **floydWarshallUsingCSR(int * values, int * col, int * rowPtr)
{
    // int matrix[nV][nV], 
    int i, j, k;
    int **d = new int *[nV];

    for (i = 0; i < nV; i++)
    {
        d[i] = new int[nV];
    }
        // for (j = 0; j < nV; j++)
        // {
        //     matrix[i][j] = graph[i][j];
        //     // d[i] = new int[nV];
        // }

    // Adding vertices individually
    for (k = 0; k < nV; k++)
    {
        for (i = 0; i < nV; i++)
        {
            for (j = 0; j < nV; j++)
            {
                // if (matrix[i][k] + matrix[k][j] < matrix[i][j])
                //     matrix[i][j] = matrix[i][k] + matrix[k][j];
                // d[i][j] = matrix[i][j];
                if (values[i+k] + values[k+j] < values[i+j])
                    d[i][j] = values[i+k] + values[k+j];
                //  = matrix[i][j];
            }
        }
    }

    // printMatrix(matrix);

    return d;
}

void printMatrix(int matrix[][nV])
{
    for (int i = 0; i < nV; i++)
    {
        for (int j = 0; j < nV; j++)
        {
            if (matrix[i][j] == INF)
                printf("%4s", "INF");
            else
                printf("%4d", matrix[i][j]);
        }
        printf("\n");
    }
}


vector<vector<int>> subsets(const vector<int> &set)
{
    // Output
    vector<vector<int>> ss;
    // If empty set, return set containing empty set
    if (set.empty())
    {
        // ss.push_back(set);
        return ss;
    }

    // If only one element, return itself and empty set
    if (set.size() == 1)
    {
        // vector<int> empty;
        // ss.push_back(empty);
        ss.push_back(set);
        return ss;
    }

    // Otherwise, get all but last element
    vector<int> allbutlast;
    for (unsigned int i = 0; i < (set.size() - 1); i++)
    {
        allbutlast.push_back(set[i]);
    }
    // Get subsets of set formed by excluding the last element of the input set
    vector<vector<int>> ssallbutlast = subsets(allbutlast);
    // First add these sets to the output
    for (unsigned int i = 0; i < ssallbutlast.size(); i++)
    {
        ss.push_back(ssallbutlast[i]);
    }
    // Now add to each set in ssallbutlast the last element of the input
    for (unsigned int i = 0; i < ssallbutlast.size(); i++)
    {
        ssallbutlast[i].push_back(set[set.size() - 1]);
    }
    // Add these new sets to the output
    for (unsigned int i = 0; i < ssallbutlast.size(); i++)
    {
        ss.push_back(ssallbutlast[i]);
    }

    return ss;
}

int findIndex(vector<int> subset, map<int, int> m){
    int sum = 0;
    for(int i = 0; i<subset.size(); i++){
        sum += subset[i];
    }

    return m.find(sum)->second;
}

vector<int> remainingElements(vector<int> subset, vector<int> s){
    vector<int> ss;

    for(int i = 0; i< subset.size(); i++)
        if(find(s.begin(), s.end(), subset[i]) == s.end())
            ss.push_back(subset[i]);

    
    return ss;
}

vector<int> rElements(vector<int> subset, vector<int> s, map<int, int> m, vector<vector<int> > res){
    vector<int> ss;

    int sum1 = 0;
    for(int i = 0; i< subset.size(); i++)
        sum1 += subset[i];

    int sum2 = 0;
    for(int i = 0; i< s.size(); i++)
        sum2 += s[i];

    int answer = sum1 - sum2;
    ss = res[m.find(answer)->second];
    return ss;
}

void printDP(vector<int> vertices, vector<vector<int> > res, int ** dp){
    cout << "DP :" <<endl;
    for (int i = 0; i < vertices.size(); i++) {
        cout << i << "\t";
        for (int x = 0; x < res.size(); x++) {
            if(dp[i][x] != 999)
                cout << dp[i][x] << "\t";
            else
                cout << 0 << "\t";
        }
        cout << endl;
    }
}

int main()
{
    //   int graph[nV][nV] = {{0, 3, INF, 5},
    //              {2, 0, INF, 4},
    //              {INF, 1, 0, INF},
    //              {INF, INF, 2, 0}};
    //   floydWarshall(graph);
    //
    //   int graphs[nV][nV]={
    //         {0, 10, 3, INF, INF},//A
    //         {INF, 0, 1, 2, INF},//B
    //         {INF, 4, 0, 8, 2},//C
    //         {INF, INF, INF, 0, 7},//D
    //         {INF, INF, INF, 9, 0}//E
    //         };

    Subset subsetInstance;
    int graph[nV][nV] =
        {
            {0, 4, 8, INF, INF, INF, INF, INF, INF, INF},   // 0
            {4, 0, INF, 2, 3, INF, INF, INF, INF, INF},     // 1
            {8, INF, 0, INF, 5, 7, INF, INF, INF, INF},     // 2
            {INF, 2, INF, 0, INF, INF, 6, INF, INF, INF},   // 3
            {INF, 3, 5, INF, 0, 1, 1, 3, INF, INF},         // 4
            {INF, INF, 7, INF, 1, 0, INF, 2, 4, INF},       // 5
            {INF, INF, INF, 6, 1, INF, 0, INF, INF, 3},     // 6
            {INF, INF, INF, INF, 3, 2, INF, 0, INF, INF},   // 7
            {INF, INF, INF, INF, INF, 4, INF, INF, 0, INF}, // 8
            {INF, INF, INF, INF, INF, INF, 3, INF, INF, 0}  // 9

        };

    int **d = floydWarshall(graph, 0, 3);



    vector<int> vertices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    vector<int> terminals = {7, 8, 9};

    vector<vector<int> > res = subsetInstance.subsets(terminals);
    subsetInstance.print(res);

    map<int, int> subsetToIndex;

    int index = 0;
    cout << "MAPPPPPP" << endl;
    for (int i = 0; i < res.size(); i++){
        int sum = 0;
        for(int j = 0; j < res[i].size(); j++){
            sum += res[i][j];
        }
        cout << sum << "\t" << index << endl;
        subsetToIndex.insert(pair<int, int>(sum, index));
        index++;
    }
    cout << "**********************" << endl;
    const int subsetSize = res.size();


    int ** dp = new int*[nV];

    for (int i = 0; i < vertices.size(); i++) 
        dp[i] = new int[subsetSize];

    for (int i = 0; i < vertices.size(); i++) {
        for (int x = 0; x < res.size(); x++) {
            if(res[x].size() == 1){
                dp[i][x] = d[i][res[x][0]];
            }
            else{
                dp[i][x] = 999;
            }
        }
    }


    printDP(vertices, res, dp);
    cout << "**********************" << endl;


     for (int i = 2; i <= terminals.size(); i++) {                                                  // for(k = 2:T)
         for (int s = 0; s < res.size(); s++) {
            if(res[s].size() == i){                                                                 //     for(every subset s of size k)

                for(int r = 0; r < vertices.size(); r++){                                           //         for(every root vertex r in V)

                    if(!(find(terminals.begin(), terminals.end(), vertices[r]) != terminals.end())){

                        vector<vector<int> > subset = subsetInstance.subsets(res[s]);
                        for(int ss = 0; ss < subset.size(); ss++){                                      //             for(every subset ss of s)
                            
                            for(int v = 0; v < vertices.size(); v++){            
                                if(!(find(terminals.begin(), terminals.end(), vertices[r]) != terminals.end()) && !(find(terminals.begin(), terminals.end(), vertices[v]) != terminals.end())){                       //                 for(every vertex v)
                                    int indexofS = findIndex(res[s], subsetToIndex);
                                    int indexofSS = findIndex(subset[ss], subsetToIndex);
                                    
                                    vector<int> sMinusSS = remainingElements(res[s], subset[ss]);
                                    int indexOfSMinsusSS = findIndex(sMinusSS, subsetToIndex);
                                    int result = dp[v][indexofSS] + dp[v][indexOfSMinsusSS] + d[r][v];
                                    if(dp[r][indexofS] > result) //dp[r][indexofS] == 0 && 
                                        dp[r][indexofS] = dp[v][indexofSS] + dp[v][indexOfSMinsusSS] + d[r][v]; //                     DP[r, s] min= DP[v, ss] + DP[v, s - ss] + dist(r, v)
                                }
                            }
                        }
                    }
                    
                }
                
            }
        }
     }

    


    printDP(vertices, res, dp);


    vector<int> v = rElements(vertices, {7,8}, subsetToIndex, res);

    vector<int> v2 = remainingElements(vertices, {7,8});

    for(int i = 0; i< v.size(); i++)
        cout<< v[i] << "\t";

    cout<<endl;

    
    for(int i = 0; i< v2.size(); i++)
        cout<< v2[i] << "\t";
}