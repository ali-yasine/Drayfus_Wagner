// T : # of terminal vertices in the query
// V : vertices

// Pre-compute APSP to get DP[root, subset] for all singleton subsets

// for(k = 2:T)
    //     for(every subset s of size k)
    //         for(every root vertex r in V)
//             // find DP[r, s]
//             for(every subset ss of s)
//                 for(every vertex v)
//                     DP[r, s] min= DP[v, ss] + DP[v, s / ss] + dist(r, v)

#include<iostream>
#include<climits>
using namespace std;

int miniDist(int distance[], bool Tset[]) // finding minimum distance
{
    int minimum=INT_MAX,ind;
              
    for(int k=0;k<6;k++) 
    {
        if(Tset[k]==false && distance[k]<=minimum)      
        {
            minimum=distance[k];
            ind=k;
        }
    }
    return ind;
}

void DijkstraAlgo(int graph[6][6],int src) // adjacency matrix 
{
    int distance[6]; // // array to calculate the minimum distance for each node                             
    bool Tset[6];// boolean array to mark visited and unvisited for each node
    
     
    for(int k = 0; k<6; k++)
    {
        distance[k] = INT_MAX;
        Tset[k] = false;    
    }
    
    distance[src] = 0;   // Source vertex distance is set 0               
    
    for(int k = 0; k<6; k++)                           
    {
        int m=miniDist(distance,Tset); 
        Tset[m]=true;
        for(int k = 0; k<6; k++)                  
        {
            // updating the distance of neighbouring vertex
            if(!Tset[k] && graph[m][k] && distance[m]!=INT_MAX && distance[m]+graph[m][k]<distance[k])
                distance[k]=distance[m]+graph[m][k];
        }
    }
    cout<<"Vertex\t\tDistance from source vertex"<<endl;
    for(int k = 0; k<6; k++)                      
    { 
        char str=65+k; 
        cout<<str<<"\t\t\t"<<distance[k]<<endl;
    }
}

int main()
{
    // int graph[6][6]={
    //     {0, 1, 2, 0, 0, 0},//A
    //     {1, 0, 0, 5, 1, 0},//B
    //     {2, 0, 0, 2, 3, 0},//C
    //     {0, 5, 2, 0, 2, 2},//D
    //     {0, 1, 3, 2, 0, 1},//E
    //     {0, 0, 0, 2, 1, 0}};
    // DijkstraAlgo(graph,0);
        

    // int terminals[3] = {1, 4, 6};

    cout<< "hello";


    return 0;
}


//      1   4   6   1,4   1,6   4,6   1,4,6
//  0   
//  1
//  2
//  3
//  4
//  5