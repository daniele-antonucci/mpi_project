#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#include <mpi.h>

#define ROOT 0
#define INF 10000
#define min(a, b) (a < b ? a : b)

typedef struct
{
    // Processes & Vertices
    int P, N, q, Ngrid;
    // Process Ids
    int Pid, algPid;
    // process for floyd algo
    int row, col;
} GRID;

/******************************************************************************************
* concatenateStringst: Join 2 strings 
    char *s1 - first string
    char *s2 - second string
******************************************************************************************/
char* concatenateStrings(const char *s1, const char *s2)
{
    char *result = malloc(strlen(s1) + strlen(s2) + 1); 
    if(result == NULL) {
        fprintf(stderr, "Error allocating filename string. ");
        exit(1);
    }
    else {
        strcpy(result, s1);
        strcat(result, s2);
        return result;
    }

}

/******************************************************************************************
* checkSqrt: Checks if its possible to calculate floyd on a NxN grid with P processes
    GRID *process - struct with all processrmation about the graph and MPI Processes
******************************************************************************************/
int checkSqrt(GRID *process)
{
    //The matrix is processable only if i can divide it into grids
    if ((process->q * process->q == (process->P - 1) && process->N % process->q == 0))
        return 1;
    return 0;
}

/******************************************************************************************
* readMatrixFromFile: read the distance graph from a file
    f - File opened
    N - Vertices of the graph
******************************************************************************************/
int *readMatrixFromFile(FILE *f, int N)
{
    int i, j;
    int *A = (int *)malloc(N * N * sizeof(int));

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            fscanf(f, "%d", &A[i * N + j]);
        }
    }
    return (int *)A;
}

/******************************************************************************************
* printMatrix: print the given matrix with N vertices
    matrix - Vector of the graph
    N - Vertices of the graph
******************************************************************************************/
void printMatrix(int *matrix, int N)
{
    int j;
    for (int i = 0; i < N; i++)
    {
        for (j = 0; j < N - 1; j++)
        {
            printf("%d ", matrix[i * N + j]);
        }
        printf("%d\n", matrix[i * N + j]);
    }
    fflush(stdout);
}

/******************************************************************************************
* sendGrid: Sends the corrispective grid of the distance graph to every MPI Processes
          (generates qxq grid)
    matrix - Vector of the graph
    N - Vertices of the graph
    q - sqrt(P)
******************************************************************************************/
void sendGrid(int *matrix, int N, int q)
{
    int Ngrid = N / q;
    int *grid_matrix = malloc(Ngrid * Ngrid * sizeof(int));

    int id = 1;
    for (int k = 0; k < q; k++)
    {
        for (int l = 0; l < q; l++)
        {
            for (int i = 0; i < Ngrid; i++)
            {
                for (int j = 0; j < Ngrid; j++)
                {
                    // The i-th value of the grid is equal to the value of matrix shifted based on q and Ngrid
                    grid_matrix[i * Ngrid + j] = matrix[(i + (k * Ngrid)) * N + j + (l * Ngrid)];
                }
            }
            MPI_Send(grid_matrix, Ngrid * Ngrid, MPI_INT, id, 1, MPI_COMM_WORLD);
            id++;
        }
    }
    free(grid_matrix);
}

/******************************************************************************************
* floydAlgo: Applies the floyd algorithm 
    matrix - Vector of the graph
    pos - Vector of the row/col grid
    res - Vector of the result grid
    Ngrid - Vertices of the grid
******************************************************************************************/
void floydAlgo(int *grid, int *pos, int *res, int Ngrid)
{
    for (int i = 0; i < Ngrid; i++)
    {
        for (int j = 0; j < Ngrid; j++)
        {
            for (int k = 0; k < Ngrid; k++)
            {
                if (res[i * Ngrid + j] > grid[i * Ngrid + k] + pos[k * Ngrid + j])
                    res[i * Ngrid + j] = grid[i * Ngrid + k] + pos[k * Ngrid + j];
            }
        }
    }
}

/******************************************************************************************
* floydWarshall2D: Initiates the communicators and apply the floyd algorithm every steps for each process
    process - Stored values 
    matrix - Starting matrix
    row_grid -
    col_grid -
    result - Result matrix for each process
    Ngrid - Vertices of the grid
******************************************************************************************/
void floydWarshall2D(GRID *process, int *matrix, int Ngrid, MPI_Comm comm, int *solution)
{
    // Communicators
    MPI_Group group, row_group, col_group;
    MPI_Comm row_comm, col_comm;

    // Row and Col values
    int *row_grid = (int *)malloc((Ngrid * Ngrid) * sizeof(int));
    int *col_grid = (int *)malloc((Ngrid * Ngrid) * sizeof(int));
    int *result = (int *)malloc((Ngrid * Ngrid) * sizeof(int));

    // Bit values
    process->row = (process->algPid / process->q);
    process->col = (process->algPid % process->q);

    //Need to associate ranks both for row_group and col_group
    int row_ranks[process->q], col_ranks[process->q];

    for (int p = 0; p < process->q; p++)
    {
        row_ranks[p] = (process->row * process->q) + p;
        col_ranks[p] = ((process->col + p * process->q) % (process->q * process->q));
    }

    MPI_Comm_group(comm, &group);   
    MPI_Group_incl(group, process->q, row_ranks, &row_group);
    MPI_Group_incl(group, process->q, col_ranks, &col_group);
    MPI_Comm_create(comm, row_group, &row_comm);
    MPI_Comm_create(comm, col_group, &col_comm);

    if ((process->algPid / process->q) == (process->algPid % process->q))
        memcpy(row_grid, matrix, (Ngrid * Ngrid) * sizeof(int));

    memcpy(result, matrix, (Ngrid * Ngrid) * sizeof(int));
    int src_rank = (process->row + 1) % process->q;
    int dest_rank = ((process->row - 1) + process->q) % process->q;

    // ln is based on the division of the distance graph
    // more process > more grids > more cycle to determine the solution
    int ln = (Ngrid * process->q) << 1;
    for (int d = 2; d < ln; d = d << 1)
    {
        memcpy(col_grid, matrix, (Ngrid * Ngrid) * sizeof(int));
        for (int step = 0; step < process->q; step++)
        {
            int src = (process->row + step) % process->q;

            if (src == process->col)
            {
                //Comunicating the current matrix, then proceed to update
                MPI_Bcast(matrix, Ngrid * Ngrid, MPI_INT, src, row_comm);
                floydAlgo(matrix, col_grid, result, Ngrid);
            }
            else
            {
                //Comunicating the current matrix, then proceed to update
                MPI_Bcast(row_grid, Ngrid * Ngrid, MPI_INT, src, row_comm);
                floydAlgo(row_grid, col_grid, result, Ngrid);
            }
            if (step < process->q - 1)
                MPI_Sendrecv_replace(col_grid, Ngrid * Ngrid, MPI_INT, dest_rank, 1, src_rank, 1, col_comm, MPI_STATUS_IGNORE);
        }
        // copy the result of operations to the actual grid
        memcpy(matrix, result, (Ngrid * Ngrid) * sizeof(int));
    }
}

/******************************************************************************************
* reorderMatrix: used when we need to create a matrix using multiple grids. It is actually made row-wise 
        (not allocating grid by grid)
    process - Stored values 
    dist - Distance graph not ordered
    sol - Final graph ordered
    Ngrid - Vertices of each grid
******************************************************************************************/
void reorderMatrix(GRID *process, int *dist, int *sol, int Ngrid)
{
    int x, y, pos, actual_x, actual_y;

    for (int i = 0; i < (process->P -1); i++)
    {
        //get coordinates on the matrix
        x = i / process->q;
        y = i % process->q;
        pos = i * Ngrid * Ngrid;

        for (int row = 0; row < Ngrid; row++)
        {
            for (int col = 0; col < Ngrid; col++)
            {
                actual_x = x * Ngrid + row;
                actual_y = y * Ngrid + col;

                if (sol[row * Ngrid + col + pos] == INF)
                    dist[actual_x * process->N + actual_y] = 0;
                else
                    dist[actual_x * process->N + actual_y] = sol[row * Ngrid + col + pos];
            }
        }
    }
}