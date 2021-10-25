/*************************************
 Blocked Floyd-Warshall 2D Algorithm

 Daniele Antonucci
*************************************/
#include "functions_mpi.h"

int main(int argc, char *argv[])
{
    // Communication info
    GRID info_process;
    // Matrix of costs
    int *distances, *grid, *solution;
    int *row_grid, *col_grid, *res_grid;
    // Timer values
    double start, end;

    // Initializing communication
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &(info_process.P));
    MPI_Comm_rank(MPI_COMM_WORLD, &(info_process.Pid));

    /* ---------------- ROOT SECTION ---------------- */
    if (info_process.Pid == ROOT)
    {
        if (strlen(argv[1]) == 0)
        {
            fprintf(stderr, "File name is not declared, please execute mpiexec with 1 argument (ex. mpiexec -np 4 ./project_mpi input.txt");
            exit(1);
        }
        else
        {
            char *extension = ".txt";
            char *filename = concatenateStrings(argv[1], extension);

            // Read distance graph from file
            FILE *f;
            f = fopen(filename, "r");

            if (f == NULL)
            {
                fprintf(stderr, "Cannot open file %s\n", filename);
                exit(1);
            }

            // Gets the vertices
            fscanf(f, "%d", &info_process.N);
            info_process.q = sqrt(info_process.P);

            // Checking if possible to divide the matrix with P processes
            if (!checkSqrt(&info_process))
            {
                fprintf(stderr, "Matrix of size %d cannot be solved with %d processes.\nAborting...\n", info_process.N, info_process.P);
                MPI_Abort(MPI_COMM_WORLD, ROOT);
                exit(1);
            }

            // Read the matrix and close the file
            distances = readMatrixFromFile(f, info_process.N);
            fclose(f);

            printf("ROOT - Distance graph: \n");
            printMatrix(distances, info_process.N);
            printf("-----------------------------------------\n\n");
        }
    }

    // Every one gets the vertices and the q value
    MPI_Bcast(&(info_process.N), 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&(info_process.q), 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    // if its possible to divide the graph, the root process will do the work and send it to others
    if (info_process.Pid == ROOT && info_process.q > 1)
        sendGrid(distances, info_process.N, info_process.q);

    // Syncing the processes
    MPI_Barrier(MPI_COMM_WORLD);

    /* ---------------- COMMON SECTION ---------------- */
    // Vertices assigned for each process
    int Ngrid = info_process.N / info_process.q;

    // Every processes receive his part
    grid = (int *)malloc(Ngrid * Ngrid * sizeof(int));

    // q = 1 equals P = 1, one process needs to compute the whole matrix
    if (info_process.q > 1)
        MPI_Recv(grid, Ngrid * Ngrid, MPI_INT, ROOT, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    else
        grid = distances;

    /* ---------------- ALGORITM SECTION ---------------- */
    // Floyd algorithm blocked 2D
    start = MPI_Wtime();
    floydWarshall2D(&info_process, grid, Ngrid);
    end = MPI_Wtime();

    // Execution time for each process
    double time_spent = end - start;
    //printf("PROCESS %d - Time execution %1.3f\n", info_process.Pid, time_spent);

    // Syncing the processes
    MPI_Barrier(MPI_COMM_WORLD);

    solution = (int *)malloc(info_process.N * info_process.N * sizeof(int));
    MPI_Gather(grid, Ngrid * Ngrid, MPI_INT, solution, Ngrid * Ngrid, MPI_INT, ROOT, MPI_COMM_WORLD);

    /* ---------------- PRINT AND CLEAR SECTION ---------------- */

    if (info_process.Pid == ROOT)
    {
        // Gather function distribute elements row-wise instead of matrix-wise
        // It needs a reorder of elements
        reorderMatrix(&info_process, distances, solution, Ngrid);

        // Printing the final result
        printf("ROOT - Distance solution graph: \n");
        printMatrix(distances, info_process.N);
        printf("-----------------------------------------\n\n");
    }

    MPI_Finalize();
    return 0;
}
