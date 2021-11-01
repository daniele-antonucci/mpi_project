/*************************************
 Blocked Floyd-Warshall 2D Algorithm

 Daniele Antonucci
*************************************/
#include "functions_mpi.h"

int main(int argc, char *argv[])
{
    // Communication info
    GRID process;
    // Communicators
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm alg_comm;
    // Matrix of costs
    int *distances, *grid, *solution;
    // Timer values
    double start, end;

    // Initializing communication
    MPI_Init(&argc, &argv);
    MPI_Comm_size(comm, &(process.P));
    MPI_Comm_rank(comm, &(process.Pid));

    
    /* ---------------- ROOT SECTION: INITIALIZING ---------------- */
    if (process.Pid == ROOT)
    {
        // Checking for input file 
        if (strlen(argv[1]) == 0)
        {
            fprintf(stderr, "File .txt is not found, please execute mpiexec with 1 argument (ex. mpiexec -np 4 ./project_mpi input_X");
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
            fscanf(f, "%d", &process.N);
            // ROOT is not used for computation
            process.q = sqrt(process.P - 1);

            // Checking if possible to divide the matrix with P processes
            if (!checkSqrt(&process))
            {
                fprintf(stderr, "Matrix of size %d cannot be solved with %d processes.\nAborting...\n", process.N, process.P);
                MPI_Abort(comm, ROOT);
                exit(1);
            }

            // Read the matrix and close the file
            distances = readMatrixFromFile(f, process.N);
            fclose(f);

            // Printing the matrix
            printf("ROOT - Distance graph: \n");
            printMatrix(distances, process.N);
            printf("-----------------------------------------\n\n");
        }

        // If its possible to divide the graph, the root process will do the work and send it to others
        if(process.q > 1) sendGrid(distances, process.N, process.q);

        // Timing the execution
        start = MPI_Wtime();
    }

    /* ---------------- COMMON SECTION ---------------- */
    // Everyone gets the vertices and the q value
    MPI_Bcast(&(process.N), 1, MPI_INT, ROOT, comm);
    MPI_Bcast(&(process.q), 1, MPI_INT, ROOT, comm);    

    // Vertices assigned for each process
    process.Ngrid = process.N / process.q;
    // Matrix for each process
    grid = (int *)malloc(process.Ngrid * process.Ngrid * sizeof(int));

    // q = 1 equals P = 1, one process needs to compute the whole matrix
    if(process.Pid != ROOT) {
        if (process.q > 1)
            MPI_Recv(grid, process.Ngrid * process.Ngrid, MPI_INT, ROOT, 1, comm, MPI_STATUS_IGNORE);
        else
            grid = distances;

    }

    /* ---------------- ALGORITM SECTION ---------------- */
    solution = (int *)malloc(process.N * process.N * sizeof(int));

    // Creating communicator for everyone except ROOT
    int flag = process.Pid % (process.P + 1) > 0;
    MPI_Comm_split(comm, flag?0:MPI_UNDEFINED, process.Pid , &alg_comm);
    if(process.Pid != ROOT) MPI_Comm_rank(alg_comm, &process.algPid);

    // ROOT Process starts the timer and waits the solution matrix from the process 1 
    if(process.Pid == ROOT) {
        MPI_Recv(solution, process.N * process.N, MPI_INT, 1, 1, comm, MPI_STATUS_IGNORE);
    }
    else {
        floydWarshall2D(&process, grid, process.Ngrid, alg_comm, solution);
    }

    //Join every grid in a single matrix
    if(process.Pid != ROOT) {
        // First, gather all the grids into a single matrix on the ROOT process in the algorithm
        MPI_Gather(grid, process.Ngrid * process.Ngrid, MPI_INT, solution, process.Ngrid * process.Ngrid, MPI_INT, ROOT, alg_comm);
        //Then send it to ROOT process of comm
        if(process.algPid == ROOT) MPI_Send(solution, process.N * process.N, MPI_INT, ROOT, 1, comm);
    }

    /* ---------------- PRINT AND CLEAR SECTION ---------------- */
    if (process.Pid == ROOT)
    {       
        end = MPI_Wtime();
        double time_spent = end - start;
        printf("Time execution %f\n\n", time_spent);
        // Gather function distribute elements row-wise instead of matrix-wise
        // It needs a reorder of elements
        reorderMatrix(&process, distances, solution, process.Ngrid);

        // Printing the final result
        printf("Distance solution graph: \n");
        printMatrix(distances, process.N);
        printf("-----------------------------------------\n\n");

    }
    MPI_Finalize();
    return 0;
}
