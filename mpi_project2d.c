/*************************************
 Blocked Floyd-Warshall 2D Algorithm 

 Daniele Antonucci
*************************************/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <limits.h>

#define ROOT 0
#define INF 10000
#define min(a, b) (a < b) ? a : b

typedef struct {
    //Processes & Vertices
    int P, N, q;
    //Process Ids
    int Pid, Pid_2d;
    //Info for floyd algo
    int row, col;
}GRID;

int check_sqrt(GRID *info);
int *generate_matrix(int N);
void print_matrix(int matrix[], int N);
void send_grid(int *matrix, int N, int q);
int *prepare_floyd(GRID *info, int grid[], MPI_Comm comm_row, MPI_Comm comm_col);
void floyd_warshall(int A[], int B[], int C[], int N);
int floyd_2d(GRID *info, int grid[], MPI_Comm comm_2d, MPI_Comm comm_col, MPI_Comm comm_row);

int main(int argc, char* argv[]) {
    //Communication info
    GRID info;
    //Matrix of costs
    int *distances;
    //2D Values for grid floyd
    int ndims[2], coords[2];
    int dim_row[2] = {1, 0};
    int dim_col[2] = {0, 1};
    int periods[2] = {1, 1};
    MPI_Comm comm_2d;
    MPI_Comm comm_row, comm_col;
    
    //Initializing communication
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &(info.P));
    MPI_Comm_rank(MPI_COMM_WORLD, &(info.Pid));

    info.q = sqrt(info.P);
    ndims[0] = ndims[1] = info.q;
    //Creating the 2D communicator
    MPI_Cart_create(MPI_COMM_WORLD, 2, ndims, periods, 1, &comm_2d);
	MPI_Comm_rank(comm_2d, &(info.Pid_2d));
	MPI_Cart_coords(comm_2d, info.Pid_2d, 2, coords);
    //Sub comunicators for row and col
    info.row = coords[0];
    info.col = coords[1];
    MPI_Cart_sub(comm_2d, dim_row, &comm_row);
    MPI_Cart_sub(comm_2d, dim_col, &comm_col);

    /* ---------------- ROOT SECTION ---------------- */
    if(info.Pid == ROOT) {
        printf("Vertices? ");
        fflush(stdout);
        //Get the number of vertices -> NxN distance matrix
        if(!scanf("%d", &(info.N))) {
            fprintf(stderr, "Cannot create a matrix with the inserted value");
            MPI_Abort(MPI_COMM_WORLD, ROOT);
            exit(1);
        };
        //Checking if possible to divide the matrix with P processes
        if (!check_sqrt(&info)) {
            fprintf(stderr, "Matrix of size %d cannot be solved with %d processes.\nAborting...\n", info.N, info.P);
            MPI_Abort(MPI_COMM_WORLD, ROOT);
            exit(1);
        }

       //Root generates the entire matrix somehow (at the moment randomly)
       distances = generate_matrix(info.N);
       printf("ROOT - Distance graph: \n");
       print_matrix(distances, info.N);
    }
    //Every one gets the grid associated
    MPI_Bcast(&(info.N), 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    if(info.Pid == ROOT) send_grid(distances, info.N, info.q);
    MPI_Barrier(MPI_COMM_WORLD);

    /* ---------------- COMMON SECTION ---------------- */
    int Ngrid = info.N / info.q;
    int *grid;

    //Every processes receive his part
    grid = (int *) malloc(Ngrid * Ngrid *sizeof(int));
    MPI_Recv(grid, Ngrid * Ngrid, MPI_INT, ROOT, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    //Printing the associated grid
    printf("PROCESS %d - Grid Distance Graph:\n", info.Pid_2d);
    print_matrix(grid, Ngrid);

    //WIP - Algorithm 
    floyd_2d(&info, grid, comm_2d, comm_col, comm_row);

    //WIP - Prints the result
    if(info.Pid == ROOT){
        //print_matrix(grid, Ngrid);
        //fflush(stdout);
    }

    MPI_Comm_free(&comm_col);
    MPI_Comm_free(&comm_row);
    MPI_Comm_free(&comm_2d);
    MPI_Finalize();
    return 0;
}

int check_sqrt(GRID *info) {
    if (info->q * info->q == info->P && info->N % info->q == 0) return 1;
    return 0;
}

int *generate_matrix(int N) {
    int *a = malloc(N * N * sizeof(int));
    srand(time(NULL));

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++){
            int r = rand() % 50;
            //INF Value = no path from i to j 
            if(r == 0 && i != j) r = INF;
            
            if(i == j) a[i*N + j] = 0;
            else a[i*N + j] = r;
        }
    }
    return (int *) a;
}

void print_matrix(int matrix[], int N) {
    int j;
    for(int i = 0; i < N; i++) {
        for(j = 0; j < N - 1; j++) {
            printf("%d ", matrix[i*N + j]);
        } 
        printf("%d\n", matrix[i*N + j]);
    }
    fflush(stdout);
}

void send_grid(int matrix[], int N, int q) {
    int Ngrid = N / q;
    int *grid_matrix = malloc(Ngrid * Ngrid * sizeof(int));

    int id = 0;
    for (int k = 0; k < q; k++) {
        for (int l = 0; l < q; l++) {
            for (int i = 0; i < Ngrid; i++) {
                for (int j = 0; j < Ngrid; j++) {
                    //The i-th value of the grid is equal to the value of matrix shifted based on q and Ngrid
                    grid_matrix[i*Ngrid + j] = matrix[(i + (k * Ngrid))*N + j + (l * Ngrid)];
                }
            }
            MPI_Send(grid_matrix, Ngrid * Ngrid, MPI_INT, id, 1, MPI_COMM_WORLD);
            id++;
        }
    }
    free(grid_matrix);
}

int floyd_2d(GRID *info, int grid[], MPI_Comm comm_2d, MPI_Comm comm_col, MPI_Comm comm_row) {
    int Ngrid = info->N / info->q;
    int mycoords[2];
    int row_id, col_id, coords[2];

    //regenerating coords
    MPI_Cart_coords(comm_2d, info->Pid_2d, 2, mycoords);

    //Pointers for row and col
    int *Kr = (int *) malloc(Ngrid * sizeof(int));
    int *Kc = (int *) malloc(Ngrid * sizeof(int));

    for(int h = 0; h < info->N; h++) {
        //Gets the current coordinates 
		coords[0] = coords[1] =  h / Ngrid;


		if(h >= mycoords[0] * Ngrid && h <= (mycoords[0] + 1) * Ngrid) {
			for(int j = 0; j < Ngrid; j++) {
				Kr[j] = grid[(h % Ngrid) * Ngrid + j];
			}
		}
        //Getting rank on col communicator
        MPI_Cart_rank(comm_col, coords, &col_id);
		MPI_Barrier(comm_col);

        MPI_Bcast(&Kr[0], Ngrid, MPI_INT, col_id, comm_col);

		if(h >= mycoords[1] * Ngrid && h <= (mycoords[1] + 1) * Ngrid) {
			for(int j = 0; j < Ngrid; j++) {
				Kc[j] = grid[j*Ngrid + (h % Ngrid)];
			}
		}
        //Getting rank on row communicator
        MPI_Cart_rank(comm_row, coords, &row_id);
		MPI_Barrier(comm_row);

        MPI_Bcast(&Kc[0], Ngrid, MPI_INT, row_id, comm_row);

        //Floyd computation
        for(int i = 0; i < Ngrid; i++) {
            for(int j = 0; j < Ngrid; j++) {
                grid[i*Ngrid + j] = min(grid[i*Ngrid + j], Kc[i] + Kr[j]);

                if(info->Pid_2d == ROOT) {
					printf("Kc[%d]=%d Kr[%d]=%d \n", i, Kc[i], j, Kr[j]);
                    printf("Grid[%d] = %d\n", (i * Ngrid + j), grid[i * Ngrid + j]);
				}
            }
        }
    }

    free(Kc);
    free(Kr);
    return 0;
}