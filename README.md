# mpi_project
Floyd-Warshall Implementation using MPI

Blocked 2D version of Floyd Warshall algorithm using MPI C
It is possible to test with even vertices of the matrix and 4 processes(or N^2 processes)

Root job is to wait the result from P-1 processes.

Usage:
mpicc project_mpi.c -o project_mpi
mpiexec -np 5 ./project_mpi input_X
