/******************************************************************************************************************
*  Name: Colin Sanders                                                                                            *
*  Email: ccsanders6@crimson.ua.edu                                                                               *
*  Date: November 4, 2021                                                                                         * 
*  Course Section: CS 581                                                                                         *
*  Homework #: 4                                                                                                  *
*                                                                                                                 *
*  The objective of this homework is to design and implement the "Game                                            *
*  of Life" Program - a cellular automata simulation by John Horton                                               *
*  Conway, efficiently using the Message Passing Interface (MPI). This implementation uses                        *
*  non-blocking point-to-point MPI calls.                                                                         *
*                                                                                                                 *
*  To Compile: mpicc -g -Wall -O3 -o hw4_non_blocking hw4_non_blocking.c                                          *
*  To run:                                                                                                        *
*  mpiexec -n <NUMBER OF PROCESSES> ./hw4_non_blocking <SIZE_OF_BOARD> <MAXIMUM_GENERATIONS> <OUTPUT_DIR> (Unix)  *
*******************************************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

// BEGIN UTILS

/* function to measure time taken */
double getTime(void)
{
    struct timeval tval;

    gettimeofday(&tval, NULL);

    return ((double)tval.tv_sec + (double)tval.tv_usec / 1000000.0);
}

int *allocateArray(int N)
{
    //Allocate memory as a 1D array of N*N length
    int *arr = (int *)malloc(N * N * sizeof(int));
    return arr;
}

void initArrayRandom(int *arr, int N)
{
    int i, j;
    for (i = 0; i < N; i++)
    {
        int row = i * N;
        for (j = 0; j < N; j++)
        {
            // Take the least significant bit of the rand() function. This is really fast and tends
            // to avoid some of the issues with using rand() and modulo for range restriction.
            arr[row + j] = rand() & 1;
        }
    }
}

void copyArray(int *dest, int *src, int N)
{
    memcpy(dest, src, sizeof(int) * N * N);
}

void print2DArray(int *arr, int N)
{
    int i, j;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            printf("%d ", arr[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void printArray(int *arr, int N)
{
    int i;
    for (i = 0; i < N; i++)
    {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

// Adds up the value of each neighbor at index [x][y]
// There's probably a million ways to iterate around a cell in a fancy way (i.e. if I was
// in Java, I would create a direction enum and some helper functions to rotate around)
// But just calculatiing each neighbor index by hand is a perfectly usable solution.
//
// NEW TO MPI IMPLEMENTATION
// THESE X & Y COORDS ARE NOW LOCAL PROCESS RELATIVE
// WE NO LONGER HAVE GHOST CELLS TO PROTECT US FROM OUT OF BOUNDS INDICES
// WE ALSO NEED TO USE NEIGHBOR ROWS
int sumOfNeighbors(int *localProcessArr, int *neighborRowTop, int *neighborRowBottom, int x, int y, int N, int localN)
{
    int rowPrev = (x - 1) * N;
    int row = rowPrev + N;
    int rowNext = row + N;

    // Going to break this into a partial sum and then add them all together to make the logic a bit easier

    int sumOfRowAbove = 0;
    if (rowPrev < 0)
    { //Use neighbor row top, can assume that x = 0.

        sumOfRowAbove += neighborRowTop[y]; // Cell directly above, always included

        if (y != 0) // Always include up left if not right most cell in row
        {
            sumOfRowAbove += neighborRowTop[y - 1];
        }

        if (y != N - 1)
        { //Always include up right if not left most cell in row
            sumOfRowAbove += neighborRowTop[y + 1];
        }
    }
    else // Same as above but can use local array
    {
        sumOfRowAbove += localProcessArr[rowPrev + y]; // Cell directly above, always included

        if (y != 0) // Always include up left if not right most cell in row
        {
            sumOfRowAbove += localProcessArr[rowPrev + y - 1];
        }

        if (y != N - 1)
        { //Always include up right if not left most cell in row
            sumOfRowAbove += localProcessArr[rowPrev + y + 1];
        }
    }

    // For the current row, we don't need to worry about neighboring cells, just making sure we aren't the left or right most.
    int sumOfMyRow = 0;
    if (y != 0) // Always include up left if not right most cell in row
    {
        sumOfMyRow += localProcessArr[row + y - 1];
    }

    if (y != N - 1)
    { //Always include up right if not left most cell in row
        sumOfMyRow += localProcessArr[row + y + 1];
    }

    int sumOfRowBelow = 0;
    if (rowNext >= localN)
    {                                          // Use neighbor row bottom
        sumOfRowBelow += neighborRowBottom[y]; // Cell directly below, always included

        if (y != 0) // Always include up left if not right most cell in row
        {
            sumOfRowAbove += neighborRowBottom[y - 1];
        }

        if (y != N - 1)
        { //Always include up right if not left most cell in row
            sumOfRowAbove += neighborRowBottom[y + 1];
        }
    }
    else
    {
        sumOfRowBelow += localProcessArr[rowNext + y]; // Cell directly above, always included

        if (y != 0) // Always include up left if not right most cell in row
        {
            sumOfRowBelow += localProcessArr[rowNext + y - 1];
        }

        if (y != N - 1)
        { //Always include up right if not left most cell in row
            sumOfRowBelow += localProcessArr[rowNext + y + 1];
        }
    }

    return sumOfRowAbove + sumOfRowBelow + sumOfMyRow;
}

void writeArrToFile(int *arr, int N, FILE *filePointer)
{
    int i, j;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            fprintf(filePointer, "%d ", arr[i * N + j]);
        }

        fprintf(filePointer, "\n");
    }
}

// END ARRAY UTILS

int main(int argc, char **argv)
{
    //srand(time(NULL));
    srand(12345);

    int size, rank, N, MAX_GENERATIONS;

    double startTime = 0, endTime = 0;

    int *currentBoard = NULL;
    int *previousBoard = NULL;

    char fileName[BUFSIZ];
    FILE *filePointer = NULL;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int processData[2];
    int sendCounts[size];
    int displacement[size];
    MPI_Request reqs[4]; // required variable for non-blocking calls
    MPI_Status stats[4]; // required variable for Waitall routine

    // Restrict argument checking to just process 0, including setting up the gameboard and distributing the initial state.
    if (rank == 0)
    {
        if (argc != 4)
        {
            printf("Usage: %s <SIZE_OF_BOARD> <MAXIMUM_GENERATIONS> <OUTPUT_DIR> \n", argv[0]);
            MPI_Finalize();
            exit(-1);
        }

        N = atoi(argv[1]);
        MAX_GENERATIONS = atoi(argv[2]);

        sprintf(fileName, "%s/output.nb.%d.%d.%d", argv[3], N, MAX_GENERATIONS, size);

        if ((filePointer = fopen(fileName, "w")) == NULL)
        {
            printf("Error opening file %s for writing\n", argv[3]);
            perror("fopen");
            MPI_Finalize();
            exit(-1);
        }

        currentBoard = allocateArray(N);
        previousBoard = allocateArray(N);

        //Note a key change from the other versions: I don't assign ghost cells to the initial board as this will mess
        //with how the distribution works. Each process will keep track of their own border cells, and I need to be particularly
        //careful with the 0th and N-1th elements in a row.
        initArrayRandom(currentBoard, N);
        copyArray(previousBoard, currentBoard, N);

        //printf("Starting board:\n");
        //print2DArray(currentBoard, N);

        // Each process needs to know how big the board is, the maximum number generations, and the number of slices
        // Along with each process's slice of the board.
        processData[0] = N;
        processData[1] = MAX_GENERATIONS;
    }

    MPI_Bcast(&processData, 2, MPI_INT, 0, MPI_COMM_WORLD);

    N = processData[0];
    MAX_GENERATIONS = processData[1];

    int sum = 0;
    for (int i = 0; i < size; i++)
    {
        sendCounts[i] = N / size;
        if (i == size - 1 && N % size != 0)
        {
            sendCounts[i] += N % size;
        }

        sendCounts[i] = sendCounts[i] * N;

        displacement[i] = sum;
        sum += sendCounts[i];
    }

    int totalLocalCells = sendCounts[rank];
    int totalLocalRows = totalLocalCells / N;

    int *localBoard = (int *)malloc(totalLocalCells * sizeof(int));
    int *localPreviousBoard = (int *)malloc(totalLocalCells * sizeof(int));

    // Rows that will be traded amongst cells
    int *localRowTop = (int *)malloc(N * sizeof(int));
    int *localRowBottom = (int *)malloc(N * sizeof(int));
    int *neighborRowTop = (int *)malloc(N * sizeof(int));
    int *neighborRowBottom = (int *)malloc(N * sizeof(int));

    MPI_Scatterv(currentBoard, sendCounts, displacement, MPI_INT, localBoard, totalLocalCells, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(previousBoard, sendCounts, displacement, MPI_INT, localPreviousBoard, totalLocalCells, MPI_INT, 0, MPI_COMM_WORLD);

    //printf("Process %d received the following cells: ", rank);
    //printArray(localBoard, localN);

    int currentGeneration, i, j, index;
    int neighbors, cellStatus, localChangeFlag = 0;

    if (rank == 0)
    {
        startTime = getTime();
    }

    for (currentGeneration = 1; currentGeneration < MAX_GENERATIONS; currentGeneration++)
    {
        //Populate local rows, initialize neighbor "ghost" cells to 0 (particularly useful for our first and last process, as they'll be avoiding some of the sends and recvs as below):
        for (i = 0; i < N; i++)
        {
            localRowTop[i] = localPreviousBoard[i];
            localRowBottom[i] = localPreviousBoard[N * (totalLocalRows - 1) + i];
            neighborRowTop[i] = 0;
            neighborRowBottom[i] = 0;
        }

        // Distribute local rows and receive neighboring rows. See logic examples.txt for the solution details.
        int requestCount = 0;
        if (rank != 0)
        {
            MPI_Irecv(neighborRowTop, N, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &reqs[requestCount++]);
        }

        if (rank != size - 1)
        {
            MPI_Irecv(neighborRowBottom, N, MPI_INT, rank + 1, 1, MPI_COMM_WORLD, &reqs[requestCount++]);
        }

        if (rank != 0)
        {
            MPI_Isend(localRowTop, N, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, &reqs[requestCount++]);
        }

        if (rank != size - 1)
        {
            MPI_Isend(localRowBottom, N, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &reqs[requestCount++]); //tag 0 for bottom local edge / top
        }

        // Only do work for the rows that do not require neighbor rows, allowing the non-blocking sends to occur
        for (i = 1; i < totalLocalRows - 1; i++)
        {
            for (j = 0; j < N; j++)
            {
                index = i * N + j;
                neighbors = sumOfNeighbors(localPreviousBoard, neighborRowTop, neighborRowBottom, i, j, N, totalLocalCells);
                cellStatus = localPreviousBoard[index];

                if (cellStatus == 1)
                {
                    if (neighbors < 2)
                    {
                        // Current cell dies of loneliness
                        localBoard[index] = 0;
                        localChangeFlag = 1;
                    }
                    else if (neighbors > 3)
                    {
                        // Current cell dies of overpopulation
                        localBoard[index] = 0;
                        localChangeFlag = 1;
                    }
                    else
                    {
                        localBoard[index] = 1;
                    }
                }
                else
                {
                    if (neighbors == 3)
                    {
                        //Current cell is not alive with exactly three neighbors, birth at this cell
                        localBoard[index] = 1;
                        localChangeFlag = 1;
                    }
                    else
                    {
                        localBoard[index] = 0;
                    }
                }
            }
        }

        MPI_Waitall(requestCount, reqs, stats);

        // Now that the requests are done, process the neighbor rows
        for (j = 0; j < N; j++)
        {
            //do the top row first
            i = 0;
            index = i * N + j;
            neighbors = sumOfNeighbors(localPreviousBoard, neighborRowTop, neighborRowBottom, i, j, N, totalLocalCells);
            cellStatus = localPreviousBoard[index];

            if (cellStatus == 1)
            {
                if (neighbors < 2)
                {
                    // Current cell dies of loneliness
                    localBoard[index] = 0;
                    localChangeFlag = 1;
                }
                else if (neighbors > 3)
                {
                    // Current cell dies of overpopulation
                    localBoard[index] = 0;
                    localChangeFlag = 1;
                }
                else
                {
                    localBoard[index] = 1;
                }
            }
            else
            {
                if (neighbors == 3)
                {
                    //Current cell is not alive with exactly three neighbors, birth at this cell
                    localBoard[index] = 1;
                    localChangeFlag = 1;
                }
                else
                {
                    localBoard[index] = 0;
                }
            }

            //then process the bottom row
            i = totalLocalRows - 1;
            index = i * N + j;
            neighbors = sumOfNeighbors(localPreviousBoard, neighborRowTop, neighborRowBottom, i, j, N, totalLocalCells);
            cellStatus = localPreviousBoard[index];

            if (cellStatus == 1)
            {
                if (neighbors < 2)
                {
                    // Current cell dies of loneliness
                    localBoard[index] = 0;
                    localChangeFlag = 1;
                }
                else if (neighbors > 3)
                {
                    // Current cell dies of overpopulation
                    localBoard[index] = 0;
                    localChangeFlag = 1;
                }
                else
                {
                    localBoard[index] = 1;
                }
            }
            else
            {
                if (neighbors == 3)
                {
                    //Current cell is not alive with exactly three neighbors, birth at this cell
                    localBoard[index] = 1;
                    localChangeFlag = 1;
                }
                else
                {
                    localBoard[index] = 0;
                }
            }
        }

        // Similarly to OpenMP implementation, we need to add all of the local change flags to a global change flag and use this
        // to determine if all processes should break.
        int globalChangeFlag = 0;
        MPI_Allreduce(&localChangeFlag, &globalChangeFlag, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (globalChangeFlag == 0)
        {
            break;
        }

        int *temp = localPreviousBoard;
        localPreviousBoard = localBoard;
        localBoard = temp;
        localChangeFlag = 0;
    }

    // Regather the board to print to file.
    MPI_Gatherv(localBoard, totalLocalCells, MPI_INT, currentBoard, sendCounts, displacement, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        endTime = getTime();

        if (N < 10)
        {
            printf("Ending board:\n");
            print2DArray(currentBoard, N);
        }

        printf("Execution took %f s and lasted %d generations\n", endTime - startTime, currentGeneration);
        printf("Writing output to file: %s\n", fileName);
        writeArrToFile(currentBoard, N, filePointer);

        free(currentBoard);
        free(previousBoard);
    }

    // garbage collection
    free(localBoard);
    free(localPreviousBoard);
    free(localRowTop);
    free(localRowBottom);
    free(neighborRowTop);
    free(neighborRowBottom);

    MPI_Finalize();
    return 0;
}

/**
 * This function is a scriptable version of my program that allows me to pass in preset data for me to use 
 * while testing. (i.e. I can call my program 100 times automatically with certain data and assert correctness)
 */
/*
int main_test_bed(int randomSeed, int preSeeded, int *testData, int N, int MAX_GENERATIONS, int size, int rank)
{
    srand(randomSeed);

    double startTime, endTime;

    int *currentBoard = NULL;
    int *previousBoard = NULL;

    int sendCounts[size];
    int displacement[size];
    MPI_Request reqs[4]; // required variable for non-blocking calls
    MPI_Status stats[4]; // required variable for Waitall routine

    if (rank == 0)
    {
        currentBoard = allocateArray(N);
        previousBoard = allocateArray(N);

        initArrayRandom(currentBoard, N);

        if (preSeeded == 1)
        {
            copyArray(currentBoard, testData, N);
        }

        copyArray(previousBoard, currentBoard, N);
    }

    int sum = 0;
    for (int i = 0; i < size; i++)
    {
        sendCounts[i] = N / size;
        if (i == size - 1 && N % size != 0)
        {
            sendCounts[i] += N % size;
        }

        sendCounts[i] = sendCounts[i] * N;

        displacement[i] = sum;
        sum += sendCounts[i];
    }

    int totalLocalCells = sendCounts[rank];
    int totalLocalRows = totalLocalCells / N;

    int *localBoard = (int *)malloc(totalLocalCells * sizeof(int));
    int *localPreviousBoard = (int *)malloc(totalLocalCells * sizeof(int));

    // Rows that will be traded amongst cells
    int *localRowTop = (int *)malloc(N * sizeof(int));
    int *localRowBottom = (int *)malloc(N * sizeof(int));
    int *neighborRowTop = (int *)malloc(N * sizeof(int));
    int *neighborRowBottom = (int *)malloc(N * sizeof(int));

    MPI_Scatterv(currentBoard, sendCounts, displacement, MPI_INT, localBoard, totalLocalCells, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(previousBoard, sendCounts, displacement, MPI_INT, localPreviousBoard, totalLocalCells, MPI_INT, 0, MPI_COMM_WORLD);

    //printf("Process %d received the following cells: ", rank);
    //printArray(localBoard, localN);

    int currentGeneration, i, j, index;
    int neighbors, cellStatus, localChangeFlag = 0;

    if (rank == 0)
    {
        startTime = getTime();
    }

    for (currentGeneration = 1; currentGeneration < MAX_GENERATIONS; currentGeneration++)
    {
        //Populate local rows, initialize neighbor "ghost" cells to 0 (particularly useful for our first and last process, as they'll be avoiding some of the sends and recvs as below):
        for (i = 0; i < N; i++)
        {
            localRowTop[i] = localPreviousBoard[i];
            localRowBottom[i] = localPreviousBoard[N * (totalLocalRows - 1) + i];
            neighborRowTop[i] = 0;
            neighborRowBottom[i] = 0;
        }

        int requestCount = 0;
        if (rank != 0)
        {
            MPI_Irecv(neighborRowTop, N, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &reqs[requestCount++]);
        }

        if (rank != size - 1)
        {
            MPI_Irecv(neighborRowBottom, N, MPI_INT, rank + 1, 1, MPI_COMM_WORLD, &reqs[requestCount++]);
        }

        if (rank != 0)
        {
            MPI_Isend(localRowTop, N, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, &reqs[requestCount++]);
        }

        if (rank != size - 1)
        {
            MPI_Isend(localRowBottom, N, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &reqs[requestCount++]); //tag 0 for bottom local edge / top
        }

        for (i = 1; i < totalLocalRows - 1; i++)
        {
            for (j = 0; j < N; j++)
            {
                index = i * N + j;
                neighbors = sumOfNeighbors(localPreviousBoard, neighborRowTop, neighborRowBottom, i, j, N, totalLocalCells);
                cellStatus = localPreviousBoard[index];

                if (cellStatus == 1)
                {
                    if (neighbors < 2)
                    {
                        // Current cell dies of loneliness
                        localBoard[index] = 0;
                        localChangeFlag = 1;
                    }
                    else if (neighbors > 3)
                    {
                        // Current cell dies of overpopulation
                        localBoard[index] = 0;
                        localChangeFlag = 1;
                    }
                    else
                    {
                        localBoard[index] = 1;
                    }
                }
                else
                {
                    if (neighbors == 3)
                    {
                        //Current cell is not alive with exactly three neighbors, birth at this cell
                        localBoard[index] = 1;
                        localChangeFlag = 1;
                    }
                    else
                    {
                        localBoard[index] = 0;
                    }
                }
            }
        }

        MPI_Waitall(requestCount, reqs, stats);

        // Now that the requests are done, process the neighbor rows
        for (j = 0; j < N; j++)
        {
            //do the top row first
            i = 0;
            index = i * N + j;
            neighbors = sumOfNeighbors(localPreviousBoard, neighborRowTop, neighborRowBottom, i, j, N, totalLocalCells);
            cellStatus = localPreviousBoard[index];

            if (cellStatus == 1)
            {
                if (neighbors < 2)
                {
                    // Current cell dies of loneliness
                    localBoard[index] = 0;
                    localChangeFlag = 1;
                }
                else if (neighbors > 3)
                {
                    // Current cell dies of overpopulation
                    localBoard[index] = 0;
                    localChangeFlag = 1;
                }
                else
                {
                    localBoard[index] = 1;
                }
            }
            else
            {
                if (neighbors == 3)
                {
                    //Current cell is not alive with exactly three neighbors, birth at this cell
                    localBoard[index] = 1;
                    localChangeFlag = 1;
                }
                else
                {
                    localBoard[index] = 0;
                }
            }

            //then process the bottom row
            i = totalLocalRows - 1;
            index = i * N + j;
            neighbors = sumOfNeighbors(localPreviousBoard, neighborRowTop, neighborRowBottom, i, j, N, totalLocalCells);
            cellStatus = localPreviousBoard[index];

            if (cellStatus == 1)
            {
                if (neighbors < 2)
                {
                    // Current cell dies of loneliness
                    localBoard[index] = 0;
                    localChangeFlag = 1;
                }
                else if (neighbors > 3)
                {
                    // Current cell dies of overpopulation
                    localBoard[index] = 0;
                    localChangeFlag = 1;
                }
                else
                {
                    localBoard[index] = 1;
                }
            }
            else
            {
                if (neighbors == 3)
                {
                    //Current cell is not alive with exactly three neighbors, birth at this cell
                    localBoard[index] = 1;
                    localChangeFlag = 1;
                }
                else
                {
                    localBoard[index] = 0;
                }
            }
        }

        // Similarly to OpenMP implementation, we need to add all of the local change flags to a global change flag and use this
        // to determine if all processes should break.
        int globalChangeFlag = 0;
        MPI_Allreduce(&localChangeFlag, &globalChangeFlag, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (globalChangeFlag == 0)
        {
            break;
        }

        int *temp = localPreviousBoard;
        localPreviousBoard = localBoard;
        localBoard = temp;
        localChangeFlag = 0;
    }

    // Regather the board to print to file.
    MPI_Gatherv(localBoard, totalLocalCells, MPI_INT, currentBoard, sendCounts, displacement, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        endTime = getTime();

        if (N < 10)
        {
            printf("Ending board:\n");
            print2DArray(currentBoard, N);
        }

        printf("Current test execution took %f ms and lasted %d generations\n", endTime - startTime, currentGeneration);

        free(currentBoard);
        free(previousBoard);
    }

    // garbage collection
    free(localBoard);
    free(localPreviousBoard);
    free(localRowTop);
    free(localRowBottom);
    free(neighborRowTop);
    free(neighborRowBottom);

    return currentGeneration;
}*/