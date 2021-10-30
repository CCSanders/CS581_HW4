/************************************************************************************************************
*  Name: Colin Sanders                                                                                      *
*  Email: ccsanders6@crimson.ua.edu                                                                         *
*  Date: November 4, 2021                                                                                   * 
*  Course Section: CS 581                                                                                   *
*  Homework #: 4                                                                                            *
*                                                                                                           *
*  The objective of this homework is to design and implement the "Game                                      *
*  of Life" Program - a cellular automata simulation by John Horton                                         *
*  Conway, efficiently using the Message Passing Interface (MPI).                                           *
*                                                                                                           *
*  To Compile: mpicc -g -Wall -O3 -o hw4 hw4.c                                                              *
*  To run:                                                                                                  *
*    mpiexec -n <NUMBER OF PROCESSES> ./hw4 <SIZE_OF_BOARD> <MAXIMUM_GENERATIONS> <OUTPUT_DIR> (Unix)       *
*************************************************************************************************************/

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
    for (i = 0; i < N; i++){
        printf("%d ", arr[i]);
    }
    printf("\n");
}

// Adds up the value of each neighbor at index [x][y]
// There's probably a million ways to iterate around a cell in a fancy way (i.e. if I was
// in Java, I would create a direction enum and some helper functions to rotate around)
// But just calculatiing each neighbor index by hand is a perfectly usable solution.
int sumOfNeighbors(int *arr, int x, int y, int N)
{
    int rowPrev = (x - 1) * N;
    int row = rowPrev + N;
    int rowNext = row + N;

    return arr[rowPrev + y - 1] + arr[rowPrev + y] + arr[rowPrev + y + 1] + arr[row + y - 1] + arr[row + y + 1] + arr[rowNext + y - 1] + arr[rowNext + y] + arr[rowNext + y + 1];
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

    int *currentBoard = NULL;
    int *previousBoard = NULL;

    int processData[2];

    char fileName[BUFSIZ];
    FILE* filePointer;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

        sprintf(fileName, "%s/output..%d%d", argv[3], N, MAX_GENERATIONS);

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

        // Each process needs to know how big the board is, the maximum number generations, and the number of slices
        // Along with each process's slice of the board.
        processData[0] = N;
        processData[1] = MAX_GENERATIONS;
    }

    MPI_Bcast(&processData, 2, MPI_INT, 0, MPI_COMM_WORLD);

    N = processData[0];
    MAX_GENERATIONS = processData[1];

    int localN = (N * N) / size;
    int *localBoard = (int *)malloc(localN * sizeof(int));
    MPI_Scatter(currentBoard, localN, MPI_INT, localBoard, localN, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Process %d received the following cells: ", rank);
    printArray(localBoard, localN);

    /*

    int currentGeneration, i, j, index;
    int neighbors, cellStatus, changeFlag = 0;

    double startTime = getTime();

    for (currentGeneration = 1; currentGeneration <= MAX_GENERATIONS; currentGeneration++)
    {
        for (i = 1; i < N - 1; i++)
        {
            for (j = 1; j < N - 1; j++)
            {
                index = i * N + j;
                neighbors = sumOfNeighbors(previousBoard, i, j, N);
                cellStatus = previousBoard[index];

                if (cellStatus == 1)
                {
                    if (neighbors < 2)
                    {
                        // Current cell dies of loneliness
                        currentBoard[index] = 0;
                        changeFlag = 1;
                    }
                    else if (neighbors > 3)
                    {
                        // Current cell dies of overpopulation
                        currentBoard[index] = 0;
                        changeFlag = 1;
                    }
                    else
                    {
                        currentBoard[index] = 1;
                    }
                }
                else
                {
                    if (neighbors == 3)
                    {
                        //Current cell is not alive with exactly three neighbors, birth at this cell
                        currentBoard[index] = 1;
                        changeFlag = 1;
                    }
                    else
                    {
                        currentBoard[index] = 0;
                    }
                }
            }
        }

        if (changeFlag == 0)
        {
            break;
        }

        int *temp = previousBoard;
        previousBoard = currentBoard;
        currentBoard = temp;
        changeFlag = 0;
    }

    double endTime = getTime();
    printf("Execution took %f s and lasted %d generations\n", endTime - startTime, currentGeneration);

    printf("Writing output to file: %s\n", fileName);
    writeArrToFile(currentBoard, N, filePointer);
    */

    //free(currentBoard);
    //free(previousBoard);

    MPI_Finalize();
    return 0;
}

/**
 * This function is a scriptable version of my program that allows me to pass in preset data for me to use 
 * while testing. (i.e. I can call my program 100 times automatically with certain data and assert correctness)
 */
/*
int main_test_bed(int randomSeed, int preSeeded, char *testData, int N, int MAX_GENERATIONS)
{
    clock_t startTime = clock();

    srand(randomSeed);

    char *currentBoard = NULL;
    char *previousBoard = NULL;

    currentBoard = allocateArray(N);
    previousBoard = allocateArray(N);

    initArrayRandomWithGhostCells(currentBoard, N);

    if (preSeeded == 1)
    {
        copyArray(currentBoard, testData, N);
    }

    copyArray(previousBoard, currentBoard, N);

    int currentGeneration = 1;
    while (currentGeneration < MAX_GENERATIONS && tick(currentBoard, previousBoard, N) == 1)
    {
        char *temp = previousBoard;
        previousBoard = currentBoard;
        currentBoard = temp;

        currentGeneration++;
    }

    if (N < 7)
    {
        print2DArray(currentBoard, N);
    }

    free(currentBoard);
    free(previousBoard);

    clock_t endTime = clock();
    double totalTime = (double)(endTime - startTime) * 1000.0 / CLOCKS_PER_SEC;
    printf("Current test execution took %f ms and lasted %d generations\n", totalTime, currentGeneration);

    return currentGeneration;
}
*/