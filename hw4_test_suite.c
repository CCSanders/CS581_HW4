/****************************************************************************** 
*  Name: Colin Sanders                                                        *
*  Email: ccsanders6@crimson.ua.edu                                           *
*  Date: November 4, 2021                                                     * 
*  Course Section: CS 581                                                     *
*  Homework #: 4                                                              *
*                                                                             *
*  The objective of this homework is to design and implement the "Game        *
*  of Life" Program - a cellular automata simulation by John Horton           *
*  Conway. This is a helper file that I used for testing                      *
*                                                                             *
*  To Compile: mpicc -g -Wall -O3 -o hw4tests hw4_test_suite.c                *
*      NOTE: MUST COMMENT OUT MAIN FUNCTION IN hw4.C FOR THIS TO COMPILE      *
*  To run:                                                                    *
*      mpiexec -n <NUMBER OF PROCESSES> ./hw4tests                            *
******************************************************************************/

#include <assert.h>
#include <mpi.h>
#include "hw4.c"

int main(int argc, char **argv)
{
    int size, rank;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //These are tests for correctness taken from wikipedia:
    //https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life#Examples_of_patterns

    //The still tests should make the change detector flag not flip in generation 1, ending the simulation immediately.
    if (4 >= size)
    {
        int test_still_block[] = {
            0, 0, 0, 0,
            0, 1, 1, 0,
            0, 1, 1, 0,
            0, 0, 0, 0};

        assert(main_test_bed(0, 1, test_still_block, 4, 25, size, rank) == 1);
    }

    if (5 >= size)
    {
        int test_still_boat[] = {
            0, 0, 0, 0, 0,
            0, 1, 1, 0, 0,
            0, 1, 0, 1, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 0, 0};

        assert(main_test_bed(0, 1, test_still_boat, 5, 25, size, rank) == 1);

        int test_still_tub[] = {
            0, 0, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 1, 0, 1, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 0, 0};

        assert(main_test_bed(0, 1, test_still_tub, 5, 25, size, rank) == 1);

        //The oscillator tests should make the change detector flag flip indefinitely, ending the simulation on max iterations
        int test_oscil_blinker[] = {
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 1, 1, 1, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0};

        assert(main_test_bed(0, 1, test_oscil_blinker, 5, 50, size, rank) == 50);
    }

    if (6 >= size)
    {
        int test_oscil_beacon[] = {
            0, 0, 0, 0, 0, 0,
            0, 1, 1, 0, 0, 0,
            0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 1, 1, 0,
            0, 0, 0, 0, 0, 0};

        assert(main_test_bed(0, 1, test_oscil_beacon, 6, 50, size, rank) == 50);
    }

    // Sanity test: make sure that the program executes the same way every time
    // We will use the R-pentomino, determine how long it takes to stablize,
    // and then run it again 100 times and make sure that it continues to end on
    // the same generation.

    if (9 >= size)
    {
        int test_r_pentomino[] = {
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 0, 0, 0,
            0, 0, 0, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0};
        int sanity_generations = main_test_bed(0, 1, test_r_pentomino, 9, 500, size, rank);
        for (int i = 0; i < 100; i++)
        {
            assert(main_test_bed(0, 1, test_r_pentomino, 9, 500, size, rank) == sanity_generations);
        }
    }

    if (6 >= size)
    {
        int test_hw4[] = {
            0, 0, 0, 1, 0, 1,
            0, 0, 1, 0, 1, 1,
            1, 0, 0, 0, 1, 1,
            1, 1, 1, 0, 0, 1,
            1, 1, 1, 0, 1, 1,
            1, 1, 0, 0, 0, 1};
        main_test_bed(12345, 1, test_hw4, 6, 10, size, rank);
    }

    if(rank == 0) {
        printf("All tests completed successfully\n");
    }

    MPI_Finalize();
    return 0;
}