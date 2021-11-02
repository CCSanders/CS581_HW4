#!/bin/bash
module load openmpi/4.0.5-gnu-pmi2
srun --mpi=pmi2 ./hw4nb 5000 5000 /scratch/$USER
srun --mpi=pmi2 ./hw4nb 5000 5000 /scratch/$USER
srun --mpi=pmi2 ./hw4nb 5000 5000 /scratch/$USER
