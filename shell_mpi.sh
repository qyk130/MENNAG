#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=24:mpiprocs=24:mem=30gb

#PBS -A UQ-EAIT-ITEE

module purge
module load OpenMPI/2.0.2


cd MENNAG

mpiexec python3 -m mpi4py.futures src/train.py  -t BipedalWalker-v3 -c settings/bipedal_small.json -g 1000 -o out/bipedout --mpi
