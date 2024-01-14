#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=12:mpiprocs=12:mem=60gb

#PBS -A UQ-EAIT-ITEE

module purge
module load OpenMPI/2.0.2
module load gcc/11.2.0

g++ --version

cd MENNAG

echo $TASK
mpiexec python3 -m mpi4py.futures src/train.py -n 24 -t $TASK --reseed $RESEED -c settings/$SETTING.json -g $GEN -d out/$TASK$SETTING -o $INDEX --mpi --task_num $TASKNUM
