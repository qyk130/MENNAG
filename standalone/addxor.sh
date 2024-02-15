#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name=addxor
#SBATCH --time=24:00:00
#SBATCH --partition=general
#SBATCH --account=a_gallagher
#SBATCH -o royal.output
#SBATCH -e royal.error

module purge
module load python/3.11.3
module load openmpi/4.1.5

source /home/s4354963/env/bin/activate

srun -n 96 python -m mpi4py.futures addxor_test.py

