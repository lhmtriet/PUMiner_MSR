#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --time=01:00:00
#SBATCH --mem=400MB
#SBATCH --array=1-30
#SBATCH --err="results/pu_%a.err"
#SBATCH --output="results/pu_%a.out"
#SBATCH --job-name="randlearn"

module load Anaconda3/5.0.1
source activate main

## Echo job id into results file
echo "array_job_index: $SLURM_ARRAY_TASK_ID"

python3 edge_test_models.py