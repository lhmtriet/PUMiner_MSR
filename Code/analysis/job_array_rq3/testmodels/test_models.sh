#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --time=05:00:00
#SBATCH --mem=10000MB
#SBATCH --array=1-100
#SBATCH --err="results/pul_%a.err"
#SBATCH --output="results/pul_%a.out"
#SBATCH --job-name="predslong"

module load Anaconda3/5.0.1
source activate main

## Echo job id into results file
echo "array_job_index: $SLURM_ARRAY_TASK_ID"

python3 test_models.py