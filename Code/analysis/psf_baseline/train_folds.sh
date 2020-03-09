#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --time=00:50:00
#SBATCH --mem=40MB
#SBATCH --array=1-10
#SBATCH --err="results/d2v_%a.err"
#SBATCH --output="results/d2v_%a.out"
#SBATCH --job-name="d2v"

module load Anaconda3/5.0.1
source activate main

## Echo job id into results file
echo "array_job_index: $SLURM_ARRAY_TASK_ID"

IFS=',' read -ra par <<< `sed -n ${SLURM_ARRAY_TASK_ID}p train_folds.csv`

python3 train_folds.py "${par[0]}"