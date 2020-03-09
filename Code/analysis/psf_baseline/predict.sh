#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=00:40:00
#SBATCH --mem=150MB
#SBATCH --array=1-10
#SBATCH --err="results/pred_%a.err"
#SBATCH --output="results/pred_%a.out"
#SBATCH --job-name="pred"

module load Anaconda3/5.0.1
source activate main

## Echo job id into results file
echo "array_job_index: $SLURM_ARRAY_TASK_ID"

IFS=',' read -ra par <<< `sed -n ${SLURM_ARRAY_TASK_ID}p predict.csv`

python3 predict.py "${par[0]}"