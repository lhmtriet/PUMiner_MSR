#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time=01:20:00
#SBATCH --mem=300MB
#SBATCH --array=1-28
#SBATCH --err="results/d2v_%a.err"
#SBATCH --output="results/d2v_%a.out"
#SBATCH --job-name="train_d2v"

module load Anaconda3/5.0.1
source activate main

## Echo job id into results file
echo "array_job_index: $SLURM_ARRAY_TASK_ID"
IFS=',' read -ra par <<< `sed -n ${SLURM_ARRAY_TASK_ID}p sets.csv`

python3 train_d2v.py "${par[0]}"