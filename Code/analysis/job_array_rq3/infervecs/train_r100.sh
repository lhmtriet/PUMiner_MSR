#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time=5:00:00
#SBATCH --mem=128000MB
#SBATCH --array=1-8
#SBATCH --err="results3/r100_long_%a.err"
#SBATCH --output="results3/r100_long_%a.out"
#SBATCH --job-name="r100_long"

module load Anaconda3/5.0.1
source activate main

## Echo job id into results file
echo "array_job_index: $SLURM_ARRAY_TASK_ID"
IFS=',' read -ra par <<< `sed -n ${SLURM_ARRAY_TASK_ID}p train_r100.csv`

python3 train_r100.py "${par[0]}"