#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --time=00:40:00
#SBATCH --mem=150MB
#SBATCH --array=1-50
#SBATCH --err="results/vald_%a.err"
#SBATCH --output="results/vald%a.out"
#SBATCH --job-name="vldn"

module load Anaconda3/5.0.1
source activate main

## Echo job id into results file
echo "array_job_index: $SLURM_ARRAY_TASK_ID"

IFS=',' read -ra par <<< `sed -n ${SLURM_ARRAY_TASK_ID}p validation.csv`

python3 validation.py "${par[0]}"
