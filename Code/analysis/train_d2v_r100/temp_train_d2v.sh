#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --time=18:00:00
#SBATCH --mem=15000MB
#SBATCH --array=1-4
#SBATCH --err="results/d2v_%a.err"
#SBATCH --output="results/d2v_%a.out"
#SBATCH --job-name="rtc"

module load Anaconda3/5.0.1
source activate main

## Echo job id into results file
echo "array_job_index: $SLURM_ARRAY_TASK_ID"

IFS=',' read -ra par <<< `sed -n ${SLURM_ARRAY_TASK_ID}p d2v_models.csv`

python3 train_d2v_r100.py "${par[0]}"