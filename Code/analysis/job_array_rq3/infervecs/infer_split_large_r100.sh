#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=1:00:00
#SBATCH --mem=15000MB
#SBATCH --array=1-100
#SBATCH --err="results/vecs_%a.err"
#SBATCH --output="results/vecs_%a.out"
#SBATCH --job-name="infervec"

module load Anaconda3/5.0.1
source activate main

## Echo job id into results file
echo "array_job_index: $SLURM_ARRAY_TASK_ID"
IFS=',' read -ra par <<< `sed -n ${SLURM_ARRAY_TASK_ID}p infer_split_large_r100.csv`

python3 infer_split_large_r100.py "${par[0]}"