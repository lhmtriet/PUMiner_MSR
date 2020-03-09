#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=5:00:00
#SBATCH --mem=20000MB
#SBATCH --array=1-28
#SBATCH --err="results/vecs_%a.err"
#SBATCH --output="results/vecs_%a.out"
#SBATCH --job-name="infervec"

module load Anaconda3/5.0.1
source activate main

## Echo job id into results file
echo "array_job_index: $SLURM_ARRAY_TASK_ID"
IFS=',' read -ra par <<< `sed -n ${SLURM_ARRAY_TASK_ID}p infer_vecs.csv`

python3 infer_vecs.py "${par[0]}"