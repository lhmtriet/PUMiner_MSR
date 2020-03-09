#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=06:00:00
#SBATCH --mem=500MB
#SBATCH --err="error/oc_test.err"
#SBATCH --out="result/oc_test.out"

# Execute the program
module load Anaconda3/5.0.1
source activate main
python3 oc_validation.py "train"
source deactivate
