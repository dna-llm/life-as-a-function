#!/bin/bash

#SBATCH -A account_name
#SBATCH --job-name evo 
#SBATCH --partition grete:shared
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user youremail@email.com

#SBATCH -o slurm_out/testing.out
#SBATCH -e slurm_out/testing.err

#SBATCH --time=24:00:00
#SBATCH --array=0

#SBATCH -G A100:1 -C 80gb

module load cuda 

export HF_HOME=$WORK/huggingface/

cd $WORK/life-as-a-function/models/evo
uv run evo_finetune.py