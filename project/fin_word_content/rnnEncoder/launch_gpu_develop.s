#!/bin/bash

#SBATCH --job-name=nlu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --gres=gpu:p40:1
#SBATCH --time=04:00:00
#SBATCH --output=out.%j


module purge
module load pytorch/python3.6/0.3.0_4
export MPLBACKEND='pdf'

python main.py --hpc --n-epochs 5 --num-words 50000 --lr 0.001

