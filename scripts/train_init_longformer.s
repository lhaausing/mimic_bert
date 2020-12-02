#!/bin/bash

#SBATCH --partition=gpu4_dev
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:4
#SBATCH --mail-type=END
#SBATCH --mail-user=xl3119@nyu.edu

source activate mimic_bert
python3 2_finetune_init_lf.py
