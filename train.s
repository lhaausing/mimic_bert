#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:p40:1
#SBATCH --time=48:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=multi_file_train
#SBATCH --mail-type=END
#SBATCH --mail-user=gw1107@nyu.edu
#SBATCH --output=/scratch/gw1107/capstone/output_log/test_multifiles_train.log
 
source activate NYU_DL 
python3 2_finetune_on_pretrained_longformer.py
