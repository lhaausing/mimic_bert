#!/bin/bash

#SBATCH --partition=gpu8_medium
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --gres=gpu:8
#SBATCH --mail-type=END
#SBATCH --mail-user=xl3119@nyu.edu
#SBATCH --output=/gpfs/scratch/xl3119/capstone/out_log/longformer_mimic_tokenizer.log

module load anaconda3/gpu/5.2.0
module load cuda/10.1.105
module load gcc/8.1.0
source activate mimic_bert
export PYTHONPATH=/gpfs/share/apps/anaconda3/gpu/5.2.0/envs/mimic_bert/lib/python3.8/site-packages:$PYTHONPATH

cd /gpfs/scratch/xl3119/capstone/mimic_bert


python3 2_finetune_init_lf.py
