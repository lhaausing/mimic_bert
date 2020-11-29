#!/bin/bash

#SBATCH --partition=gpu4_dev
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:4
#SBATCH --mail-type=END
#SBATCH --mail-user=xl3119@nyu.edu

model_name=bert-base-uncased
batch_size=32
n_gpu=4
seeds=6-23-28-36-66
checkpt_path=/gpfs/scratch/xl3119/checkpoints
save_preds_dir=/gpfs/scratch/xl3119/preds

module load anaconda3/gpu/5.2.0
module load cuda/10.1.105
module load gcc/8.1.0
source activate bento
export PYTHONPATH=/gpfs/share/apps/anaconda3/gpu/5.2.0/envs/bento/lib/python3.8/site-packages:$PYTHONPATH

cd /gpfs/scratch/xl3119/tf_icd_coding

python get_preds.py \
  --seeds ${seeds} \
  --data_dir ../data \
  --model_name ${model_name} \
  --batch_size ${batch_size} \
  --n_gpu ${n_gpu} \
  --checkpt_path ${checkpt_path} \
  --save_preds_dir ${save_preds_dir}
