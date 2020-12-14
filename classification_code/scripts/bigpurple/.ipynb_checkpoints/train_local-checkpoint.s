#!/bin/bash

#SBATCH --partition=gpu8_medium
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:8
#SBATCH --mail-type=END
#SBATCH --mail-user=xl3119@nyu.edu

model_name=bert-base-uncased
batch_size=32
n_gpu=8
n_epochs=60
seed=28
checkpt_path=../checkpoints/local_bs${batch_size}_seed${seed}

echo "type: ngram, model: ${model_name}, batch_size: ${batch_size}, seed: ${seed}"

module load anaconda3/gpu/5.2.0
module load cuda/10.1.105
module load gcc/8.1.0
source activate bento
export PYTHONPATH=/gpfs/share/apps/anaconda3/gpu/5.2.0/envs/bento/lib/python3.8/site-packages:$PYTHONPATH

cd /gpfs/scratch/xl3119/tf_icd_coding

python run.py \
  --seed ${seed} \
  --data_dir ../data \
  --model_name ${model_name} \
  --n_epochs ${n_epochs} \
  --batch_size ${batch_size} \
  --n_gpu ${n_gpu} \
  --checkpt_path ${checkpt_path} \
  --load_data_cache \
  --save_best_f \
  --save_best_auc
