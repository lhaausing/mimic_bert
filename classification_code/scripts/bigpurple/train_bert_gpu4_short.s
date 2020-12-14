#!/bin/bash

#SBATCH --partition=gpu4_short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:4
#SBATCH --mail-type=END
#SBATCH --mail-user=hl3664@nyu.edu

model_name=Bert_base
batch_size=32
n_gpu=4
n_epochs=45
seed=28
max_len=512
layer=13
tokenizer=Bert_base
checkpt_path=checkpoints/local_bs${batch_size}_seed${seed}

echo "model: ${model_name}, batch_size: ${batch_size}, seed: ${seed}, freeze :${layer}, partition: gpu4_short, max_len: 512
"

module load anaconda3/gpu/5.2.0
module load cuda/10.1.105
module load gcc/8.1.0
source activate bento
export PYTHONPATH=/gpfs/share/apps/anaconda3/gpu/5.2.0/envs/bento/lib/python3.8/site-packages:$PYTHONPATH


python run.py \
  --seed ${seed} \
  --max_len ${max_len} \
  --data_dir data/mimic_preprocessed_data \
  --model_name ${model_name} \
  --n_epochs ${n_epochs} \
  --batch_size ${batch_size} \
  --n_gpu ${n_gpu} \
  --checkpt_path ${checkpt_path} \
  --load_data_cache \
  --save_best_f \
  --save_best_auc \
  --tokenizer ${tokenizer} \
  --freeze ${layer}
