#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:p40:4
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=bert_icd_pred
#SBATCH --mail-type=END
#SBATCH --mail-user=xl3119@nyu.edu
#SBATCH --output=/scratch/xl3119/tf_icd/ngram_ns28_bs32_seed66.log

overlay_ext3=/scratch/xl3119/tf_icd/overlay-10GB-400K.ext3
model_name=bert_base
batch_size=48
ngram_size=28
n_gpu=4
n_epochs=60
seed=66
checkpt_path=../global_${model_name}_ns${ngram_size}_mp${maxpool_size}_seed${seed}sepcls_invw.pt

cd /scratch/xl3119/tf_icd/tf_icd_coding

singularity \
    exec --nv --overlay $overlay_ext3:ro \
    /beegfs/work/public/singularity/centos-7.8.2003.sif  \
    /bin/bash -c "module load  anaconda3/5.3.1; \
                  module load cuda/10.1.105; \
                  module load gcc/6.3.0; \
                  source /share/apps/anaconda3/5.3.1/etc/profile.d/conda.sh; \
                  conda activate /ext3/cenv; \
                  python3 run.py \
                              --model_name ../${model_name} \
                              --n_epochs ${n_epochs} \
                              --batch_size ${batch_size} \
                              --ngram_size ${ngram_size} \
                              --use_ngram \
                              --n_gpu ${n_gpu} \
                              --checkpt_path ${checkpt_path} \
                              --save_best_f1 \
                              --save_best_auc \
                              --data_dir ../data \
                              #--load_data_cache \
                              --seed ${seed} "
