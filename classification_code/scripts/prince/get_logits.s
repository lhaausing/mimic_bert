#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=get_logits
#SBATCH --mail-type=END
#SBATCH --mail-user=xl3119@nyu.edu
#SBATCH --output=/scratch/xl3119/tf_icd/get_logits.log

overlay_ext3=/scratch/xl3119/tf_icd/overlay-10GB-400K.ext3
model_name=bert_base
model_dir=../checkpt/bert_base_bs32_ns28_60.pt
batch_size=8
ngram_size=28
n_gpu=4

singularity \
    exec --nv --overlay $overlay_ext3:ro \
    /beegfs/work/public/singularity/centos-7.8.2003.sif  \
    /bin/bash -c "module load  anaconda3/5.3.1; \
                  module load cuda/10.1.105; \
                  module load gcc/6.3.0; \
                  source /share/apps/anaconda3/5.3.1/etc/profile.d/conda.sh; \
                  conda activate /ext3/cenv; \
                  python3 get_logits.py \
                              --data_dir ../data \
                              --model_name ../${model_name} \
                              --model_dir ${model_dir} \
                              --batch_size ${batch_size} \
                              --ngram_size ${ngram_size} \
                              --use_ngram \
                              --sep_cls \
                              --n_gpu ${n_gpu} "
