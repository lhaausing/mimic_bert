import logging
import argparse
import os
from os.path import join
import math
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from torch.utils.data import ConcatDataset
import glob
import numpy as np

from transformers import TextDataset, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer
# use longformer directly instead of using create long model for Roberta
from transformers import LongformerForMaskedLM, LongformerConfig, BertTokenizer
from transformers import TrainingArguments, HfArgumentParser
from transformers.modeling_longformer import LongformerSelfAttention

# Choose GPU
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def use_embeddings_fasttext(model, word_embeddings):
    emb_tensor = torch.from_numpy(word_embeddings)
    size = emb_tensor.size()
    emb_class = nn.Embedding(size[0], size[1],padding_idx=0)
    emb_class.weight.data = emb_tensor
    model.longformer.embeddings.word_embeddings = emb_class

    return model

def pretrain_and_evaluate(args, model, tokenizer, eval_only, model_path=None):
    # train from scrath if model_path=None
    def _dataset(file_path):
        return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=tokenizer.max_len)

    val_dataset = _dataset(args.val_datapath)
    if eval_only:
        print("Assign validation dataset")
        train_dataset = val_dataset
    else:
        logger.info(f'Loading and tokenizing training data is usually slow: {args.train_datapath}')

        train_dataset = ConcatDataset([_dataset(f) for f in glob.glob('/gpfs/scratch/xl3119/capstone/data/splited_train/*')])

    print("Creating data collator with mlm")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    print("Start Trainer")
    trainer = Trainer(model=model, args=args, data_collator=data_collator,
                      train_dataset=train_dataset, eval_dataset=val_dataset, prediction_loss_only=True,)

    if not eval_only:
        trainer.train(model_path=model_path) # None train from scratch
        trainer.save_model(args.output_dir) # save model to the output_dir




    # Evaluation
    results = {}

    logger.info("*** Evaluate ***")

    eval_loss = trainer.evaluate()
    eval_loss = eval_loss['eval_loss']

    perplexity = math.exp(eval_loss)
    results["perplexity"] = perplexity
    results["bpc"] = eval_loss/math.log(2)

    output_eval_file = os.path.join(training_args.output_dir, "eval_results_mlm.txt")
    with open(output_eval_file, "a") as writer:
        writer.write("***** Eval results *****")
        logger.info("***** Eval results *****")
        for key, value in results.items():
            logger.info(f"  {key} = {value}")
            writer.write(f"{key} = {value}\n")


@dataclass
class ModelArgs:
    attention_window: int = field(default=512, metadata={"help": "Size of attention window"})
    max_pos: int = field(default=4096, metadata={"help": "Maximum position"})

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    parser = HfArgumentParser((TrainingArguments, ModelArgs,))

    training_args, model_args = parser.parse_args_into_dataclasses(look_for_args_file=False, args=[
    '--output_dir', '/gpfs/scratch/xl3119/capstone/checkpoints/longformer_mimic_tokenizer',
    '--warmup_steps', '500',
    '--learning_rate', '0.0005',
    '--weight_decay', '0.01',
    '--adam_epsilon', '1e-6',
    '--max_steps', '150000',
    '--logging_steps', '1000',
    '--save_steps', '1000',
    '--max_grad_norm', '5.0',
    '--per_gpu_eval_batch_size', '2',
    '--per_gpu_train_batch_size', '1',  # 32GB gpu with fp32
    '--gradient_accumulation_steps', '4',
    #'--evaluate_during_training', # this is removed to reduce training time
    '--do_train',
    ])
    train_fn = '/gpfs/scratch/xl3119/capstone/data/Preproc0_clinical_sentences_all_without_number_train_patients.txt'
    val_fn = '/gpfs/scratch/xl3119/capstone/data/Preproc0_clinical_sentences_all_without_number_val_patients.txt'
    # these are small file for test
#     train_fn = './Preprocessed_Data/test_clinical_sentences_all_with_number_train.txt'
#     val_fn = './Preprocessed_Data/test_clinical_sentences_all_with_number_val.txt'
    training_args.val_datapath = val_fn
    training_args.train_datapath = train_fn

##################### use pretrianed longformer in transformer
    init_config = LongformerConfig.from_json_file('config_files/longformer_base_4096/config.json')
    mimic_tokenizer = BertTokenizer.from_pretrained('mimic_tokenizer')
    word_embeddings =  np.loadtxt(join('/gpfs/scratch/xl3119/capstone/wd_emb',"word_embedding_matrix.txt"))
    longformer_model = LongformerForMaskedLM(init_config)
    longformer_model = use_embeddings_fasttext(longformer_model, word_embeddings)
    # longformer_tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

    logger.info('Train and eval with Longformer pretrained ...')
    pretrain_and_evaluate(training_args, longformer_model, mimic_tokenizer, eval_only=False, model_path=None\
                          #,model_path=training_args.output_dir # Local path to the model if the model to train has been ins tantiated from a local path.
                         )
