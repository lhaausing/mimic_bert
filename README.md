# mimic_bert
This is a repository for all stufss during the training BERT in medical notes.

## Preprocessing
`1_mimic_preprocess_with_digits.ipynb` and ``1_mimic_preprocess_without_digits.ipynb`` preprocess the raw text data and split it with paragraph. Then it's saved as `Preproc0_clinical_sentences_all_with_number.txt` and `Preproc0_clinical_sentences_all_without_number.txt`. One removes all digits,  and the other keeps all digits.
Then split the two files into train and eval sperately. 

## Train Longformer

### With longformer tokenizer

#### Split the train data into differen files 

`split -a 2 -l 256000 -d Preproc0_clinical_sentences_all_with_number_train.txt ./splited_train/train_`

#### Eval on longformer
`2_finetune_on_pretrained_longformer-evalonbase.py` run vanilla longformer with `Preproc0_clinical_sentences_all_with_number_train & _ val` data for evaluation only. 

Performance (Evaluation for validataion file):

* perplexity = 134.2684929739575

* bpc = 7.068976995932668


#### Train and eval o 
`2_finetune_on_pretrained_longformer.py` train longformer for 3000 steps. with `Preproc0_clinical_sentences_all_with_number_train & _ val` data for MLM, evaluate and save the final models in folder `/Clinical-longformer-pretrain-models`. 

Performance (Evaluation for validataion file):

* perplexity = 7.681640762093989

* bpc = 2.941414496690642

