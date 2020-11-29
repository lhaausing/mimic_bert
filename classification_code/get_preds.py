# coding=utf-8

import os
import re
import sys
import glob
import pickle
import random
import logging
import argparse
from tqdm import tqdm
from os.path import join

import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Dataset, DataLoader

import transformers
from transformers import AutoTokenizer, AutoModel

from data import *
from utils import *
from models import *

set_global_logging_level(logging.ERROR, ["transformers", "nlp", "torch", "tensorflow", "tensorboard", "wandb"])

def get_preds_and_metrics(args, model, val_loader, pt):
    model.eval()
    total_loss = 0.
    num_examples = 0
    criterion = torch.nn.BCEWithLogitsLoss()
    k = 5
    y = []
    yhat = []
    yhat_raw = []

    with torch.no_grad():
        for idx, (input_ids, attn_mask, labels) in tqdm(enumerate(val_loader)):
            if args.use_ngram:
                ngram_encoding = get_ngram_encoding(args, attn_mask.to(args.device), args.ngram_size).cpu()
                logits = model(input_ids, ngram_encoding)
                loss = criterion(logits.to(args.device), labels.to(args.device))

                total_loss += loss.item() * logits.size()[0]
                num_examples += logits.size()[0]

                y.append(labels.cpu().detach().numpy())
                yhat.append(np.round(torch.sigmoid(logits).cpu().detach().numpy()))
                yhat_raw.append(torch.sigmoid(logits).cpu().detach().numpy())
            else:
                input_ids, attn_mask, list_labels = get_val_snippets(args, input_ids, attn_mask, labels)
                batch_loss = 0.
                num_snippets = 0
                all_preds = []
                for i in range(len(input_ids)):
                    logits = model(input_ids[i].to(args.device), attn_mask[i].to(args.device))
                    loss = criterion(logits.to(args.device), list_labels[i].to(args.device))

                    num_snippets += input_ids[i].size(0)
                    batch_loss += loss.item() * input_ids[i].size(0)

                    logits = torch.mean(torch.sigmoid(logits), dim=0)
                    all_preds.append(logits.unsqueeze(0))
                #Report results
                total_loss += batch_loss
                logits = torch.cat(all_preds, dim=0)
                num_examples += num_snippets

                y.append(labels.cpu().detach().numpy())
                yhat.append(np.round(logits.cpu().detach().numpy()))
                yhat_raw.append(logits.cpu().detach().numpy())

        # Compute scores with results
        y = np.concatenate(y, axis=0)
        yhat = np.concatenate(yhat, axis=0)
        yhat_raw = np.concatenate(yhat_raw, axis=0)
        metrics = all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)

        print('validation loss is {}.'.format(total_loss/num_examples))
        print("[MACRO] acc, prec, rec, f1, auc")
        print("{}, {}, {}, {}, {}".format(metrics["acc_macro"],
                                          metrics["prec_macro"],
                                          metrics["rec_macro"],
                                          metrics["f1_macro"],
                                          metrics["auc_macro"]))
        print("[MICRO] accuracy, precision, recall, f-measure, AUC")
        print("{}, {}, {}, {}, {}".format(metrics["acc_micro"],
                                          metrics["prec_micro"],
                                          metrics["rec_micro"],
                                          metrics["f1_micro"],
                                          metrics["auc_micro"]))

        for metric, val in metrics.items():
            if metric.find("rec_at") != -1:
                print("{}: {}".format(metric, val))
        sys.stdout.flush()

        pickle.dump(yhat_raw, open(join(args.save_preds_dir,pt+'_preds.pkl'),'wb'))
        pickle.dump(y, open(join(args.save_preds_dir,'top_50_y.pkl'),'wb'))

def main():
    parser = argparse.ArgumentParser()

    #required parameters
    parser.add_argument("--model_name", default=None, type=str, required=True,
                        help="Model name or directory from transformers library or local dir. Tokenizer uses the same name or dir.")
    parser.add_argument("--use_ngram", action="store_true",
                        help="Whether use ngram_embeddings.")
    parser.add_argument("--n_epochs", default=30, type=int,
                        help="Number of epochs of training.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size for training and validation.")
    parser.add_argument("--ngram_size", default=32, type=int,
                        help="Size of the N-Gram that one's using.")
    parser.add_argument("--max_len", default=384, type=int,
                        help="sliding window stride. Should be <=510.")
    parser.add_argument("--n_gpu", default=1, type=int,
                        help="Suggested to train on multiple gpus if batch size > 8 and n-gram size < 32.")
    parser.add_argument("--device", default="cuda:0", type=str,
                        help="Normally this doesn't matter.")
    parser.add_argument("--seeds", type=str, default="6-23-28-36-66",
                        help="You need to provide a bunch of seeds here, splitted by _ .")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. It should contain a training set and a validation set.")
    parser.add_argument("--checkpt_path", default="/gpfs/scratch/xl3119/checkpoints", type=str,
                        help="Path to saved checkpoints.")
    parser.add_argument("--save_preds_dir", default="/gpfs/scratch/xl3119/preds", type=str, required=True,
                        help="dir for saved preds.")
    args = parser.parse_args()

    #define tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    #get the exact seeds
    seeds = args.seeds.split("-")
    #get dataloaders
    if args.use_ngram:
        val_dataset = load_cache(args)[1]
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=args.batch_size,
                                collate_fn=val_dataset.mimic3_col_func,
                                shuffle=False)
    else:
        val_dataset = load_tensor_cache(args)[1]
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=args.batch_size,
                                shuffle=False)

    if args.use_ngram:
        all_best_f1_names = ['ngram_bs{}_seed{}_best_f1'.format(args.batch_size, seed) for seed in seeds]
        all_best_auc_names = ['ngram_bs{}_seed{}_best_auc'.format(args.batch_size, seed) for seed in seeds]
        model =  NGramTransformer(args.model_name,args.ngram_size)
    else:
        all_best_f1_names = ['local_bs{}_seed{}_best_f1'.format(args.batch_size, seed) for seed in seeds]
        all_best_auc_names = ['local_bs{}_seed{}_best_auc'.format(args.batch_size, seed) for seed in seeds]
        model = snippet_bert(args.model_name)

    model = model.to(args.device)

    if args.n_gpu > 1:
        device_ids = [_ for _ in range(args.n_gpu)]
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    #get snippets for best f1 models
    for pt in all_best_f1_names:
        state_dict = torch.load(join(args.checkpt_path,pt+'.pt')).state_dict()
        model.load_state_dict(state_dict)
        get_preds_and_metrics(args, model, val_loader, pt)

    #get snippets for best auc models
    for pt in all_best_auc_names:
        state_dict = torch.load(join(args.checkpt_path,pt+'.pt')).state_dict()
        model.load_state_dict(state_dict)
        get_preds_and_metrics(args, model, val_loader, pt)

if __name__ == '__main__':
    main()
