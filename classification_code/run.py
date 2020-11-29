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

def eval(args, model, val_loader):
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
                for _ in range(len(input_ids)):
                    logits = model(input_ids[_].to(args.device), attn_mask[_].to(args.device))
                    loss = criterion(logits.to(args.device), list_labels[_].to(args.device))

                    num_snippets += input_ids[_].size(0)
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

        return metrics

def train(args, train_loader, val_loader):
    # Define model, parallel training, optimizer.
    if args.use_ngram: model = NGramTransformer(args.model_name,args.ngram_size)
    else: model = snippet_bert(args.model_name)
    model = model.to(args.device)

    if args.n_gpu > 1:
        device_ids = [_ for _ in range(args.n_gpu)]
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],lr=float(args.lr),eps=float(args.eps))

    history_metrics = []
    start_epoch = 0
    best_f1 = 0.
    best_auc = 0.
    if args.train_from_checkpt:
        model = torch.load(args.checkpt_path+'_last.pt')
        history = pickle.load(open(args.checkpt_path+'_hist.pkl','rb'))
        history_metrics = history['hist'][-1]['epoch']
        best_f1 = hist['best_f1']
        best_auc = hist['best_auc']
        start_epoch = history_metrics[-1]['epoch']

    criterion = torch.nn.BCEWithLogitsLoss()
    model.zero_grad()

    #Train
    for i in range(start_epoch, args.n_epochs):
        total_loss = 0.
        num_examples = 0
        for idx, (input_ids, attn_mask, labels) in tqdm(enumerate(train_loader)):

            model.train()
            if args.use_ngram:
                ngram_encoding = get_ngram_encoding(args, attn_mask.to(args.device), args.ngram_size).cpu()
                logits = model(input_ids, ngram_encoding)
                loss = criterion(logits.to(args.device), labels.to(args.device))

                loss.backward()
                optimizer.step()
                model.zero_grad()

                num_examples += logits.size()[0]
                total_loss += loss.item() * logits.size()[0]

            else:
                input_ids, attn_mask, labels = get_train_snippets(args, input_ids, attn_mask, labels)

                batch_loss = 0.
                num_snippets = 0
                for _ in range(len(input_ids)):

                    logits = model(input_ids[_], attn_mask[_])
                    loss = criterion(logits.to(args.device), labels[_].to(args.device))

                    loss.backward()
                    optimizer.step()
                    model.zero_grad()

                    #Aggregatings losses
                    num_snippets += input_ids[_].size(0)
                    batch_loss += loss.item() * input_ids[_].size(0)

                num_examples += num_snippets
                total_loss += batch_loss

        print('')
        print('epoch: {}'.format(i+1))
        print('train loss is {}.'.format(total_loss / num_examples))
        sys.stdout.flush()

        metrics = eval(args, model, val_loader)

        if args.save_best_f1:
            if metrics["f1_micro"] > best_f1:
                best_f1 = metrics["f1_micro"]
                torch.save(model, args.checkpt_path+'_best_f1.pt')
        if args.save_best_f1:
            if metrics["auc_micro"] > best_auc:
                best_auc = metrics["auc_micro"]
                torch.save(model, args.checkpt_path+'_best_auc.pt')

        history_metrics.append({'epoch':i, 'metrics':metrics})
        torch.save(model, args.checkpt_path+'_last.pt')
        pickle.dump({'best_f1': best_f1,
                     'best_auc': best_auc,
                     'hist':history_metrics}, open(args.checkpt_path+'_hist.pkl','wb'))


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def main():
    parser = argparse.ArgumentParser()

    #required parameters
    parser.add_argument("--model_name", default=None, type=str, required=True,
                        help="Model name or directory from transformers library or local dir. Tokenizer uses the same name or dir.")
    parser.add_argument("--attention", action="store_true",
                        help="Whether plug in the attention layer after the Transformers LM.")
    parser.add_argument("--use_ngram", action="store_true",
                        help="Whether use ngram_embeddings.")
    parser.add_argument("--n_epochs", default=30, type=int,
                        help="Number of epochs of training.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size for training and validation.")
    parser.add_argument("--ngram_size", default=32, type=int,
                        help="Size of the N-Gram that one's using.")
    parser.add_argument("--maxpool_size", default=32, type=int,
                        help="Size of the Max-pooling. Probably need to be larger than 28.")
    parser.add_argument("--max_len", default=384, type=int,
                        help="sliding window stride. Should be <=510.")
    parser.add_argument("--n_gpu", default=1, type=int,
                        help="Suggested to train on multiple gpus if batch size > 8 and n-gram size < 32.")
    parser.add_argument("--lr", default="2e-5", type=str,
                        help="Learning Rate.")
    parser.add_argument("--eps", default="1e-8", type=str,
                        help="Epsilon.")
    parser.add_argument("--device", default="cuda:0", type=str,
                        help="Normally this doesn't matter.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. It should contain a training set and a validation set.")
    parser.add_argument("--load_data_cache", action="store_true",
                        help="load_data_cache.")
    parser.add_argument("--checkpt_path", default="./model", type=str,
                        help="Saving dir of the final checkpoint.")
    parser.add_argument("--train_from_checkpt", action="store_true",
                        help="Do you want to train for checkpoints you have? You'll also have a eval metric history.")
    parser.add_argument("--save_best_f1", action="store_true",
                        help="save best f1 checkpoints.")
    parser.add_argument("--save_best_auc", action="store_true",
                        help="save best auc checkpoints.")

    args = parser.parse_args()
    set_seed(args)

    #define tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.use_ngram:
        if args.load_data_cache:
            train_dataset, val_dataset, test_dataset = load_cache(args)
        else:
            #load csv file
            print('Start preprocessing ngram bert data.')
            sys.stdout.flush()
            train_dataset, val_dataset, test_dataset = load_data_and_save_cache(args, tokenizer)
            print('finished')
            sys.stdout.flush()

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  collate_fn=train_dataset.mimic3_col_func,
                                  shuffle=True)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=args.batch_size,
                                collate_fn=val_dataset.mimic3_col_func,
                                shuffle=True)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=args.batch_size,
                                 collate_fn=test_dataset.mimic3_col_func,
                                 shuffle=True)
    else:
        if args.load_data_cache:
            train_dataset, val_dataset, test_dataset = load_tensor_cache(args)
        else:
            print('Start preprocessing local bert data.')
            sys.stdout.flush()
            train_dataset, val_dataset, test_dataset = load_data_and_save_tensor_cache(args, tokenizer)
            print('finished')
            sys.stdout.flush()

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=args.batch_size,
                                shuffle=True)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True)

    print('Data loader is loaded, start training.')
    sys.stdout.flush()
    #train
    train(args, train_loader, val_loader)

if __name__ == '__main__':
    main()
