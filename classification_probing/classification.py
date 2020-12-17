import sys
import os
import re
import sys
import glob
import pickle
import random
from data import *
from utils import *
from models import *

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
from transformers import BertTokenizer, LongformerTokenizer


class ClassficationProbing(nn.Module):
    def __init__(self,n_class=50):
        super().__init__()
        self.c = n_class
        self.hid = 768

        self.layers=nn.Linear(self.hid,self.c)

  def forward(self,x):
        logits=self.layers(x)
        return logits


def load_data(args):
    train_dataset, val_dataset, test_dataset = load_tensor_cache(args)
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
                            args.checkpt_path
    with open(args.checkpt_path+"_"+args.model_name+'_probing_train.pkl', 'rb') as ifp:
        train_layers_file = pickle.load(ifp)
    with open(args.checkpt_path+"_"+args.model_name+'_probing_val.pkl', "rb") as input_file:
        val_layers_file = pickle.load(input_file)

    train_layers=train_layers_file['all_layers']
    val_layers=val_layers_file['all_layers']
    return train_layers,val_layers,train_loader,val_loader

def eval(args, model, val_loader,val_layer):
    model.eval()
    total_loss = 0.
    num_examples = 0
    criterion = torch.nn.BCEWithLogitsLoss()
    k = 5
    y = []
    yhat = []
    yhat_raw = []

    with torch.no_grad():
        for layer_re,(idx, (input_ids, attn_masks, labels)) in zip(val_layer,enumerate(val_loader)):
            all_preds = []
            for i in range(len(input_ids)):   
        
                layer=layer_re[i]
                target=labels[i].unsqueeze(0)

                logits = model(layer).unsqueeze(0)

                loss = criterion(logits.to(args.device), target.to(args.device))
                logits = torch.sigmoid(logits)
            
                batch_loss = 0.
                num_snippets = 0

                num_snippets += input_ids[0].size(0)
                batch_loss += loss.item() * input_ids[0].size(0)
                all_preds.append(logits)

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
        # print("[MACRO] acc, prec, rec, f1, auc")
        # print("{}, {}, {}, {}, {}".format(metrics["acc_macro"],
        #                                   metrics["prec_macro"],
        #                                   metrics["rec_macro"],
        #                                   metrics["f1_macro"],
        #                                   metrics["auc_macro"]))
        # print("[MICRO] accuracy, precision, recall, f-measure, AUC")
        # print("{}, {}, {}, {}, {}".format(metrics["acc_micro"],
        #                                   metrics["prec_micro"],
        #                                   metrics["rec_micro"],
        #                                   metrics["f1_micro"],
        #                                   metrics["auc_micro"]))

        for metric, val in metrics.items():
            if metric.find("rec_at") != -1:
                print("{}: {}".format(metric, val))
        
        metrics["val_loss"]=total_loss/num_examples

        return metrics


def train(args, train_loader, val_loader,train_layer,val_layer):
    model = ClassficationProbing(50)
    model = model.to(args.device)

    if args.n_gpu > 1:
        device_ids = [_ for _ in range(args.n_gpu)]
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],lr=float(args.lr),eps=float(args.eps))

    history_metrics = []
    start_epoch = 0
    best_f1 = 0.
    best_auc = 0.
    best_val=0
    overall_res=[]

    criterion = torch.nn.BCEWithLogitsLoss()
    model.zero_grad()

    #Train
    for i in range(start_epoch, args.n_epochs):
        total_loss = 0.
        num_examples = 0
        for layer_tp,(idx, (input_ids, attn_masks, labels)) in zip(train_layer,enumerate(train_loader)):

            model.train()
            for j in range(len(input_ids)):
                layer=layer_tp[j]
                logits = model(layer).unsqueeze(0)

                target=labels[j].unsqueeze(0)
                loss = criterion(logits.to(args.device), target.to(args.device))
                logits = torch.mean(torch.sigmoid(logits), dim=0)


                batch_loss = 0.
                num_snippets = 0

                loss.backward()
                optimizer.step()
                model.zero_grad()

                #Aggregatings losses
                num_snippets += input_id.size(0)
                batch_loss += loss.item() * input_id.size(0)

            num_examples += num_snippets
            total_loss += batch_loss
        
        print('')
        print('epoch: {}'.format(i+1))
        print('train loss is {}.'.format(total_loss / num_examples))
        sys.stdout.flush()
        metrics = eval(args, model, val_loader,val_layer)

        if best_val>metrics["val_loss"]:
          break
        else:
          best_val=metrics["val_loss"]
    
    return metrics

def process(args,train_layers,val_layers,train_loader,val_loader):
    n=len(train_layers)
    total_res=[]

    for layer_id in range(n):
        model=ClassficationProbing(50).to(args.device)
        optimizer = optim.Adam(model.parameters(),lr=float(args.lr),eps=float(args.eps))
        criterion = torch.nn.BCEWithLogitsLoss()
        print("\n")
        print("=======================================================")
        print("layer id: ",layer_id)
        train_layer=train_layers[layer_id]
        val_layer=val_layers[layer_id]

        res=train(args, train_loader, val_loader,train_layer,val_layer)
        print("[MACRO] acc, prec, rec, f1, auc")
        print("{}, {}, {}, {}, {}".format(res["acc_macro"],
                                        res["prec_macro"],
                                        res["rec_macro"],
                                        res["f1_macro"],
                                        res["auc_macro"]))
        total_res.append(res)
        return total_res

if name=="__main__":
    parser = argparse.ArgumentParser()

    #required parameters
    parser.add_argument("--model_name", default=None, type=str, required=True,
                        help="Model name or directory from transformers library or local dir. Tokenizer uses the same name or dir.")
    parser.add_argument("--n_epochs", default=30, type=int,
                        help="Number of epochs of training.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size for training and validation.")
    parser.add_argument("--max_len", default=512, type=int,
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
    parser.add_argument("--tokenizer", default="Longformer", type=str,
                        help="Saving dir of the final checkpoint.")

    args = parser.parse_args()
    print("Processing data...")
    sys.stdout.flush()
    train_layers,val_layers,train_loader,val_loader=load_data(args)
    res=process(args,train_layers,val_layers,train_loader,val_loader)
    pickle.dump({"res":res}, open(args.checkpt_path+'_'+args.model_name+'_res.pkl','wb'))


