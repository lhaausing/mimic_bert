import os
import re
import sys
import glob
import json
import pickle
import random
import logging
import argparse
from tqdm import tqdm
from os.path import join

import math
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Dataset, DataLoader

import transformers
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer,LongformerTokenizer

from data import *
from models import *

def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)

def load_cache(args):
    train_dataset = pickle.load(open(join(args.data_dir,'train_50.pkl'),'rb'))
    val_dataset = pickle.load(open(join(args.data_dir,'dev_50.pkl'),'rb'))
    test_dataset = pickle.load(open(join(args.data_dir,'test_50.pkl'),'rb'))

    return train_dataset, val_dataset, test_dataset

def load_tensor_cache(args):
    train_dataset = pickle.load(open(join(args.data_dir,'train_50_tensor.pkl'),'rb'))
    val_dataset = pickle.load(open(join(args.data_dir,'dev_50_tensor.pkl'),'rb'))
    test_dataset = pickle.load(open(join(args.data_dir,'test_50_tensor.pkl'),'rb'))

    return train_dataset, val_dataset, test_dataset

def load_data_and_save_cache(args, tokenizer):
    train_df = pd.read_csv(join(args.data_dir,'train_50.csv'),engine='python')
    val_df = pd.read_csv(join(args.data_dir,'dev_50.csv'),engine='python')
    test_df = pd.read_csv(join(args.data_dir,'test_50.csv'),engine='python')

    #load text
    train_texts = [elem[6:-6] for elem in train_df['TEXT']]
    val_texts = [elem[6:-6] for elem in val_df['TEXT']]
    test_texts = [elem[6:-6] for elem in test_df['TEXT']]

    #load and transform labels
    with open(join(args.data_dir,'TOP_50_CODES.csv'),'r') as f:
        idx2code = [elem[:-1] for elem in f.readlines()]
        f.close()
    code2idx = {elem:i for i, elem in enumerate(idx2code)}

    train_codes = [[code2idx[code] for code in elem.split(';')] for elem in train_df['LABELS']]
    val_codes = [[code2idx[code] for code in elem.split(';')] for elem in val_df['LABELS']]
    test_codes = [[code2idx[code] for code in elem.split(';')] for elem in test_df['LABELS']]

    train_labels = [sum([torch.arange(50) == torch.Tensor([code]) for code in sample]) for sample in train_codes]
    val_labels = [sum([torch.arange(50) == torch.Tensor([code]) for code in sample]) for sample in val_codes]
    test_labels = [sum([torch.arange(50) == torch.Tensor([code]) for code in sample]) for sample in test_codes]

    #build dataset and dataloader
    train_dataset = mimic3_dataset(train_texts, train_labels, args.ngram_size, tokenizer, args.use_ngram)
    val_dataset = mimic3_dataset(val_texts, val_labels, args.ngram_size, tokenizer, args.use_ngram)
    test_dataset = mimic3_dataset(test_texts, test_labels, args.ngram_size, tokenizer, args.use_ngram)

    pickle.dump(train_dataset, open(join(args.data_dir,'train_50.pkl'),'wb'))
    pickle.dump(val_dataset, open(join(args.data_dir,'dev_50.pkl'),'wb'))
    pickle.dump(test_dataset, open(join(args.data_dir,'test_50.pkl'),'wb'))

    return train_dataset, val_dataset, test_dataset

def tokenize(args):
    #define tokenizer: checkpoint folder name
    print('Tokenizer: ', args.tokenizer)
    # immediate print in hpc 
    sys.stdout.flush()
    if args.tokenizer=="Bert":
        tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
    if args.tokenizer=="Longformer":
        tokenizer=LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_df = pd.read_csv(join(args.data_dir,'train_50.csv'),engine='python')
    val_df = pd.read_csv(join(args.data_dir,'dev_50.csv'),engine='python')
    test_df = pd.read_csv(join(args.data_dir,'test_50.csv'),engine='python')

    print("load text")
    sys.stdout.flush()
    train_texts = [elem[6:-6] for elem in train_df['TEXT']]
    val_texts = [elem[6:-6] for elem in val_df['TEXT']]
    test_texts = [elem[6:-6] for elem in test_df['TEXT']]

    train_inputs = tokenizer(train_texts, return_tensors='pt', padding=True)
    val_inputs = tokenizer(val_texts, return_tensors='pt', padding=True)
    test_inputs = tokenizer(test_texts, return_tensors='pt', padding=True)

    print("load and transform labels")
    sys.stdout.flush()
    with open(join(args.data_dir,'TOP_50_CODES.csv'),'r') as f:
        idx2code = [elem[:-1] for elem in f.readlines()]
        f.close()
    code2idx = {elem:i for i, elem in enumerate(idx2code)}

    train_codes = [[code2idx[code] for code in elem.split(';')] for elem in train_df['LABELS']]
    val_codes = [[code2idx[code] for code in elem.split(';')] for elem in val_df['LABELS']]
    test_codes = [[code2idx[code] for code in elem.split(';')] for elem in test_df['LABELS']]

    train_labels = [sum([torch.arange(50) == torch.Tensor([code]) for code in sample]) for sample in train_codes]
    val_labels = [sum([torch.arange(50) == torch.Tensor([code]) for code in sample]) for sample in val_codes]
    test_labels = [sum([torch.arange(50) == torch.Tensor([code]) for code in sample]) for sample in test_codes]

    train_labels = torch.cat([elem.unsqueeze(0) for elem in train_labels], dim=0).type('torch.FloatTensor')
    val_labels = torch.cat([elem.unsqueeze(0) for elem in val_labels], dim=0).type('torch.FloatTensor')
    test_labels = torch.cat([elem.unsqueeze(0) for elem in test_labels], dim=0).type('torch.FloatTensor')

    print("build dataset and dataloader")
    sys.stdout.flush()
    train_dataset = TensorDataset(train_inputs["input_ids"], train_inputs["attention_mask"], train_labels)
    val_dataset = TensorDataset(val_inputs["input_ids"], val_inputs["attention_mask"], val_labels)
    test_dataset = TensorDataset(test_inputs["input_ids"], test_inputs["attention_mask"], test_labels)

    pickle.dump(train_dataset, open(join(args.data_dir,'train_50_tensor.pkl'),'wb'))
    pickle.dump(val_dataset, open(join(args.data_dir,'dev_50_tensor.pkl'),'wb'))
    pickle.dump(test_dataset, open(join(args.data_dir,'test_50_tensor.pkl'),'wb'))

def load_data_and_save_tensor_cache(args, tokenizer):
    train_df = pd.read_csv(join(args.data_dir,'train_50.csv'),engine='python')
    val_df = pd.read_csv(join(args.data_dir,'dev_50.csv'),engine='python')
    test_df = pd.read_csv(join(args.data_dir,'test_50.csv'),engine='python')

    #load text
    train_texts = [elem[6:-6] for elem in train_df['TEXT']]
    val_texts = [elem[6:-6] for elem in val_df['TEXT']]
    test_texts = [elem[6:-6] for elem in test_df['TEXT']]

    train_inputs = tokenizer(train_texts, return_tensors='pt', padding=True)
    val_inputs = tokenizer(val_texts, return_tensors='pt', padding=True)
    test_inputs = tokenizer(test_texts, return_tensors='pt', padding=True)

    #load and transform labels
    with open(join(args.data_dir,'TOP_50_CODES.csv'),'r') as f:
        idx2code = [elem[:-1] for elem in f.readlines()]
        f.close()
    code2idx = {elem:i for i, elem in enumerate(idx2code)}

    train_codes = [[code2idx[code] for code in elem.split(';')] for elem in train_df['LABELS']]
    val_codes = [[code2idx[code] for code in elem.split(';')] for elem in val_df['LABELS']]
    test_codes = [[code2idx[code] for code in elem.split(';')] for elem in test_df['LABELS']]

    train_labels = [sum([torch.arange(50) == torch.Tensor([code]) for code in sample]) for sample in train_codes]
    val_labels = [sum([torch.arange(50) == torch.Tensor([code]) for code in sample]) for sample in val_codes]
    test_labels = [sum([torch.arange(50) == torch.Tensor([code]) for code in sample]) for sample in test_codes]

    train_labels = torch.cat([elem.unsqueeze(0) for elem in train_labels], dim=0).type('torch.FloatTensor')
    val_labels = torch.cat([elem.unsqueeze(0) for elem in val_labels], dim=0).type('torch.FloatTensor')
    test_labels = torch.cat([elem.unsqueeze(0) for elem in test_labels], dim=0).type('torch.FloatTensor')

    #build dataset and dataloader
    train_dataset = TensorDataset(train_inputs["input_ids"], train_inputs["attention_mask"], train_labels)
    val_dataset = TensorDataset(val_inputs["input_ids"], val_inputs["attention_mask"], val_labels)
    test_dataset = TensorDataset(test_inputs["input_ids"], test_inputs["attention_mask"], test_labels)

    pickle.dump(train_dataset, open(join(args.data_dir,'train_50_tensor.pkl'),'wb'))
    pickle.dump(val_dataset, open(join(args.data_dir,'dev_50_tensor.pkl'),'wb'))
    pickle.dump(test_dataset, open(join(args.data_dir,'test_50_tensor.pkl'),'wb'))

    return train_dataset, val_dataset, test_dataset

def get_ngram_encoding(args, attn_mask=None, ngram_size=None, sep_cls=True):

    sent_lens = torch.sum(attn_mask,1)
    if sep_cls:
        sent_lens -= 1
    max_sent_len = torch.max(sent_lens).item()
    num_ngram = [math.ceil(elem / ngram_size) for elem in sent_lens.tolist()]
    max_num_ngram = max(num_ngram)
    arange_t = torch.arange(max_sent_len).to(args.device)

    ngram_pos = [[min(j * ngram_size, sent_lens[i].item()) for j in range(elem+1)] for i, elem in enumerate(num_ngram)]
    for i in range(len(ngram_pos)):
        ngram_pos[i] = ngram_pos[i] + [-1]*(max_num_ngram+1-len(ngram_pos[i]))
    ngram_encoding = [torch.cat([((arange_t>=elem[i])*(arange_t<elem[i+1])).unsqueeze(0).to(args.device) for i in range(max_num_ngram)]).unsqueeze(0) for elem in ngram_pos]
    ngram_encoding = torch.cat(ngram_encoding)

    if sep_cls:
        size = ngram_encoding.size()
        zero_pos = torch.zeros(size[0],size[1],1,dtype=torch.bool).to(args.device)
        cls_pos = torch.BoolTensor([[[1]+[0]*(size[2])]]*size[0]).to(args.device)
        ngram_encoding = torch.cat([zero_pos, ngram_encoding], dim=2)
        ngram_encoding = torch.cat([cls_pos, ngram_encoding], dim=1)

    return ngram_encoding.type(torch.FloatTensor)

def get_train_snippets(args, input_ids, attn_masks, labels):

    lens = (torch.sum(attn_masks, dim=1)-2).type(torch.IntTensor).tolist()
    n_sni = [args.max_len*(int(elem/args.max_len)+1) for elem in lens]
    #Add max_len to tensor
    input_ids = torch.cat([input_ids, torch.Tensor([0]).repeat(input_ids.size(0),args.max_len).type(torch.LongTensor)], dim=1)
    attn_masks = torch.cat([attn_masks, torch.Tensor([0]).repeat(attn_masks.size(0),args.max_len).type(torch.LongTensor)], dim=1)
    #Extract discharge summary ids
    input_ids = [(input_ids[i,1:n_sni[i]+1],input_ids[i,n_sni[i]+1]) for i in range(input_ids.size(0))]
    attn_masks = [(attn_masks[i,1:n_sni[i]+1],attn_masks[i,n_sni[i]+1]) for i in range(attn_masks.size(0))]
    #Transform ids
    input_ids = [(elem[0].view(-1, args.max_len), elem[1].item()) for elem in input_ids]
    attn_masks = [(elem[0].view(-1, args.max_len), elem[1].item()) for elem in attn_masks]
    #Insert CLS and SEP ids
    input_ids = [(torch.Tensor([[101]]*elem[0].size(0)), elem[0], torch.Tensor([[102]]*(elem[0].size(0)-1)+[[elem[1]]])) for elem in input_ids]
    attn_masks = [(torch.Tensor([[1]]*elem[0].size(0)), elem[0], torch.Tensor([[1]]*(elem[0].size(0)-1)+[[elem[1]]])) for elem in attn_masks]
    input_ids = [torch.cat([elem[0].type(torch.LongTensor), elem[1].type(torch.LongTensor), elem[2].type(torch.LongTensor)], dim=1) for elem in input_ids]
    attn_masks = [torch.cat([elem[0].type(torch.LongTensor), elem[1].type(torch.LongTensor), elem[2].type(torch.LongTensor)], dim=1) for elem in attn_masks]
    #Get discharge summary labels
    labels = [labels[i,:].repeat(input_ids[i].size(0),1) for i in range(labels.size(0))]

    assert (len(input_ids)==len(attn_masks)==len(labels))
    input_ids = [text[i,:] for text in input_ids for i in range(text.size(0))]
    attn_masks = [text[i,:] for text in attn_masks for i in range(text.size(0))]
    labels = [text[i,:] for text in labels for i in range(text.size(0))]

    assert (len(input_ids)==len(attn_masks)==len(labels))
    ids = [_ for _ in range(len(labels))]
    random.shuffle(ids)
    input_ids = [input_ids[i].unsqueeze(0) for i in ids]
    attn_masks = [attn_masks[i].unsqueeze(0) for i in ids]
    labels = [labels[i].unsqueeze(0) for i in ids]

    final_input_ids = []
    final_attn_masks = []
    final_labels = []

    num_blocks = int(math.ceil(len(ids)/args.batch_size))
    input_ids = [torch.cat(input_ids[args.batch_size*i:args.batch_size*(i+1)], dim=0) for i in range(num_blocks)]
    attn_masks = [torch.cat(attn_masks[args.batch_size*i:args.batch_size*(i+1)], dim=0) for i in range(num_blocks)]
    labels = [torch.cat(labels[args.batch_size*i:args.batch_size*(i+1)], dim=0) for i in range(num_blocks)]

    return input_ids, attn_masks, labels

def get_val_snippets(args, input_ids, attn_masks, labels):

    lens = (torch.sum(attn_masks, dim=1)-2).type(torch.IntTensor).tolist()
    n_sni = [args.max_len*(int(elem/args.max_len)+1) for elem in lens]
    input_ids = torch.cat([input_ids, torch.Tensor([0]).repeat(input_ids.size(0),args.max_len).type(torch.LongTensor)], dim=1)
    attn_masks = torch.cat([attn_masks, torch.Tensor([0]).repeat(attn_masks.size(0),args.max_len).type(torch.LongTensor)], dim=1)
    #Extract discharge summary ids
    input_ids = [(input_ids[i,1:n_sni[i]+1],input_ids[i,n_sni[i]+1]) for i in range(input_ids.size(0))]
    attn_masks = [(attn_masks[i,1:n_sni[i]+1],attn_masks[i,n_sni[i]+1]) for i in range(attn_masks.size(0))]
    #Transform ids
    input_ids = [(elem[0].view(-1, args.max_len), elem[1].item()) for elem in input_ids]
    attn_masks = [(elem[0].view(-1, args.max_len), elem[1].item()) for elem in attn_masks]
    #Insert CLS and SEP ids
    input_ids = [(torch.Tensor([[101]]*elem[0].size(0)), elem[0], torch.Tensor([[102]]*(elem[0].size(0)-1)+[[elem[1]]])) for elem in input_ids]
    attn_masks = [(torch.Tensor([[1]]*elem[0].size(0)), elem[0], torch.Tensor([[1]]*(elem[0].size(0)-1)+[[elem[1]]])) for elem in attn_masks]
    input_ids = [torch.cat([elem[0].type(torch.LongTensor), elem[1].type(torch.LongTensor), elem[2].type(torch.LongTensor)], dim=1) for elem in input_ids]
    attn_masks = [torch.cat([elem[0].type(torch.LongTensor), elem[1].type(torch.LongTensor), elem[2].type(torch.LongTensor)], dim=1) for elem in attn_masks]
    #Get discharge summary labels
    labels = [labels[i,:].repeat(input_ids[i].size(0),1) for i in range(labels.size(0))]

    return input_ids, attn_masks, labels

def union_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)

def intersect_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)

def macro_accuracy(yhat, y):
    num = intersect_size(yhat, y, 0) / (union_size(yhat, y, 0) + 1e-10)
    return np.mean(num)

def macro_precision(yhat, y):
    num = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
    return np.mean(num)

def macro_recall(yhat, y):
    num = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
    return np.mean(num)

def macro_f1(yhat, y):
    prec = macro_precision(yhat, y)
    rec = macro_recall(yhat, y)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1


def all_macro(yhat, y):
    return macro_accuracy(yhat, y), macro_precision(yhat, y), macro_recall(yhat, y), macro_f1(yhat, y)

def micro_accuracy(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / union_size(yhatmic, ymic, 0)

def micro_precision(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / yhatmic.sum(axis=0)

def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / ymic.sum(axis=0)

def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1

def all_micro(yhatmic, ymic):
    return micro_accuracy(yhatmic, ymic), micro_precision(yhatmic, ymic), micro_recall(yhatmic, ymic), micro_f1(yhatmic, ymic)

def auc_metrics(yhat_raw, y, ymic):
    if yhat_raw.shape[0] <= 1:
        return
    fpr = {}
    tpr = {}
    roc_auc = {}
    #get AUC for each label individually
    relevant_labels = []
    auc_labels = {}
    for i in range(y.shape[1]):
        #only if there are true positives for this label
        if y[:,i].sum() > 0:
            fpr[i], tpr[i], _ = roc_curve(y[:,i], yhat_raw[:,i])
            if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                auc_score = auc(fpr[i], tpr[i])
                if not np.isnan(auc_score):
                    auc_labels["auc_%d" % i] = auc_score
                    relevant_labels.append(i)

    #macro-AUC: just average the auc scores
    aucs = []
    for i in relevant_labels:
        aucs.append(auc_labels['auc_%d' % i])
    roc_auc['auc_macro'] = np.mean(aucs)

    #micro-AUC: just look at each individual prediction
    yhatmic = yhat_raw.ravel()
    fpr["micro"], tpr["micro"], _ = roc_curve(ymic, yhatmic)
    roc_auc["auc_micro"] = auc(fpr["micro"], tpr["micro"])

    return roc_auc

def recall_at_k(yhat_raw, y, k):
    #num true labels in top k predictions / num true labels
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]

    #get recall at k for each example
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y[i,tk].sum()
        denom = y[i,:].sum()
        vals.append(num_true_in_top_k / float(denom))

    vals = np.array(vals)
    vals[np.isnan(vals)] = 0.

    return np.mean(vals)

def precision_at_k(yhat_raw, y, k):
    #num true labels in top k predictions / k
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]

    #get precision at k for each example
    vals = []
    for i, tk in enumerate(topk):
        if len(tk) > 0:
            num_true_in_top_k = y[i,tk].sum()
            denom = len(tk)
            vals.append(num_true_in_top_k / float(denom))

    return np.mean(vals)

def all_metrics(yhat, y, k=8, yhat_raw=None, calc_auc=True):
    """
        Inputs:
            yhat: binary predictions matrix
            y: binary ground truth matrix
            k: for @k metrics
            yhat_raw: prediction scores matrix (floats)
        Outputs:
            dict holding relevant metrics
    """
    names = ["acc", "prec", "rec", "f1"]

    #macro
    macro = all_macro(yhat, y)

    #micro
    ymic = y.ravel()
    yhatmic = yhat.ravel()
    micro = all_micro(yhatmic, ymic)

    metrics = {names[i] + "_macro": macro[i] for i in range(len(macro))}
    metrics.update({names[i] + "_micro": micro[i] for i in range(len(micro))})

    #AUC and @k
    if yhat_raw is not None and calc_auc:
        #allow k to be passed as int or list
        if type(k) != list:
            k = [k]
        for k_i in k:
            rec_at_k = recall_at_k(yhat_raw, y, k_i)
            metrics['rec_at_%d' % k_i] = rec_at_k
            prec_at_k = precision_at_k(yhat_raw, y, k_i)
            metrics['prec_at_%d' % k_i] = prec_at_k
            metrics['f1_at_%d' % k_i] = 2*(prec_at_k*rec_at_k)/(prec_at_k+rec_at_k)

        roc_auc = auc_metrics(yhat_raw, y, ymic)
        metrics.update(roc_auc)

    return metrics

def print_metrics(metrics):
    print()
    if "auc_macro" in metrics.keys():
        print("[MACRO] accuracy, precision, recall, f-measure, AUC")
        print("%.4f, %.4f, %.4f, %.4f, %.4f" % (metrics["acc_macro"], metrics["prec_macro"], metrics["rec_macro"], metrics["f1_macro"], metrics["auc_macro"]))
    else:
        print("[MACRO] accuracy, precision, recall, f-measure")
        print("%.4f, %.4f, %.4f, %.4f" % (metrics["acc_macro"], metrics["prec_macro"], metrics["rec_macro"], metrics["f1_macro"]))

    if "auc_micro" in metrics.keys():
        print("[MICRO] accuracy, precision, recall, f-measure, AUC")
        print("%.4f, %.4f, %.4f, %.4f, %.4f" % (metrics["acc_micro"], metrics["prec_micro"], metrics["rec_micro"], metrics["f1_micro"], metrics["auc_micro"]))
    else:
        print("[MICRO] accuracy, precision, recall, f-measure")
        print("%.4f, %.4f, %.4f, %.4f" % (metrics["acc_micro"], metrics["prec_micro"], metrics["rec_micro"], metrics["f1_micro"]))
    for metric, val in metrics.items():
        if metric.find("rec_at") != -1:
            print("%s: %.4f" % (metric, val))
    print()