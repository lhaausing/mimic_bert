# coding=utf-8

import os
import glob
import logging
import argparse
from os.path import join

import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils import *

class mimic3_dataset(Dataset):

    def __init__(self, texts, labels, ngram_size, tokenizer, use_ngram = False):
        self.texts = texts
        self.idx = list(range(len(labels)))
        self.labels = labels
        self.tokenizer = tokenizer
        self.ngram_size = ngram_size

        assert (len(self.texts) == len(self.labels))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, key):
        return [self.idx[key], self.texts[key], self.labels[key]]

    def mimic3_col_func(self, batch):
        logging.getLogger("transformers.tokenization").setLevel(logging.ERROR)
        inputs = self.tokenizer([elem[1] for elem in batch], return_tensors='pt', padding=True)
        input_ids = inputs["input_ids"]
        attn_mask = inputs['attention_mask']
        labels = torch.cat([elem[2].unsqueeze(0) for elem in batch], dim=0).type('torch.FloatTensor')

        return (input_ids, attn_mask, labels)
