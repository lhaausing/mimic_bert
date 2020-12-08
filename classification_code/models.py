import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from os.path import join

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import AutoTokenizer, AutoModel

class Attn_Layer(nn.Module):
    """
    Calculate attention for each label.
    After transposition, attn.size() = (batch_size, class_size, max_sent_len)
    """

    def __init__(self, hid, class_size):
        super().__init__()
        self.hid = hid
        self.c = 1
        self.w = nn.Linear(self.hid, self.c)

    def forward(self, input_embeds):
        attn_w = self.w(input_embeds)
        attn_w = torch.transpose(attn_w, 1, 2)
        attn_w = F.softmax(attn_weights, dim=2)

        embeds = torch.bmm(attn_w, input_embeds)

        return embeds

class Attn_Out_Layer(nn.Module):
    """Calculate logits before applying sigmoid func."""

    def __init__(self, hid, class_size):
        super().__init__()
        self.hid = hid
        self.c = class_size
        self.out_w = nn.Linear(self.hid, 1)

    def forward(self, input_embeds):
        logits = self.out_w(input_embeds)
        logits = logits.view(-1, self.c)

        return logits

class NGramTransformer(nn.Module):

    def __init__(self, model_name='', ngram_size = 32, n_class = 50):
        super().__init__()
        self.ngram_size = ngram_size
        self.model_name = model_name
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.out_layer = nn.Linear(self.hidden_size, n_class)
        self.wd_emb = self.bert.embeddings.word_embeddings

    def forward(self, input_ids=None, ngram_encoding=None):
        embeds = torch.bmm(ngram_encoding, self.wd_emb(input_ids))
        embeds, cls_embeds  = self.bert(inputs_embeds=embeds)
        logits = self.out_layer(embeds[:,0,:])

        return logits

class NGramTransformer_Attn(nn.Module):

    def __init__(self, model_name='', ngram_size = 32, n_class = 50,device= 'cuda:0'):
        super().__init__()
        self.model_name = model_name
        self.bert = AutoModel.from_pretrained(model_name)
        self.hid = self.bert.config.hidden_size
        self.c = n_class
        self.ngram_size = ngram_size

        self.wd_emb = self.bert.embeddings.word_embeddings
        self.attn_layer = Attn_Layer(self.hid, self.c)
        self.out_layer = Attn_Out_Layer(self.hid, self.c)

    def forward(self, input_ids=None, ngram_encoding=None):
        embeds = torch.bmm(ngram_encoding, self.wd_emb(input_ids))
        embeds, cls_embeds  = self.bert(inputs_embeds=embeds)
        embeds = self.attn_layer(embeds)
        logits = self.out_layer(embeds)

        return logits

class cnn_bert(nn.Module):
    def __init__(self, model_name='', ngram_size = 16, mp_size = 32, n_class = 50, device= 'cuda:0', sep_cls = True, use_attn = False):
        super().__init__()
        #Transformers Encoder
        self.bert = AutoModel.from_pretrained(model_name)

        #some_names
        self.model_name = model_name
        self.hid = self.bert.config.hidden_size
        self.c = n_class
        self.ngram_size = ngram_size
        self.mp_size = mp_size
        self.use_attn = use_attn
        self.sep_cls = sep_cls

        #layers
        self.wd_emb = self.bert.embeddings.word_embeddings
        self.conv = nn.Conv1d(self.hid, self.hid, self.ngram_size)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(self.mp_size)
        if self.use_attn:
            self.attn = Attn_Layer(self.hid, self.c)
            self.out = Attn_Out_Layer(self.hid, self.c)
        else:
            self.out = nn.Linear(self.hid, self.c)

    def forward(self, input_ids=None):
        x = self.wd_emb(input_ids)
        x = x.permute(0,2,1)
        if self.sep_cls:
            x_cls, x = x[:,:,0], x[:,:,1:]
            x = self.conv(x)
            x = self.relu(x)
            x = torch.cat((x_cls.unsqueeze(2),x),2)
        else:
            x = self.conv(x)
            x = self.relu(x)
        x = self.maxpool(x)
        x, x_cls  = self.bert(inputs_embeds=x.permute(0,2,1))
        if self.use_attn:
            x = self.attn(x)
            logits = self.out(x)
        else:
            logits = self.out(x[:,0,:])

        return logits

class local_bert(nn.Module):
    def __init__(self, model_name='', n_class = 50, stride = 256):
        super().__init__()
        #Transformers Encoder
        self.bert = AutoModel.from_pretrained(model_name)

        #hyperparams
        self.model_name = model_name
        self.c = n_class
        self.hid = self.bert.config.hidden_size
        self.max_len = self.bert.config.max_position_embeddings
        self.stride = stride

        #model blocks
        self.attn_w = nn.Linear(self.hid, 1)
        self.fc = nn.Linear(self.hid, self.c)

    def forward(self, input_ids):
        #Calculate window spans
        b_max_len = input_ids.size()[1]
        accumul_pos = 0
        input_windows = []
        while accumul_pos < (b_max_len - self.max_len):
            input_windows.append((accumul_pos, accumul_pos + self.max_len))
            accumul_pos += self.stride
        input_windows.append((accumul_pos, self.max_len))

        #Pass input_windows ids to BERT
        #Concatenate all pooled outputs
        x_cls = []
        for _ in input_windows:
            temp_x_cls = self.bert(input_ids[:,_[0]:_[1]])[1]
            x_cls.append(temp_x_cls)
        x_cls = torch.cat([_.unsqueeze(1) for _ in x_cls], dim=1)

        #Attention Layer
        attn_w = self.attn_w(x_cls)
        attn_w = torch.transpose(attn_w, 1, 2)
        attn_w = F.softmax(attn_weights, dim=2)
        x_cls = torch.bmm(attn_w, x_cls)
        x_cls = x_cls.view(-1, self.hid)

        #Fully Connected Layer
        logits = self.fc(x_cls)

        return logits

class snippet_model(nn.Module):
    def __init__(self, model_name='', n_class = 50,layer=0,type='Longformer'):
        super().__init__()
        #Transformers Encoder
        self.model = AutoModel.from_pretrained(model_name)
        self.freeze=False 

        # !!! different layers
        if type=="Longformer":
            if layer>0 and layer<=12:
                self.freeze=True
                ct=0
                for child in self.model.children():
                    if ct==1: # except the classification layer
                        newmodel=child
                        cnt=0
                        for newchild in newmodel.children():
                            if cnt==1: # except embedding and pool layer
                                finalmodel=newchild.layer
                                cnt_layer=1
                                for finalchild in finalmodel:
                                    if layer>=cnt_layer:
                                        for param in child.parameters():
                                            param.requires_grad = False
                                    cnt_layer+=1
                            cnt+=1
                    ct+=1

        #hyperparams
        self.model_name = model_name
        self.c = n_class
        self.hid = self.model.config.hidden_size

        #model blocks
        self.fc = nn.Linear(self.hid, self.c)

    def forward(self, input_ids, attn_masks,length):
        #Calculate sample windows

        #Pass input_windows ids to BERT
        #Concatenate all pooled outputs
        tmp,x_cls= self.model(input_ids=input_ids,attention_mask=attn_masks)
        if self.freeze:
            # !!!from bert 得到 tensor (batch size, seq length, hid dimension)
            # 如果是用整个句子得到的tensor的话就要把整个句子用max pooling或者attention mechanism得到一个跟cls的tensor维度一样的东西再接fully connected layer
            # meanpooling:
            # print(tmp.shape)
            # sys.stdout.flush()
            # get the mean
            tmp=tmp.sum(dim=1)
            x=torch.div(tmp,length)
        else:
            x=x_cls


        #Fully Connected Layer
        logits = self.fc(x)

        return logits

