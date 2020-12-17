import math
import random
import numpy as np
from tqdm import tqdm
import sys
from os.path import join

import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel
from transformers import LongformerModel, LongformerTokenizer

class snippet_model(nn.Module):
    def __init__(self, model_name='', n_class = 50,probing=False):
        super().__init__()
        #Transformers Encoder
        
        if model_name=="Bert_base":
            self.model=BertModel.from_pretrained('bert-base-uncased')
        elif model_name=="Longformer_base":
            self.model= LongformerModel.from_pretrained('allenai/longformer-base-4096')
        else:
            self.model = AutoModel.from_pretrained(model_name)

        # !!! different layers
        self.probing=probing
        if self.probing:
            for child in self.model.children():
                for param in child.parameters():
                    param.requires_grad = False

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
        if self.probing:
            # !!!from bert 得到 tensor (batch size, seq length, hid dimension)
            # 如果是用整个句子得到的tensor的话就要把整个句子用max pooling或者attention mechanism得到一个跟cls的tensor维度一样的东西再接fully connected layer
            # meanpooling:
            # print(tmp.shape)
            # sys.stdout.flush()
            # get the mean
                
            embeddings=self.model.embeddings(input_ids)
            word_embeddings=self.model.embeddings.word_embeddings(input_ids)

            layer_res=[torch.div(word_embeddings.sum(dim=1),length)]

            layer=embeddings.unsqueeze(0)
            # print("version:",transformers.__version__)
            # print(self.model)
            # sys.stdout.flush()


            # get each layer vectors
            for each_layer in range(len(self.model.encoder.layer)):
                is_index_masked = attn_masks < 0
                is_index_global_attn =attn_masks > 0
                is_global_attn = is_index_global_attn.flatten().any().item()
                if self.model_name=="longformer" or self.model_name=="longformer_mimic":
                    layer=self.model.encoder.layer[each_layer](layer[0],attn_masks,is_index_masked,is_index_global_attn,is_global_attn)
                elif self.model_name=="bert_mimic":
                    layer=self.model.encoder.layer[each_layer](layer[0],attn_masks)
                layer_res.append(torch.div(layer[0].sum(dim=1),length))
            
            # calculate logits from final layer
            logits=self.fc(self.model.pooler(layer[0]))


            return layer_res,logits

                
        else:
            tmp,x_cls= self.model(input_ids=input_ids,attention_mask=attn_masks)
            x=x_cls

            #Fully Connected Layer
            logits = self.fc(x)

            return logits

class ClassficationProbing(nn.Module):
    def __init__(self,n_class=50,model_name='longformer'):
        self.model=AutoModel.from_pretrained(model_name)
        self.c = n_class
        self.hid = self.model.config.hidden_size

        #model blocks
        self.fc = nn.Linear(self.hid, self.c)

    def forward(x):
        tmp,x_cls=x
        logits = self.fc(x_cls)
        return logits

