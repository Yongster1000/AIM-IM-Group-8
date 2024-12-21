import os
import pandas as pd
from statistics import median,mean,stdev
#from scipy import stats as s
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
#import src.transformer as transformer
from torch.nn.modules import transformer
from tqdm import tqdm
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc
import numpy as np
from typing import List, Optional, Tuple, Union
import math
from torch.optim import AdamW, Adam
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from scipy import stats
import random
from torch.utils.data.sampler import SubsetRandomSampler
import importlib
import argparse


### All attributes/features we need to look in raw data file for summarizing 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
attr_list = ['SEX', 'edu', 'DM_FAM', 'smoking', 'DRK', 'betel', 'SPORT', 'cardio_b'] + ['AGE', 'SBP', 'DBP', 'HR', 'Weight', 'Height', 'BMI', 'WHR', 'T_CHO', 'TG', 'HDL', 'LDL']
#cat_8_num_12 
# %%
#from arkrde Arnab De in https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/7
class FocalLoss(nn.Module):
  def __init__(self, weight=None, gamma=2., reduction='none'):
    nn.Module.__init__(self)
    self.weight = weight
    self.gamma = gamma
    self.reduction = reduction
      
  def forward(self, input_tensor, target_tensor):
    log_prob = F.log_softmax(input_tensor, dim=-1)
    prob = torch.exp(log_prob)
    return F.nll_loss(
        ((1 - prob) ** self.gamma) * log_prob, 
        target_tensor, 
        weight=self.weight,
        reduction = self.reduction
    )

#parameters
def get_dataloader(data_set, batch_size, n_workers = 0):
    return DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=n_workers,
        pin_memory=True, #default: False
        #collate_fn=collate_batch,
    )
'''
def collate_batch(batch_list):
    batch_size = len(batch_list)
    #print([i.shape for i in batch_list])
    #data = pad_sequence([i[:, :-1] for i in batch_list], batch_first = True)
    list_to_tuple = tuple([i[:-1].unsqueeze(dim = 0) for i in batch_list])
    data = torch.cat(list_to_tuple, dim = 0)
    #print(data.size()) #torch.Size([32, 420])
    labels = torch.Tensor([i[-1] for i in batch_list])
    
    #print(labels.size())
    #print(labels)
    return data, labels
'''

batch_size = 256
feature_dimension = 42 

class Classifier(nn.Module):
    def __init__(self, d_model=144, n_class=2, dropout=0.1, cls = None, num_range = 0):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        #feature_dimension = 42
        
        #self.pre_embedding = nn.Embedding(len(attr_list), d_model)
        #self.embedding = nn.Embedding.from_pretrained(torch.cat((self.pre_embedding.weight, torch.zeros(1, d_model)), dim = 0), freeze = False)
        self.nhead = 1
        #self.embedding_for_value = nn.Embedding(12 * num_range + 17, d_model, padding_idx = len(attr_list)) #the first one is replaced by CLS token
        #self.embedding_for_value = nn.Embedding((12 + 20) * num_range + 17, d_model)
        self.embedding_for_value = nn.Embedding((12) * num_range + 17, d_model)
        #self.embedding_for_cat = nn.Embedding(17, d_model)
        #self.embedding = nn.Embedding(len(attr_list) - 1, d_model)
        #nn.init.zeros_(self.embedding.weight[-1, :])
        self.d_model = d_model
        
        self.encoder_layer = transformer.TransformerEncoderLayer(
        d_model=d_model, dim_feedforward=144, nhead = self.nhead, batch_first=True#, activation = F.gelu
        )
        self.encoder = transformer.TransformerEncoder(self.encoder_layer, num_layers = 1, enable_nested_tensor=False
        )
        self.total_embedding = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # Project the the dimension of features from d_model into speaker nums.
        
        self.pred_layer = nn.Sequential(
          nn.Linear(d_model, d_model//2),
          nn.GELU(),
          nn.Dropout(0.45),
          nn.Linear(d_model//2, n_class)
          )
        
        #self.pos_enc = self.pos_embed((len(attr_list) + 20))
        self.pos_enc = self.pos_embed(len(attr_list))
        #self.pos_enc = self.pos_embed()
        #self.pos_enc = self.pos_embed(24 * (len(attr_list) - 1))

    def pos_embed(self, max_position_embeddings):
      def even_code(pos, idx):
        return np.sin(pos / (10000 ** (2 * idx / self.d_model)))
      def odd_code(pos, idx):
        return np.cos(pos / (10000 ** (2 * idx / self.d_model)))

      # initialize position embedding table
      lookup_table = np.zeros((max_position_embeddings, self.d_model), dtype=np.float32)

      # reset table parameters with hard encoding
      # set even dimension
      for pos in range(max_position_embeddings):
          for idx in np.arange(0, self.d_model, step=2):
              lookup_table[pos, idx] = even_code(pos, idx)
      # set odd dimension
      for pos in range(max_position_embeddings):
          for idx in np.arange(1, self.d_model, step=2):
              lookup_table[pos, idx] = odd_code(pos, idx)
      

      return torch.tensor(lookup_table).to(device)
    

    def forward(self, mels, mask = None, transformer_explainer = None):

        #B: batches; N: # of features; S: condensed_size
        #print('mels.size() =', mels.size()) #(B, (N+1)*S)
        #go_round = torch.cat([torch.arange(1, len(attr_list))] * condensed_size).unsqueeze(0).expand(mask.shape[0], -1).to(device)
        #cat_round = torch.Tensor([0, 1]).to(device)
        #go_round = torch.cat([torch.arange(len(attr_list) - 1)] * condensed_size).unsqueeze(0).expand(mask.shape[0], -1).to(device)
        #go_round: [1, 2, 3, ..., 42, 1, 2, ..., 42, ..., 1, 2, 3, ..., 42] for each batch
        #go_round[mask == True] = len(attr_list)
        #print('zero vector? ', self.embedding(torch.Tensor([43]).int().to(device)))
        embed_vectors = self.embedding_for_value(mels.int()) #(B, (N+1)*S, D)

        #print(self.embedding(torch.Tensor([0]).int().to(device)))
        #print('mels =', mels[0, :])
        if torch.isnan(mels).sum() > 0:
            print('nan occurs in 1')
        
        #out = mels.unsqueeze(-1) * embed_vectors #(B, (N+1)*S, D)
        #val_emb = self.val_emb(mels.unsqueeze(-1))
        #out = val_emb + embed_vectors
        #out = self.total_embedding(out)
        out = embed_vectors
        #print('out.shape =', out.shape)
        #out = out + self.pos_enc
        #out = out + torch.cat([self.pos_enc] * 24, dim = 0).unsqueeze(0)
        #print(torch.cat([self.pos_enc[i, :].unsqueeze(0).expand(len(attr_list) - 1, -1) for i in range(self.pos_enc.shape[0])], dim = 0).shape)
        #print(out.shape, self.pos_enc.shape)
        out = out + self.pos_enc.unsqueeze(0).expand(out.shape[0], -1, -1)
        
        #cls_tokens = self.embedding(torch.arange(1).to(device)).unsqueeze(0).expand(mels.shape[0], -1, -1)
        #out = torch.cat((torch.zeros_like(cls_tokens), out), dim = 1).to(device) #(B, (N+1)*S + 1, D)
        
        #mask.shape #(B, (N+1)*S)
        #expanded_mask = torch.cat((torch.zeros((mask.shape[0], 1)), mask), dim = 1).to(device)
        #expanded_mask = mask.unsqueeze(1).repeat(self.nhead, mask.shape[-1], 1).to(device)
        #mask = mask.unsqueeze(1).repeat(1, mask.shape[-1], 1).to(device)
        #expanded_mask = torch.cat([mask[i, :, :].repeat(self.nhead, 1, 1) for i in range(mask.shape[0])], dim = 0)
        #expanded_mask.shape #(B, (N+1)*S + 1)
        #mask = mask.to(device)
        #out = self.encoder(out, mask = expanded_mask)
        #mask = torch.cat((torch.zeros((mask.shape[0], 1)), mask), dim = 1).to(device)
        #out = self.encoder(out, src_key_padding_mask = mask.to(device))
        out = self.encoder(out)
        #out_filter = mask.unsqueeze(-1).expand(-1, -1, out.shape[2]).to(device)
        
        #stats = out[:, 0, :]
        stats = out.mean(dim = 1)
        #filtering = out * (1 - out_filter * 1)
        
        #stats = filtering.sum(dim = 1)/(1 - mask * 1).sum(dim = 1, keepdim = True)
        #
        #out = self.pred_layer(out)
        out = self.pred_layer(stats)
        #out = torch.sigmoid(out)
        #out = self.dropout(out)
        #print(out.shape)
        return out

def model_fn(batch, model, criterion, device, test = False, mask = None, only_logits = False, transformer_explainer = None):
    """Forward a batch through the model."""
    #print(batch.shape)
    #data, labels = torch.split(batch, [20 + 20, 1], dim = 1)
    data, labels = torch.split(batch, [20, 1], dim = 1)
    #print(labels[:, 0].shape)
    labels = labels[:, 0].type(torch.LongTensor)

    #data = data.to(device, dtype = torch.float32)
    data = data.to(device)
    labels = labels.to(device)
    
    #print(labels)
    if transformer_explainer == None:
        outs = model(data, mask = mask)
    
    loss = criterion(outs, labels)
    copy_out = outs
    #print(copy_out)
    softmax = nn.Softmax(dim = 1)
    pos_prob = softmax(copy_out.cpu().detach())[:,1]
    #print(pos_prob)
    preds = outs.argmax(1)
    #print(labels.cpu().numpy())
    # Compute accuracy.
    #accuracy = torch.mean((preds == labels).float())
    #average_precision_score_ = average_precision_score(labels.cpu().detach().numpy(), pos_prob.numpy(), average = 'samples')
    if transformer_explainer == None:
        if not only_logits:
            return loss, labels, preds, pos_prob
        else:
            return labels, outs
    
        

def valid(dataloader, model, criterion, device, word = "valid", multiple_test_required = False, mask = None, transformer_explainer = None):
    """Validate on validation set."""
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc=word, unit=" uttr")
    all_labels = []
    all_pos_prob = []
    all_preds = []
    all_loss = []
    #all_logits = []
    store = None
    count = 0
    for i, batch in enumerate(dataloader):      
        with torch.no_grad():
            is_test = word == "test"
            #piece_of_mask = mask[count:count+batch_size, :]
            #print('batch.size() =', batch[-1].size(), end = " ")
            #print('piece_of_mask.size() =', piece_of_mask.size())
            if transformer_explainer == None:
                loss, labels, preds, pos_prob = model_fn(batch, model, criterion, device, test = is_test)
                all_preds.extend(list(preds))
                
            
        count += batch_size

        running_loss += loss.item()
        all_loss.append(loss.item())
        all_labels.extend(list(labels))
        all_pos_prob.extend(list(pos_prob))
        
        pbar.update(batch_size)
        pbar.set_postfix(
          loss=f"{running_loss / (i+1):.2f}"
        )
        
    
    pbar.close()
    label_copy = labels

    
    model.train()
    
    all_labels = torch.Tensor(all_labels)
    all_preds = torch.Tensor(all_preds)
    all_pos_prob = torch.Tensor(all_pos_prob)
    all_loss = torch.Tensor(all_loss)
    #print(all_pos_prob.numpy())
    precision, recall, _ = precision_recall_curve(all_labels.cpu().detach().numpy(), all_pos_prob.numpy())
    TP = sum(all_labels * all_preds)
    FN = sum(all_labels * (1-all_preds))
    TN = sum((1-all_labels) * (1-all_preds))
    FP = sum((1-all_labels) * all_preds)
    if transformer_explainer == None:
        return auc(recall, precision), roc_auc_score(all_labels.cpu().detach().numpy(), all_pos_prob.numpy(), average = 'samples'), (TP, FN, TN, FP), float(all_loss.mean())
    else:
        return auc(recall, precision), roc_auc_score(all_labels.cpu().detach().numpy(), all_pos_prob.numpy(), average = 'samples'), (TP, FN, TN, FP), store
    #return average_precision_score(label_copy.cpu().detach().numpy(), pos_prob.numpy(), average = 'samples'), roc_auc_score(labels.cpu().detach().numpy(), pos_prob.numpy(), average = 'samples')

def get_cosine_schedule_with_warmup(
  optimizer: Optimizer,
  num_warmup_steps: int,
  num_training_steps: int,
  num_cycles: float = 0.5,
  last_epoch: int = -1,
):
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
        )
        return max(
        0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
def classify_label(train_set_w_label):
  label_0 = []
  label_1 = []
  for element in train_set_w_label:
    if element[0][-1] == 1:
      label_1.append(element)
    else:
      label_0.append(element)
  print('label_0_portion =', len(label_0)/len(train_set_w_label))
  return label_0, label_1

def to_embed_num(tensor_matrix):
  #splited_matrix = torch.split(tensor_matrix, [8, 12 + 20, 1], dim = 1) #cat, num, target
  splited_matrix = torch.split(tensor_matrix, [8, 12, 1], dim = 1) #cat, num, target
  cat_list = ['SEX', 'edu', 'DM_FAM', 'smoking', 'DRK', 'betel', 'SPORT', 'cardio_b'] #8
  #num_list = ['AGE', 'SBP', 'DBP', 'HR', 'Weight', 'Height', 'BMI', 'WHR', 'T_CHO', 'TG', 'HDL', 'LDL'] + [1] * 20
  num_list = ['AGE', 'SBP', 'DBP', 'HR', 'Weight', 'Height', 'BMI', 'WHR', 'T_CHO', 'TG', 'HDL', 'LDL']
   #attr_list =  + ['target']
  count = 10
  for i in range(len(cat_list)):
    num_choice = list(set(splited_matrix[0][:, i].tolist()))
    for choice in num_choice:
      splited_matrix[0][:, i][splited_matrix[0][:, i] == choice] = count 
      count += 1
  #print('count = ', count)
  splited_matrix[0][:, :] -= 10
  #count = 10
  for i in range(len(num_list)):
    num_range = [float("-inf"), -1, -0.5, 0, 0.5, 1, float("inf")]
    #num_range = [float("-inf"), -0.75, 0, 0.75, float("inf")]
    #num_range = [float("-inf"), -2, -0.13, 0.06, 0, 0.06, 0.13, 2, float("inf")]
    num_range_dict = dict(zip(num_range, range(count, count + len(num_range) - 1)))
    #print(num_range_dict)
    count += len(num_range) - 1
    for j in range(len(num_range) - 1):
      splited_matrix[1][:, i][(splited_matrix[1][:, i] - num_range[j]) * (splited_matrix[1][:, i] - num_range[j + 1]) <= 0] = num_range[j]
    #print(splited_matrix[1][:5,0:2])
    for j in range(len(num_range) - 1):
      splited_matrix[1][:, i][splited_matrix[1][:, i] == num_range[j]] = num_range_dict[num_range[j]] 
    #print(splited_matrix[1][:5,0:2])
  splited_matrix[1][:, :] -= 10
  #print(splited_matrix[1][:5,0:2])
  return torch.cat(splited_matrix, dim = 1)
def confidence_interval(array):
  # Example data (You would replace this with your actual data)
  data = array

  # Step 1: Calculate sample mean (x̄)
  sample_mean = np.mean(data).item()

  # Step 2: Calculate sample standard deviation (s)
  sample_std_dev = np.std(data, ddof=1).item()  # ddof=1 for sample standard deviation

  # Step 3: Calculate sample size (n)
  n = len(data)

  # Step 4: Calculate standard error (SE)
  SE = sample_std_dev / np.sqrt(n).item()

  # Step 5: Use z-value for 95% confidence interval (z* ≈ 1.96)
  z_critical = 1.96

  # Step 6: Calculate margin of error
  margin_of_error = z_critical * SE

  # Step 7: Calculate confidence interval
  confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

  # Output the result
  print(f"Sample Mean: {sample_mean}")
  print(f"Sample Standard Deviation: {sample_std_dev}")
  print(f"Standard Error: {SE}")
  print(f"Critical z-value: {z_critical}")
  print(f"Margin of Error: {margin_of_error}")
  print(f"95% Confidence Interval: {confidence_interval}")
    

def main(
    execution_file,
    random_state,
    model_file,
    include_methylation,
    number_of_fold
):
    preprocess_module = importlib.import_module(name=execution_file[:-3])
    model_module = importlib.import_module(name = model_file[:-3])
    original_pre_config = preprocess_module.parse_args()
    model_config = model_module.parse_args()
    auprc_list = []
    for each in random_state:
      
      original_pre_config["random_state"] = each
      original_pre_config["include_methylation"] = include_methylation
      original_pre_config["number_of_fold"] = number_of_fold
      list_result = preprocess_module.main(**original_pre_config)
      for element in list_result:
        model_config["df_train"] = element[0]
        model_config["df_test"] = element[1]
        model_config["include_methylation"] = original_pre_config["include_methylation"] 
        auprc_list.append(model_module.main(**model_config))
    
    auprc_list = np.array(auprc_list)
    confidence_interval(auprc_list)

def parse_args(model_file):
  """arguments"""
  config = {
    "execution_file": "data_preprocessing.py",
    "random_state": [i for i in range(30)],
    "model_file": model_file,
    "include_methylation": True,
    "number_of_fold": 5 #larger than 1
  }

  return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', type=str, help='the name of the file.', required=True)
    #parser.add_argument('-s', '--save_name', type=str, help='the name of the save file.', required=True)
    config = parser.parse_args()
    main(**parse_args(config.file_name))
     
## %%
