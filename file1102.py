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


### All attributes/features we need to look in raw data file for summarizing 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
attr_list = ['SEX', 'edu', 'DM_FAM', 'smoking', 'DRK', 'betel', 'SPORT', 'cardio_b'] + ['AGE', 'SBP', 'DBP', 'HR', 'Weight', 'Height', 'BMI', 'WHR', 'T_CHO', 'TG', 'HDL', 'LDL'] + ['target']
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
        self.embedding_for_value = nn.Embedding(12 * num_range + 17, d_model)
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
        
        self.pos_enc = self.pos_embed((len(attr_list) - 1))
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
  splited_matrix = torch.split(tensor_matrix, [8, 12, 1], dim = 1) #cat, num, target
  cat_list = ['SEX', 'edu', 'DM_FAM', 'smoking', 'DRK', 'betel', 'SPORT', 'cardio_b'] #8
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
    

def main(
    input_dir,
    summary_type, 
    save,
    save_path,
    batch_size,
    n_workers,
    valid_steps,
    warmup_steps,
    save_steps,
    total_steps,
    train,
    test,
    explainer, 
    num_range
):
    if not save:
        df_train = pd.read_csv("train_standard.csv") 
        df_train = df_train[attr_list]
        df_test = pd.read_csv("test_standard.csv")
        df_test = df_test[attr_list]
        torch.save(torch.from_numpy(df_train.to_numpy()), "train_data.pt")
        torch.save(torch.from_numpy(df_test.to_numpy()), "test_data.pt")
        #print(torch.from_numpy(df_train.to_numpy()).shape)
    else:
        #return
        if train:
            train_set = torch.load('train_data.pt')
            #print(train_set.shape)
            #print(train_set[:5, :10])
            train_set = to_embed_num(train_set)
            #print(train_set[:5, :10])
            #return 
            
            #train_mask = torch.load('train0924_mask.pt')
            #valid_mask = torch.load('val0924_mask.pt')
            #print('probe =', train_set[0][:2, :])
            
            #train_set_w_label_and_mask = preprocess(train_set, train_mask, 'train')
            #valid_set_w_label_and_mask = preprocess(valid_set, valid_mask, 'valid')
            #print(train_set_w_label[0])
            #train_set_w_label_and_mask.extend(valid_set_w_label_and_mask)
            #print(len(train_set_w_label_and_mask))
            #label_0, label_1 = classify_label(train_set_w_label_and_mask)
            print(f"[Info]: Finish loading data!", flush = True)
            #ref: https://ithelp.ithome.com.tw/articles/10277163
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #device = torch.device("cpu")
            model = Classifier(num_range = num_range).to(device)
            criterion = nn.CrossEntropyLoss()
            criterion = FocalLoss(reduction = 'mean')

            optimizer = Adam(model.parameters(), lr=3e-5)#1e-3
            #scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
            print(f"[Info]: Finish creating model!",flush = True)

            best_average_precision_score = -1.0
            best_state_dict = None
            best_loss = 1e+05

            pbar = tqdm(total=total_steps, ncols=0, desc="Train", unit=" step")
            step = 0
            count_epoch = 0
            #for step in range(total_steps):
            
            # set testing data size
            valid_split = 0.2
            # need shuffle or not
            shuffle_dataset = True
            # random seed of shuffle
            #random_seed = random_seed

            # creat data indices for training and testing splits
            dataset_size = train_set.shape[0]
            indices = list(range(dataset_size))
            # count out split size
            split = int(np.floor(valid_split * dataset_size))
            
            while step < total_steps:
              clock = 0
              count = 0
              #print([train_set_w_label[i].shape for i in range(10)])
              #print(train_set_mask.shape)
              #print(train_set_w_label[0].device)
              '''
              random.shuffle(new_train_dataset)
              
              #new_train_dataset = train_set_w_label_and_mask
              #new_valid_dataset = valid_set_w_label_and_mask
              #train_dataset_0.extend(train_dataset_1)
              #valid_dataset_0.extend(valid_dataset_1)
              

              train_set_w_label = [i[0] for i in new_train_dataset]
              valid_set_w_label = [i[0] for i in new_valid_dataset]
              
              train_set_mask = torch.cat(tuple([i[1] for i in new_train_dataset]), dim = 0)
              valid_set_mask = torch.cat(tuple([i[1] for i in new_valid_dataset]), dim = 0)
              train_loader = get_dataloader(train_set_w_label, batch_size)
              valid_loader = get_dataloader(valid_set_w_label, batch_size)
              '''
              
              if shuffle_dataset:
                  #np.random.seed(random_seed)
                  np.random.shuffle(indices)
              train_indices, valid_indices = indices[split:], indices[:split]

              # creating data samplers and loaders:
              train_sampler = SubsetRandomSampler(train_indices)
              valid_sampler = SubsetRandomSampler(valid_indices)

              train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
              valid_loader = DataLoader(train_set, batch_size=batch_size, sampler=valid_sampler)
              train_iterator = iter(train_loader)
              #print('p, ', padded_train_mask.size())
              #print('train_set_mask.shape = ', train_set_mask.shape)
              while True:
                  #print('count = ', count)
              # Get data
                try:
                    batch = next(train_iterator) #[data, label]
                    #len(attr_list) = 43
                    #mask = train_set_mask[count:count + batch_size, :]
                    count += batch_size
                except StopIteration:
                    #print('stopiteration\ncount =', count)
                    break
                  
                #print(model_fn(batch, model, criterion, device))
                loss, _, _, _ = model_fn(batch, model, criterion, device)
                #print('loss =', loss)
                batch_loss = loss.item()
                #batch_accuracy = accuracy.item()
                optimizer.zero_grad()  
                # Updata model
                loss.backward()
                optimizer.step()
                #scheduler.step()
                

                # Log
                
                pbar.update()
          
                pbar.set_postfix(
                    loss=f"{batch_loss:.2f}",
                    step=step + 1,
                )
                
              # Do validation
              
                  
              pbar.close()
              clock += 1
              if clock == 1:
                clock = 1
                valid_average_precision_score, auroc, _, loss = valid(valid_loader, model, criterion, device, multiple_test_required=True)

                # keep the best model
                if valid_average_precision_score > best_average_precision_score:
                    best_average_precision_score = valid_average_precision_score
                    best_state_dict = model.state_dict()
                    '''
                    valid_average_precision_score, auroc, _, loss = valid(valid_loader, model, criterion, device, multiple_test_required=True, mask = padded_valid_mask)


                    # keep the best model
                    if loss < best_loss:
                        best_loss = loss
                        best_state_dict = model.state_dict()
                    '''
                    step = 0
                else:
                  step += 1
                '''
                if loss >= best_loss:
                  step += 1
                else:
                  best_loss = loss
                  step = 0
                  '''
                #print('loss =', loss)
                # Save the best model so far.
                if (step + 1) % save_steps == 0 and best_state_dict is not None:
                  torch.save(best_state_dict, save_path)
                  pbar.write(f"Step {count_epoch + 1}, best model saved. (AUPRC={best_average_precision_score:.4f}, AUROC={auroc:.4f})")

              count_epoch += 1
              print('count =', count_epoch)
              if count_epoch == total_steps:
                break
              pbar.close()
        if test:
            '''
            test_set = pd.read_csv(test_csv, usecols=[i for i in attr_list]).to_numpy()#.to(device)
            _ = Phy_dataset(test_set, name = "test0924")
            '''
            test_set = torch.load("test_data.pt") #Note we conduct the explainability evaluation method on valid dataset
            test_set = to_embed_num(test_set)
            #test_mask = torch.load("test0924_mask.pt")
            #test_set, test_mask = input_shrinking(test_set, condensed_size)
            #test_set, test_mask = do_normalization(test_set, test_mask)
            #test_set_w_label = preprocess(test_set, test_mask, 'test')
            
            #new_test_set_w_label = [i[0] for i in test_set_w_label]
            test_loader = DataLoader(test_set, batch_size=batch_size)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #device = torch.device("cpu")
            #test_set_mask = torch.cat(tuple([i[1] for i in test_set_w_label]), dim = 0)

            model = Classifier(n_class=2, num_range = num_range).to(device)
            model.load_state_dict(torch.load(save_path))
            model.eval()
            
            criterion = nn.CrossEntropyLoss()
            
            auprc, auroc, multiple_test, *args = valid(test_loader, model, criterion, device, word = "test"
                                                    , multiple_test_required = True)
            #print(batch_accuracy)
            #print('percentage of positive = ', sum(label)/2400)
            TP, FN, TN, FP = multiple_test
            '''
            TP = sum(all_labels.to(device) * all_preds)
            FN = sum(all_labels.to(device) * (1-all_preds))
            TN = sum((1-all_labels).to(device) * (1-all_preds))
            FP = sum((1-all_labels).to(device)* all_preds)'''
            print(TP, FP, TN, FN)
            print('auprc =', auprc)
            print('auroc =', auroc)
            print('Sensitivity = ', TP/(TP+FN))
            print('Specificity = ', TN/(TN+FP))
            print('PPV = ', TP/(TP+FP))
            print('NPV = ', TN/(TN+FN))
                

def parse_args():
  """arguments"""
  config = {
    "input_dir": "physionet.org/files/challenge-2012/1.0.0/set-c/",
    "summary_type": "mean",
    "save": True,
    "save_path": "model20241106.ckpt",#0716, 6 layer#0805, 43
    "batch_size": 256,
    "n_workers": 0,
    "valid_steps": 200,#no use
    "warmup_steps": 100,#no use
    "save_steps": 1,#1000
    "total_steps": 350,#7000#for early stopping
    "train": False,
    "test": True,
    "explainer": None,
    "num_range": 6
  }

  return config

if __name__ == "__main__":
    main(**parse_args())
     
## %%
