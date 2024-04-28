import pandas as pd
import numpy as np
import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm
from datasets import load_dataset
import random
import time

class textDataset(Dataset):
  def __init__(self, texts, label, tokenizer, max_length):
        self.texts = texts
        self.label = label
        self.tokenizer = tokenizer
        self.max_length = max_length

  def __len__(self):
      return len(self.label)

  def __getitem__(self, idx):

      text1 = self.texts[idx]
      inputs = self.tokenizer.encode_plus(
            text1 ,
            None,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
            truncation=True
        )
      ids = inputs["input_ids"]
      token_type_ids = inputs["token_type_ids"]
      mask = inputs["attention_mask"]

      return {
          'ids': torch.tensor(ids, dtype=torch.long, device=device),
          'mask': torch.tensor(mask, dtype=torch.long, device=device),
          'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long, device=device),
          'target': torch.tensor(self.label[idx], dtype=torch.long, device=device)
        }

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.out = nn.Linear(768, 1)

    def forward(self,ids,mask,token_type_ids):
        _,o2= self.bert_model(ids,attention_mask=mask,token_type_ids=token_type_ids, return_dict=False)

        out= self.out(o2)

        return out

if __name__ == "__main__":
    NUM_ATTENTION_KEPT = 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")
    dataset = load_dataset("stanfordnlp/sst2")
    data = dataset.with_format("torch")

    # train_text = [each['sentence'] for each in data['train']]
    # train_label = [each['label'] for each in data['train']]

    val_text = [each['sentence'] for each in data['validation']]
    val_label = [each['label'] for each in data['validation']]


    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

    # trainset= textDataset(train_text, train_label, tokenizer, max_length=100)
    # train_loader=DataLoader(dataset=trainset,batch_size=32)

    valset= textDataset(val_text, val_label, tokenizer, max_length=100)
    val_loader=DataLoader(dataset=valset,batch_size=32)

    model=BERT().to(device)
    model.bert_model.encoder.layer = model.bert_model.encoder.layer[:6]

    model.load_state_dict(torch.load("./checkpoint/layer_prune2.pt"))
    #Initialize Optimizer
    
    model.eval()
    num_correct = 0
    num_samples = 0
    
    with torch.no_grad():
        start_time = time.time()

        for batch, dl in enumerate(val_loader):
            ids=dl['ids']
            token_type_ids=dl['token_type_ids']
            mask= dl['mask']
            label=dl['target']
            label = label.unsqueeze(1)
            output=model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids)
            label = label.type_as(output)
            pred = torch.where(output >= 0, 1, 0)
            num_correct += sum(1 for a, b in zip(pred, label) if a[0] == b[0])
            num_samples += pred.shape[0]
        end_time = time.time()

        acc = num_correct / num_samples
            

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f} in {end_time - start_time}s')


