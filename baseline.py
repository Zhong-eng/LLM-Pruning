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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")
    dataset = load_dataset("stanfordnlp/sst2")
    data = dataset.with_format("torch")

    train_text = [each['sentence'] for each in data['train']]
    train_label = [each['label'] for each in data['train']]

    val_text = [each['sentence'] for each in data['validation']]
    val_label = [each['label'] for each in data['validation']]


    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

    trainset= textDataset(train_text, train_label, tokenizer, max_length=100)
    train_loader=DataLoader(dataset=trainset,batch_size=32)

    valset= textDataset(val_text, val_label, tokenizer, max_length=100)
    val_loader=DataLoader(dataset=valset,batch_size=32)

    model=BERT().to(device)

    loss_fn = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    #Initialize Optimizer
    optimizer= optim.Adam(model.parameters(), lr= 0.000001)
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/baseline'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for i in range(15):
            model.train()
            for batch, dl in enumerate(train_loader):
                if batch < 1 + 1 + 10:
                    prof.step()  # Need to call this at each step to notify profiler of steps' boundary.
                ids=dl['ids']
                token_type_ids=dl['token_type_ids']
                mask= dl['mask']
                label=dl['target']
                label = label.unsqueeze(1)

                optimizer.zero_grad()

                output=model(
                    ids=ids,
                    mask=mask,
                    token_type_ids=token_type_ids)
                label = label.type_as(output)

                loss=loss_fn(output,label)
                loss.backward()

                optimizer.step()
            model.eval()
            num_correct = 0
            num_samples = 0
            with torch.no_grad():
                for batch, dl in enumerate(val_loader):
                    ids=dl['ids']
                    token_type_ids=dl['token_type_ids']
                    mask= dl['mask']
                    label=dl['target']
                    label = label.unsqueeze(1)

                    optimizer.zero_grad()

                    output=model(
                        ids=ids,
                        mask=mask,
                        token_type_ids=token_type_ids)
                    label = label.type_as(output)
                    pred = torch.where(output >= 0, 1, 0)
                    num_correct += sum(1 for a, b in zip(pred, label) if a[0] == b[0])
                    num_samples += pred.shape[0]
                acc = num_correct / num_samples
                print(f'##Epoch {i+1}: Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), './checkpoint/baseline.pt')
                    print("Best checkpoint saved!")

