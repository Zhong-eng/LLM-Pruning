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
from transformers import BertTokenizer
import time
EPOCH = 30
profiling_file_name = './log/bert-base_L2_epoch' + str(EPOCH)

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
    
# class BERTL2Pruned(nn.Module):
#     def __init__(self, prune_heads):
#         super(BERTL2Pruned, self).__init__()
#         self.bert_model = transformers.BertModel.from_pretrained("bert-base-uncased")
#         self.out = nn.Linear(768, 1)
#         self.prune_heads = prune_heads
#         self.apply_pruning()

#     def forward(self, ids, mask, token_type_ids):
#         _, pooled_output = self.bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
#         output = self.out(pooled_output)
#         return output

#     def apply_pruning(self):
#         with torch.no_grad():
#             for i, layer in enumerate(self.bert_model.encoder.layer):
#                 self.prune_layer(layer, self.prune_heads)

#     def prune_layer(self, layer, prune_heads):
#         attention = layer.attention.self
#         all_weights = [attention.query.weight, attention.key.weight, attention.value.weight]
#         for weight in all_weights:
#             norms = torch.norm(weight, p=2, dim=-1)  # Compute L2 norms
#             top_k, indices = torch.topk(norms, prune_heads)  # Find top-k important weights
#             mask = torch.ones_like(norms)
#             mask[indices] = 1
#             weight.data *= mask.unsqueeze(1)  # Apply pruning

class BERTStructurallyPruned(nn.Module):
    def __init__(self, original_model, attention_head_size, num_attention_heads):
        super(BERTStructurallyPruned, self).__init__()
        self.all_head_size = int(attention_head_size * num_attention_heads)
        self.bert_model = self.prune_bert_model(original_model, attention_head_size, num_attention_heads)
        self.out = nn.Linear(768, 1)  

    def forward(self, ids, mask, token_type_ids):
        _, pooled_output = self.bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output = self.out(pooled_output)
        return output

    def prune_bert_model(self, original_model, attention_head_size, num_attention_heads):
        # config = original_model.config
        # new_config = transformers.BertConfig.from_dict(config.to_dict())
        # new_config.num_attention_heads = num_attention_heads
        # new_config.hidden_size = self.all_head_size  # Adjust hidden size based on new heads

        pruned_model = transformers.BertModel(original_model.config)
        
        with torch.no_grad():
            for layer_orig, layer_pruned in zip(original_model.encoder.layer, pruned_model.encoder.layer):
                layer_pruned.attention.self.attention_head_size = attention_head_size
                layer_pruned.attention.self.num_attention_heads = num_attention_heads
                layer_pruned.attention.self.all_head_size = attention_head_size * num_attention_heads

                # Adjust query, key, value weights and biases
                # for name in ['query', 'key', 'value']:
                query_layer = getattr(layer_orig.attention.self, 'query')
                key_layer = getattr(layer_orig.attention.self, 'key')

                query_indices = self.select_top_weights(query_layer.weight.data, (self.all_head_size + 768) // 2)

                q_pruned_layer = getattr(layer_pruned.attention.self, "query")
                q_pruned_layer.weight = nn.Parameter(query_layer.weight.data[query_indices, :])
                q_pruned_layer.bias = nn.Parameter(query_layer.bias.data[query_indices])

                k_pruned_layer = getattr(layer_pruned.attention.self, "key")
                k_pruned_layer.weight = nn.Parameter(key_layer.weight.data[query_indices, :])
                k_pruned_layer.bias = nn.Parameter(key_layer.bias.data[query_indices])
                
                key_indices = self.select_top_weights(k_pruned_layer.weight.data, self.all_head_size)

                q_pruned_layer.weight = nn.Parameter(q_pruned_layer.weight.data[key_indices, :])
                q_pruned_layer.bias = nn.Parameter(q_pruned_layer.bias.data[key_indices])

                k_pruned_layer.weight = nn.Parameter(k_pruned_layer.weight.data[key_indices, :])
                k_pruned_layer.bias = nn.Parameter(k_pruned_layer.bias.data[key_indices])
                
                value_layer = getattr(layer_orig.attention.self, 'value')
                v_indices = self.select_top_weights(query_layer.weight.data, self.all_head_size)

                v_pruned_layer = getattr(layer_pruned.attention.self, "value")
                v_pruned_layer.weight = nn.Parameter(value_layer.weight.data[v_indices, :])
                v_pruned_layer.bias = nn.Parameter(value_layer.bias.data[v_indices])

                print("pruned_layer:", q_pruned_layer.weight.shape)
                
                # Adjust dense layer in the output of the attention block
                out_layer_orig = layer_orig.attention.output.dense
                out_layer_pruned = layer_pruned.attention.output.dense
                out_layer_pruned.weight = nn.Parameter(out_layer_orig.weight.data[:, v_indices])
                print("out_layer_pruned:", out_layer_pruned.weight.shape)
                out_layer_pruned.bias = nn.Parameter(out_layer_orig.bias.data)

        return pruned_model

    def select_top_weights(self, weights, k):
        # Assuming selecting top-k weights by some norm or criteria
        norms = torch.norm(weights, p=2, dim=1)
        _, indices = torch.topk(norms, k)
        indices, _ = torch.sort(indices)
        return indices


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    # Data Preparation
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_dataset("stanfordnlp/sst2")
    data = dataset.with_format("torch")

    train_text = [each['sentence'] for each in data['train']]
    train_label = [each['label'] for each in data['train']]
    val_text = [each['sentence'] for each in data['validation']]
    val_label = [each['label'] for each in data['validation']]

    # Dataset and DataLoader
    trainset = textDataset(train_text, train_label, tokenizer, max_length=100)
    train_loader = DataLoader(dataset=trainset, batch_size=32)
    valset = textDataset(val_text, val_label, tokenizer, max_length=100)
    val_loader = DataLoader(dataset=valset, batch_size=32)

    # Model Initialization
    prune_heads = 640  # Number of columns/rows to retain in the attention layers
    # model = BERTL2Pruned(prune_heads).to(device)
    original_model = BERT().to('cpu')
    # original_model.load_state_dict(torch.load("./checkpoint/baseline.pt"))
    original_model = original_model.bert_model

    model = BERTStructurallyPruned(original_model, attention_head_size=64, num_attention_heads=10).to(device)

    # Training Configuration
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.000001)
    
    best_acc = 0.0
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiling_file_name),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for i in range(EPOCH):
        # for i in range(2):
            model.train()
            start = time.time()
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
            end = time.time()
            print(f"training time for one epoch: {end - start}s")
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

                    optimizer.zero_grad()

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
                
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), "./checkpoint/l2_prune-10-64-new.pt")
                print(f'##Epoch {i+1}: Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f} in {end_time - start_time}s')

    model.eval()
    
    with torch.no_grad():
        start_time = time.time()
        num_correct = 0
        num_samples = 0
        for batch, dl in enumerate(train_loader):
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
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")




