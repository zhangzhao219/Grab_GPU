import sys
import torch
import pandas as pd
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.distributed as dist

dist.init_process_group("nccl")
rank, world_size = dist.get_rank(), dist.get_world_size()
device_id = rank % torch.cuda.device_count()
device = torch.device(device_id)
tokenizer = AutoTokenizer.from_pretrained("pretrained/prajjwal1/bert-tiny")

class Dataset_Reader(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = []
        data_file = pd.read_csv(file_path)
        data_file = pd.concat([data_file for i in range(1000)])
        tokenized_text = tokenizer(
            text=data_file["text"].tolist(),
            padding="max_length",
            max_length=512,
            truncation=True,
        )
        for index in range(len(data_file)):
            self.data.append({
                "input_ids": torch.tensor(tokenized_text["input_ids"][index]),
                "attention_mask": torch.tensor(
                    tokenized_text["attention_mask"][index]
                ),
                "labels": data_file["stars"].tolist()[index],
            })
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]

train_dataloader = DataLoader(Dataset_Reader("data.csv", tokenizer), batch_size=int(sys.argv[1]))
model = AutoModelForSequenceClassification.from_pretrained("pretrained/prajjwal1/bert-tiny", num_labels=5).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 100000000

model.train()
for epoch in tqdm(range(num_epochs)):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()