from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn

data = pd.read_csv('../data/pid_passage_map.csv')
model_name = 'castorini/doc2query-t5-base-msmarco'
#model_name = 'doc2query/msmarco-t5-base-v1'
model = T5ForConditionalGeneration.from_pretrained(model_name)
#model = nn.DataParallel(model)
tokenizer = T5Tokenizer.from_pretrained(model_name)

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

class PassageDataset(Dataset):
    def __init__(self, passages, tokenizer, max_length=320):
        self.passages = passages
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.passages)

    def __getitem__(self, idx):
        passage = self.passages[idx]
        input_ids = self.tokenizer.encode(passage, max_length=self.max_length, truncation=True, return_tensors="pt")[0]

        # Pad the input_ids to the max_length
        padding_length = max(0, self.max_length - input_ids.size(0))
        input_ids = torch.nn.functional.pad(input_ids, (0, padding_length), value=self.tokenizer.pad_token_id)

        return input_ids

def generate_queries(model, input_ids, tokenizer, max_length=64, num_return_sequences=2):
    outputs = model.generate(
        input_ids=input_ids.to(device),  # Move input_ids to GPU
        max_length=max_length,
        do_sample=True,
        top_p=0.95,
        num_return_sequences=num_return_sequences
    )
    queries = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return queries

passages = data['passage'][:1000000]

passage_dataset = PassageDataset(passages, tokenizer)
dataloader = DataLoader(passage_dataset, batch_size=256, shuffle=False, num_workers=8)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

ls = []
with torch.no_grad():
    # Generate queries for each passage
    all_queries = []
    for batch in tqdm(dataloader):
        queries = generate_queries(model, batch, tokenizer)
        k = 0
        for i in range(int(len(queries) / 2)):
            ls.append([queries[k], queries[k + 1]])
            k = k + 2

df = pd.DataFrame({'passage_expanded': ls})
df.to_csv('expanded_passage1.csv')

