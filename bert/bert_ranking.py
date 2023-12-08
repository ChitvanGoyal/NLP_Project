import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
import numpy as np
import io

df = pd.read_csv('../data/top_1000_BM_score.csv')
df = df.sort_values(['qid', 'bm25_score'], ascending=False)
df['rank_BM'] = df.groupby('qid')['bm25_score'].rank(ascending=False).astype(int)
df = df[df['rank_BM']<=30]

class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, model, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        query = self.df.iloc[idx]['query']
        passage = self.df.iloc[idx]['passage']

        tokenized_passages = self.tokenizer.encode_plus(query, passage, return_tensors='pt', padding=True, truncation=True)

        input_ids = tokenized_passages['input_ids']

        padding_length = max(0, self.max_length - input_ids.size(1))
        input_ids = torch.nn.functional.pad(input_ids, (0, padding_length), value=self.tokenizer.pad_token_id)

        attention_mask = tokenized_passages['attention_mask']
        padding_length = max(0, self.max_length - attention_mask.size(1))
        attention_mask = torch.nn.functional.pad(attention_mask, (0, padding_length), value=0)  # Padding attention_mask with 0

        token_type_ids = tokenized_passages['token_type_ids']
        padding_length = max(0, self.max_length - token_type_ids.size(1))
        token_type_ids = torch.nn.functional.pad(token_type_ids, (0, padding_length), value=0)  # Padding attention_mask with 0

        return {
            'input_ids': input_ids.squeeze(),
            'attention_mask': attention_mask.squeeze(),
            'token_type_ids': token_type_ids.squeeze()
        }


model_name = "Capreolus/bert-base-msmarco"
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained("Capreolus/bert-base-msmarco")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))
model = nn.DataParallel(model)
model.to(device)
model.eval()


custom_dataset = CustomDataset(df, tokenizer, model)


batch_size = 512 # Choose an appropriate batch size
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

all_logits=[]
for batch_tokenized_passages in tqdm(dataloader):
    a=batch_tokenized_passages
    with torch.no_grad():
        outputs = model(input_ids= batch_tokenized_passages['input_ids'].to(device),
                        attention_mask = batch_tokenized_passages['attention_mask'].to(device),
                        token_type_ids = batch_tokenized_passages['token_type_ids'].to(device)
                        )

    logits = outputs.logits
    logits_list = logits.tolist()
    all_logits.extend(logits_list)

file_path = 'logits_30.txt'  # Replace with the actual path to your file
with open(file_path, 'w') as txt_file:
    for logits_list in all_logits:
        txt_file.write(' '.join(map(str, logits_list)) + '\n')
# store file

#file_path = 'logits_10.txt'  # Replace with the actual path to your file

ls = []
# Read the contents of the file
with open(file_path, 'r') as file:
    file_contents = file.read()
    
df1 = pd.read_csv(io.StringIO(file_contents), sep=' ', header=None, names=['label_1', 'label_2'])

import numpy as np
logits=np.array(df1)
label_probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)


######################### Bert MRR
df['bert_score']=label_probabilities.T[1]
#df['bert_score'] = df1['label_1']
print(df['bert_score'].head())
#df = df.sort_values(['qid', 'bert_score'], ascending=False)
df['rank_bert'] = df.groupby('qid')['bert_score'].rank(ascending=False).astype(int)

total_mrr = 0.0
r=[]

for unique_id in df['qid'].unique():

    current_id_df = df[df['qid'] == unique_id]

    current_id_df = current_id_df.sort_values(by='rank_bert')

    rank_first_relevant = current_id_df.loc[current_id_df['relevance'] == 1, 'rank_bert'].min()

    mrr = 1 / rank_first_relevant if pd.notnull(rank_first_relevant) else 0
    
    # Set number of documents to consider rank_first_relevant<10, 1000 default
    if rank_first_relevant<=10:

        total_mrr += mrr
    r.append(rank_first_relevant)

# Calculate Mean Reciprocal Rank
mean_mrr = total_mrr / len(df['qid'].unique())

print('mean mrr bert = ' + str(mean_mrr))




###################### BM 25 MRR


#df['bert_score']=label_probabilities.T[0]

df = df.sort_values(['qid', 'bm25_score'], ascending=False)
df['rank_bm25'] = df.groupby('qid')['bm25_score'].rank(ascending=False).astype(int)

total_mrr = 0.0
r=[]

for unique_id in df['qid'].unique():

    current_id_df = df[df['qid'] == unique_id]

    current_id_df = current_id_df.sort_values(by='rank_bm25')

    rank_first_relevant = current_id_df.loc[current_id_df['relevance'] == 1, 'rank_bm25'].min()

    mrr = 1 / rank_first_relevant if pd.notnull(rank_first_relevant) else 0

    # Set number of documents to consider rank_first_relevant<10, 1000 default
    if rank_first_relevant<=10:

        total_mrr += mrr
    r.append(rank_first_relevant)

# Calculate Mean Reciprocal Rank
mean_mrr = total_mrr / len(df['qid'].unique())

print('mean mrr bm25 = ' + str(mean_mrr))
