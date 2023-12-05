import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import Dataset as HFDataset

class MSMARCODataset(Dataset):
    def __init__(self, dataframe, tokenizer, candidate_document_set_size, max_token_length=512):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        self.candidate_documents_size = candidate_document_set_size
        self.grouped = self._group_data()

    def _group_data(self):
        # Group by 'qid' and select top N docs based on BM25 score
        return self.dataframe.groupby('qid', group_keys=False)\
            .apply(lambda x: x.nlargest(self.candidate_documents_size, 'bm25_score'))

    def __len__(self):
        return len(self.grouped['qid'].unique())
    
    def __getitem__(self, idx):
        # Map idx to the corresponding qid
        unique_qids = self.grouped['qid'].unique()
        if idx >= len(unique_qids):
            raise IndexError("Index out of bounds")

        query_id = unique_qids[idx]

        # Get the group corresponding to this query ID
        group = self.grouped[self.grouped['qid'] == query_id]
        query_text = group['query'].iloc[0]  # The query is the same for all passages in the group

        query_encoding = self.tokenizer(
            query_text,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_token_length,
            return_tensors="pt"
        )

        passages = group['passage'].tolist()
        passage_encodings = [self.tokenizer(
            passage,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_token_length,
            return_tensors="pt"
        ) for passage in passages]

        labels = torch.tensor(group['relevance'].tolist())

        return {
            'query_input_ids': query_encoding['input_ids'].squeeze(0),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(0),
            'passage_input_ids': [pe['input_ids'].squeeze(0) for pe in passage_encodings],
            'passage_attention_mask': [pe['attention_mask'].squeeze(0) for pe in passage_encodings],
            'labels': labels
        }

def create_data_loader(dataframe, tokenizer, batch_size, max_token_length=512):
    dataset = MSMARCODataset(dataframe, tokenizer, max_token_length)
    return DataLoader(dataset, batch_size=batch_size)

# Example usage
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

df = pd.read_csv("filtered_top1000_dev_with_labels_bmscore.tsv")
data_loader = create_data_loader(df, tokenizer, batch_size=32)
