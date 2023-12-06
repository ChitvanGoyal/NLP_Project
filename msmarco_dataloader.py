import pandas as pd
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import Dataset as HFDataset
from tqdm.auto import tqdm

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


def collate_fn(batch, tokenizer):
    query_input_ids = [item['query_input_ids'] for item in batch]
    query_attention_masks = [item['query_attention_mask'] for item in batch]
    all_passage_input_ids = []
    all_passage_attention_masks = []

    for item in batch:
        # Concatenate all passages for each query
        concatenated_passages_input_ids = torch.cat(item['passage_input_ids'], dim=0)
        concatenated_passages_attention_mask = torch.cat(item['passage_attention_mask'], dim=0)

        # Padding if necessary to match the max sequence length
        max_length = tokenizer.model_max_length
        if concatenated_passages_input_ids.size(0) < max_length:
            required_padding = max_length - concatenated_passages_input_ids.size(0)
            concatenated_passages_input_ids = torch.cat([
                concatenated_passages_input_ids,
                torch.full((required_padding,), tokenizer.pad_token_id, dtype=torch.long)
            ])
            concatenated_passages_attention_mask = torch.cat([
                concatenated_passages_attention_mask,
                torch.zeros(required_padding, dtype=torch.long)
            ])

        all_passage_input_ids.append(concatenated_passages_input_ids)
        all_passage_attention_masks.append(concatenated_passages_attention_mask)

    query_input_ids = torch.stack(query_input_ids)
    query_attention_masks = torch.stack(query_attention_masks)
    all_passage_input_ids = torch.stack(all_passage_input_ids)
    all_passage_attention_masks = torch.stack(all_passage_attention_masks)

    return {
        'query_input_ids': query_input_ids,
        'query_attention_mask': query_attention_masks,
        'passage_input_ids': all_passage_input_ids,
        'passage_attention_mask': all_passage_attention_masks
    }

def create_data_loader(dataframe, tokenizer, batch_size, candidate_document_set_size, max_token_length=512):
    dataset = MSMARCODataset(dataframe, tokenizer, candidate_document_set_size, max_token_length)
    collate_fn_with_tokenizer = partial(collate_fn, tokenizer=tokenizer)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_with_tokenizer)


def main():
    tokenizer = BertTokenizer.from_pretrained('Capreolus/bert-base-msmarco')
    df = pd.read_csv(r"C:\Users\Andrew Deur\Documents\NYU\DS-GA 1011 NLP\Project\filtered_top1000_dev_with_labels_bmscore.tsv")
    data_loader = create_data_loader(df, tokenizer, batch_size=32, candidate_document_set_size=20)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_name = "Capreolus/bert-base-msmarco"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    scores = []
    with torch.no_grad():  # Disable gradient computation
        
        for batch in tqdm(data_loader):
            query_input_ids = batch['query_input_ids'].to(device)
            query_attention_mask = batch['query_attention_mask'].to(device)
            passage_input_ids = batch['passage_input_ids'].to(device)
            passage_attention_mask = batch['passage_attention_mask'].to(device)

            # Concatenate along the sequence length dimension
            combined_input_ids = torch.cat([query_input_ids, passage_input_ids], dim=1)
            combined_attention_mask = torch.cat([query_attention_mask, passage_attention_mask], dim=1)

            # Ensure the combined sequence does not exceed the maximum length
            max_length = model.config.max_position_embeddings
            if combined_input_ids.size(1) > max_length:
                combined_input_ids = combined_input_ids[:, :max_length]
                combined_attention_mask = combined_attention_mask[:, :max_length]

            # Run model inference
            outputs = model(input_ids=combined_input_ids, attention_mask=combined_attention_mask)

            # Extract relevance scores
            probabilities = torch.softmax(outputs.logits, dim=1)
            batch_scores = probabilities[:, 1]  # Assuming label '1' is relevant
            scores.extend(batch_scores.flatten().tolist())


if __name__ == '__main__':
    main()