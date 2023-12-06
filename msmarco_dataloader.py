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
    passage_input_ids = [item['passage_input_ids'] for item in batch]
    passage_attention_masks = [item['passage_attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Find the maximum number of passages and maximum passage length in this batch
    max_passage_count = max(len(p) for p in passage_input_ids)
    padded_passage_input_ids = []
    padded_passage_attention_masks = []

    for p_input_ids, p_attention_masks in zip(passage_input_ids, passage_attention_masks):
        # Pad each passage list to the same length
        padded_p_input_ids = pad_sequence(p_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        padded_p_attention_masks = pad_sequence(p_attention_masks, batch_first=True, padding_value=0)

        # Pad the list itself to have the same number of passages
        while len(padded_p_input_ids) < max_passage_count:
            padded_p_input_ids = torch.cat([padded_p_input_ids, torch.zeros((1, padded_p_input_ids.size(1)), dtype=torch.long)], dim=0)
            padded_p_attention_masks = torch.cat([padded_p_attention_masks, torch.zeros((1, padded_p_attention_masks.size(1)), dtype=torch.long)], dim=0)

        padded_passage_input_ids.append(padded_p_input_ids)
        padded_passage_attention_masks.append(padded_p_attention_masks)

    query_input_ids = torch.stack(query_input_ids)
    query_attention_masks = torch.stack(query_attention_masks)
    labels = torch.stack(labels)

    return {
        'query_input_ids': query_input_ids,
        'query_attention_mask': query_attention_masks,
        'passage_input_ids': torch.stack(padded_passage_input_ids),
        'passage_attention_mask': torch.stack(padded_passage_attention_masks),
        'labels': labels
    }

def create_data_loader(dataframe, tokenizer, batch_size, candidate_document_set_size, max_token_length=512):
    dataset = MSMARCODataset(dataframe, tokenizer, candidate_document_set_size, max_token_length)

    collate_fn_with_tokenizer = partial(collate_fn, tokenizer=tokenizer)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_with_tokenizer)


def main():
    # Example usage
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

            batch_scores = []
            for p_input_ids, p_attention_mask in zip(batch['passage_input_ids'], batch['passage_attention_mask']):
                
                # Ensure the tensors are of the same batch size
                min_batch_size = min(query_input_ids.size(0), p_input_ids.size(0))
                query_input_ids_resized = query_input_ids[:min_batch_size]
                query_attention_mask_resized = query_attention_mask[:min_batch_size]
                p_input_ids_resized = p_input_ids[:min_batch_size].to(device)
                p_attention_mask_resized = p_attention_mask[:min_batch_size].to(device)

                # Concatenate along the sequence length dimension
                combined_input_ids = torch.cat([query_input_ids_resized, p_input_ids_resized], dim=1)
                combined_attention_mask = torch.cat([query_attention_mask_resized, p_attention_mask_resized], dim=1)

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

                # Flatten the batch_scores tensor and extend the main scores list
                scores.extend(batch_scores.flatten().tolist())


if __name__ == '__main__':
    main()