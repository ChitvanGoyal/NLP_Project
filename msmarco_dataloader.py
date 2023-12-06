import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import Dataset as HFDataset # Rename to avoid conflict with PyTorch Datasets
from tqdm.auto import tqdm

class MSMARCODataset(Dataset):
    
    def __init__(self, dataframe, tokenizer, max_token_length=512):
        
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        
        row = self.dataframe.iloc[idx]
        query_text = str(row['query'])
        passage_text = str(row['passage'])

        # Encode the query-passage pair
        encoding = self.tokenizer.encode_plus(
            query_text,
            passage_text,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_token_length,
            return_tensors="pt"
        )

        label = torch.tensor(row['relevance'])

        return {
            'query_id': row['qid'],
            'passage_id': row['pid'],
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': label
        }


def collate_fn(batch):
    input_ids_batch = torch.stack([item['input_ids'] for item in batch])
    attention_mask_batch = torch.stack([item['attention_mask'] for item in batch])
    labels_batch = torch.stack([item['label'] for item in batch])
    query_ids = torch.tensor([item['query_id'] for item in batch])
    passage_ids = torch.tensor([item['passage_id'] for item in batch]) 

    return {
        'input_ids': input_ids_batch,
        'attention_mask': attention_mask_batch,
        'labels': labels_batch,
        'query_ids': query_ids,
        'passage_ids': passage_ids
    }


def create_data_loader(dataframe, tokenizer, batch_size, max_token_length=512):
    dataset = MSMARCODataset(dataframe, tokenizer, max_token_length)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)


def calculate_mrr(ground_truths, predictions):
    reciprocal_ranks = []

    for query_id in ground_truths.keys():
        if query_id in predictions:
            try:
                rank = predictions[query_id].index(ground_truths[query_id]) + 1
                reciprocal_ranks.append(1 / rank)
            except ValueError:
                reciprocal_ranks.append(0)
        else:
            reciprocal_ranks.append(0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def main():
    tokenizer = BertTokenizer.from_pretrained('Capreolus/bert-base-msmarco')
    df = pd.read_csv(r"C:\Users\Andrew Deur\Documents\NYU\DS-GA 1011 NLP\Project\filtered_top1000_dev_with_labels_bmscore.tsv")
    data_loader = create_data_loader(df, tokenizer, batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_name = "Capreolus/bert-base-msmarco"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Populate ground_truths with correct relevant passage for each query
    relevant_df = df[df['relevance'] == 1]
    ground_truths = dict(zip(relevant_df['qid'], relevant_df['pid'])) # Create the dictionary mapping query_id to passage_id
    
    query_passage_scores = {}
    
    with torch.no_grad():  # Disable gradient computation
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            query_ids = batch['query_ids']
            passage_ids = batch['passage_ids']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=1) # Extract relevance scores
            batch_scores = probabilities[:, 1]  # Assuming label '1' is relevant
            
            for query_id, score, passage_id in zip(query_ids, batch_scores, passage_ids):
                if query_id not in query_passage_scores:
                    query_passage_scores[query_id] = []
                query_passage_scores[query_id].append((score.item(), batch['passage_ids']))  # Assuming passage IDs

        
    # Re-rank passages for each query
    predictions = {}
    for query_id, scores in query_passage_scores.items():
        ranked_passages = sorted(scores, key=lambda x: x[0], reverse=True)
        predictions[query_id] = [x[1] for x in ranked_passages]

    # Calculate MRR
    mrr_score = calculate_mrr(ground_truths, predictions)
    print(f"MRR Score: {mrr_score}")
    


if __name__ == '__main__':
    main()