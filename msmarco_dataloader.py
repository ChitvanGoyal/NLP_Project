import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
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

        query_encoding = self.tokenizer(
            query_text,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_token_length,
            return_tensors="pt"
        )

        passage_encoding = self.tokenizer(
            passage_text,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_token_length,
            return_tensors="pt"
        )

        label = torch.tensor(row['relevance'])

        return {
            'query_input_ids': query_encoding['input_ids'].squeeze(0),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(0),
            'passage_input_ids': passage_encoding['input_ids'].squeeze(0),
            'passage_attention_mask': passage_encoding['attention_mask'].squeeze(0),
            'label': label
        }


def collate_fn(batch):
    max_length = 512  # Adjust based on your model's max length
    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []

    for item in batch:
        # Concatenate the query and passage input ids and attention masks
        combined_input_ids = torch.cat([item['query_input_ids'], item['passage_input_ids']])[:max_length]
        combined_attention_mask = torch.cat([item['query_attention_mask'], item['passage_attention_mask']])[:max_length]

        # Calculate padding length for this concatenated pair
        padding_length = max_length - combined_input_ids.size(0)

        # Pad the concatenated input ids and attention masks to max_length
        padded_input_ids = torch.cat([combined_input_ids, torch.zeros(padding_length, dtype=torch.long)])
        padded_attention_mask = torch.cat([combined_attention_mask, torch.zeros(padding_length, dtype=torch.long)])

        input_ids_batch.append(padded_input_ids)
        attention_mask_batch.append(padded_attention_mask)
        labels_batch.append(item['label'])

    # Convert lists to tensors
    input_ids_batch = torch.stack(input_ids_batch)
    attention_mask_batch = torch.stack(attention_mask_batch)
    labels_batch = torch.stack(labels_batch)

    return {
        'input_ids': input_ids_batch,
        'attention_mask': attention_mask_batch,
        'labels': labels_batch
    }


def create_data_loader(dataframe, tokenizer, batch_size, max_token_length=512):
    dataset = MSMARCODataset(dataframe, tokenizer, max_token_length)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)


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

    scores = []
    with torch.no_grad():  # Disable gradient computation
        
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Run model inference
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Extract relevance scores
            probabilities = torch.softmax(outputs.logits, dim=1)
            batch_scores = probabilities[:, 1]  # Assuming label '1' is relevant
            scores.extend(batch_scores.flatten().tolist())


if __name__ == '__main__':
    main()