import pandas as pd
from tqdm import tqdm
import numpy as np

# Top 1000 passage for each query by BM score
top1000_dev_path = '../Data/top1000.dev.tar' 
top1000_dev_df = pd.read_csv(top1000_dev_path, sep='\t', compression='tar', header=None, names=['qid', 'pid', 'query', 'passage'])
print('top1000 extracted')

# Relavance labels(0/1) for queery and passage pairs
qrel_dev_path = '../Data/qrels.dev.tsv' 
qrel_dev_df = pd.read_csv(qrel_dev_path, sep='\t', header=None, names=['qid', 'Q0', 'pid', 'relevance'])
print('qrel extracted')

# merge relevance labels
merged_df = pd.merge(top1000_dev_df, qrel_dev_df, on=['qid', 'pid'], how='left')
merged_df['relevance'].fillna(0, inplace=True)
merged_df['relevance'] = merged_df['relevance'].astype(int)
print('merged')

# save file location
output_path = '../Data/top1000_dev_with_labels.tsv'


# remove queries with no relavance labels
unique_queries_qrel_dev = merged_df[merged_df['relevance']==1]['qid'].unique()
filtered_merged_df = merged_df[merged_df['qid'].isin(unique_queries_qrel_dev)]
print('filtered')

filtered_merged_df.to_csv(output_path, index=False)
print('saved')

