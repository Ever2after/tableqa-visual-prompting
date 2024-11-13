import os
import pandas as pd
from utils.data_loader import download_and_cache_dataset, load_embeddings
from config import Config

class WikiTQDataset:
    """
    WikiTableQuestions 데이터셋 클래스.
    """
    
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or Config.DATASET_PATH
        self.dataset = self._load_dataset()
        self.embeddings = self._load_embeddings()

    def _load_dataset(self):
        """데이터셋 로드."""
        dataset_url = 'hf://datasets/TableQAKit/WTQ/test.json'
        local_filename = 'wikitq_test.json'
        return download_and_cache_dataset(dataset_url, local_filename, self.cache_dir)

    def _load_embeddings(self):
        """임베딩 로드."""
        embedding_file = os.path.join(self.cache_dir, 'wikitq_embeddings.pkl')
        return load_embeddings(embedding_file, len(self.dataset))

    def get_item(self, index):
        """
        인덱스에 해당하는 항목 반환.
        """
        if index >= len(self.dataset):
            raise IndexError(f"Index {index} out of bounds for dataset with {len(self.dataset)} entries.")
        
        row = self.dataset.iloc[index]
        table = pd.DataFrame(row.table['rows'], columns=row.table['header'])
        question = row['question']
        answer = row['seq_out']
        embedding = self.embeddings[index]

        return {
            "table": table,
            "question": question,
            "answer": answer,
            "embedding": embedding
        }

    def __len__(self):
        return len(self.dataset)
