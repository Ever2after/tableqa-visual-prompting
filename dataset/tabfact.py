import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import pickle

from utils.data_loader import download_and_cache_dataset, load_embeddings
from utils.score import get_tableqa_embeddings
from utils.logger import setup_logger
from config import Config

logger = setup_logger("DataLoader", log_file='data_loader.log')

class TabFactDataset:
    """
    TabFact 데이터셋 클래스.
    """
    
    def __init__(self, cache_dir=None, split='test'):
        self.cache_dir = cache_dir or Config.DATASET_PATH
        self.split = split
        self.dataset = self._load_dataset()
        self.embeddings = self._load_embeddings()

    def _load_dataset(self):
        """데이터셋 로드."""
        dataset_url = 'ibm/tab_fact'
        local_filename = f'tabfact_{self.split}.json'
        return download_and_cache_dataset(dataset_url, local_filename, self.cache_dir, self.split)

    def _load_embeddings(self):
        """임베딩 로드."""
        embedding_file = os.path.join(self.cache_dir, f'tabfact_{self.split}_embeddings.pkl')
        
        embeddings = load_embeddings(embedding_file, len(self.dataset))

        if not embeddings[0]:
            # caculate embeddings in order 
            logger.info("Calculating embeddings for TabFact dataset.")

            f = lambda x: get_tableqa_embeddings(pd.DataFrame(x['table']['rows'], columns=x['table']['header']), x['statement'])
            with ThreadPoolExecutor() as executor:
                embeddings = list(executor.map(f, self.dataset.iloc[:1000].to_dict(orient='records')))
            with open(embedding_file, 'wb') as f:
                pickle.dump(embeddings, f)
        
        return embeddings

    def get_item(self, index):
        """
        인덱스에 해당하는 항목 반환.
        """
        if index >= len(self.dataset):
            raise IndexError(f"Index {index} out of bounds for dataset with {len(self.dataset)} entries.")
        
        row = self.dataset.iloc[index]
        table = pd.DataFrame(row.table['rows'], columns=row.table['header'])
        question = f"Answer whether the following statement is true or false: {row['statement']}"
        answer = 'true' if row['label'] == 1 else 'false'
        embedding = self.embeddings[index]

        return {
            "table": table,
            "question": question,
            "answer": answer,
            "embedding": embedding
        }

    def __len__(self):
        return len(self.dataset)
