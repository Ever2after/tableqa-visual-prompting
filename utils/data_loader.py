import os
import pandas as pd
import pickle
from datasets import load_dataset
from utils.logger import setup_logger

# 로거 설정
logger = setup_logger("DataLoader", log_file='data_loader.log')

def download_and_cache_dataset(dataset_url, local_filename, cache_dir, split, json_lines=True):
    """
    데이터셋을 Hugging Face URL에서 다운로드하거나 로컬 캐시에서 로드.

    Parameters:
        - dataset_url (str): Hugging Face Dataset URL
        - local_filename (str): 로컬에 저장할 파일 경로
        - cache_dir (str): 캐시 디렉토리 경로

    Returns:
        - pd.DataFrame: 로드된 데이터셋
    """
    local_path = os.path.join(cache_dir, local_filename)

    if os.path.exists(local_path):
        logger.info(f"Loading dataset from local cache: {local_path}")
        return pd.read_json(local_path, lines=json_lines)

    logger.info(f"Downloading dataset from: {dataset_url}")
    dataset = load_dataset(dataset_url, split=split, cache_dir=cache_dir, trust_remote_code=True)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    dataset.to_json(local_path)
    logger.info(f"Dataset cached locally at: {local_path}")

    return dataset.to_pandas()

def load_embeddings(embedding_file, num_entries):
    """
    임베딩 캐시를 로드하거나 기본값을 설정.

    Parameters:
        - embedding_file (str): 로컬 임베딩 파일 경로
        - num_entries (int): 데이터셋 항목 수

    Returns:
        - list: 임베딩 리스트
    """
    if os.path.exists(embedding_file):
        logger.info(f"Loading embeddings from cache: {embedding_file}")
        with open(embedding_file, 'rb') as f:
            return pickle.load(f)
    else:
        logger.warning(f"Embedding cache not found: {embedding_file}")
        return [None] * num_entries
