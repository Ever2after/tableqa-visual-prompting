from model.gpt import openai_client
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def _get_embedding(input_str):
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=input_str,
        encoding_format="float"
    )
    return response.data[0].embedding

def _cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    dot_product = np.dot(v1, v2)
    magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot_product / magnitude if magnitude != 0 else 0

def _get_column_embeddings(df):
    def _process_column(column_name, column_data):
        column_string = f"{column_name} " + ' '.join(column_data.apply(str).values)
        return _get_embedding(column_string)
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(
            _process_column, 
            df.columns, 
            [df.iloc[:, i] for i in range(len(df.columns))]
        ))
    return dict(zip(range(len(df.columns)), results))

def _get_row_embeddings(df):
    def _process_row(row_data):
        row_string = ' '.join(row_data.apply(str).values)
        return _get_embedding(row_string)
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(_process_row, [df.iloc[i] for i in range(len(df))]))
    return dict(zip(range(len(df)), results))

def get_table_score(table, query, top_k=10, column_top_k=5, row_top_k=5, cached_embeddings=None):
    if cached_embeddings:
        column_embeddings, row_embeddings, query_embedding = cached_embeddings
    else:
        column_embeddings = _get_column_embeddings(table)  
        row_embeddings = _get_row_embeddings(table)        
        query_embedding = _get_embedding(query)    
    
    scores = pd.DataFrame(0.0, index=range(len(row_embeddings)), columns=range(len(table.columns)), dtype=float)

    row_scores = {i: _cosine_similarity(row_embeddings[i], query_embedding) for i in range(len(table))}
    col_scores = {i: _cosine_similarity(column_embeddings[i], query_embedding) for i in range(len(table.columns))}

    if top_k:
        for i in range(len(row_scores)):
            scores.iloc[i, :] += row_scores[i]
        for i in range(len(col_scores)):
            scores.iloc[:, i] += col_scores[i] 

        # 각 cell의 score을 sort하고 상위 top_k개의 cell의 score 제외 나머지는 0으로
        sorted_scores = scores.stack().sort_values(ascending=False)
        threshold_score = sorted_scores.iloc[top_k - 1] if len(sorted_scores) > top_k else sorted_scores.iloc[-1]
        scores = scores.where(scores >= threshold_score, 0)
    else:
        top_k_rows = sorted(row_scores, key=row_scores.get, reverse=True)[:row_top_k]
        for i in top_k_rows:
            scores.iloc[i, :] += row_scores[i]
        
        top_k_cols = sorted(col_scores, key=col_scores.get, reverse=True)[:column_top_k]
        for i in top_k_cols:
            scores.iloc[:, i] += col_scores[i]

    return scores.astype(float)