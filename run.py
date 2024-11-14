import pandas as pd
import argparse
import os
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from data.wikitq import WikiTQDataset
from data.tabfact import TabFactDataset
from evaluate import evaluate_predictions
from model.gpt import GPT
from model.gemini import Gemini
from config import Config
from utils.logger import setup_logger
from utils.table_generator import generate_table_image
from utils.score import get_table_score
import random

# 로거 설정
logger = setup_logger("Run", log_file='run.log')

def get_dataset_class(dataset_name):
    """
    데이터셋 이름에 따라 적절한 데이터셋 클래스를 반환.
    """
    dataset_classes = {
        'wikitq': WikiTQDataset,
        'tabfact': TabFactDataset
    }
    if dataset_name not in dataset_classes:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return dataset_classes[dataset_name]

def save_image(base64_image):
    a = random.randint(1, 100)
    """
    base64 인코딩된 이미지를 파일로 저장.
    """
    image = Image.open(io.BytesIO(base64.b64decode(base64_image)))
    image.save(f'images/{a}_table_image.png')
    

def get_answer(table, query, model, mode, plain, top_k, column_top_k, row_top_k, cached_embeddings):
    model_instance = GPT() if model == 'gpt' else Gemini()

    if mode == 'text':
        return model_instance.get_tableqa_answer(mode='text', table=table.to_markdown(), query=query)
    
    score = get_table_score(table, query, top_k, column_top_k, row_top_k, cached_embeddings) if not plain else None

    try:
        table_image = generate_table_image(table, score)
        # save_image(table_image)
    except Exception as e:
        print(e)
        return False

    return model_instance.get_tableqa_answer(mode='image', table=table_image, query=query)

def process_item(i, dataset, model, mode, plain, top_k, column_top_k, row_top_k):
    item = dataset.get_item(i)
    logger.info(f"Processing item {i} with question: {item['question']}")
    
    table = item['table']
    query = item['question']
    answer = item['answer']
    cached_embeddings = item['embedding']

    prediction = get_answer(
        table,
        query,
        model,
        mode,
        plain,
        top_k,
        column_top_k,
        row_top_k,
        cached_embeddings
    )
    return {'index': i, 'prediction': prediction, 'answer': answer}

def run_and_save_predictions(dataset, model, mode, plain=False, top_k=10, column_top_k=5, row_top_k=5, output_path='predictions.csv', length=100, offset=0):
    logger.info(f"Running predictions with model: {model}, mode: {mode}, plain: {plain}, top_k: {top_k}, column_top_k: {column_top_k}, row_top_k: {row_top_k}")
    predictions = []

    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_index = {
            executor.submit(
                process_item, 
                i, 
                dataset, 
                model,
                mode,
                plain, 
                top_k, 
                column_top_k, 
                row_top_k
            ): i 
            for i in range(offset, offset + min(len(dataset), length))
        }

        for future in as_completed(future_to_index):
            try:
                result = future.result()
                predictions.append(result)
            except Exception as e:
                logger.error(f"Error processing item {future_to_index[future]}: {e}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(predictions).to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")

def main():
    # 명령줄 인자 설정
    parser = argparse.ArgumentParser(description="Table QA Model Execution and Evaluation")
    parser.add_argument('--dataset', type=str, default='wikitq', help='Dataset name: wikitq, tabfact, etc.')
    parser.add_argument('--model', type=str, default='gpt', help='Model to use: gpt or gemini')
    parser.add_argument('--mode', type=str, default='image', help='Execution mode: text or image')
    parser.add_argument('--plain', action='store_true', help='Use plain tables without scoring')
    parser.add_argument('--top_k', type=int, default=10, help='Top K cells to consider for scoring')
    parser.add_argument('--column_top_k', type=int, default=5, help='Top K columns to consider for scoring')
    parser.add_argument('--row_top_k', type=int, default=5, help='Top K rows to consider for scoring')
    parser.add_argument('--output_path', type=str, default='predictions.csv', help='Path to save predictions')
    parser.add_argument('--length', type=int, default=100, help='Number of examples to process')
    parser.add_argument('--offset', type=int, default=0, help='Index offset for starting example')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate predictions against ground truth')
    args = parser.parse_args()

    logger.info(f"Starting run with model: {args.model}, dataset: {args.dataset}")

    # 데이터셋 클래스 인스턴스 생성
    dataset_class = get_dataset_class(args.dataset)
    dataset = dataset_class()

    # 예측 수행 및 저장
    output_path = os.path.join(Config.RESULTS_PATH, args.output_path)
    run_and_save_predictions(
        dataset,
        args.model,
        args.mode,
        args.plain,
        args.top_k,
        args.column_top_k,
        args.row_top_k,
        output_path,
        args.length,
        args.offset
    )

    # 평가 수행 (옵션)
    if args.evaluate:
        metrics = evaluate_predictions(output_path, dataset)
        logger.info(f"Evaluation metrics: {metrics}")
        print(f"Evaluation metrics: {metrics}")

if __name__ == "__main__":
    main()
