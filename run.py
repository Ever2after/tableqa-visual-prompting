import pandas as pd
import argparse
import os
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import random

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from dataset.wikitq import WikiTQDataset
from dataset.tabfact import TabFactDataset
from dataset.finqa import FinQADataset
from dataset.robut_wikitq import RobutWikiTQDataset
from dataset.fetaqa import FeTaQDataset
from model.gpt import GPT
from model.gemini import Gemini
from config import Config
from utils.logger import setup_logger
from utils.table_generator import generate_table_image
from utils.score import get_table_score
from utils.evaluate import evaluate_predictions
from utils.selector import LLMSelector


# 로거 설정
logger = setup_logger("Run", log_file='run.log')

def get_dataset_class(dataset_name):
    """
    데이터셋 이름에 따라 적절한 데이터셋 클래스를 반환.
    """
    dataset_classes = {
        'wikitq': WikiTQDataset,
        'tabfact': TabFactDataset,
        'finqa': FinQADataset,
        'robut_wikitq': RobutWikiTQDataset,
        'fetaqa': FeTaQDataset
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

def _predict_row(i, item, model, mode, cot, plain, top_k, column_top_k, row_top_k):
    if i%100 == 0:
        logger.info(f"Processing item {i}")
    
    table = item['table']
    query = item['question']
    cached_embeddings = item['embedding']
    
    # Initialize model instance
    model_instance = GPT() if model == 'gpt' else Gemini()
    
    # If mode is text, directly get the answer using the textual representation of the table
    if mode == 'text':
        return {
            'index': i,
            'prediction': model_instance.get_tableqa_answer(
                mode='text',
                table=table.to_markdown(),
                query=query,
                cot=cot
            )
        }
    
    # For image mode, optionally calculate table scores if not in plain mode
    score = None
    if not plain:
        score = get_table_score(table, query, top_k, column_top_k, row_top_k, cached_embeddings)
    
    try:
        # Generate table image with scores if available
        table_image = generate_table_image(table, score)
    except Exception as e:
        print(f"Error generating table image for item {i}: {e}")
        return {'index': i, 'prediction': False}
    
    # Get the prediction based on the generated table image
    prediction = model_instance.get_tableqa_answer(
        mode='image',
        table=table_image,
        query=query,
        cot=cot
    )
    
    return {'index': i, 'prediction': prediction}

def predict(dataset, model, mode, cot=False, plain=False, top_k=10, column_top_k=5, row_top_k=5, length=100, offset=0, max_workers=4):
    logger.info(f"Running predictions with model: {model}, mode: {mode}, cot: {cot}, plain: {plain}, top_k: {top_k}, column_top_k: {column_top_k}, row_top_k: {row_top_k}")
    predictions = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                _predict_row, 
                i,
                getattr(dataset, 'get_item')(i),
                model,
                mode,
                cot,
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

    logger.info(f"Predictions completed for {len(predictions)} items.")

    return pd.DataFrame(predictions)


def _select_row(selector, dataset, text_predictions, image_predictions, index):
    logger.info(f"Selecting item {index}")
    text_pred = text_predictions.loc[index, 'prediction']
    image_pred = image_predictions.loc[index, 'prediction']
    table = dataset.get_item(index)['table'].to_markdown()
    query = dataset.get_item(index)['question']
    selected_pred = selector.select(table, query, [text_pred, image_pred])
    return {'index': index, 'selected_index': selected_pred, 'prediction': [text_pred, image_pred][selected_pred]}

def select(dataset, text_predictions, image_predictions):
    selector = LLMSelector()
    selected_predictions = []
    
    indices = text_predictions.index  # Assuming indices match between text_predictions and image_predictions
    
    with ThreadPoolExecutor() as executor:
        # Map the function to the thread pool
        futures = [
            executor.submit(_select_row, selector, dataset, text_predictions, image_predictions, i)
            for i in indices
        ]
        
        # Collect results as they are completed
        for future in futures:
            selected_predictions.append(future.result())
    
    return pd.DataFrame(selected_predictions)

def main():
    # 명령줄 인자 설정
    parser = argparse.ArgumentParser(description="Table QA Model Execution and Evaluation")
    parser.add_argument('--predict', action='store_true', help='Run predictions')
    parser.add_argument('--dataset', type=str, default='wikitq', help='Dataset name: wikitq, tabfact, etc.')
    parser.add_argument('--split', type=str, default='test', help='Dataset split: train, dev, test, etc.')
    parser.add_argument('--model', type=str, default='gpt', help='Model to use: gpt or gemini')
    parser.add_argument('--mode', type=str, default='image', help='Execution mode: text or image')
    parser.add_argument('--cot', action='store_true', help='Use cot (context of thinking) for LLM')
    parser.add_argument('--plain', action='store_true', help='Use plain tables without scoring')
    parser.add_argument('--top_k', type=int, default=10, help='Top K cells to consider for scoring')
    parser.add_argument('--column_top_k', type=int, default=5, help='Top K columns to consider for scoring')
    parser.add_argument('--row_top_k', type=int, default=5, help='Top K rows to consider for scoring')
    parser.add_argument('--length', type=int, default=100, help='Number of examples to process')
    parser.add_argument('--offset', type=int, default=0, help='Index offset for starting example')
    parser.add_argument('--max_workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--output_path', type=str, help='Path to save results')

    parser.add_argument('--evaluate', action='store_true', help='Evaluate predictions against ground truth')
    parser.add_argument('--prediction_path', type=str, default='predictions.csv', help='Path to predictions for evaluation')
    parser.add_argument('--metric', type=str, default='accuracy', help='Evaluation metric: accuracy, sbleu, rougel')

    parser.add_argument('--select', action='store_true', help='Select a more correct answer')
    parser.add_argument('--text_pred_path', type=str, default='text_predictions.csv', help='Path to text predictions')
    parser.add_argument('--image_pred_path', type=str, default='image_predictions.csv', help='Path to image predictions')

    args = parser.parse_args()

    logger.info(f"Starting run with model: {args.model}, dataset: {args.dataset}")

    # 데이터셋 클래스 인스턴스 생성
    dataset_class = get_dataset_class(args.dataset)
    dataset = dataset_class(split=args.split)

    # 예측 수행 및 저장
    if args.predict:
        predictions = predict(
            dataset,
            args.model,
            args.mode,
            args.cot,
            args.plain,
            args.top_k,
            args.column_top_k,
            args.row_top_k,
            args.length,
            args.offset,
            args.max_workers
        )

        default_output_path = f'{args.dataset}_{args.split}_{args.model}_{args.mode}'
        if args.cot:
            default_output_path += '_cot'
        if args.mode == 'image':
            if args.plain:
                pass
            elif args.top_k > 0:
                default_output_path += f'_top{args.top_k}'
            elif args.column_top_k > 0 or args.row_top_k > 0:
                default_output_path += f'_c{args.column_top_k}_r{args.row_top_k}'
        default_output_path += f'_{args.offset}_{args.offset + args.length - 1}'
        default_output_path += '.csv'

        output_path = args.output_path or default_output_path
        output_path = os.path.join(Config.RESULTS_PATH, output_path)

        os.makedirs(Config.RESULTS_PATH, exist_ok=True)
        predictions.to_csv(output_path, index=False)

        logger.info(f"Predictions saved to {output_path}")

    # 평가 수행
    if args.evaluate:
        predictions = pd.read_csv(args.prediction_path, index_col=0)
        metrics = evaluate_predictions(predictions, dataset, args.metric)

        # save score
        predictions.to_csv(args.prediction_path)

        # logging
        logger.info(f"Evaluation metrics: {metrics}")
    
    # 선택 수행
    if args.select:
        text_predictions = pd.read_csv(args.text_pred_path, index_col=0)
        image_predictions = pd.read_csv(args.image_pred_path, index_col=0)

        selected_predictions = select(dataset, text_predictions, image_predictions)

        output_path = os.path.join(Config.RESULTS_PATH, args.output_path or 'selected_predictions.csv')
        selected_predictions.to_csv(output_path, index=False)

        logger.info(f"Selection completed. Selected predictions saved to {output_path}")

if __name__ == "__main__":
    main()
