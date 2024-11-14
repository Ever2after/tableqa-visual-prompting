import pandas as pd
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from data.wikitq import WikiTQDataset
from data.tabfact import TabFactDataset
from evaluate import evaluate_predictions
from model.gpt import GPT
from model.gemini import Gemini
from config import Config
from utils.logger import setup_logger
from utils.table_generator import generate_table_image
from utils.score import get_table_score

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

def get_answer(table, query, model='gpt', plain=False, top_k=10, column_top_k=5, row_top_k=5, cached_embeddings=None):
    if not plain:
        score = get_table_score(table, query, top_k=top_k, column_top_k=column_top_k, row_top_k=row_top_k, cached_embeddings=cached_embeddings)
    else:
        score = None
    
    try:
        table_image = generate_table_image(table, score)
    except Exception as e:
        print(e)
        return False

    if model == 'gpt':
        model_ = GPT()
    elif model == 'gemini':
        model_ = Gemini()

    answer = model_.get_tableqa_answer(mode='image', table=table_image, question=query)
    
    return answer

def run_and_save_predictions(dataset, model_name, output_file, length=100, offset=0, plain=False, top_k=10, column_top_k=5, row_top_k=5):
    """
    모델을 실행하여 예측값을 생성하고 결과를 저장.

    Parameters:
        - dataset: 데이터셋 클래스 인스턴스
        - model_name: 사용할 모델 이름 ('gpt' 또는 'gemini')
        - output_file: 예측값을 저장할 파일 경로
        - plain: 점수 없이 원본 테이블만 사용하는 경우 True
        - top_k, column_top_k, row_top_k: 스코어링 관련 설정
    """
    logger.info(f"Running model: {model_name}")
    predictions = []

    def process_item(i):
        item = dataset.get_item(i)
        logger.info(f"Processing item {i} with question: {item['question']}")
        table = item['table']
        question = item['question']
        answer = item['answer']
        cached_embeddings = item['embedding']

        prediction = get_answer(
            table=table,
            query=question,
            model=model_name,
            plain=plain,
            top_k=top_k,
            column_top_k=column_top_k,
            row_top_k=row_top_k,
            cached_embeddings=cached_embeddings
        )
        return {'index': i, 'prediction': prediction, 'answer': answer}

    with ThreadPoolExecutor() as executor:
        future_to_index = {executor.submit(process_item, i): i for i in range(offset, offset + min(len(dataset), length))}
        for future in as_completed(future_to_index):
            try:
                result = future.result()
                predictions.append(result)
            except Exception as e:
                logger.error(f"Error processing item {future_to_index[future]}: {e}")

    # 결과 저장
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pd.DataFrame(predictions).to_csv(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")

def main():
    # 명령줄 인자 설정
    parser = argparse.ArgumentParser(description="Table QA Model Execution and Evaluation")
    parser.add_argument('--model', type=str, default='gpt', help='Model to use: gpt or gemini')
    parser.add_argument('--dataset', type=str, default='wikitq', help='Dataset name: wikitq, tabfact, etc.')
    parser.add_argument('--predictions', type=str, default='predictions.csv', help='Path to save predictions')
    parser.add_argument('--length', type=int, default=100, help='Number of examples to process')
    parser.add_argument('--offset', type=int, default=0, help='Index offset for starting example')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate predictions against ground truth')
    parser.add_argument('--plain', action='store_true', help='Use plain tables without scoring')
    parser.add_argument('--top_k', type=int, default=10, help='Top K cells to consider for scoring')
    parser.add_argument('--column_top_k', type=int, default=5, help='Top K columns to consider for scoring')
    parser.add_argument('--row_top_k', type=int, default=5, help='Top K rows to consider for scoring')
    args = parser.parse_args()

    logger.info(f"Starting run with model: {args.model}, dataset: {args.dataset}")

    # 데이터셋 클래스 인스턴스 생성
    dataset_class = get_dataset_class(args.dataset)
    dataset = dataset_class()

    # 예측 수행 및 저장
    output_path = os.path.join(Config.RESULTS_PATH, args.predictions)
    run_and_save_predictions(
        dataset=dataset,
        model_name=args.model,
        output_file=output_path,
        length=args.length,
        offset=args.offset,
        plain=args.plain,
        top_k=args.top_k,
        column_top_k=args.column_top_k,
        row_top_k=args.row_top_k
    )

    # 평가 수행 (옵션)
    if args.evaluate:
        metrics = evaluate_predictions(output_path, dataset)
        logger.info(f"Evaluation metrics: {metrics}")
        print(f"Evaluation metrics: {metrics}")

if __name__ == "__main__":
    main()
