# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def evaluate_predictions(predictions_file, dataset_name):
    """
    예측값과 실제 정답을 비교하여 메트릭 평가.

    Parameters:
        - predictions_file (str): 모델의 예측값을 저장한 파일 경로

    Returns:
        - metrics (dict): 평가 메트릭 (정확도, Precision, Recall, F1-score)
    """
    # 예측값 및 실제 정답 로드
    predictions = pd.read_csv(predictions_file)

    # 메트릭 계산
    predictions['score'] = predictions.apply(lambda x: 1 if x['answer'] in x['prediction'] else 0, axis=1)
    
    # output file 저장 (csv)
    predictions.to_csv(predictions_file, index=False)
    
    metrics = {
        "accuracy": predictions["score"].mean()
    }

    return metrics
