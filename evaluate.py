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

    # 'seq_out' 열을 기준으로 비교
    y_true = predictions['answer']
    y_pred = predictions['prediction']

    # 메트릭 계산
    metrics = {
        "accuracy": sum(y_true == y_pred) / len(y_true),
    }

    return metrics
