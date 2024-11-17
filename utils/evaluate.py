# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer

def evaluate_predictions(predictions, dataset, metric='accuracy'):
    # 메트릭 계산
    def score(pred, ans):
        if metric == 'accuracy':
            pred = str(pred).lower().strip()
            ans = str(ans).lower().strip()
            return int(ans in pred)
        elif metric == 'sbleu':
            return sacrebleu.corpus_bleu([pred], [[ans]]).score / 100
        elif metric == 'rougel':
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = scorer.score(pred, ans)
            return scores['rougeL'].fmeasure
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    # 정답과 예측값 비교
    for i, x in predictions.iterrows():
        pred = x['prediction']
        ans = dataset.get_item(i)['answer']
        predictions.loc[i, 'score'] = score(pred, ans)
        predictions.loc[i, 'answer'] = ans

    metrics = {
        metric: predictions["score"].mean()
    }

    return metrics
