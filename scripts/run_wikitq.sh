#!/bin/bash

# 셸 스크립트 시작
echo "Starting WikiTQ evaluation process..."

# 환경 변수 설정
MODEL="gpt"  # 사용할 모델: gpt 또는 gemini
DATASET="wikitq"  # 데이터셋 이름
PREDICTIONS_PATH="wikitq_predictions.csv"  # 예측값 저장 경로
EVALUATE=true  # 평가 수행 여부

# 디렉토리 이동 (필요 시 스크립트의 루트 경로로)
cd "$(dirname "$0")/.." || exit 1

# Run the Python script for prediction and optional evaluation
if $EVALUATE; then
  echo "Running model with evaluation..."
  python run.py --model $MODEL --dataset $DATASET --predictions $PREDICTIONS_PATH --length 10 --top_k 0 --evaluate 
else
  echo "Running model without evaluation..."
  python run.py --model $MODEL --dataset $DATASET --predictions $PREDICTIONS_PATH
fi

# 완료 메시지
echo "Process completed. Predictions saved to $PREDICTIONS_PATH"
