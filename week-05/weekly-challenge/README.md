# Week 05 Weekly Challenge

2026년 6월 8일 과제를 실행 가능한 코드와 노트북으로 정리했다.

## 구현 내용

| 과제 | 구현 파일 |
|---|---|
| ResNet으로 새 이미지 데이터셋 분류 | `image_model_comparison.py` |
| VGG16 사전 학습 모델 전이 학습 | `image_model_comparison.py` |
| 동일 데이터셋에서 ResNet18/VGG16 비교 | `image_model_comparison.py` |
| GridSearch/RandomSearch 하이퍼파라미터 튜닝 | `hyperparameter_search.py` |
| 다음 단어 예측 및 자기회귀 문장 생성 | `chatbot/model.py`, `chatbot/train.py` |
| FastAPI 웹 서비스 | `chatbot/app.py` |
| 전체 실습 흐름 | `weekly_challenge_2026-06-08.ipynb` |

기존 `05.py`는 초기 연습 코드로 그대로 두었다.

## 환경 설치

Python 3.11~3.13 사용을 권장한다.

```bash
cd weeks/week-05/weekly-challenge
python3 -m pip install -r requirements.txt
```

## 1. 이미지 모델 비교

기본 데이터셋은 실제 이미지 데이터인 CIFAR-10이다. 두 모델은 같은 학습/검증/테스트 인덱스를 사용한다.

```bash
python3 image_model_comparison.py \
  --dataset cifar10 \
  --models both \
  --epochs 3
```

개인 이미지 데이터셋은 클래스별 폴더 구조로 준비한다.

```text
my_images/
├── train/
│   ├── class_a/
│   └── class_b/
└── test/
    ├── class_a/
    └── class_b/
```

```bash
python3 image_model_comparison.py \
  --dataset imagefolder \
  --image-dir ./my_images \
  --models both
```

빠른 코드 검증은 사전 학습 가중치와 다운로드를 끈 가상 이미지로 실행할 수 있다.

```bash
python3 image_model_comparison.py \
  --dataset fake \
  --models resnet18 \
  --no-pretrained \
  --epochs 1 \
  --train-limit 16 \
  --validation-limit 8 \
  --test-limit 8 \
  --batch-size 4 \
  --device cpu
```

결과는 `artifacts/image_models/`에 저장된다.

- `comparison_results.json`: 모델별 정확도, 손실, 시간, 파라미터 수
- `model_comparison.png`: 테스트 정확도와 실행 시간 비교
- `{model}_history.png`: 학습 과정
- `{model}_confusion_matrix.png`: 실제 클래스와 예측 클래스 비교
- `{model}_best.pt`: 가장 높은 검증 정확도의 모델 가중치

## 2. 하이퍼파라미터 탐색

```bash
python3 hyperparameter_search.py
```

동일한 가상 분류 데이터와 RandomForest 모델에 대해 GridSearchCV와 RandomizedSearchCV를 비교한다. 결과는 `artifacts/hyperparameter_search/`에 저장된다.

## 3. 다음 단어 챗봇

먼저 한국어 말뭉치로 LSTM 다음 단어 모델을 학습한다.

```bash
python3 -m chatbot.train
```

학습이 끝나면 FastAPI 서버를 실행한다.

```bash
python3 -m uvicorn chatbot.app:app --reload --port 8000
```

- 웹 화면: `http://127.0.0.1:8000`
- API 문서: `http://127.0.0.1:8000/docs`
- 상태 확인: `http://127.0.0.1:8000/health`
- 다음 단어 후보: `POST /api/next-word`
- 문장 생성: `POST /api/generate`

API 요청 예시:

```bash
curl -X POST http://127.0.0.1:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"오늘 저녁에는","max_new_tokens":20,"temperature":0.8,"top_k":8}'
```

이 챗봇은 질문의 의미를 이해해 답하는 대규모 언어 모델이 아니다. 작은 말뭉치에서 다음 단어 확률을 학습하고, 예측 단어를 다시 입력에 붙이는 자기회귀 방식을 확인하는 과제용 모델이다.

입력 단어가 말뭉치에 거의 없으면 다음 단어 예측이 어색해질 수 있다. 예를 들어 `오늘 오전`을 자연스럽게 처리하려면 `chatbot/corpus.txt`에 오전 문맥 예시가 있어야 한다. 현재 코드는 사용자가 입력한 문장은 그대로 보존하고, 모델이 새로 생성한 단어만 뒤에 붙이도록 처리한다.
