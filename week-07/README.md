# Week 7 LangChain RAG 프로젝트

<details>
<summary><strong>회고</strong></summary>

- 기존 RAG를 LangChain LCEL로 옮기면서 문서 로딩, 검색, 프롬프트, 생성 단계를 조합 가능한 구성 요소로 관리할 수 있었다. 코드 구조는 명확해졌지만 추상화 단계가 늘어난 만큼 체인 내부 데이터 흐름을 이해하는 것이 중요했다.
- LangSmith Tracing을 연결하니 최종 답변만 보는 것보다 검색 문서와 각 실행 단계를 함께 확인하는 편이 오류 원인을 찾는 데 효과적이었다. Dataset 기반 평가를 통해 변경 전후 결과를 반복해서 비교할 수 있다는 점도 유용했다.
- 자기 모델, Gemma, Gemini를 같은 질문으로 비교하면서 모델의 크기보다 학습 목표와 실제 작업의 일치가 더 중요하다는 점을 확인했다. 자동완성 모델은 문장 이어쓰기에는 적합했지만 사실 기반 RAG 답변에는 맞지 않았다.
- LangChain으로 옮겼다고 답변 품질이 자동으로 좋아지지는 않았다. 같은 평가 Dataset으로 검색 결과와 답변을 비교하면서 프레임워크 변경과 품질 개선은 따로 검증해야 한다는 점을 배웠다.

</details>

## 과제 진행 현황

- [x] 개인 프로젝트 RAG 파이프라인을 LangChain 기반으로 마이그레이션
- [x] LangChain RAG를 FastAPI REST API로 배포
- [x] LangSmith Tracing과 Dataset 기반 평가 구현

## 프로젝트 개요

6주차 개인 프로젝트의 RAG를 LangChain LCEL 기반 2-step RAG로 마이그레이션하고 FastAPI와 LangSmith 평가를 연결했다. 다음 단어 자동완성과 문서 기반 질의응답을 하나의 웹 애플리케이션에서 제공한다.

RAG 파이프라인은 다음 순서로 동작한다.

```text
Document
→ RecursiveCharacterTextSplitter
→ GoogleGenerativeAIEmbeddings
→ InMemoryVectorStore
→ RunnableParallel / RunnablePassthrough
→ ChatPromptTemplate
→ Claude Haiku
```

생성 모델은 Anthropic을 우선 사용한다. Anthropic 호출이 실패하면 Gemini 2.5 Flash-Lite로 전환하고, 해당 서버 프로세스에서는 이후 요청도 Gemini로 처리한다.

## 프로젝트 구조

```text
week-07/
├── README.md                          # 회고, 과제, 실행, 평가 결과 통합 문서
└── weekly-challenge/
    ├── langchain_rag/                 # LangChain RAG와 FastAPI
    ├── evaluation/                    # 평가 Dataset과 결과
    ├── tests/                         # 비용 없는 단위/API 테스트
    ├── evaluate_langsmith.py          # Dataset 동기화 및 실험 실행
    ├── upload_model_comparison_langsmith.py
    └── requirements.txt
```

## 설치 및 실행

```bash
cd weeks/week-07/weekly-challenge
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m uvicorn langchain_rag.app:app --reload --port 8001
```

API 키와 모델은 저장소 루트의 공통 `.env`에서 관리한다. 7주차 전용 설정은 같은 파일의 `WEEK7_*` 변수를 사용한다.

```dotenv
WEEK7_DOCUMENTS_DIR=RAG_chatbot/knowledge
WEEK7_CHUNK_SIZE=800
WEEK7_CHUNK_OVERLAP=120
WEEK7_TOP_K=4
WEEK7_AUTOCOMPLETE_PROVIDER=gemini
```

- 웹 화면: `http://127.0.0.1:8001`
- API 문서: `http://127.0.0.1:8001/docs`
- 질의: `POST /api/query`
- 재인덱싱: `POST /api/documents/reindex`

API 응답의 `provider`와 `model` 필드에서 실제 사용된 생성 모델을 확인할 수 있다. `/health`의 `active_provider`에서도 현재 활성 제공자를 확인할 수 있다.

## 다음 단어 자동완성

자동완성은 `gemini-2.5-flash-lite`를 우선 사용한다. Gemini 호출 실패 또는 할당량 소진 시 루트의 `chatbot/artifacts/chatbot.pt`와 다음 단어 인덱스로 전환된다. 응답의 `provider`, `model`, `fallback` 필드에서 실제 사용 모델을 확인할 수 있다.

자기 모델은 한국어 다음 단어 자동완성에 맞춰 학습한 문자 단위 Transformer다. 입력 문맥이 다음 단어 인덱스에 있으면 관측 빈도를 사용하고, 없으면 Transformer가 문자를 반복 생성한다.

### 자동완성 예시

| 입력 | 생성 결과 |
|---|---|
| 오늘 오전 | 오늘 오전 계획은 FastAPI 화면 확인입니다 |
| 엄마 | 엄마 생각이 나서 따뜻한 문자를 보냈습니다 |
| 주말에는 | 주말에는 가족과 맛있는 음식을 먹고 싶어요 |
| 내일 오후에는 | 내일 오후에는 발표 자료를 정리할 계획입니다 |
| 모델 학습 결과를 | 모델 학습 결과를 확인하고 수정합니다 |

### 학습 데이터와 모델 구조

학습 데이터는 `autocomplete_corpus.txt`의 일반 한국어 자동완성 문장이다. Q/A 쌍과 상담 답변 데이터는 포함하지 않았다.

| 항목 | 값 |
|---|---:|
| 학습 문장 | 2,698개 |
| 학습 토큰 | 74,037개 |
| 문자 어휘 | 354개 |
| Epoch | 1 |
| Step | 800 |
| Best validation loss | 3.107 |
| Block size | 128 |
| Embedding dimension | 128 |
| Attention head | 4개 |
| Transformer layer | 3개 |

모델은 현재 문자까지를 입력으로 받고 다음 문자를 정답으로 두어 cross entropy loss를 최소화했다. 문자 Embedding, causal mask를 적용한 3층 `TransformerEncoder`, 선형 출력층으로 구성된다.

## LangSmith 평가

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="your-langsmith-key"
export LANGSMITH_ENDPOINT="https://apac.api.smith.langchain.com"
export LANGSMITH_PROJECT="kakao-bootcamp-week7-rag"
python3 evaluate_langsmith.py
python3 upload_model_comparison_langsmith.py
```

`evaluate_langsmith.py`는 12개 질문을 LangSmith Dataset에 동기화하고 현재 RAG 파이프라인의 `answer_token_f1`과 `source_hit`을 계산한다. `upload_model_comparison_langsmith.py`는 자기 모델, Gemma 4 E2B, Gemini 2.5 Flash의 답변을 같은 Dataset의 별도 Experiment로 올린다.

비교 Experiment에는 F1, groundedness, 출처 표기율, 검색 출처 적중률, 실제 호출 지연이 기록된다.

- 현재 RAG 실험 결과: `weekly-challenge/evaluation/results_langsmith.json`
- 모델 비교 결과: `weekly-challenge/evaluation/results_langsmith_model_comparison.json`

```bash
pytest -q
```

## 모델 비교 평가 결과

동일한 12개 RAG 질문에서는 로컬 Gemma 4 E2B가 답변 F1 `0.4585`, groundedness `0.7909`로 Gemini 2.5 Flash보다 높았다. 두 모델 모두 12개 질문을 오류 없이 처리하고 모든 답변에 출처를 표시했다.

| 모델 | 성공 | 답변 Token F1 | Groundedness | 출처 표기율 | 지연 중앙값 |
|---|---:|---:|---:|---:|---:|
| Gemma 4 E2B | 12/12 | 0.4585 | 0.7909 | 100% | 1.11초 |
| Gemini 2.5 Flash | 12/12 | 0.3843 | 0.6917 | 100% | 1.95초 |
| 자기 모델 | 12/12 | 0.0000 | 0.1319 | 0% | 0.05초 |

이 데이터 범위에서는 Gemma 4 E2B를 로컬 RAG 생성 모델로 사용할 근거가 있다. 다만 질문이 프로젝트 내부 문서 3개에서 만들어졌으므로 범용 성능 우위로 해석하면 안 된다.

Gemma의 첫 호출은 모델 로딩 때문에 약 10.6초가 걸렸지만, 이후 대부분 0.7~2.5초에 응답했다. 따라서 평균 지연 1.94초보다 중앙값 1.11초가 warm 상태의 사용 경험을 더 잘 나타낸다.

답변 Token F1은 기준 답변과 생성 답변에 함께 등장하는 한국어·영문·숫자 토큰의 정밀도와 재현율을 조화 평균한 값이다. Groundedness는 생성 답변 토큰 중 검색 문맥에서도 확인되는 토큰의 비율이다. 검색 출처 적중률은 세 모델 모두 100%였지만 동일한 검색 결과를 사용했으므로 생성 모델 자체의 성능 지표로 보지 않는다.

자기 모델은 자동완성 문장에서는 문맥을 유지했지만 문서 질의응답 F1은 `0.0000`이었다. 이는 모델 실패라기보다 학습 목표와 평가 작업이 다르다는 결과다. 질문과 정답 쌍으로 학습하지 않았기 때문에 사실 기반 RAG 답변을 기대하면 안 된다.

## 평가 방법

평가 데이터는 기능, API, RAG 검색, 자기 모델 구조, 학습 데이터, 한계에 관한 질문과 기준 답변 12개다. 각 질문에 대해 Gemini Embedding과 `InMemoryVectorStore`로 같은 상위 2개 문서 청크를 검색했다.

- 자기 모델: `chatbot/artifacts/chatbot.pt`
- Gemma: Ollama `gemma4:e2b`, temperature 0.1
- Gemini: API `gemini-2.5-flash`, temperature 0.1
- 평가 지표: 답변 Token F1, groundedness, 출처 표기율, 검색 출처 적중률, 응답 지연

## 한계

1. 질문 12개가 프로젝트 내부 문서 3개에서 만들어져 표본이 작고 도메인이 좁다.
2. Token F1은 같은 의미의 다른 표현을 낮게 평가할 수 있다.
3. Groundedness는 토큰 중복 기반이라 문장의 사실관계를 완전히 판정하지 않는다.
4. 모델별 생성은 질문당 한 번만 실행해 온도와 시점에 따른 변동성을 측정하지 않았다.
5. 자기 모델은 자동완성 모델이므로 RAG LLM과의 비교는 용도 적합성 진단으로만 해석한다.
6. Gemma와 Gemini의 차이는 이 평가 데이터에서만 관찰된 결과이며 일반적인 우열이나 통계적 유의성을 뜻하지 않는다.

## 앞으로의 방향

1. 자기 모델은 자동완성 기능으로 유지하고 문장 말뭉치를 확대한다.
2. 문자 tokenizer를 subword tokenizer로 바꾸고 충분히 학습하되 과적합을 함께 확인한다.
3. 외부 문서와 답변 불가 질문을 포함해 평가 데이터를 최소 30~50개로 확장한다.
4. 같은 질문을 모델별로 3회 이상 실행해 평균, 중앙값, 변동성을 비교한다.
5. Token F1 외에 사람 평가 또는 LLM judge로 정확성, 완결성, 문맥 충실도를 평가한다.
6. Gemma는 cold start와 메모리 사용량을, Gemini는 호출 비용과 할당량을 기록한다.

## 추가로 확인할 질문

- 자기 모델의 목표를 다음 단어 추천으로 유지할지 대화형 질의응답으로 다시 정의할지 결정해야 한다.
- 실제 사용자 문장에서도 다음 단어 인덱스의 효과가 유지되는지 확인해야 한다.
- Gemma 로컬 실행 비용과 Gemini API 비용을 포함했을 때 어떤 모델이 운영에 적합한지 비교해야 한다.

인증 키는 코드와 문서에 작성하지 않고 환경 변수로만 전달한다.
