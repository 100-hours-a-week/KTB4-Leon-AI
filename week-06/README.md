# Week 6 RAG 프로젝트

<details>
<summary><strong>회고</strong></summary>

- RAG를 직접 구현하면서 답변 품질은 생성 모델뿐 아니라 검색 결과와 청크 구성에 크게 좌우된다는 점을 확인했다.
- 관련성 기준이 높으면 자연스러운 질문을 놓치고, 너무 낮으면 무관한 문서가 검색됐다. 실제 질문을 기준으로 임계값을 조정하는 과정이 필요했다.
- Gemini 과부하와 포트 충돌을 겪으면서 코드 밖의 실행 환경도 서비스 안정성에 영향을 준다는 점을 배웠다. 이 과정에서 Anthropic 우선, Gemini 대체 구조와 명확한 오류 처리를 추가했다.
- 문서를 합치거나 구조를 변경한 뒤에는 이전 검색 결과를 그대로 신뢰할 수 없었다. 문서가 바뀔 때마다 검색 평가를 다시 실행해야 했다.
</details>

## 과제 진행 현황

- [x] Gemini API 또는 공개 가중치 모델을 사용해 문서 로딩부터 응답 생성까지 RAG 아키텍처 구축
- [x] FastAPI REST API 배포
  - [x] (선택) SSE 스트리밍 구현
- [x] 검색 hit rate, MRR, token F1, groundedness, latency 기반 평가 구현
- [ ] (선택) Graph RAG 조사 및 적용
- [X] 5주차 챗봇에 RAG 아키텍처 적용

Graph RAG는 구현에 실패한 것이 아니라 선택 과제로 남겨 두었다. 현재 검색 원본이 단일 README이고 청크도 소수라 엔티티·관계 추출, 그래프 저장소, 그래프 탐색을 추가해도 검색 품질 향상보다 복잡도와 비용이 커진다고 판단했다. 인물·조직·기술 사이의 관계를 여러 문서에 걸쳐 추론해야 하는 규모로 데이터가 늘어나면 적용할 가치가 있다.

<details>
<summary><strong>트러블슈팅</strong></summary>

위클리 챌린지 6주차에서 발생한 문제와 해결 과정은 다음과 같다.

### 1. 오프라인 테스트 생성기가 무관한 답을 반환함

- 증상: 문서와 관계없는 일상 질문에도 가장 점수가 높은 문장을 답변으로 반환했다.
- 원인: 초기 오프라인 생성기는 검색 점수가 낮아도 후보 중 하나를 반드시 선택했다.
- 해결: 최고 검색 점수가 관련성 기준보다 낮으면 생성을 실행하지 않고 문서 범위 밖 질문이라고 안내하도록 변경했다.

### 2. 자연어 질문이 관련성 기준에서 탈락함

- 증상: 기본 예시 질문은 답하지만 `서버는 어떻게 실행해?` 같은 표현은 답하지 못했다.
- 원인: 문서를 하나로 합친 뒤 검색 최고점이 기존 기준 `0.08` 바로 아래로 떨어졌다.
- 해결: 다양한 자연어 질문을 통과시키도록 `0.05`로 기준을 조정하고, 검색 문맥에 질문과 공통된 핵심어가 없으면 생성 단계에서도 답변을 거부하도록 이중으로 검증했다.

### 3. 필요한 실행 명령이 다른 청크에 있어 답을 놓침

- 증상: FastAPI 설명 청크는 찾았지만 실제 `uvicorn` 실행 명령이 있는 인접 청크는 제거됐다.
- 원인: 각 청크에 관련성 기준을 따로 적용해 보조 문맥이 사라졌다.
- 해결: 청크 크기를 1,200자, 겹침을 200자로 조정했다. 최고 결과가 기준을 통과하면 상위 4개 청크 전체를 생성 모델에 전달한다.

### 4. Gemini 일시 과부하로 500 오류 발생

- 증상: Gemini가 `503 UNAVAILABLE`을 반환하면 웹 화면에는 일반 서버 오류만 표시됐다.
- 원인: 외부 모델 과부하 예외 처리와 대체 제공자가 없었다.
- 해결: Anthropic을 우선 사용하고 실패하면 Gemini로 전환하도록 구성했다. 모델 오류는 API에서 처리 가능한 오류로 변환했다.

### 5. 문서 통합 후 기존 로더가 파일을 읽지 못함

- 증상: 여러 Markdown을 단일 README로 합치자 기존 `documents` 디렉터리 기반 로더를 그대로 사용할 수 없었다.
- 원인: 로더가 디렉터리 입력만 지원했다.
- 해결: 단일 파일과 디렉터리를 모두 읽도록 로더를 확장하고 평가 기대 출처를 `README.md`로 통일했다.

### 6. 기본 포트 충돌

- 증상: 8000 포트를 5주차 챗봇 서버가 사용 중이었다.
- 해결: 실행 중인 서버를 종료하지 않고 6주차 RAG 서버를 8001 포트에서 실행했다.

</details>

## 프로젝트 개요

문서 로딩, 검색, 생성 모델 답변, FastAPI 배포, 정량 평가를 포함한 RAG 프로젝트다. 이 파일 하나가 프로젝트 설명서이자 RAG 검색 대상 문서로 사용된다.

생성 모델은 Anthropic을 먼저 사용하고 호출에 실패하면 Gemini로 전환한다. 한 번 전환된 서버 프로세스는 이후 요청에서도 Gemini를 사용한다. 로컬 Ollama의 공개 가중치 모델도 선택할 수 있다.

## RAG 파이프라인 기초

RAG(Retrieval-Augmented Generation)는 외부 문서를 검색한 뒤 검색 결과를 생성 모델의 문맥으로 제공하는 구조다. 모델을 다시 학습하지 않고도 최신 문서나 조직 내부 지식을 답변에 반영할 수 있다.

일반적인 RAG 파이프라인은 다음 순서로 동작한다.

1. 문서 로딩
2. 텍스트 정제
3. 청킹
4. 인덱싱
5. 검색
6. 프롬프트 구성
7. LLM 응답 생성

이 프로젝트는 Markdown, TXT, JSON, CSV 문서를 읽고 겹침이 있는 문자 단위 청크로 나눈다. 검색 단계에서는 한국어 형태소 분석기 없이도 동작하도록 문자 n-gram TF-IDF 벡터와 코사인 유사도를 사용한다.

생성 단계에서는 검색 점수가 높은 청크와 출처 번호를 Anthropic, Gemini 또는 로컬 Ollama 모델에 전달한다. 생성 모델은 검색 문맥에 없는 내용을 추측하지 않고 답변 문장에 출처 번호를 표시하도록 지시받는다.

검색 최고 점수가 `RAG_MIN_RELEVANCE_SCORE` 기본값 `0.05`보다 낮으면 문서 밖 질문으로 판단해 답변 생성을 건너뛴다. 검색을 통과해도 문맥에 근거가 없으면 생성 모델이 답변을 거부한다.

## FastAPI 배포 가이드

서버는 상태 확인, 문서 목록, 문서 재인덱싱, 일반 질의, 스트리밍 질의 API를 제공한다.

- `GET /health`: 문서 수, 청크 수, 활성 생성 모델 확인
- `GET /api/documents`: 인덱스 문서와 통계 반환
- `POST /api/documents/reindex`: 검색 문서를 다시 읽고 인덱스 교체
- `POST /api/query`: 질문, 답변, 검색 출처와 유사도 점수를 JSON으로 반환
- `POST /api/query/stream`: 출처와 생성 토큰을 Server-Sent Events로 전송

외부 배포에서는 HTTPS, 인증, 요청 크기 제한, 호출량 제한, 로그 개인정보 제거가 추가로 필요하다. 문서가 바뀌면 재인덱싱 API를 호출하거나 서버를 재시작한다.

## 프로젝트 구조

```text
week-06/
├── README.md                       # 회고, 과제, 실행 문서 및 RAG 검색 원본
└── weekly-challenge/
    ├── evaluation/                 # 평가 질문과 결과
    ├── rag_app/
    │   ├── app.py                  # FastAPI REST/SSE API
    │   ├── config.py               # 제공자와 검색 설정
    │   ├── documents.py            # 문서 로더와 청커
    │   ├── generator.py            # Anthropic/Gemini/Ollama 생성기
    │   ├── pipeline.py             # RAG 오케스트레이션
    │   └── retriever.py            # 문자 n-gram TF-IDF 검색
    ├── tests/
    ├── evaluate.py
    └── requirements.txt
```

## 설치 및 실행

```bash
cd weeks/week-06/weekly-challenge
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

저장소 루트의 공통 `.env`를 설정한 뒤 서버를 실행한다.

```bash
python3 -m uvicorn rag_app.app:app --reload --port 8001
```

질의 예시:

```bash
curl -X POST http://127.0.0.1:8001/api/query \
  -H "Content-Type: application/json" \
  -d '{"question":"RAG 파이프라인은 어떤 단계로 구성되나요?","top_k":4}'
```

## 평가

RAG 품질은 검색과 생성을 분리해서 평가한다. 검색 평가는 hit rate와 reciprocal rank를, 생성 평가는 기준 답변과의 token F1 및 검색 문맥에 근거한 단어 비율을 사용한다. 실제 사용자 질문, 기준 답변, 기대 출처로 평가 데이터셋을 계속 보강해야 한다.

```bash
python3 evaluate.py
python3 evaluate.py --provider gemini
pytest -q
```

결과는 `weekly-challenge/evaluation/results.json`에 저장되고 평가 데이터는 `weekly-challenge/evaluation/questions.json`에서 관리한다.

## 설계 메모

- 별도 형태소 분석기나 벡터 DB 없이 실행되도록 문자 n-gram TF-IDF를 사용한다.
- 프로세스 시작 시 단일 `README.md`를 메모리에 인덱싱한다.
- 대용량 운영 환경에서는 FAISS, Qdrant, Elasticsearch 같은 영속 벡터 저장소로 교체해야 한다.
- 생성 프롬프트는 검색 문맥 밖의 추측을 금지하고 출처 표기를 요구한다.
- API 키는 환경 변수로만 주입하고 Git 저장소에 커밋하지 않는다.
