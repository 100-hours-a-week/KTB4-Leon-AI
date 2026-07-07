# Week 8 LangGraph RAG Agent

<details>
<summary><strong>회고</strong></summary>

LangChain이 아직 준비되지 않은 상태에서도 LangGraph로 바로 RAG 에이전트를 구성했고, 7주차 LangChain RAG를 StateGraph 기반으로 마이그레이션하며 실행 흐름을 명시적으로 분리했다.
plan → search/list → generate → verify → rewrite의 상태 기반 그래프 구조로 도구 선택과 출처 검증 루프를 안정적으로 만들었다.

</details>

## 과제 진행 현황

- [x] 7주차 LangChain RAG를 LangGraph `StateGraph`로 마이그레이션
- [x] 조건부 도구 선택과 답변 검증 루프를 갖춘 AI Agent로 확장
- [x] FastAPI REST API와 웹 화면으로 배포

## 핵심 구조

```text
START
  ↓
plan ── 질문에 맞는 도구 선택
  ├─ search_knowledge
  ├─ list_capabilities
  └─ list_documents
          ↓
       generate
          ↓
        verify ── 출처 누락 ─→ rewrite ─┐
          │                              │
          └──────── 통과 ─→ END ←────────┘
```

7주차의 문서 로딩, 청킹, 검색, 생성 단계를 `StateGraph` 노드로 분리했다. Agent의 `plan` 노드는 질문을 분석해 필요한 도구를 선택한다. `verify` 노드는 문서 근거가 있는 답변에 출처가 포함됐는지 검사하고, 누락 시 `rewrite` 노드로 한 번 되돌린다.

## Agent 도구

- `search_knowledge`: `RAG_chatbot/knowledge` 문서를 검색한다.
- `list_capabilities`: Agent 기능과 REST API를 안내한다.
- `list_documents`: 현재 인덱싱된 문서와 청크 수를 확인한다.

## 모델 구성

Anthropic Claude Haiku를 먼저 사용하고 호출에 실패하면 Gemini Flash-Lite로 전환한다. 전환된 서버 프로세스는 이후 요청에서도 Gemini를 유지한다. 테스트에서는 API 비용이 없는 결정론적 생성기를 주입한다.

## 실행

```bash
cd weeks/week-08/weekly-challenge
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m uvicorn langgraph_agent.app:app --reload --port 8002
```

API 키와 LangGraph 설정은 저장소 루트의 공통 `.env`에서 읽는다. 최초 한 번만 루트의 `.env.example`을 `.env`로 복사하고 실제 키를 입력한다.

## REST API

- `POST /api/agent/query`: Agent 질의
- `POST /api/query`: 호환용 질의 별칭
- `GET /api/graph`: 그래프 노드와 edge 확인
- `POST /api/documents/reindex`: 지식 문서 재인덱싱
- `GET /health`: 서버와 인덱스 상태 확인

응답에는 최종 답변뿐 아니라 선택한 `route`, 실행한 `tools_used`, 검색 `sources`, 실제 `provider`와 `model`, 그래프 실행 `steps`가 포함된다. 웹 화면은 사용자에게 답변만 보여준다.

## 검증

```bash
pytest -q
```

테스트는 지식 검색 경로, 기능 안내 도구, 문서 목록 도구, FastAPI 응답, 검증·수정 루프 구조를 확인한다.
