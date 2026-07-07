from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .graph import LangGraphRAGAgent


app = FastAPI(
    title="Week 8 LangGraph RAG Agent",
    description="StateGraph 기반 도구 선택형 RAG Agent API",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=10)


class SourceResponse(BaseModel):
    source: str
    chunk_id: str
    score: float
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    route: str
    tools_used: list[str]
    sources: list[SourceResponse]
    provider: str
    model: str
    steps: list[str]


@lru_cache(maxsize=1)
def get_agent() -> LangGraphRAGAgent:
    return LangGraphRAGAgent()


def agent_or_503() -> LangGraphRAGAgent:
    try:
        return get_agent()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ready", **agent_or_503().stats()}


@app.get("/api/graph")
def graph_definition() -> dict:
    return agent_or_503().graph_definition()


@app.post("/api/documents/reindex")
def reindex() -> dict:
    try:
        return {"status": "reindexed", **agent_or_503().reindex()}
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _query(request: QueryRequest) -> QueryResponse:
    try:
        result = agent_or_503().query(request.question, request.top_k)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        route=result.route,
        tools_used=result.tools_used,
        sources=[SourceResponse(**source.__dict__) for source in result.sources],
        provider=result.provider,
        model=result.model,
        steps=result.steps,
    )


@app.post("/api/agent/query", response_model=QueryResponse)
def agent_query(request: QueryRequest) -> QueryResponse:
    return _query(request)


@app.post("/api/query", response_model=QueryResponse)
def query_alias(request: QueryRequest) -> QueryResponse:
    return _query(request)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LangGraph RAG Agent</title>
  <style>
    :root { --bg:#f3f5f7; --panel:#fff; --ink:#18212b; --muted:#68737f; --line:#d8dee4; --accent:#2563eb; --soft:#eff6ff; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    main { width:min(760px,calc(100% - 32px)); margin:48px auto; }
    h1 { margin:0; font-size:28px; }
    .description { margin:8px 0 20px; color:var(--muted); line-height:1.55; }
    .panel { padding:28px; border:1px solid var(--line); border-radius:10px; background:var(--panel); }
    label { display:block; margin-bottom:8px; font-weight:650; }
    textarea { width:100%; min-height:120px; resize:vertical; padding:13px; border:1px solid var(--line); border-radius:7px; font:inherit; line-height:1.55; }
    textarea:focus { outline:3px solid var(--soft); border-color:var(--accent); }
    button { margin-top:12px; padding:10px 16px; border:0; border-radius:7px; background:var(--accent); color:#fff; font:inherit; font-weight:650; cursor:pointer; }
    button:disabled { opacity:.55; cursor:wait; }
    .answer { min-height:120px; margin-top:22px; padding:17px; border-left:4px solid var(--accent); background:var(--soft); white-space:pre-wrap; line-height:1.7; }
    @media(max-width:700px){ main{margin:24px auto}.panel{padding:20px} }
  </style>
</head>
<body>
  <main>
    <h1>LangGraph RAG Agent</h1>
    <p class="description">질문을 분석해 필요한 도구를 선택하고 문서 근거로 답합니다.</p>
    <section class="panel">
      <form id="form">
        <label for="question">질문</label>
        <textarea id="question" required>문서를 추가하려면 어떻게 하나요?</textarea>
        <button id="submit" type="submit">Agent에게 질문하기</button>
      </form>
      <div id="answer" class="answer" aria-live="polite">답변이 여기에 표시됩니다.</div>
    </section>
  </main>
  <script>
    const form=document.querySelector('#form'),button=document.querySelector('#submit'),answer=document.querySelector('#answer');
    form.addEventListener('submit',async event=>{
      event.preventDefault();button.disabled=true;answer.textContent='Agent 실행 중...';
      try{
        const response=await fetch('/api/agent/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:document.querySelector('#question').value,top_k:4})});
        const data=await response.json();if(!response.ok)throw new Error(data.detail||'요청 실패');answer.textContent=data.answer;
      }catch(error){answer.textContent=error.message}finally{button.disabled=false}
    });
  </script>
</body>
</html>
"""
