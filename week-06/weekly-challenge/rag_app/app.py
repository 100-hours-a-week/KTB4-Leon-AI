from __future__ import annotations

import json
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .pipeline import RAGPipeline


app = FastAPI(
    title="Week 6 Gemini RAG",
    description="문서 검색 결과를 근거로 Gemini가 답변하는 RAG API",
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
    sources: list[SourceResponse]
    provider: str
    model: str


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def _pipeline_or_503() -> RAGPipeline:
    try:
        return get_pipeline()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    pipeline = _pipeline_or_503()
    return {"status": "ready", **pipeline.stats()}


@app.get("/api/documents")
def documents() -> dict:
    return _pipeline_or_503().stats()


@app.post("/api/documents/reindex")
def reindex() -> dict:
    try:
        return {"status": "reindexed", **_pipeline_or_503().reindex()}
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        result = _pipeline_or_503().query(request.question, request.top_k)
        return QueryResponse(
            question=result.question,
            answer=result.answer,
            sources=[SourceResponse(**source.__dict__) for source in result.sources],
            provider=result.provider,
            model=result.model,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/query/stream")
def query_stream(request: QueryRequest) -> StreamingResponse:
    pipeline = _pipeline_or_503()
    try:
        sources, tokens = pipeline.stream_query(request.question, request.top_k)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    def events():
        source_data = [source.__dict__ for source in sources]
        yield f"event: sources\ndata: {json.dumps(source_data, ensure_ascii=False)}\n\n"
        try:
            for token in tokens:
                data = json.dumps({"token": token}, ensure_ascii=False)
                yield f"event: token\ndata: {data}\n\n"
            yield "event: done\ndata: {}\n\n"
        except RuntimeError as exc:
            data = json.dumps({"detail": str(exc)}, ensure_ascii=False)
            yield f"event: error\ndata: {data}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Week 6 RAG</title>
  <style>
    :root { color-scheme: light; --bg:#f4f6f8; --panel:#fff; --ink:#17212b; --muted:#66717d; --line:#d7dde3; --accent:#087f5b; --soft:#e6fcf5; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    main { width:min(760px,calc(100% - 32px)); margin:48px auto; }
    header { margin-bottom:8px; }
    h1 { margin:0; font-size:28px; letter-spacing:0; }
    .scope { margin:8px 0 20px; color:var(--muted); line-height:1.55; }
    .workspace { border:1px solid var(--line); border-radius:10px; background:var(--panel); overflow:hidden; }
    .query { padding:28px; }
    label { display:block; font-weight:650; margin-bottom:8px; }
    textarea { width:100%; min-height:120px; resize:vertical; padding:13px; border:1px solid var(--line); border-radius:6px; font:inherit; line-height:1.55; }
    textarea:focus { outline:3px solid var(--soft); border-color:var(--accent); }
    button { margin-top:12px; padding:10px 15px; border:0; border-radius:6px; color:#fff; background:var(--accent); font:inherit; font-weight:650; cursor:pointer; }
    button:disabled { opacity:.55; cursor:wait; }
    .answer { min-height:120px; margin-top:22px; padding:16px; background:var(--soft); border-left:4px solid var(--accent); white-space:pre-wrap; line-height:1.7; }
    @media(max-width:700px){ main{margin:24px auto}.query{padding:20px} }
  </style>
</head>
<body>
  <main>
    <header><h1>문서 기반 RAG 질의응답</h1></header>
    <p class="scope">일반 대화 챗봇이 아닙니다. 등록된 RAG·FastAPI 문서를 검색하고, 관련 근거가 있을 때만 답합니다.</p>
    <div class="workspace">
      <section class="query">
        <form id="form">
          <label for="question">질문</label>
          <textarea id="question" required placeholder="예: RAG 파이프라인은 어떤 단계로 구성되나요?">RAG 파이프라인은 어떤 단계로 구성되나요?</textarea>
          <button id="submit" type="submit">문서에서 답변 찾기</button>
        </form>
        <div id="answer" class="answer" aria-live="polite">답변이 여기에 표시됩니다.</div>
      </section>
    </div>
  </main>
  <script>
    const form=document.querySelector('#form'), button=document.querySelector('#submit');
    const answer=document.querySelector('#answer');
    form.addEventListener('submit',async event=>{
      event.preventDefault(); button.disabled=true; answer.textContent='답변 생성 중...';
      try {
        const response=await fetch('/api/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:document.querySelector('#question').value,top_k:4})});
        const data=await response.json(); if(!response.ok) throw new Error(data.detail||'요청 실패');
        answer.textContent=data.answer;
      } catch(error) { answer.textContent=error.message; } finally { button.disabled=false; }
    });
  </script>
</body>
</html>
"""
