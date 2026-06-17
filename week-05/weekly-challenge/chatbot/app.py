from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from chatbot.model import generate_text, load_checkpoint, predict_next_words


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(
    os.getenv("CHATBOT_MODEL_PATH", BASE_DIR / "artifacts" / "chatbot.pt")
)

app = FastAPI(
    title="다음 단어 생성 챗봇",
    description="LSTM 다음 단어 모델을 재귀적으로 호출해 문장을 생성합니다.",
    version="1.0.0",
)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=200)
    max_new_tokens: int = Field(default=24, ge=1, le=60)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_k: int = Field(default=8, ge=1, le=30)


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str


class NextWordRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=200)
    top_k: int = Field(default=5, ge=1, le=20)


class NextWordCandidate(BaseModel):
    word: str
    probability: float


class NextWordResponse(BaseModel):
    prompt: str
    candidates: list[NextWordCandidate]


@lru_cache(maxsize=1)
def get_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"학습 모델이 없습니다: {MODEL_PATH}. 먼저 `python -m chatbot.train`을 실행하세요."
        )
    return load_checkpoint(MODEL_PATH, device="cpu")


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ready" if MODEL_PATH.exists() else "model_missing",
        "model_exists": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
    }


@app.post("/api/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    try:
        model, vocabulary, config, _ = get_model_bundle()
        torch.manual_seed(42)
        generated = generate_text(
            model,
            vocabulary,
            config,
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            device="cpu",
        )
        return GenerateResponse(
            prompt=request.prompt,
            generated_text=generated,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/next-word", response_model=NextWordResponse)
def next_word(request: NextWordRequest) -> NextWordResponse:
    try:
        model, vocabulary, config, _ = get_model_bundle()
        candidates = predict_next_words(
            model,
            vocabulary,
            config,
            request.prompt,
            top_k=request.top_k,
            device="cpu",
        )
        return NextWordResponse(
            prompt=request.prompt,
            candidates=[
                NextWordCandidate(word=word, probability=probability)
                for word, probability in candidates
            ],
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>다음 단어 생성 챗봇</title>
  <style>
    :root {
      color-scheme: light;
      --surface: #f7f8fb;
      --panel: #ffffff;
      --ink: #1f2430;
      --muted: #6f768a;
      --line: #dfe3ec;
      --accent: #2e4780;
      --accent-soft: #eaf1fe;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--surface);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(760px, calc(100% - 32px));
      margin: 56px auto;
    }
    h1 { margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }
    .subtitle { margin: 0 0 24px; color: var(--muted); }
    .panel {
      padding: 24px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    label { display: block; margin-bottom: 8px; font-weight: 650; }
    textarea {
      width: 100%;
      min-height: 112px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 14px;
      color: var(--ink);
      font: inherit;
      outline: none;
    }
    textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
    .controls {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin: 16px 0;
    }
    input {
      width: 100%;
      margin-top: 6px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      font: inherit;
    }
    button {
      border: 0;
      border-radius: 6px;
      padding: 11px 16px;
      background: var(--accent);
      color: white;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }
    button:disabled { opacity: .55; cursor: wait; }
    .result {
      min-height: 88px;
      margin-top: 20px;
      padding: 16px;
      border-left: 4px solid var(--accent);
      background: var(--accent-soft);
      line-height: 1.7;
      white-space: pre-wrap;
    }
    @media (max-width: 620px) {
      main { margin: 28px auto; }
      .controls { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <h1>다음 단어 생성 챗봇</h1>
    <p class="subtitle">입력 문맥 뒤에 올 단어를 반복 예측해 문장을 완성합니다.</p>
    <section class="panel">
      <form id="form">
        <label for="prompt">시작 문장</label>
        <textarea id="prompt" required>오늘 저녁에는</textarea>
        <div class="controls">
          <label>생성 단어 수<input id="maxTokens" type="number" min="1" max="60" value="24"></label>
          <label>Temperature<input id="temperature" type="number" min="0.1" max="2" step="0.1" value="0.8"></label>
          <label>Top-k<input id="topK" type="number" min="1" max="30" value="8"></label>
        </div>
        <button id="submit" type="submit">문장 생성</button>
      </form>
      <div id="result" class="result">생성 결과가 여기에 표시됩니다.</div>
    </section>
  </main>
  <script>
    const form = document.querySelector("#form");
    const button = document.querySelector("#submit");
    const result = document.querySelector("#result");
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      button.disabled = true;
      result.textContent = "생성 중...";
      try {
        const response = await fetch("/api/generate", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            prompt: document.querySelector("#prompt").value,
            max_new_tokens: Number(document.querySelector("#maxTokens").value),
            temperature: Number(document.querySelector("#temperature").value),
            top_k: Number(document.querySelector("#topK").value)
          })
        });
        const data = await response.json();
        result.textContent = response.ok ? data.generated_text : data.detail;
      } catch (error) {
        result.textContent = "요청에 실패했습니다.";
      } finally {
        button.disabled = false;
      }
    });
  </script>
</body>
</html>
"""
