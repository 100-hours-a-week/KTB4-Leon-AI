from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .autocomplete import GeminiAutocomplete
from .config import REPO_ROOT, Settings
from .pipeline import LangChainRAG


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from RAG_chatbot.chatbot.model import (  # noqa: E402
    generate_text_from_index,
    generate_text_hybrid,
    load_checkpoint,
    predict_next_words,
)


AUTOCOMPLETE_MODEL_PATH = Path(
    os.getenv(
        "CHATBOT_MODEL_PATH",
        REPO_ROOT / "RAG_chatbot" / "artifacts" / "chatbot.pt",
    )
)
INDEX_PATH = Path(__file__).with_name("index.html")

app = FastAPI(
    title="한국어 자동완성 및 LangChain RAG",
    description="다음 단어 자동완성과 문서 기반 질의응답 API",
    version="1.1.0",
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
    trace_id: str
    provider: str
    model: str


class NextWordRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=200)
    top_k: int = Field(default=5, ge=1, le=20)


class NextWordCandidate(BaseModel):
    word: str
    rank: int


class NextWordResponse(BaseModel):
    prompt: str
    candidates: list[NextWordCandidate]
    provider: str
    model: str
    fallback: bool


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=200)
    max_new_tokens: int = Field(default=8, ge=1, le=30)
    top_k: int = Field(default=5, ge=1, le=20)


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    provider: str
    model: str
    fallback: bool


@lru_cache(maxsize=1)
def get_pipeline() -> LangChainRAG:
    return LangChainRAG()


@lru_cache(maxsize=1)
def get_autocomplete_bundle():
    if not AUTOCOMPLETE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"학습 모델이 없습니다: {AUTOCOMPLETE_MODEL_PATH}"
        )
    return load_checkpoint(AUTOCOMPLETE_MODEL_PATH, device="cpu")


@lru_cache(maxsize=1)
def get_gemini_autocomplete() -> GeminiAutocomplete:
    settings = Settings.from_env()
    if not settings.api_key:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")
    return GeminiAutocomplete(settings.api_key, settings.autocomplete_model)


def get_autocomplete_settings() -> Settings:
    return Settings.from_env()


def pipeline_or_503() -> LangChainRAG:
    try:
        return get_pipeline()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def autocomplete_or_503():
    try:
        return get_autocomplete_bundle()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    settings = get_autocomplete_settings()
    return {
        "status": "ready",
        "autocomplete_provider": settings.autocomplete_provider,
        "autocomplete_model": settings.autocomplete_model,
        "autocomplete_model_exists": AUTOCOMPLETE_MODEL_PATH.exists(),
        **pipeline_or_503().stats(),
    }


@app.get("/api/documents")
def documents() -> dict:
    return pipeline_or_503().stats()


@app.post("/api/documents/reindex")
def reindex() -> dict:
    try:
        return {"status": "reindexed", **pipeline_or_503().reindex()}
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        result = pipeline_or_503().query(request.question, request.top_k)
        return QueryResponse(
            question=result.question,
            answer=result.answer,
            sources=[SourceResponse(**source.__dict__) for source in result.sources],
            trace_id=result.trace_id,
            provider=result.provider,
            model=result.model,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/next-word", response_model=NextWordResponse)
def next_word(request: NextWordRequest) -> NextWordResponse:
    settings = get_autocomplete_settings()
    if settings.autocomplete_provider == "gemini":
        try:
            words = get_gemini_autocomplete().predict_next_words(
                request.prompt,
                request.top_k,
            )
            return NextWordResponse(
                prompt=request.prompt,
                candidates=[
                    NextWordCandidate(word=word, rank=index)
                    for index, word in enumerate(words, start=1)
                ],
                provider="gemini",
                model=settings.autocomplete_model,
                fallback=False,
            )
        except Exception as exc:
            if not settings.autocomplete_local_fallback:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

    model, tokenizer, config, metadata = autocomplete_or_503()
    candidates = predict_next_words(
        model,
        tokenizer,
        config,
        request.prompt,
        top_k=request.top_k,
        next_word_index=metadata.get("next_word_index"),
        device="cpu",
    )
    return NextWordResponse(
        prompt=request.prompt,
        candidates=[
            NextWordCandidate(word=word, rank=index)
            for index, (word, _) in enumerate(candidates, start=1)
        ],
        provider="local",
        model="character-transformer",
        fallback=settings.autocomplete_provider != "local",
    )


@app.post("/api/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    settings = get_autocomplete_settings()
    if settings.autocomplete_provider == "gemini":
        try:
            generated = get_gemini_autocomplete().continue_sentence(
                request.prompt,
                request.max_new_tokens,
            )
            return GenerateResponse(
                prompt=request.prompt,
                generated_text=generated,
                provider="gemini",
                model=settings.autocomplete_model,
                fallback=False,
            )
        except Exception as exc:
            if not settings.autocomplete_local_fallback:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

    model, tokenizer, config, metadata = autocomplete_or_503()
    next_word_index = metadata.get("next_word_index")
    if next_word_index:
        generated = generate_text_from_index(
            next_word_index,
            request.prompt,
            max_new_words=request.max_new_tokens,
            top_k=request.top_k,
        )
        if generated.strip() != request.prompt.strip():
            return GenerateResponse(
                prompt=request.prompt,
                generated_text=generated,
                provider="local",
                model="character-transformer",
                fallback=settings.autocomplete_provider != "local",
            )

    generated = generate_text_hybrid(
        model,
        tokenizer,
        config,
        request.prompt,
        max_new_words=request.max_new_tokens,
        top_k=request.top_k,
        next_word_index=None,
        device="cpu",
    )
    return GenerateResponse(
        prompt=request.prompt,
        generated_text=generated,
        provider="local",
        model="character-transformer",
        fallback=settings.autocomplete_provider != "local",
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_PATH.read_text(encoding="utf-8")
