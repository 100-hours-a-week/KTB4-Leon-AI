import asyncio
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import httpx

from rag_app.app import app
from rag_app.config import PROJECT_DIR, Settings
from rag_app.documents import Document, split_document
from rag_app.generator import FallbackGenerator
from rag_app.pipeline import RAGPipeline


def offline_settings(documents_dir: Path = PROJECT_DIR.parent / "README.md") -> Settings:
    return Settings(documents_dir=documents_dir, provider="offline")


def test_split_document_has_overlap() -> None:
    document = Document(source="sample.txt", text="가" * 180)
    chunks = split_document(document, chunk_size=100, chunk_overlap=20)

    assert len(chunks) == 2
    assert chunks[0].text[-20:] == chunks[1].text[:20]


def test_pipeline_retrieves_expected_document() -> None:
    pipeline = RAGPipeline(offline_settings())

    result = pipeline.query("재인덱싱 API 경로는 무엇인가요?", top_k=2)

    assert result.sources[0].source == "README.md"
    assert result.answer
    assert result.provider == "offline"


def test_pipeline_rejects_question_outside_documents() -> None:
    pipeline = RAGPipeline(offline_settings())

    result = pipeline.query("오늘 날씨 어때?", top_k=2)

    assert "찾지 못했습니다" in result.answer


def test_pipeline_accepts_natural_language_question() -> None:
    pipeline = RAGPipeline(offline_settings())

    result = pipeline.query("서버는 어떻게 실행해?", top_k=4)

    assert result.sources
    assert result.sources[0].source == "README.md"


def test_generator_falls_back_and_stays_on_fallback() -> None:
    class FailingGenerator:
        provider = "anthropic"
        model = "primary"
        calls = 0

        def generate(self, prompt: str) -> str:
            self.calls += 1
            raise RuntimeError("quota exhausted")

        def stream(self, prompt: str) -> Iterator[str]:
            raise RuntimeError("quota exhausted")
            yield ""

    class WorkingGenerator:
        provider = "gemini"
        model = "fallback"

        def generate(self, prompt: str) -> str:
            return "fallback answer"

        def stream(self, prompt: str) -> Iterator[str]:
            yield "fallback answer"

    primary = FailingGenerator()
    chain = FallbackGenerator([primary, WorkingGenerator()])

    assert chain.generate("question") == "fallback answer"
    assert chain.provider == "gemini"
    assert chain.generate("second question") == "fallback answer"
    assert primary.calls == 1


def test_query_api() -> None:
    async def request() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/query",
                json={"question": "RAG는 어떤 단계로 구성되나요?", "top_k": 2},
            )
            stream = await client.post(
                "/api/query/stream",
                json={"question": "RAG는 어떤 단계로 구성되나요?", "top_k": 2},
            )
            return response, stream

    pipeline = RAGPipeline(offline_settings())
    with patch("rag_app.app.get_pipeline", return_value=pipeline):
        response, stream = asyncio.run(request())

    assert response.status_code == 200
    assert response.json()["sources"]
    assert stream.status_code == 200
    assert "event: sources" in stream.text
    assert "event: token" in stream.text
    assert "event: done" in stream.text
