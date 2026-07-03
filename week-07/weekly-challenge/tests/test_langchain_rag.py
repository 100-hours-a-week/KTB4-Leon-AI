import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langchain_rag.autocomplete import (
    CandidatePayload,
    CompletionPayload,
    GeminiAutocomplete,
)
from langchain_rag.app import app
from langchain_rag.config import Settings
from langchain_rag.pipeline import LangChainRAG
from evaluate_langsmith import (
    answer_token_f1,
    source_hit,
    sync_dataset,
    wait_for_experiment_results,
)
from upload_model_comparison_langsmith import (
    actual_latency_ms,
    build_cached_target,
    citation_present,
    groundedness,
)


def build_pipeline(documents_dir: Path) -> LangChainRAG:
    return LangChainRAG(
        Settings(documents_dir=documents_dir, api_key="test"),
        embeddings=DeterministicFakeEmbedding(size=64),
        llm=FakeListChatModel(responses=["문서 기반 답변입니다. [출처 1]"]),
    )


def test_langchain_pipeline(tmp_path: Path) -> None:
    (tmp_path / "guide.md").write_text(
        "문서는 knowledge 폴더에 추가하고 재인덱싱합니다.", encoding="utf-8"
    )
    pipeline = build_pipeline(tmp_path)

    result = pipeline.query("문서를 어디에 추가하나요?", top_k=1)

    assert result.answer == "문서 기반 답변입니다. [출처 1]"
    assert result.sources[0].source == "guide.md"
    assert result.provider == "anthropic"
    assert result.trace_id
    assert pipeline.stats()["framework"] == "LangChain LCEL"


def test_query_api(tmp_path: Path) -> None:
    (tmp_path / "guide.txt").write_text("RAG 테스트 문서", encoding="utf-8")
    pipeline = build_pipeline(tmp_path)

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.post(
                "/api/query", json={"question": "RAG란 무엇인가요?", "top_k": 1}
            )

    with patch("langchain_rag.app.get_pipeline", return_value=pipeline):
        response = asyncio.run(request())

    assert response.status_code == 200
    assert response.json()["sources"][0]["source"] == "guide.txt"


def test_autocomplete_api() -> None:
    class BrokenGeminiAutocomplete:
        def predict_next_words(self, prompt: str, top_k: int):
            raise RuntimeError("quota exhausted")

        def continue_sentence(self, prompt: str, max_new_words: int):
            raise RuntimeError("quota exhausted")

    async def request() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            next_word_response = await client.post(
                "/api/next-word",
                json={"prompt": "오늘 오전", "top_k": 2},
            )
            generate_response = await client.post(
                "/api/generate",
                json={"prompt": "오늘 오전", "max_new_tokens": 4, "top_k": 2},
            )
            return next_word_response, generate_response

    bundle = (object(), object(), object(), {"next_word_index": {}})
    settings = Settings(
        api_key="test",
        autocomplete_provider="gemini",
        autocomplete_local_fallback=True,
    )
    with (
        patch(
            "langchain_rag.app.get_autocomplete_settings",
            return_value=settings,
        ),
        patch(
            "langchain_rag.app.get_gemini_autocomplete",
            return_value=BrokenGeminiAutocomplete(),
        ),
        patch("langchain_rag.app.get_autocomplete_bundle", return_value=bundle),
        patch(
            "langchain_rag.app.predict_next_words",
            return_value=[("계획은", 0.75), ("일정은", 0.25)],
        ),
        patch(
            "langchain_rag.app.generate_text_hybrid",
            return_value="오늘 오전 계획은 코드 점검입니다",
        ),
    ):
        next_word_response, generate_response = asyncio.run(request())

    assert next_word_response.status_code == 200
    assert next_word_response.json()["candidates"][0] == {
        "word": "계획은",
        "rank": 1,
    }
    assert next_word_response.json()["provider"] == "local"
    assert next_word_response.json()["fallback"] is True
    assert generate_response.status_code == 200
    assert (
        generate_response.json()["generated_text"]
        == "오늘 오전 계획은 코드 점검입니다"
    )
    assert generate_response.json()["provider"] == "local"


def test_gemini_autocomplete_service() -> None:
    class FakeModels:
        def __init__(self) -> None:
            self.responses = [
                SimpleNamespace(
                    parsed=CandidatePayload(
                        candidates=["공부합니다", "산책합니다", "공부합니다"]
                    ),
                    text=None,
                ),
                SimpleNamespace(
                    parsed=CompletionPayload(
                        completion="오늘 저녁에는 친구와 산책합니다."
                    ),
                    text=None,
                ),
            ]

        def generate_content(self, **kwargs):
            return self.responses.pop(0)

    service = GeminiAutocomplete(
        "test",
        "gemini-2.5-flash-lite",
        client=SimpleNamespace(models=FakeModels()),
    )

    candidates = service.predict_next_words("오늘 저녁에는", 3)
    completion = service.continue_sentence("오늘 저녁에는", 5)

    assert candidates == ("공부합니다", "산책합니다")
    assert completion == "오늘 저녁에는 친구와 산책합니다."


def test_home_defaults_to_autocomplete() -> None:
    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.get("/")

    response = asyncio.run(request())

    assert response.status_code == 200
    assert 'data-mode="autocomplete" aria-selected="true"' in response.text
    assert "다음 단어 예측" in response.text


def test_evaluators() -> None:
    assert answer_token_f1(
        {"answer": "문서를 다시 인덱싱한다"},
        {"answer": "문서를 인덱싱한다"},
    ) > 0.5
    assert source_hit(
        {"sources": ["rag_faq.md"]},
        {"expected_source": "rag_faq.md"},
    )


def test_cached_comparison_target_and_evaluators() -> None:
    target = build_cached_target(
        [
            {
                "question": "RAG란?",
                "answer": "문서를 검색해 답합니다. [출처 1]",
                "retrieved_sources": ["rag_faq.md"],
                "model": "Gemma 4 E2B",
                "model_id": "gemma4:e2b",
                "groundedness": 0.75,
                "citation_present": True,
                "latency_ms": 123.4,
                "error": None,
            }
        ]
    )

    outputs = target({"question": "RAG란?"})

    assert groundedness(outputs) == 0.75
    assert citation_present(outputs)
    assert actual_latency_ms(outputs) == 123.4
    assert outputs["cached_evaluation_result"] is True


def test_dataset_sync_removes_stale_examples() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.deleted = []
            self.created = []

        def has_dataset(self, *, dataset_name: str) -> bool:
            return True

        def read_dataset(self, *, dataset_name: str):
            return SimpleNamespace(id="dataset-id")

        def list_examples(self, *, dataset_id: str):
            return [SimpleNamespace(id="stale-example")]

        def delete_example(self, example_id: str) -> None:
            self.deleted.append(example_id)

        def create_examples(self, *, dataset_id: str, examples: list) -> None:
            self.created.extend(examples)

    client = FakeClient()
    sync_dataset(
        client,
        "test-dataset",
        [
            {
                "question": "질문",
                "answer": "답변",
                "expected_source": "guide.md",
                "category": "기능",
            }
        ],
    )

    assert client.deleted == ["stale-example"]
    assert client.created[0]["metadata"] == {"category": "기능"}


def test_waits_until_all_feedback_is_complete() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def get_experiment_results(self, *, name: str) -> dict:
            self.calls += 1
            count = 4 if self.calls == 1 else 12
            return {
                "run_stats": {"run_count": 12},
                "feedback_stats": {
                    "answer_token_f1": {"n": count},
                    "source_hit": {"n": count},
                },
            }

    client = FakeClient()
    with patch("evaluate_langsmith.time.sleep"):
        result = wait_for_experiment_results(
            client,
            "experiment",
            expected_runs=12,
            attempts=2,
        )

    assert client.calls == 2
    assert result["feedback_stats"]["source_hit"]["n"] == 12


def test_falls_back_to_gemini_when_anthropic_quota_is_exhausted(
    tmp_path: Path,
) -> None:
    class QuotaExhaustedModel:
        calls = 0

        def invoke(self, prompt):
            self.calls += 1
            error = RuntimeError("credit balance is too low")
            error.status_code = 400
            raise error

    (tmp_path / "guide.md").write_text("RAG는 문서를 검색합니다.", encoding="utf-8")
    primary = QuotaExhaustedModel()
    pipeline = LangChainRAG(
        Settings(
            documents_dir=tmp_path,
            api_key="test",
            anthropic_api_key="test",
        ),
        embeddings=DeterministicFakeEmbedding(size=64),
        llm=primary,
        fallback_llm=FakeListChatModel(
            responses=[
                "Gemini 대체 답변입니다. [출처 1]",
                "Gemini 후속 답변입니다. [출처 1]",
            ]
        ),
    )

    result = pipeline.query("RAG란 무엇인가요?", top_k=1)
    second = pipeline.query("RAG는 무엇을 검색하나요?", top_k=1)

    assert result.answer == "Gemini 대체 답변입니다. [출처 1]"
    assert result.provider == "gemini"
    assert result.model == "gemini-2.5-flash-lite"
    assert second.provider == "gemini"
    assert primary.calls == 1
    assert pipeline.stats()["active_provider"] == "gemini"
