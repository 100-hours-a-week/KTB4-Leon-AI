from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from threading import RLock

from .config import Settings
from .documents import load_documents, split_documents
from .generator import FallbackGenerator, build_prompt, create_generator
from .retriever import SearchResult, TfidfRetriever


NO_RELEVANT_CONTEXT = (
    "제공된 문서에서 관련 내용을 찾지 못했습니다. "
    "이 앱은 RAG와 FastAPI 문서에 관한 질문에만 답합니다."
)


@dataclass(frozen=True)
class Source:
    source: str
    chunk_id: str
    score: float
    text: str


@dataclass(frozen=True)
class QueryResult:
    question: str
    answer: str
    sources: list[Source]
    provider: str
    model: str


class RAGPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.retriever = TfidfRetriever()
        primary_generator = create_generator(
            provider=self.settings.provider,
            api_key=self.settings.api_key,
            model=self.settings.model,
            ollama_base_url=self.settings.ollama_base_url,
            anthropic_base_url=self.settings.anthropic_base_url,
        )
        if self.settings.fallback_provider and self.settings.fallback_model:
            fallback_generator = create_generator(
                provider=self.settings.fallback_provider,
                api_key=self.settings.fallback_api_key,
                model=self.settings.fallback_model,
                ollama_base_url=self.settings.ollama_base_url,
                anthropic_base_url=self.settings.anthropic_base_url,
            )
            self.generator = FallbackGenerator(
                [primary_generator, fallback_generator]
            )
        else:
            self.generator = primary_generator
        self._document_count = 0
        self._source_names: list[str] = []
        self._lock = RLock()
        self.reindex()

    def reindex(self) -> dict[str, int | list[str]]:
        documents = load_documents(self.settings.documents_dir)
        if not documents:
            raise ValueError(
                f"지원하는 문서가 없습니다: {self.settings.documents_dir}"
            )
        chunks = split_documents(
            documents,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.retriever.index(chunks)
        with self._lock:
            self._document_count = len(documents)
            self._source_names = [document.source for document in documents]
        return self.stats()

    def retrieve(self, question: str, top_k: int | None = None) -> list[SearchResult]:
        results = self.retriever.search(
            question,
            top_k=top_k or self.settings.default_top_k,
        )
        if not results or results[0].score < self.settings.min_relevance_score:
            return []
        return results

    @staticmethod
    def _make_sources(results: list[SearchResult]) -> list[Source]:
        return [
            Source(
                source=result.chunk.source,
                chunk_id=result.chunk.chunk_id,
                score=round(result.score, 6),
                text=result.chunk.text,
            )
            for result in results
        ]

    def query(self, question: str, top_k: int | None = None) -> QueryResult:
        question = question.strip()
        results = self.retrieve(question, top_k)
        if not results:
            return QueryResult(
                question=question,
                answer=NO_RELEVANT_CONTEXT,
                sources=[],
                provider=self.generator.provider,
                model=self.generator.model,
            )
        answer = self.generator.generate(build_prompt(question, results))
        if not answer:
            raise RuntimeError("생성 모델이 빈 응답을 반환했습니다.")
        return QueryResult(
            question=question,
            answer=answer,
            sources=self._make_sources(results),
            provider=self.generator.provider,
            model=self.generator.model,
        )

    def stream_query(
        self, question: str, top_k: int | None = None
    ) -> tuple[list[Source], Iterator[str]]:
        question = question.strip()
        results = self.retrieve(question, top_k)
        if not results:
            return [], iter([NO_RELEVANT_CONTEXT])
        return self._make_sources(results), self.generator.stream(
            build_prompt(question, results)
        )

    def stats(self) -> dict[str, int | str | list[str]]:
        with self._lock:
            return {
                "documents": self._document_count,
                "chunks": self.retriever.chunk_count,
                "sources": list(self._source_names),
                "provider": self.generator.provider,
                "model": self.generator.model,
            }
