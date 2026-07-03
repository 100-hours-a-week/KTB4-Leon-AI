from __future__ import annotations

from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings


SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}


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
    trace_id: str
    provider: str
    model: str


def load_documents(directory: Path) -> list[Document]:
    if not directory.exists():
        raise FileNotFoundError(f"문서 디렉터리가 없습니다: {directory}")
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    documents = []
    for path in paths:
        text = path.read_text(encoding="utf-8").replace("\ufeff", "").strip()
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": str(path.relative_to(directory))},
                )
            )
    return documents


class LangChainRAG:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embeddings: Embeddings | None = None,
        llm: BaseChatModel | None = None,
        fallback_llm: BaseChatModel | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        if not 0 <= self.settings.chunk_overlap < self.settings.chunk_size:
            raise ValueError("chunk_overlap은 0 이상 chunk_size 미만이어야 합니다.")
        if embeddings is None and not self.settings.api_key:
            raise RuntimeError("임베딩 생성에 GOOGLE_API_KEY 또는 GEMINI_API_KEY가 필요합니다.")
        self.embeddings = embeddings or GoogleGenerativeAIEmbeddings(
            model=self.settings.embedding_model,
            api_key=self.settings.api_key,
            output_dimensionality=768,
        )
        if llm is None:
            if self.settings.primary_provider != "anthropic":
                raise ValueError("RAG_PRIMARY_PROVIDER는 anthropic만 지원합니다.")
            if not self.settings.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY가 필요합니다.")
            self.llm = ChatAnthropic(
                model=self.settings.anthropic_model,
                api_key=self.settings.anthropic_api_key,
                temperature=0.1,
                max_tokens=800,
            )
        else:
            self.llm = llm
        if self.settings.fallback_provider != "gemini":
            raise ValueError("RAG_FALLBACK_PROVIDER는 gemini만 지원합니다.")
        self.fallback_llm = fallback_llm
        if self.fallback_llm is None and llm is None and self.settings.api_key:
            self.fallback_llm = ChatGoogleGenerativeAI(
                model=self.settings.model,
                api_key=self.settings.api_key,
                temperature=0.1,
                max_tokens=800,
                thinking_budget=0,
            )
        self._provider_lock = RLock()
        self._fallback_active = False
        self._lock = RLock()
        self._vector_store: InMemoryVectorStore | None = None
        self._document_count = 0
        self._chunk_count = 0
        self._source_names: list[str] = []
        self._build_chain()
        self.reindex()

    def _build_chain(self) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 검색된 문서를 근거로 답하는 친절한 한국어 챗봇입니다.\n"
                    "답변 규칙:\n"
                    "1. 아래 검색 문맥에 있는 내용만 근거로 사용하고, 문맥에 없는 사실을 지어내지 마세요.\n"
                    "2. 문맥에 질문과 관련된 내용이 조금이라도 있으면 그 내용을 바탕으로 분명하고 자신 있게 답하세요. "
                    "관련 내용이 있는데도 '문서에 없다'며 발을 빼지 마세요.\n"
                    "3. 문맥에 질문과 관련된 내용이 전혀 없을 때만 '제공된 문서에서는 답을 찾을 수 없습니다'라고 답하세요.\n"
                    "4. 한 답변 안에서 '모른다'와 '안다'를 동시에 말하지 마세요. 둘 중 하나만 고르세요.\n"
                    "5. 근거로 사용한 문장 끝에는 [출처 N] 형식으로 출처 번호를 표시하세요.\n\n"
                    "검색 문맥:\n{context}",
                ),
                ("human", "{question}"),
            ]
        )
        retrieve = RunnableLambda(self._retrieve).with_config(
            run_name="retrieve_documents"
        )
        prepare = RunnableParallel(
            question=itemgetter("question"),
            retrieved=retrieve,
        )
        add_context = RunnablePassthrough.assign(
            context=RunnableLambda(self._format_context).with_config(
                run_name="format_context"
            )
        )
        generation = (
            prompt.with_config(run_name="grounded_prompt")
            | RunnableLambda(self._generate_answer).with_config(
                run_name="anthropic_with_gemini_fallback"
            )
        )
        self.chain = (
            prepare
            | add_context
            | RunnablePassthrough.assign(generation=generation)
        ).with_config(run_name="week7_langchain_rag")

    def _generate_answer(self, prompt: Any) -> dict[str, str]:
        parser = StrOutputParser()
        with self._provider_lock:
            fallback_active = self._fallback_active

        if fallback_active:
            if self.fallback_llm is None:
                raise RuntimeError("Gemini 대체 모델이 설정되지 않았습니다.")
            response = self.fallback_llm.invoke(prompt)
            return {
                "answer": parser.invoke(response),
                "provider": "gemini",
                "model": self.settings.model,
            }

        try:
            response = self.llm.invoke(prompt)
            return {
                "answer": parser.invoke(response),
                "provider": (
                    "anthropic"
                    if self.settings.primary_provider == "anthropic"
                    else "injected"
                ),
                "model": self.settings.anthropic_model,
            }
        except Exception:
            if self.fallback_llm is None:
                raise
            with self._provider_lock:
                self._fallback_active = True
            response = self.fallback_llm.invoke(prompt)
            return {
                "answer": parser.invoke(response),
                "provider": "gemini",
                "model": self.settings.model,
            }

    def reindex(self) -> dict[str, Any]:
        documents = load_documents(self.settings.documents_dir)
        if not documents:
            raise ValueError(f"지원하는 문서가 없습니다: {self.settings.documents_dir}")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"{chunk.metadata['source']}#{index}"

        vector_store = InMemoryVectorStore(embedding=self.embeddings)
        vector_store.add_documents(chunks)
        with self._lock:
            self._vector_store = vector_store
            self._document_count = len(documents)
            self._chunk_count = len(chunks)
            self._source_names = sorted(
                {document.metadata["source"] for document in documents}
            )
        return self.stats()

    def _retrieve(self, inputs: dict[str, Any]) -> list[tuple[Document, float]]:
        question = str(inputs["question"]).strip()
        top_k = int(inputs.get("top_k", self.settings.top_k))
        with self._lock:
            if self._vector_store is None:
                raise RuntimeError("벡터 인덱스가 준비되지 않았습니다.")
            return self._vector_store.similarity_search_with_score(
                question, k=top_k
            )

    @staticmethod
    def _format_context(inputs: dict[str, Any]) -> str:
        retrieved: list[tuple[Document, float]] = inputs["retrieved"]
        return "\n\n".join(
            f"[출처 {index}: {document.metadata['source']}]\n{document.page_content}"
            for index, (document, _) in enumerate(retrieved, start=1)
        )

    @staticmethod
    def _sources(retrieved: list[tuple[Document, float]]) -> list[Source]:
        return [
            Source(
                source=str(document.metadata["source"]),
                chunk_id=str(document.metadata["chunk_id"]),
                score=round(float(score), 6),
                text=document.page_content,
            )
            for document, score in retrieved
        ]

    def query(self, question: str, top_k: int | None = None) -> QueryResult:
        question = question.strip()
        if not question:
            raise ValueError("질문이 비어 있습니다.")
        run_id = uuid4()
        output = self.chain.invoke(
            {"question": question, "top_k": top_k or self.settings.top_k},
            config={
                "run_id": run_id,
                "tags": ["week-07", "langchain-rag"],
                "metadata": {"top_k": top_k or self.settings.top_k},
            },
        )
        generation = output["generation"]
        return QueryResult(
            question=question,
            answer=str(generation["answer"]).strip(),
            sources=self._sources(output["retrieved"]),
            trace_id=str(run_id),
            provider=str(generation["provider"]),
            model=str(generation["model"]),
        )

    def stats(self) -> dict[str, Any]:
        with self._provider_lock:
            active_provider = "gemini" if self._fallback_active else "anthropic"
        with self._lock:
            return {
                "documents": self._document_count,
                "chunks": self._chunk_count,
                "sources": list(self._source_names),
                "primary_provider": self.settings.primary_provider,
                "primary_model": self.settings.anthropic_model,
                "fallback_provider": (
                    self.settings.fallback_provider if self.fallback_llm else None
                ),
                "fallback_model": self.settings.model if self.fallback_llm else None,
                "active_provider": active_provider,
                "embedding_model": self.settings.embedding_model,
                "framework": "LangChain LCEL",
            }
