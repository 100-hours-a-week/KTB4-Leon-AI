from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}


@dataclass(frozen=True)
class Source:
    source: str
    chunk_id: str
    score: float
    text: str


class KnowledgeIndex:
    def __init__(
        self,
        knowledge_dir: Path,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> None:
        self.knowledge_dir = knowledge_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._chunks: list[Document] = []
        self._lock = RLock()
        self.reindex()

    def _load_documents(self) -> list[Document]:
        if not self.knowledge_dir.exists():
            raise FileNotFoundError(
                f"지식 문서 디렉터리가 없습니다: {self.knowledge_dir}"
            )
        documents: list[Document] = []
        for path in sorted(self.knowledge_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8").replace("\ufeff", "").strip()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(path.relative_to(self.knowledge_dir))},
                    )
                )
        return documents

    def reindex(self) -> dict[str, int | list[str]]:
        documents = self._load_documents()
        if not documents:
            raise ValueError(f"지식 문서가 없습니다: {self.knowledge_dir}")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"{chunk.metadata['source']}#{index}"

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(chunk.page_content for chunk in chunks)
        with self._lock:
            self._vectorizer = vectorizer
            self._matrix = matrix
            self._chunks = chunks
        return self.stats()

    def search(self, question: str, top_k: int = 4) -> list[Source]:
        question = question.strip()
        if not question:
            raise ValueError("질문이 비어 있습니다.")
        with self._lock:
            if self._vectorizer is None or self._matrix is None:
                raise RuntimeError("문서 인덱스가 준비되지 않았습니다.")
            vector = self._vectorizer.transform([question])
            scores = linear_kernel(vector, self._matrix).ravel()
            indexes = scores.argsort()[::-1][: min(top_k, len(self._chunks))]
            return [
                Source(
                    source=str(self._chunks[index].metadata["source"]),
                    chunk_id=str(self._chunks[index].metadata["chunk_id"]),
                    score=round(float(scores[index]), 6),
                    text=self._chunks[index].page_content,
                )
                for index in indexes
                if scores[index] > 0
            ]

    def stats(self) -> dict[str, int | list[str]]:
        with self._lock:
            sources = sorted(
                {str(chunk.metadata["source"]) for chunk in self._chunks}
            )
            return {
                "documents": len(sources),
                "chunks": len(self._chunks),
                "sources": sources,
            }
