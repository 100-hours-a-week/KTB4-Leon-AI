from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .documents import Chunk


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class TfidfRetriever:
    """Character n-gram retrieval works for Korean without a tokenizer service."""

    def __init__(self) -> None:
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix: csr_matrix | None = None
        self._chunks: list[Chunk] = []
        self._lock = RLock()

    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("인덱싱할 문서 청크가 없습니다.")
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(chunk.text for chunk in chunks)
        with self._lock:
            self._vectorizer = vectorizer
            self._matrix = matrix.tocsr()
            self._chunks = list(chunks)

    def search(self, query: str, top_k: int = 4) -> list[SearchResult]:
        query = query.strip()
        if not query:
            raise ValueError("검색 질문이 비어 있습니다.")
        if top_k <= 0:
            raise ValueError("top_k는 1 이상이어야 합니다.")

        with self._lock:
            if self._vectorizer is None or self._matrix is None:
                raise RuntimeError("문서 인덱스가 준비되지 않았습니다.")
            query_vector = self._vectorizer.transform([query])
            scores = linear_kernel(query_vector, self._matrix).ravel()
            ranked = scores.argsort()[::-1][: min(top_k, len(self._chunks))]
            return [
                SearchResult(chunk=self._chunks[index], score=float(scores[index]))
                for index in ranked
                if scores[index] > 0
            ]

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)
