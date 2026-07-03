from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_SUFFIXES = {".csv", ".json", ".md", ".markdown", ".txt"}


@dataclass(frozen=True)
class Document:
    source: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str
    position: int


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", "").replace("\r\n", "\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _load_json(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    return json.dumps(data, ensure_ascii=False, indent=2)


def _load_csv(path: Path) -> str:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)
        return "\n".join(
            " | ".join(f"{key}: {value}" for key, value in row.items())
            for row in rows
        )


def load_document(path: Path, root: Path | None = None) -> Document:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"지원하지 않는 문서 형식입니다: {suffix}")

    text = _load_json(path) if suffix == ".json" else (
        _load_csv(path) if suffix == ".csv" else path.read_text(encoding="utf-8")
    )
    source = str(path.relative_to(root)) if root else path.name
    return Document(source=source, text=normalize_text(text))


def load_documents(path: Path) -> list[Document]:
    if not path.exists():
        raise FileNotFoundError(f"문서 경로가 없습니다: {path}")

    if path.is_file():
        return [load_document(path, path.parent)]

    paths = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES
    )
    documents = [load_document(candidate, path) for candidate in paths]
    return [document for document in documents if document.text]


def split_document(
    document: Document,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size는 1 이상이어야 합니다.")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap은 0 이상 chunk_size 미만이어야 합니다.")

    text = document.text
    chunks: list[Chunk] = []
    start = 0
    position = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            search_start = start + int(chunk_size * 0.6)
            boundaries = [
                text.rfind(separator, search_start, end)
                for separator in ("\n\n", "\n", ". ", "다. ")
            ]
            boundary = max(boundaries)
            if boundary > start:
                end = boundary + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=f"{document.source}#{position}",
                    source=document.source,
                    text=chunk_text,
                    position=position,
                )
            )
            position += 1
        if end >= len(text):
            break
        next_start = end - chunk_overlap
        start = next_start if next_start > start else end
    return chunks


def split_documents(
    documents: list[Document],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    return [
        chunk
        for document in documents
        for chunk in split_document(document, chunk_size, chunk_overlap)
    ]
