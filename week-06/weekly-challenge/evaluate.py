from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from rag_app.config import PROJECT_DIR, Settings
from rag_app.pipeline import RAGPipeline


TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+")


def tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text)}


def token_f1(answer: str, reference: str) -> float:
    answer_tokens = tokens(answer)
    reference_tokens = tokens(reference)
    if not answer_tokens or not reference_tokens:
        return 0.0
    common = len(answer_tokens & reference_tokens)
    precision = common / len(answer_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if common else 0.0


def groundedness(answer: str, contexts: list[str]) -> float:
    answer_tokens = tokens(answer)
    context_tokens = tokens(" ".join(contexts))
    return len(answer_tokens & context_tokens) / len(answer_tokens) if answer_tokens else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 검색 및 답변 품질을 평가합니다.")
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_DIR / "evaluation" / "questions.json"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_DIR / "evaluation" / "results.json"
    )
    parser.add_argument("--provider", choices=("offline", "gemini"), default="offline")
    parser.add_argument("--top-k", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Settings.from_env()
    settings = Settings(
        documents_dir=base.documents_dir,
        provider=args.provider,
        model=(
            os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
            if args.provider == "gemini"
            else base.model
        ),
        api_key=(os.getenv("GEMINI_API_KEY") if args.provider == "gemini" else None),
        chunk_size=base.chunk_size,
        chunk_overlap=base.chunk_overlap,
        default_top_k=args.top_k,
    )
    pipeline = RAGPipeline(settings)
    cases = json.loads(args.dataset.read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        started = time.perf_counter()
        result = pipeline.query(case["question"], top_k=args.top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000
        source_rank = next(
            (
                rank
                for rank, source in enumerate(result.sources, start=1)
                if source.source == case["expected_source"]
            ),
            None,
        )
        rows.append(
            {
                "question": case["question"],
                "answer": result.answer,
                "retrieval_hit": source_rank is not None,
                "reciprocal_rank": 1 / source_rank if source_rank else 0.0,
                "answer_token_f1": token_f1(result.answer, case["reference"]),
                "groundedness": groundedness(
                    result.answer, [source.text for source in result.sources]
                ),
                "latency_ms": round(elapsed_ms, 2),
                "retrieved_sources": [source.source for source in result.sources],
            }
        )

    metric_names = (
        "retrieval_hit",
        "reciprocal_rank",
        "answer_token_f1",
        "groundedness",
        "latency_ms",
    )
    summary = {
        name: round(sum(float(row[name]) for row in rows) / len(rows), 4)
        for name in metric_names
    }
    report = {"provider": args.provider, "cases": len(rows), "summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"평가 결과: {args.output.resolve()}")


if __name__ == "__main__":
    main()
