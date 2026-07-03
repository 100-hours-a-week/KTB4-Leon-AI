from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import httpx
import torch
from google import genai
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_rag.config import PROJECT_DIR, REPO_ROOT, Settings
from langchain_rag.pipeline import load_documents


sys.path.insert(0, str(REPO_ROOT))

from RAG_chatbot.chatbot.model import (  # noqa: E402
    generate_text,
    generate_text_from_index,
    load_checkpoint,
)


TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+")
SOURCE_PATTERN = re.compile(r"\[[^\]]*출처\s+\d+[^\]]*\]")
SELF_MODEL_PATH = REPO_ROOT / "RAG_chatbot" / "artifacts" / "chatbot.pt"


def token_set(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text)}


def token_f1(answer: str, reference: str) -> float:
    actual = token_set(answer)
    expected = token_set(reference)
    if not actual or not expected:
        return 0.0
    common = len(actual & expected)
    if common == 0:
        return 0.0
    precision = common / len(actual)
    recall = common / len(expected)
    return 2 * precision * recall / (precision + recall)


def groundedness(answer: str, contexts: list[str]) -> float:
    answer_tokens = token_set(answer)
    context_tokens = token_set(" ".join(contexts))
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def build_prompt(question: str, retrieved: list[tuple[Document, float]]) -> str:
    context = "\n\n".join(
        f"[출처 {index}: {document.metadata['source']}]\n{document.page_content}"
        for index, (document, _) in enumerate(retrieved, start=1)
    )
    return f"""당신은 문서 기반 한국어 질의응답 모델입니다.
검색 문맥에 있는 내용만 사용해 간결하게 답하세요.
문맥에 답이 없으면 모른다고 말하세요.
답변 끝에는 근거를 [출처 N] 형식으로 표시하세요.

검색 문맥:
{context}

질문: {question}
답변:"""


class SharedRetriever:
    def __init__(self, settings: Settings) -> None:
        documents = load_documents(settings.documents_dir)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"{chunk.metadata['source']}#{index}"
        embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.api_key,
            output_dimensionality=768,
        )
        self.vector_store = InMemoryVectorStore(embedding=embeddings)
        self.vector_store.add_documents(chunks)

    def search(self, question: str, top_k: int) -> list[tuple[Document, float]]:
        return self.vector_store.similarity_search_with_score(question, k=top_k)


class SelfAutocompleteModel:
    name = "자기 모델"
    model = "character-transformer"

    def __init__(self) -> None:
        self.network, self.tokenizer, self.config, self.metadata = load_checkpoint(
            SELF_MODEL_PATH,
            device="cpu",
        )

    def generate(self, prompt: str) -> str:
        torch.manual_seed(42)
        generated = generate_text(
            self.network,
            self.tokenizer,
            self.config,
            prompt,
            max_new_tokens=100,
            temperature=0.6,
            top_k=8,
            min_new_tokens=12,
            device="cpu",
        )
        suffix = generated[len(prompt) :].strip()
        return suffix or "(답변을 생성하지 못함)"

    def autocomplete(self, prompt: str) -> str:
        next_word_index = self.metadata.get("next_word_index", {})
        return generate_text_from_index(
            next_word_index,
            prompt,
            max_new_words=8,
            top_k=5,
        )


class OllamaGemmaModel:
    name = "Gemma 4 E2B"

    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str) -> str:
        response = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0.1, "num_predict": 500},
            },
            timeout=300,
        )
        response.raise_for_status()
        return str(response.json()["message"]["content"]).strip()


class GeminiModel:
    name = "Gemini 2.5 Flash"

    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"temperature": 0.1, "max_output_tokens": 500},
        )
        return (response.text or "").strip()

    def close(self) -> None:
        self.client.close()


def evaluate_answer(
    *,
    model_name: str,
    model_id: str,
    generate: Callable[[str], str],
    prompt: str,
    reference: str,
    contexts: list[str],
    expected_source: str,
    retrieved_sources: list[str],
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        answer = generate(prompt)
        error = None
    except Exception as exc:  # Keep the other models running if one provider fails.
        answer = ""
        error = f"{type(exc).__name__}: {exc}"
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "model": model_name,
        "model_id": model_id,
        "answer": answer,
        "answer_token_f1": round(token_f1(answer, reference), 4),
        "groundedness": round(groundedness(answer, contexts), 4),
        "citation_present": bool(SOURCE_PATTERN.search(answer)),
        "retrieval_hit": expected_source in retrieved_sources,
        "latency_ms": round(elapsed_ms, 2),
        "error": error,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary = {}
    for model_name in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model_name]
        successful = [row for row in model_rows if row["error"] is None]
        denominator = len(model_rows)
        summary[model_name] = {
            "cases": denominator,
            "successful_cases": len(successful),
            "answer_token_f1": round(
                mean(row["answer_token_f1"] for row in successful), 4
            )
            if successful
            else 0.0,
            "groundedness": round(
                mean(row["groundedness"] for row in successful), 4
            )
            if successful
            else 0.0,
            "citation_rate": round(
                sum(row["citation_present"] for row in successful) / denominator,
                4,
            )
            if denominator
            else 0.0,
            "retrieval_hit_rate": round(
                sum(row["retrieval_hit"] for row in successful) / denominator,
                4,
            )
            if denominator
            else 0.0,
            "latency_ms": round(
                mean(row["latency_ms"] for row in successful), 2
            )
            if successful
            else 0.0,
            "latency_median_ms": round(
                median(row["latency_ms"] for row in successful), 2
            )
            if successful
            else 0.0,
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="세 모델의 동일 RAG 질문 비교 평가")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_DIR / "evaluation" / "dataset.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "evaluation" / "model_comparison_results.json",
    )
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--gemma-model", default="gemma4:e2b")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()
    if not settings.api_key:
        raise RuntimeError("GEMINI_API_KEY가 필요합니다.")
    cases = json.loads(args.dataset.read_text(encoding="utf-8"))
    if len(cases) < 10:
        raise ValueError("평가 데이터는 질문과 정답 10개 이상이어야 합니다.")

    retriever = SharedRetriever(settings)
    self_model = SelfAutocompleteModel()
    gemma = OllamaGemmaModel(args.gemma_model, args.ollama_url)
    gemini = GeminiModel(settings.api_key, args.gemini_model)
    models = [self_model, gemma, gemini]
    rows = []
    try:
        for case_index, case in enumerate(cases, start=1):
            retrieved = retriever.search(case["question"], args.top_k)
            prompt = build_prompt(case["question"], retrieved)
            contexts = [document.page_content for document, _ in retrieved]
            retrieved_sources = [
                str(document.metadata["source"]) for document, _ in retrieved
            ]
            for model in models:
                result = evaluate_answer(
                    model_name=model.name,
                    model_id=model.model,
                    generate=model.generate,
                    prompt=prompt,
                    reference=case["answer"],
                    contexts=contexts,
                    expected_source=case["expected_source"],
                    retrieved_sources=retrieved_sources,
                )
                rows.append(
                    {
                        "case": case_index,
                        "category": case["category"],
                        "question": case["question"],
                        "reference": case["answer"],
                        "expected_source": case["expected_source"],
                        "retrieved_sources": retrieved_sources,
                        **result,
                    }
                )
            print(f"completed={case_index}/{len(cases)}")
    finally:
        gemini.close()

    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "cases": len(cases),
        "comparison_scope": (
            "세 모델에 동일한 검색 문맥과 질문을 입력했다. 자기 모델은 자동완성 "
            "모델이므로 질의응답 점수는 용도 부적합을 확인하는 진단값이다."
        ),
        "self_model_training": {
            "architecture": self_model.metadata.get("architecture"),
            "data_type": self_model.metadata.get("data_type"),
            "texts": self_model.metadata.get("texts"),
            "tokens": self_model.metadata.get("tokens"),
            "vocabulary_size": self_model.metadata.get("vocabulary_size"),
            "epochs": self_model.metadata.get("epochs"),
            "steps": self_model.metadata.get("steps"),
            "best_validation_loss": self_model.metadata.get(
                "best_validation_loss"
            ),
        },
        "self_model_demo": [
            {
                "prompt": prompt,
                "completion": self_model.autocomplete(prompt),
            }
            for prompt in (
                "오늘 오전",
                "엄마",
                "주말에는",
                "내일 오후에는",
                "모델 학습 결과를",
            )
        ],
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"result={args.output.resolve()}")


if __name__ == "__main__":
    main()
