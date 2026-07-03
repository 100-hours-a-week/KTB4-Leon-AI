from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langsmith import Client

from evaluate_langsmith import (
    answer_token_f1,
    source_hit,
    sync_dataset,
    wait_for_experiment_results,
)
from langchain_rag.config import PROJECT_DIR


MODEL_SLUGS = {
    "자기 모델": "self-autocomplete",
    "Gemma 4 E2B": "gemma4-e2b",
    "Gemini 2.5 Flash": "gemini-2-5-flash",
}


def groundedness(outputs: dict) -> float:
    return float(outputs.get("groundedness", 0.0))


def citation_present(outputs: dict) -> bool:
    return bool(outputs.get("citation_present", False))


def actual_latency_ms(outputs: dict) -> float:
    return float(outputs.get("actual_latency_ms", 0.0))


def build_cached_target(rows: list[dict[str, Any]]) -> Callable[[dict], dict]:
    rows_by_question = {row["question"]: row for row in rows}

    def target(inputs: dict) -> dict:
        question = str(inputs["question"])
        row = rows_by_question[question]
        return {
            "answer": row["answer"],
            "sources": row["retrieved_sources"],
            "model": row["model"],
            "model_id": row["model_id"],
            "groundedness": row["groundedness"],
            "citation_present": row["citation_present"],
            "actual_latency_ms": row["latency_ms"],
            "original_error": row["error"],
            "cached_evaluation_result": True,
        }

    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="실행 완료된 3개 모델 비교 결과를 LangSmith에 업로드"
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=PROJECT_DIR / "evaluation" / "dataset.json",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=PROJECT_DIR / "evaluation" / "model_comparison_results.json",
    )
    parser.add_argument("--dataset-name", default="kakao-bootcamp-week7-rag")
    parser.add_argument("--experiment-prefix", default="week7-model-comparison")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR
        / "evaluation"
        / "results_langsmith_model_comparison.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = json.loads(args.dataset_file.read_text(encoding="utf-8"))
    comparison = json.loads(args.results_file.read_text(encoding="utf-8"))
    if len(cases) < 10:
        raise ValueError("LangSmith 비교 평가에는 질문이 10개 이상 필요합니다.")

    client = Client()
    sync_dataset(client, args.dataset_name, cases)
    dataset = client.read_dataset(dataset_name=args.dataset_name)
    expected_questions = {case["question"] for case in cases}
    experiments = {}

    for model_name, slug in MODEL_SLUGS.items():
        model_rows = [
            row for row in comparison["rows"] if row["model"] == model_name
        ]
        row_questions = {row["question"] for row in model_rows}
        if row_questions != expected_questions:
            raise ValueError(f"{model_name} 결과와 Dataset 질문이 일치하지 않습니다.")

        model_id = str(model_rows[0]["model_id"])
        result = client.evaluate(
            build_cached_target(model_rows),
            data=args.dataset_name,
            evaluators=[
                answer_token_f1,
                source_hit,
                groundedness,
                citation_present,
                actual_latency_ms,
            ],
            experiment_prefix=f"{args.experiment_prefix}-{slug}",
            description=(
                "동일 검색 문맥에서 실행한 모델 비교 결과. 생성 답변은 "
                "model_comparison_results.json에서 가져오며 actual_latency_ms가 "
                "실제 모델 호출 지연이다."
            ),
            metadata={
                "models": model_id,
                "model_name": model_name,
                "cached_outputs": True,
                "source_generated_at": comparison["generated_at"],
            },
            max_concurrency=1,
        )
        experiment = wait_for_experiment_results(
            client,
            result.experiment_name,
            expected_runs=len(cases),
            attempts=20,
        )
        project = client.read_project(
            project_name=result.experiment_name,
            include_stats=True,
        )
        experiments[model_name] = {
            "model_id": model_id,
            "experiment": result.experiment_name,
            "project_url": project.url,
            "feedback_stats": experiment.get("feedback_stats", {}),
            "run_stats": experiment.get("run_stats", {}),
        }
        print(f"{model_name}: {project.url}")

    report = {
        "dataset": args.dataset_name,
        "dataset_id": str(dataset.id),
        "cases": len(cases),
        "source_results": str(args.results_file.resolve()),
        "source_generated_at": comparison["generated_at"],
        "cached_outputs": True,
        "experiments": experiments,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"result={args.output.resolve()}")


if __name__ == "__main__":
    main()
