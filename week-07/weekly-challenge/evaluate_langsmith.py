from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from langsmith import Client

from langchain_rag.config import PROJECT_DIR
from langchain_rag.pipeline import LangChainRAG


TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+")


def token_set(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text)}


def answer_token_f1(outputs: dict, reference_outputs: dict) -> float:
    actual = token_set(str(outputs.get("answer", "")))
    expected = token_set(str(reference_outputs.get("answer", "")))
    if not actual or not expected:
        return 0.0
    common = len(actual & expected)
    if common == 0:
        return 0.0
    precision = common / len(actual)
    recall = common / len(expected)
    return 2 * precision * recall / (precision + recall)


def source_hit(outputs: dict, reference_outputs: dict) -> bool:
    return str(reference_outputs.get("expected_source", "")) in outputs.get(
        "sources", []
    )


def sync_dataset(client: Client, dataset_name: str, cases: list[dict]) -> None:
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
    else:
        dataset = client.create_dataset(
            dataset_name,
            description="Week 7 LangChain RAG의 질문, 기준 답변, 기대 출처",
        )
    examples = [
        {
            "id": str(uuid5(NAMESPACE_URL, f"{dataset_name}:{case['question']}")),
            "inputs": {"question": case["question"]},
            "outputs": {
                "answer": case["answer"],
                "expected_source": case["expected_source"],
            },
            "metadata": {"category": case.get("category", "미분류")},
        }
        for case in cases
    ]
    existing_examples = list(client.list_examples(dataset_id=dataset.id))
    existing_ids = {str(example.id) for example in existing_examples}
    desired_ids = {example["id"] for example in examples}
    for existing in existing_examples:
        if str(existing.id) not in desired_ids:
            client.delete_example(existing.id)

    new_examples = []
    for example in examples:
        if example["id"] in existing_ids:
            client.update_example(
                example["id"],
                inputs=example["inputs"],
                outputs=example["outputs"],
                metadata=example["metadata"],
                dataset_id=dataset.id,
            )
        else:
            new_examples.append(example)

    if new_examples:
        client.create_examples(dataset_id=dataset.id, examples=new_examples)


def wait_for_experiment_results(
    client: Client,
    experiment_name: str,
    expected_runs: int,
    attempts: int = 10,
) -> dict:
    experiment = {}
    for _ in range(attempts):
        experiment = client.get_experiment_results(name=experiment_name)
        run_count = experiment.get("run_stats", {}).get("run_count", 0)
        feedback_stats = experiment.get("feedback_stats", {})
        feedback_complete = feedback_stats and all(
            stats.get("n", 0) >= expected_runs for stats in feedback_stats.values()
        )
        if run_count >= expected_runs and feedback_complete:
            return experiment
        time.sleep(1)
    return experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangSmith Dataset 기반 RAG 평가")
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=PROJECT_DIR / "evaluation" / "dataset.json",
    )
    parser.add_argument("--dataset-name", default="kakao-bootcamp-week7-rag")
    parser.add_argument("--experiment-prefix", default="week7-langchain-rag")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "evaluation" / "results_langsmith.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = json.loads(args.dataset_file.read_text(encoding="utf-8"))
    client = Client()
    sync_dataset(client, args.dataset_name, cases)
    pipeline = LangChainRAG()

    def target(inputs: dict) -> dict:
        result = pipeline.query(str(inputs["question"]))
        return {
            "answer": result.answer,
            "sources": [source.source for source in result.sources],
            "trace_id": result.trace_id,
            "provider": result.provider,
            "model": result.model,
        }

    results = client.evaluate(
        target,
        data=args.dataset_name,
        evaluators=[answer_token_f1, source_hit],
        experiment_prefix=args.experiment_prefix,
        description="Claude Haiku 우선, Gemini Flash-Lite 대체 LangChain 2-step RAG 평가",
        max_concurrency=1,
    )
    print(f"dataset={args.dataset_name}")
    print(f"experiment={results.experiment_name}")
    experiment = wait_for_experiment_results(
        client,
        results.experiment_name,
        expected_runs=len(cases),
    )
    report = {
        "dataset": args.dataset_name,
        "experiment": results.experiment_name,
        "feedback_stats": experiment["feedback_stats"],
        "run_stats": experiment["run_stats"],
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"result={args.output.resolve()}")


if __name__ == "__main__":
    main()
