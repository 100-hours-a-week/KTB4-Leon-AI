from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)

from plot_style import COLORS, add_chart_header, finish_axis, use_chart_theme


def make_dataset(
    samples: int,
    features: int,
    classes: int,
    seed: int,
):
    informative = max(classes * 2, features // 2)
    informative = min(informative, features - 2)
    return make_classification(
        n_samples=samples,
        n_features=features,
        n_informative=informative,
        n_redundant=2,
        n_classes=classes,
        n_clusters_per_class=1,
        class_sep=1.2,
        flip_y=0.03,
        random_state=seed,
    )


def run_search(
    name: str,
    search,
    x_train,
    y_train,
    x_test,
    y_test,
) -> dict:
    started_at = time.perf_counter()
    search.fit(x_train, y_train)
    elapsed_seconds = time.perf_counter() - started_at
    predictions = search.predict(x_test)

    return {
        "search": name,
        "best_cv_accuracy": float(search.best_score_),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "elapsed_seconds": elapsed_seconds,
        "best_params": search.best_params_,
        "evaluated_candidates": len(search.cv_results_["params"]),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "cv_results": pd.DataFrame(search.cv_results_),
    }


def plot_comparison(results: list[dict], output_dir: Path) -> None:
    use_chart_theme()
    labels = [result["search"] for result in results]
    families = [COLORS["blue"], COLORS["orange"]]
    fills = [families[index]["base"] for index in range(len(results))]
    edges = [families[index]["dark"] for index in range(len(results))]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    width = 0.34
    x = range(len(results))
    cv_bars = axes[0].bar(
        [index - width / 2 for index in x],
        [result["best_cv_accuracy"] * 100 for result in results],
        width=width,
        color=fills,
        edgecolor=edges,
        label="Best CV",
    )
    test_bars = axes[0].bar(
        [index + width / 2 for index in x],
        [result["test_accuracy"] * 100 for result in results],
        width=width,
        color="white",
        edgecolor=edges,
        hatch="//",
        label="Test",
    )
    axes[0].set_xticks(list(x), labels)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].bar_label(cv_bars, fmt="%.1f", padding=3, fontsize=8)
    axes[0].bar_label(test_bars, fmt="%.1f", padding=3, fontsize=8)
    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    finish_axis(axes[0])

    time_bars = axes[1].bar(
        labels,
        [result["elapsed_seconds"] for result in results],
        color=fills,
        edgecolor=edges,
    )
    axes[1].set_ylabel("Search time (seconds)")
    axes[1].bar_label(time_bars, fmt="%.1fs", padding=4)
    finish_axis(axes[1])

    add_chart_header(
        fig,
        axes[0],
        "GridSearch와 RandomSearch 비교",
        "동일한 가상 분류 데이터·모델·교차검증 분할에서 정확도와 탐색 시간을 비교",
    )
    fig.subplots_adjust(top=0.76, wspace=0.34)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.12, 0.87),
        frameon=False,
        ncol=2,
    )
    fig.savefig(output_dir / "search_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="가상 분류 데이터에서 GridSearchCV와 RandomizedSearchCV를 비교합니다."
    )
    parser.add_argument("--samples", type=int, default=2500)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--random-iterations", type=int, default=12)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./artifacts/hyperparameter_search")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features, labels = make_dataset(
        args.samples,
        args.features,
        args.classes,
        args.seed,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=args.seed,
    )
    cross_validation = StratifiedKFold(
        n_splits=args.cv,
        shuffle=True,
        random_state=args.seed,
    )
    estimator = RandomForestClassifier(
        random_state=args.seed,
        n_jobs=1,
        class_weight="balanced",
    )

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid={
            "n_estimators": [80, 140],
            "max_depth": [None, 8, 14],
            "min_samples_split": [2, 6],
            "max_features": ["sqrt", 0.7],
        },
        scoring="accuracy",
        cv=cross_validation,
        n_jobs=args.jobs,
        return_train_score=True,
    )
    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions={
            "n_estimators": randint(60, 240),
            "max_depth": [None, 6, 8, 10, 14, 18],
            "min_samples_split": randint(2, 12),
            "min_samples_leaf": randint(1, 6),
            "max_features": uniform(0.35, 0.6),
        },
        n_iter=args.random_iterations,
        scoring="accuracy",
        cv=cross_validation,
        n_jobs=args.jobs,
        random_state=args.seed,
        return_train_score=True,
    )

    results = [
        run_search("GridSearch", grid_search, x_train, y_train, x_test, y_test),
        run_search("RandomSearch", random_search, x_train, y_train, x_test, y_test),
    ]

    for result in results:
        result["cv_results"].to_csv(
            output_dir / f"{result['search'].lower()}_cv_results.csv",
            index=False,
        )
    plot_comparison(results, output_dir)

    serializable_results = []
    for result in results:
        serializable_results.append(
            {key: value for key, value in result.items() if key != "cv_results"}
        )
    with (output_dir / "search_summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset": {
                    "samples": args.samples,
                    "features": args.features,
                    "classes": args.classes,
                    "train_rows": len(x_train),
                    "test_rows": len(x_test),
                },
                "results": serializable_results,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n탐색 결과")
    for result in results:
        print(
            f"- {result['search']}: test_accuracy={result['test_accuracy']:.3f}, "
            f"time={result['elapsed_seconds']:.1f}s, "
            f"candidates={result['evaluated_candidates']}"
        )
        print(f"  best_params={result['best_params']}")
    print(f"결과 저장: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
