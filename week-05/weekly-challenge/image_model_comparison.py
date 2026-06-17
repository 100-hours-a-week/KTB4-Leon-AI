from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from plot_style import COLORS, TOKENS, add_chart_header, finish_axis, use_chart_theme


@dataclass
class ModelResult:
    model: str
    best_epoch: int
    best_validation_accuracy: float
    test_accuracy: float
    test_loss: float
    elapsed_seconds: float
    trainable_parameters: int
    total_parameters: int
    history: list[dict[str, float]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def require_torchvision():
    try:
        from torchvision import datasets, models, transforms
    except ImportError as exc:
        raise RuntimeError(
            "torchvision이 필요합니다. `python -m pip install torchvision`을 실행하세요."
        ) from exc
    return datasets, models, transforms


def subset_indices(total: int, limit: int | None, seed: int) -> list[int]:
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    return indices if limit is None else indices[: min(limit, total)]


def split_indices(
    total: int,
    train_limit: int | None,
    validation_limit: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    indices = subset_indices(total, None, seed)
    validation_count = min(validation_limit, max(1, total // 5))
    validation_indices = indices[:validation_count]
    train_indices = indices[validation_count:]
    if train_limit is not None:
        train_indices = train_indices[: min(train_limit, len(train_indices))]
    return train_indices, validation_indices


def build_datasets(args: argparse.Namespace) -> tuple[Dataset, Dataset, Dataset, list[str]]:
    datasets, _, transforms = require_torchvision()
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    if args.dataset == "cifar10":
        raw_train = datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            download=True,
            transform=train_transform,
        )
        raw_validation = datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            download=True,
            transform=evaluation_transform,
        )
        raw_test = datasets.CIFAR10(
            root=args.data_dir,
            train=False,
            download=True,
            transform=evaluation_transform,
        )
        train_indices, validation_indices = split_indices(
            len(raw_train), args.train_limit, args.validation_limit, args.seed
        )
        test_indices = subset_indices(len(raw_test), args.test_limit, args.seed + 1)
        return (
            Subset(raw_train, train_indices),
            Subset(raw_validation, validation_indices),
            Subset(raw_test, test_indices),
            list(raw_train.classes),
        )

    if args.dataset == "imagefolder":
        if args.image_dir is None:
            raise ValueError("imagefolder 사용 시 `--image-dir`가 필요합니다.")
        image_root = Path(args.image_dir)
        source_root = image_root / "train" if (image_root / "train").exists() else image_root
        raw_train = datasets.ImageFolder(source_root, transform=train_transform)
        raw_evaluation = datasets.ImageFolder(source_root, transform=evaluation_transform)
        train_indices, validation_indices = split_indices(
            len(raw_train), args.train_limit, args.validation_limit, args.seed
        )

        test_root = image_root / "test"
        if test_root.exists():
            raw_test = datasets.ImageFolder(test_root, transform=evaluation_transform)
            if raw_test.classes != raw_train.classes:
                raise ValueError("train/test 폴더의 클래스 이름이 일치해야 합니다.")
            test_dataset: Dataset = Subset(
                raw_test,
                subset_indices(len(raw_test), args.test_limit, args.seed + 1),
            )
        else:
            test_count = min(args.test_limit, max(1, len(validation_indices) // 2))
            test_indices = validation_indices[:test_count]
            validation_indices = validation_indices[test_count:]
            test_dataset = Subset(raw_evaluation, test_indices)

        return (
            Subset(raw_train, train_indices),
            Subset(raw_evaluation, validation_indices),
            test_dataset,
            list(raw_train.classes),
        )

    raw_train = datasets.FakeData(
        size=max(args.train_limit or 128, 32),
        image_size=(3, 224, 224),
        num_classes=args.fake_classes,
        transform=train_transform,
        random_offset=args.seed,
    )
    raw_validation = datasets.FakeData(
        size=max(args.validation_limit, 16),
        image_size=(3, 224, 224),
        num_classes=args.fake_classes,
        transform=evaluation_transform,
        random_offset=args.seed + 10_000,
    )
    raw_test = datasets.FakeData(
        size=max(args.test_limit, 16),
        image_size=(3, 224, 224),
        num_classes=args.fake_classes,
        transform=evaluation_transform,
        random_offset=args.seed + 20_000,
    )
    return raw_train, raw_validation, raw_test, [
        f"class_{index}" for index in range(args.fake_classes)
    ]


def build_loaders(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    workers: int,
    device: torch.device,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    loader_args = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    return (
        DataLoader(
            train_dataset,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            **loader_args,
        ),
        DataLoader(validation_dataset, shuffle=False, **loader_args),
        DataLoader(test_dataset, shuffle=False, **loader_args),
    )


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool,
    fine_tune: bool,
) -> nn.Module:
    _, models, _ = require_torchvision()

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if fine_tune:
            for parameter in model.layer4.parameters():
                parameter.requires_grad = True
        return model

    if name == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        if fine_tune:
            for parameter in model.features[-5:].parameters():
                parameter.requires_grad = True
        return model

    raise ValueError(f"지원하지 않는 모델입니다: {name}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, list[int], list[int]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        total += batch_size
        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    return total_loss / total, correct / total, all_labels, all_predictions


def train_model(
    name: str,
    loaders: tuple[DataLoader, DataLoader, DataLoader],
    class_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> ModelResult:
    train_loader, validation_loader, test_loader = loaders
    set_seed(args.seed)
    model = build_model(name, len(class_names), args.pretrained, args.fine_tune).to(device)
    set_seed(args.seed)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    history: list[dict[str, float]] = []
    best_accuracy = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    started_at = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        validation_loss, validation_accuracy, _, _ = run_epoch(
            model, validation_loader, criterion, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        print(
            f"[{name}] epoch {epoch:02d}/{args.epochs} "
            f"train_acc={train_accuracy:.3f} val_acc={validation_accuracy:.3f}"
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("학습 결과를 저장하지 못했습니다.")
    model.load_state_dict(best_state)
    test_loss, test_accuracy, labels, predictions = run_epoch(
        model, test_loader, criterion, device
    )
    elapsed_seconds = time.perf_counter() - started_at

    torch.save(
        {
            "model_name": name,
            "class_names": class_names,
            "state_dict": best_state,
            "pretrained": args.pretrained,
            "fine_tune": args.fine_tune,
        },
        output_dir / f"{name}_best.pt",
    )
    plot_confusion_matrix(name, labels, predictions, class_names, output_dir)
    plot_training_history(name, history, output_dir)

    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return ModelResult(
        model=name,
        best_epoch=best_epoch,
        best_validation_accuracy=best_accuracy,
        test_accuracy=test_accuracy,
        test_loss=test_loss,
        elapsed_seconds=elapsed_seconds,
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
        history=history,
    )


def plot_training_history(
    model_name: str,
    history: list[dict[str, float]],
    output_dir: Path,
) -> None:
    use_chart_theme()
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.28)

    axes[0].plot(
        epochs,
        [row["train_loss"] for row in history],
        color=COLORS["blue"]["base"],
        marker="o",
        label="Train",
    )
    axes[0].plot(
        epochs,
        [row["validation_loss"] for row in history],
        color=COLORS["orange"]["base"],
        marker="s",
        linestyle="--",
        label="Validation",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend(frameon=False)
    finish_axis(axes[0])

    axes[1].plot(
        epochs,
        [row["train_accuracy"] * 100 for row in history],
        color=COLORS["blue"]["base"],
        marker="o",
        label="Train",
    )
    axes[1].plot(
        epochs,
        [row["validation_accuracy"] * 100 for row in history],
        color=COLORS["orange"]["base"],
        marker="s",
        linestyle="--",
        label="Validation",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 100)
    axes[1].legend(frameon=False)
    finish_axis(axes[1])

    add_chart_header(
        fig,
        axes[0],
        f"{model_name} 학습 과정",
        "같은 데이터 분할에서 학습 손실과 검증 정확도를 epoch별로 비교",
    )
    fig.savefig(output_dir / f"{model_name}_history.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    model_name: str,
    labels: list[int],
    predictions: list[int],
    class_names: list[str],
    output_dir: Path,
) -> None:
    use_chart_theme()
    matrix = confusion_matrix(labels, predictions, labels=range(len(class_names)))
    matrix = matrix.astype(float) / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        matrix,
        cmap=sns.blend_palette(
            [TOKENS["panel"], COLORS["blue"]["light"], COLORS["blue"]["base"]],
            as_cmap=True,
        ),
        vmin=0,
        vmax=1,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor=TOKENS["panel"],
        ax=ax,
        cbar_kws={"label": "Row-normalized ratio"},
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    add_chart_header(
        fig,
        ax,
        f"{model_name} 클래스별 예측 결과",
        "행은 실제 클래스, 열은 모델 예측 클래스이며 각 행의 합은 1",
    )
    fig.savefig(
        output_dir / f"{model_name}_confusion_matrix.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_model_comparison(results: Iterable[ModelResult], output_dir: Path) -> None:
    result_list = list(results)
    if not result_list:
        return
    use_chart_theme()
    model_labels = [result.model for result in result_list]
    palette = [COLORS["blue"], COLORS["orange"]]
    colors = [palette[index % len(palette)]["base"] for index in range(len(result_list))]
    edges = [palette[index % len(palette)]["dark"] for index in range(len(result_list))]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    bars = axes[0].bar(
        model_labels,
        [result.test_accuracy * 100 for result in result_list],
        color=colors,
        edgecolor=edges,
    )
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].bar_label(bars, fmt="%.1f%%", padding=4)
    finish_axis(axes[0])

    bars = axes[1].bar(
        model_labels,
        [result.elapsed_seconds for result in result_list],
        color=colors,
        edgecolor=edges,
    )
    axes[1].set_ylabel("Training and evaluation time (seconds)")
    axes[1].bar_label(bars, fmt="%.1fs", padding=4)
    finish_axis(axes[1])

    chart_title = (
        "ResNet18과 VGG16 성능 비교"
        if len(result_list) > 1
        else f"{result_list[0].model} 성능 요약"
    )
    add_chart_header(
        fig,
        axes[0],
        chart_title,
        "동일한 데이터 분할·epoch·평가지표를 사용한 테스트 정확도와 전체 실행 시간",
    )
    fig.subplots_adjust(wspace=0.32)
    fig.savefig(output_dir / "model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="동일한 이미지 데이터셋에서 ResNet18과 VGG16 전이 학습을 비교합니다."
    )
    parser.add_argument(
        "--dataset",
        choices=("cifar10", "imagefolder", "fake"),
        default="cifar10",
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--image-dir")
    parser.add_argument(
        "--models",
        choices=("both", "resnet18", "vgg16"),
        default="both",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--validation-limit", type=int, default=1000)
    parser.add_argument("--test-limit", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="./artifacts/image_models")
    parser.add_argument("--fake-classes", type=int, default=3)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fine-tune",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}, dataset={args.dataset}, pretrained={args.pretrained}")

    train_dataset, validation_dataset, test_dataset, class_names = build_datasets(args)
    model_names = ["resnet18", "vgg16"] if args.models == "both" else [args.models]
    results = []
    for name in model_names:
        loaders = build_loaders(
            train_dataset,
            validation_dataset,
            test_dataset,
            args.batch_size,
            args.workers,
            device,
            args.seed,
        )
        results.append(
            train_model(name, loaders, class_names, args, device, output_dir)
        )
    plot_model_comparison(results, output_dir)

    payload = {
        "dataset": args.dataset,
        "class_names": class_names,
        "device": str(device),
        "pretrained": args.pretrained,
        "fine_tune": args.fine_tune,
        "results": [asdict(result) for result in results],
    }
    with (output_dir / "comparison_results.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print("\n모델 비교")
    for result in results:
        print(
            f"- {result.model}: test_accuracy={result.test_accuracy:.3f}, "
            f"time={result.elapsed_seconds:.1f}s, best_epoch={result.best_epoch}"
        )
    print(f"결과 저장: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
