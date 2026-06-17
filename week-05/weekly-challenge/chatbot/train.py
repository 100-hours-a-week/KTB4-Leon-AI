from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from chatbot.model import (
    ModelConfig,
    NextWordLSTM,
    Vocabulary,
    make_context,
    save_checkpoint,
    tokenize,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_sentences(path: Path) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    sentences = [tokenize(line) for line in lines if line.strip() and not line.startswith("#")]
    return [sentence for sentence in sentences if sentence]


def build_training_tensors(
    sentences: list[list[str]],
    vocabulary: Vocabulary,
    context_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    contexts: list[list[int]] = []
    targets: list[int] = []
    pad_id = vocabulary.token_to_id["<pad>"]

    for sentence in sentences:
        sequence = vocabulary.encode(["<bos>", *sentence, "<eos>"])
        for target_index in range(1, len(sequence)):
            contexts.append(
                make_context(sequence[:target_index], context_size, pad_id)
            )
            targets.append(sequence[target_index])

    return (
        torch.tensor(contexts, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="한국어 문장에서 다음 단어를 예측하는 LSTM을 학습합니다."
    )
    parser.add_argument("--corpus", type=Path, default=base_dir / "corpus.txt")
    parser.add_argument(
        "--output",
        type=Path,
        default=base_dir.parent / "artifacts" / "chatbot.pt",
    )
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--context-size", type=int, default=6)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    sentences = load_sentences(args.corpus)
    vocabulary = Vocabulary.build(sentences)
    config = ModelConfig(
        context_size=args.context_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    )
    contexts, targets = build_training_tensors(
        sentences,
        vocabulary,
        config.context_size,
    )
    loader = DataLoader(
        TensorDataset(contexts, targets),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = NextWordLSTM(len(vocabulary), config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print(
        f"device={device}, sentences={len(sentences)}, "
        f"examples={len(contexts)}, vocabulary={len(vocabulary)}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_contexts, batch_targets in loader:
            batch_contexts = batch_contexts.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_contexts)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_targets.size(0)
            correct += (logits.argmax(dim=1) == batch_targets).sum().item()
            total += batch_targets.size(0)

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} loss={total_loss / total:.4f} "
                f"next_word_accuracy={correct / total:.3f}"
            )

    save_checkpoint(
        args.output,
        model,
        vocabulary,
        config,
        {
            "corpus": str(args.corpus),
            "sentences": len(sentences),
            "training_examples": len(contexts),
            "epochs": args.epochs,
        },
    )
    print(f"모델 저장: {args.output.resolve()}")


if __name__ == "__main__":
    main()

