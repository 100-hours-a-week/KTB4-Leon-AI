from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+|[.!?]")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.strip())


def detokenize(tokens: list[str]) -> str:
    text = " ".join(token for token in tokens if token not in SPECIAL_TOKENS)
    return re.sub(r"\s+([.!?])", r"\1", text).strip()


def join_prompt_and_generated(prompt: str, generated_tokens: list[str]) -> str:
    prompt = prompt.strip()
    suffix = detokenize(generated_tokens)
    if not suffix:
        return prompt
    if suffix[0] in ".!?":
        return f"{prompt}{suffix}"
    return f"{prompt} {suffix}"


@dataclass
class ModelConfig:
    context_size: int = 6
    embedding_dim: int = 96
    hidden_dim: int = 160
    num_layers: int = 1
    dropout: float = 0.0


class Vocabulary:
    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {index: token for token, index in token_to_id.items()}

    @classmethod
    def build(cls, sentences: list[list[str]], min_frequency: int = 1) -> "Vocabulary":
        counts = Counter(token for sentence in sentences for token in sentence)
        tokens = SPECIAL_TOKENS + sorted(
            token
            for token, count in counts.items()
            if count >= min_frequency and token not in SPECIAL_TOKENS
        )
        return cls({token: index for index, token in enumerate(tokens)})

    def encode(self, tokens: list[str]) -> list[int]:
        unknown_id = self.token_to_id["<unk>"]
        return [self.token_to_id.get(token, unknown_id) for token in tokens]

    def decode(self, token_ids: list[int]) -> list[str]:
        return [self.id_to_token.get(token_id, "<unk>") for token_id in token_ids]

    def __len__(self) -> int:
        return len(self.token_to_id)


class NextWordLSTM(nn.Module):
    def __init__(self, vocabulary_size: int, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            vocabulary_size,
            config.embedding_dim,
            padding_idx=0,
        )
        self.lstm = nn.LSTM(
            config.embedding_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(config.hidden_dim, vocabulary_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        outputs, _ = self.lstm(embedded)
        return self.output(outputs[:, -1, :])


def make_context(
    token_ids: list[int],
    context_size: int,
    pad_id: int,
) -> list[int]:
    context = token_ids[-context_size:]
    return [pad_id] * (context_size - len(context)) + context


def save_checkpoint(
    path: Path,
    model: NextWordLSTM,
    vocabulary: Vocabulary,
    config: ModelConfig,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocabulary": vocabulary.token_to_id,
            "config": asdict(config),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    device: torch.device | str = "cpu",
) -> tuple[NextWordLSTM, Vocabulary, ModelConfig, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    vocabulary = Vocabulary(checkpoint["vocabulary"])
    config = ModelConfig(**checkpoint["config"])
    model = NextWordLSTM(len(vocabulary), config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, vocabulary, config, checkpoint.get("metadata", {})


@torch.inference_mode()
def predict_next_words(
    model: NextWordLSTM,
    vocabulary: Vocabulary,
    config: ModelConfig,
    prompt: str,
    top_k: int = 5,
    device: torch.device | str = "cpu",
) -> list[tuple[str, float]]:
    prompt_ids = vocabulary.encode(["<bos>", *tokenize(prompt)])
    context = torch.tensor(
        [
            make_context(
                prompt_ids,
                config.context_size,
                vocabulary.token_to_id["<pad>"],
            )
        ],
        dtype=torch.long,
        device=device,
    )
    probabilities = torch.softmax(model(context)[0], dim=0)
    for special_token in ("<pad>", "<bos>", "<unk>"):
        probabilities[vocabulary.token_to_id[special_token]] = 0
    probabilities /= probabilities.sum()

    candidate_count = min(max(1, top_k), probabilities.numel())
    values, indices = torch.topk(probabilities, candidate_count)
    return [
        (vocabulary.id_to_token[token_id.item()], probability.item())
        for token_id, probability in zip(indices, values, strict=True)
    ]


@torch.inference_mode()
def generate_text(
    model: NextWordLSTM,
    vocabulary: Vocabulary,
    config: ModelConfig,
    prompt: str,
    max_new_tokens: int = 24,
    temperature: float = 0.8,
    top_k: int = 8,
    device: torch.device | str = "cpu",
) -> str:
    if not 0.1 <= temperature <= 2.0:
        raise ValueError("temperature는 0.1 이상 2.0 이하여야 합니다.")
    prompt_tokens = tokenize(prompt)
    prompt_ids = vocabulary.encode(["<bos>", *prompt_tokens])
    generated_ids: list[int] = []
    pad_id = vocabulary.token_to_id["<pad>"]
    eos_id = vocabulary.token_to_id["<eos>"]
    blocked_ids = {
        vocabulary.token_to_id["<pad>"],
        vocabulary.token_to_id["<bos>"],
        vocabulary.token_to_id["<unk>"],
    }

    for _ in range(max_new_tokens):
        context = torch.tensor(
            [make_context([*prompt_ids, *generated_ids], config.context_size, pad_id)],
            dtype=torch.long,
            device=device,
        )
        logits = model(context)[0] / temperature
        for token_id in blocked_ids:
            logits[token_id] = -torch.inf

        candidate_count = min(max(1, top_k), logits.numel())
        top_values, top_indices = torch.topk(logits, candidate_count)
        probabilities = torch.softmax(top_values, dim=0)
        sampled_index = torch.multinomial(probabilities, num_samples=1).item()
        next_id = top_indices[sampled_index].item()
        if next_id == eos_id:
            break
        generated_ids.append(next_id)
        if vocabulary.id_to_token.get(next_id) in {".", "!", "?"}:
            break

    return join_prompt_and_generated(prompt, vocabulary.decode(generated_ids))
