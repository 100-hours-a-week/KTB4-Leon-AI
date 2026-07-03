from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from google import genai
from pydantic import BaseModel, Field


WORD_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+")


class CandidatePayload(BaseModel):
    candidates: list[str] = Field(min_length=1, max_length=20)


class CompletionPayload(BaseModel):
    completion: str = Field(min_length=1)


def normalize_candidate(value: str) -> str:
    match = WORD_PATTERN.search(value.strip())
    return match.group(0) if match else ""


class GeminiAutocomplete:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.client = client or genai.Client(api_key=api_key)

    @lru_cache(maxsize=512)
    def predict_next_words(self, prompt: str, top_k: int) -> tuple[str, ...]:
        response = self.client.models.generate_content(
            model=self.model,
            contents=f"입력 문장: {prompt!r}\n후보 개수: {top_k}",
            config={
                "system_instruction": (
                    "당신은 한국어 입력기의 다음 어절 추천기입니다. "
                    "입력 문장을 답변하거나 고치지 말고, 바로 뒤에 자연스럽게 "
                    "올 수 있는 서로 다른 한 어절 후보만 순위대로 반환하세요. "
                    "후보에는 공백과 설명을 넣지 마세요."
                ),
                "temperature": 0.35,
                "max_output_tokens": 120,
                "response_mime_type": "application/json",
                "response_schema": CandidatePayload,
                "thinking_config": {"thinking_budget": 0},
            },
        )
        payload = response.parsed
        if not isinstance(payload, CandidatePayload):
            payload = CandidatePayload.model_validate_json(response.text or "{}")

        last_word = prompt.rstrip().split()[-1] if prompt.rstrip() else ""
        words = []
        for value in payload.candidates:
            word = normalize_candidate(value)
            if word and word != last_word and word not in words:
                words.append(word)
            if len(words) >= top_k:
                break
        if not words:
            raise ValueError("Gemini가 유효한 다음 단어 후보를 반환하지 않았습니다.")
        return tuple(words)

    @lru_cache(maxsize=256)
    def continue_sentence(self, prompt: str, max_new_words: int) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=(
                f"입력 문장: {prompt!r}\n"
                f"추가할 최대 어절 수: {max_new_words}"
            ),
            config={
                "system_instruction": (
                    "당신은 한국어 문장 자동완성기입니다. 입력 문장을 그대로 "
                    "유지하고 그 뒤를 자연스럽게 이어 한 문장으로 완성하세요. "
                    "질문에 답하거나 설명하지 말고 완성된 문장만 반환하세요."
                ),
                "temperature": 0.45,
                "max_output_tokens": 160,
                "response_mime_type": "application/json",
                "response_schema": CompletionPayload,
                "thinking_config": {"thinking_budget": 0},
            },
        )
        payload = response.parsed
        if not isinstance(payload, CompletionPayload):
            payload = CompletionPayload.model_validate_json(response.text or "{}")

        completion = payload.completion.strip()
        if completion.startswith(prompt.strip()):
            return completion
        return f"{prompt.rstrip()} {completion}".strip()
