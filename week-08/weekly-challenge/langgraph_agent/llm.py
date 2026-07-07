from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from threading import RLock
from typing import Protocol

from .config import Settings


@dataclass(frozen=True)
class Generation:
    text: str
    provider: str
    model: str


class Generator(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        system: str,
        max_tokens: int = 600,
        temperature: float = 0.1,
    ) -> Generation: ...


class ProviderRouter:
    """Use the primary model first and stay on the fallback after a failure."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._providers = [settings.primary_provider]
        if (
            settings.fallback_provider
            and settings.fallback_provider not in self._providers
        ):
            self._providers.append(settings.fallback_provider)
        self._active_index = 0
        self._lock = RLock()

    @property
    def active_provider(self) -> str:
        with self._lock:
            return self._providers[self._active_index]

    def _anthropic(
        self, prompt: str, system: str, max_tokens: int, temperature: float
    ) -> Generation:
        if not self.settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY가 설정되지 않았습니다.")
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError("anthropic 패키지가 필요합니다.") from exc

        client = Anthropic(api_key=self.settings.anthropic_api_key)
        message = client.messages.create(
            model=self.settings.anthropic_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text
            for block in message.content
            if getattr(block, "type", "") == "text"
        ).strip()
        if not text:
            raise RuntimeError("Anthropic이 빈 응답을 반환했습니다.")
        return Generation(text, "anthropic", self.settings.anthropic_model)

    def _gemini(
        self, prompt: str, system: str, max_tokens: int, temperature: float
    ) -> Generation:
        if not self.settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("google-genai 패키지가 필요합니다.") from exc

        client = genai.Client(api_key=self.settings.gemini_api_key)
        try:
            response = client.models.generate_content(
                model=self.settings.gemini_model,
                contents=f"{system}\n\n{prompt}",
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
            text = (response.text or "").strip()
        finally:
            client.close()
        if not text:
            raise RuntimeError("Gemini가 빈 응답을 반환했습니다.")
        return Generation(text, "gemini", self.settings.gemini_model)

    def generate(
        self,
        prompt: str,
        *,
        system: str,
        max_tokens: int = 600,
        temperature: float = 0.1,
    ) -> Generation:
        errors: list[str] = []
        with self._lock:
            start = self._active_index
        for index in range(start, len(self._providers)):
            provider = self._providers[index]
            try:
                if provider == "anthropic":
                    result = self._anthropic(
                        prompt, system, max_tokens, temperature
                    )
                elif provider == "gemini":
                    result = self._gemini(prompt, system, max_tokens, temperature)
                else:
                    raise RuntimeError(f"지원하지 않는 LLM 제공자입니다: {provider}")
                with self._lock:
                    self._active_index = index
                return result
            except Exception as exc:
                errors.append(f"{provider}: {type(exc).__name__}: {str(exc)[:160]}")
                with self._lock:
                    self._active_index = min(index + 1, len(self._providers) - 1)
        raise RuntimeError("사용 가능한 LLM이 없습니다 (" + " | ".join(errors) + ").")


class OfflineGenerator:
    """Deterministic generator for tests and local graph inspection."""

    provider = "offline"
    model = "deterministic-agent-test-double"

    def generate(
        self,
        prompt: str,
        *,
        system: str,
        max_tokens: int = 600,
        temperature: float = 0.1,
    ) -> Generation:
        if "ROUTE_DECISION" in system:
            question_match = re.search(
                r"질문:\s*(.+?)\n\nJSON만 출력", prompt, re.DOTALL
            )
            question = (
                question_match.group(1) if question_match else prompt
            ).strip().lower()
            if any(term in question for term in ("문서 수", "몇 개", "인덱스 상태")):
                route = "list_documents"
            elif any(term in question for term in ("기능", "할 수", "엔드포인트")):
                route = "list_capabilities"
            else:
                route = "search_knowledge"
            return Generation(
                json.dumps({"route": route, "reason": "offline routing"}),
                self.provider,
                self.model,
            )

        if "CITATION_REWRITE" in system:
            answer_match = re.search(r"기존 답변:\s*(.+?)\n\n", prompt, re.DOTALL)
            answer = answer_match.group(1).strip() if answer_match else prompt.strip()
            if "[출처" not in answer:
                answer += " [출처 1]"
            return Generation(answer, self.provider, self.model)

        context_match = re.search(r"도구 실행 결과:\s*(.+)", prompt, re.DOTALL)
        context = context_match.group(1).strip() if context_match else ""
        lines = [
            line.strip(" #-\t")
            for line in context.splitlines()
            if line.strip() and not line.startswith("[출처")
        ]
        answer = lines[0] if lines else "관련 정보가 없습니다."
        if "[출처" not in answer and context:
            answer += " [출처 1]"
        return Generation(answer, self.provider, self.model)


def create_generator(settings: Settings) -> Generator:
    if settings.primary_provider == "offline" or os.getenv(
        "LANGGRAPH_OFFLINE", ""
    ).lower() in {"1", "true", "yes"}:
        return OfflineGenerator()
    return ProviderRouter(settings)
