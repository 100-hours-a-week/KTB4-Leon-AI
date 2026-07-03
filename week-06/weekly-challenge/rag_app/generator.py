from __future__ import annotations

import json
import re
from collections.abc import Iterator
from threading import RLock
from typing import Protocol

import httpx

from .retriever import SearchResult


class Generator(Protocol):
    provider: str
    model: str

    def generate(self, prompt: str) -> str: ...

    def stream(self, prompt: str) -> Iterator[str]: ...


def build_prompt(question: str, results: list[SearchResult]) -> str:
    context = "\n\n".join(
        f"[출처 {index}: {result.chunk.source}]\n{result.chunk.text}"
        for index, result in enumerate(results, start=1)
    )
    return f"""당신은 검색된 문서를 근거로 답하는 한국어 질의응답 도우미입니다.
아래 문맥은 관련성 기준을 통과한 검색 결과입니다.
질문이 짧거나 문법적으로 불완전하면 검색 문맥을 기준으로 가장 자연스러운 의도를 해석하세요.
문맥에 질문과 관련된 내용이 있으면 핵심 답부터 분명하게 설명하세요.
'직접적인 답을 찾지 못했습니다' 같은 회피성 서론을 붙이지 마세요.
검색 문맥에 없는 사실은 추가하지 말고, 근거로 사용한 문장 끝에 [출처 N]을 표시하세요.

검색 문맥:
{context}

질문: {question}
답변:"""


class GeminiGenerator:
    provider = "gemini"

    def __init__(self, api_key: str | None, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def _client(self):
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "Gemini SDK가 없습니다. `pip install google-genai`를 실행하세요."
            ) from exc
        return genai.Client(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        client = self._client()
        try:
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={"temperature": 0.1, "max_output_tokens": 800},
                )
            except Exception as exc:
                raise RuntimeError(f"Gemini 호출에 실패했습니다: {exc}") from exc
            return (response.text or "").strip()
        finally:
            client.close()

    def stream(self, prompt: str) -> Iterator[str]:
        client = self._client()
        try:
            try:
                for chunk in client.models.generate_content_stream(
                    model=self.model,
                    contents=prompt,
                    config={"temperature": 0.1, "max_output_tokens": 800},
                ):
                    if chunk.text:
                        yield chunk.text
            except Exception as exc:
                raise RuntimeError(f"Gemini 스트리밍에 실패했습니다: {exc}") from exc
        finally:
            client.close()


class AnthropicGenerator:
    provider = "anthropic"

    def __init__(self, api_key: str | None, model: str, base_url: str) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    @property
    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY가 설정되지 않았습니다.")
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _payload(self, prompt: str, stream: bool) -> dict:
        return {
            "model": self.model,
            "max_tokens": 800,
            "temperature": 0.1,
            "stream": stream,
            "messages": [{"role": "user", "content": prompt}],
        }

    def generate(self, prompt: str) -> str:
        try:
            response = httpx.post(
                f"{self.base_url}/v1/messages",
                headers=self._headers,
                json=self._payload(prompt, stream=False),
                timeout=120,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Anthropic 호출에 실패했습니다: {exc}") from exc
        blocks = response.json().get("content", [])
        return "".join(
            str(block.get("text", ""))
            for block in blocks
            if block.get("type") == "text"
        ).strip()

    def stream(self, prompt: str) -> Iterator[str]:
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/v1/messages",
                headers=self._headers,
                json=self._payload(prompt, stream=True),
                timeout=120,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    if event.get("type") == "content_block_delta":
                        token = event.get("delta", {}).get("text")
                        if token:
                            yield str(token)
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Anthropic 스트리밍에 실패했습니다: {exc}") from exc


class FallbackGenerator:
    """Switch permanently to the next provider after the active one fails."""

    def __init__(self, generators: list[Generator]) -> None:
        if not generators:
            raise ValueError("생성 제공자가 하나 이상 필요합니다.")
        self._generators = generators
        self._active_index = 0
        self._lock = RLock()

    @property
    def provider(self) -> str:
        return self._generators[self._active_index].provider

    @property
    def model(self) -> str:
        return self._generators[self._active_index].model

    def _candidates(self) -> list[tuple[int, Generator]]:
        with self._lock:
            remaining = self._generators[self._active_index :]
            return list(enumerate(remaining, self._active_index))

    def generate(self, prompt: str) -> str:
        last_error: RuntimeError | None = None
        for index, generator in self._candidates():
            try:
                answer = generator.generate(prompt)
                with self._lock:
                    self._active_index = index
                return answer
            except RuntimeError as exc:
                last_error = exc
                with self._lock:
                    self._active_index = min(index + 1, len(self._generators) - 1)
        raise last_error or RuntimeError("사용 가능한 생성 제공자가 없습니다.")

    def stream(self, prompt: str) -> Iterator[str]:
        last_error: RuntimeError | None = None
        for index, generator in self._candidates():
            emitted = False
            try:
                for token in generator.stream(prompt):
                    emitted = True
                    yield token
                with self._lock:
                    self._active_index = index
                return
            except RuntimeError as exc:
                if emitted:
                    raise
                last_error = exc
                with self._lock:
                    self._active_index = min(index + 1, len(self._generators) - 1)
        raise last_error or RuntimeError("사용 가능한 생성 제공자가 없습니다.")


class OllamaGenerator:
    provider = "ollama"

    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    @property
    def _payload(self) -> dict:
        return {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 800},
        }

    def generate(self, prompt: str) -> str:
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={**self._payload, "prompt": prompt},
                timeout=120,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"로컬 Ollama 호출에 실패했습니다: {exc}") from exc
        return str(response.json().get("response", "")).strip()

    def stream(self, prompt: str) -> Iterator[str]:
        payload = {**self._payload, "prompt": prompt, "stream": True}
        try:
            with httpx.stream(
                "POST", f"{self.base_url}/api/generate", json=payload, timeout=120
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    token = json.loads(line).get("response")
                    if token:
                        yield str(token)
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"로컬 Ollama 스트리밍에 실패했습니다: {exc}") from exc


class OfflineExtractiveGenerator:
    """Deterministic test double used when API calls are intentionally disabled."""

    provider = "offline"
    model = "extractive-test-double"

    def generate(self, prompt: str) -> str:
        question_match = re.search(r"질문:\s*(.+?)\n답변:", prompt, re.DOTALL)
        question = question_match.group(1) if question_match else ""
        question_terms = set(re.findall(r"[가-힣A-Za-z0-9]{2,}", question.lower()))
        contexts = re.findall(
            r"\[출처 (\d+): ([^\]]+)\]\n(.*?)(?=\n\n\[출처|\n\n질문:)",
            prompt,
            re.DOTALL,
        )
        candidates: list[tuple[int, str, str]] = []
        for source_number, _, text in contexts:
            for sentence in re.split(r"(?<=[.!?다])\s+|\n+", text):
                clean = sentence.strip(" #-\t")
                if not clean:
                    continue
                terms = set(re.findall(r"[가-힣A-Za-z0-9]{2,}", clean.lower()))
                candidates.append((len(question_terms & terms), source_number, clean))
        if not candidates:
            return "제공된 문서에서 답을 찾지 못했습니다."
        score, source_number, sentence = max(candidates, key=lambda item: item[0])
        if score == 0:
            return "제공된 문서에서 답을 찾지 못했습니다."
        return f"{sentence} [출처 {source_number}]"

    def stream(self, prompt: str) -> Iterator[str]:
        answer = self.generate(prompt)
        for index in range(0, len(answer), 24):
            yield answer[index : index + 24]


def create_generator(
    provider: str,
    api_key: str | None,
    model: str,
    ollama_base_url: str = "http://127.0.0.1:11434",
    anthropic_base_url: str = "https://api.anthropic.com",
) -> Generator:
    if provider == "gemini":
        return GeminiGenerator(api_key=api_key, model=model)
    if provider == "offline":
        return OfflineExtractiveGenerator()
    if provider == "ollama":
        return OllamaGenerator(model=model, base_url=ollama_base_url)
    if provider == "anthropic":
        return AnthropicGenerator(
            api_key=api_key, model=model, base_url=anthropic_base_url
        )
    raise ValueError(f"지원하지 않는 RAG_LLM_PROVIDER입니다: {provider}")
