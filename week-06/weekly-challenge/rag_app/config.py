from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_DIR.parents[2]
load_dotenv(REPO_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    documents_dir: Path = PROJECT_DIR.parent / "README.md"
    provider: str = "gemini"
    model: str = "gemini-2.5-flash-lite"
    api_key: str | None = None
    fallback_provider: str | None = None
    fallback_model: str | None = None
    fallback_api_key: str | None = None
    anthropic_base_url: str = "https://api.anthropic.com"
    ollama_base_url: str = "http://127.0.0.1:11434"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    default_top_k: int = 4
    min_relevance_score: float = 0.05

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.getenv(
            "RAG_PRIMARY_PROVIDER", os.getenv("RAG_LLM_PROVIDER", "gemini")
        ).strip().lower()

        def model_for(name: str) -> str:
            if name == "anthropic":
                return os.getenv(
                    "ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"
                ).strip() or "claude-haiku-4-5-20251001"
            if name == "ollama":
                return os.getenv("OLLAMA_MODEL", "gemma4:e2b").strip() or "gemma4:e2b"
            return os.getenv(
                "GEMINI_MODEL", "gemini-2.5-flash-lite"
            ).strip() or "gemini-2.5-flash-lite"

        def api_key_for(name: str) -> str | None:
            if name == "anthropic":
                return os.getenv("ANTHROPIC_API_KEY") or None
            if name == "gemini":
                return os.getenv("GEMINI_API_KEY") or None
            return None

        fallback_name = os.getenv(
            "RAG_FALLBACK_PROVIDER", "gemini" if provider == "anthropic" else ""
        ).strip().lower()
        fallback_provider = (
            fallback_name if fallback_name and fallback_name != provider else None
        )
        document_path = Path(
            os.getenv("WEEK6_DOCUMENT_PATH", "weeks/week-06/README.md")
        )
        if not document_path.is_absolute():
            document_path = REPO_ROOT / document_path
        return cls(
            documents_dir=document_path.resolve(),
            provider=provider,
            model=model_for(provider),
            api_key=api_key_for(provider),
            fallback_provider=fallback_provider,
            fallback_model=(model_for(fallback_provider) if fallback_provider else None),
            fallback_api_key=(
                api_key_for(fallback_provider) if fallback_provider else None
            ),
            anthropic_base_url=os.getenv(
                "ANTHROPIC_BASE_URL", "https://api.anthropic.com"
            ).rstrip("/"),
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
            ).rstrip("/"),
            chunk_size=int(os.getenv("WEEK6_CHUNK_SIZE", "1200")),
            chunk_overlap=int(os.getenv("WEEK6_CHUNK_OVERLAP", "200")),
            default_top_k=int(os.getenv("WEEK6_TOP_K", "4")),
            min_relevance_score=float(
                os.getenv("WEEK6_MIN_RELEVANCE_SCORE", "0.05")
            ),
        )
