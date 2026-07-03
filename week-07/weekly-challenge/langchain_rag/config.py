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
    documents_dir: Path = REPO_ROOT / "RAG_chatbot" / "knowledge"
    model: str = "gemini-2.5-flash-lite"
    embedding_model: str = "models/gemini-embedding-2"
    api_key: str | None = None
    primary_provider: str = "anthropic"
    fallback_provider: str = "gemini"
    anthropic_model: str = "claude-haiku-4-5-20251001"
    anthropic_api_key: str | None = None
    autocomplete_provider: str = "gemini"
    autocomplete_model: str = "gemini-2.5-flash-lite"
    autocomplete_local_fallback: bool = True
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4

    @classmethod
    def from_env(cls) -> "Settings":
        documents_dir = Path(
            os.getenv("WEEK7_DOCUMENTS_DIR", "RAG_chatbot/knowledge")
        )
        if not documents_dir.is_absolute():
            documents_dir = REPO_ROOT / documents_dir
        return cls(
            documents_dir=documents_dir.resolve(),
            model=(
                os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()
                or "gemini-2.5-flash-lite"
            ),
            embedding_model=os.getenv(
                "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2"
            ).strip(),
            api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            primary_provider=os.getenv(
                "RAG_PRIMARY_PROVIDER", "anthropic"
            ).strip().lower()
            or "anthropic",
            fallback_provider=os.getenv(
                "RAG_FALLBACK_PROVIDER", "gemini"
            ).strip().lower()
            or "gemini",
            anthropic_model=os.getenv(
                "ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"
            ).strip()
            or "claude-haiku-4-5-20251001",
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            autocomplete_provider=os.getenv(
                "WEEK7_AUTOCOMPLETE_PROVIDER", "gemini"
            ).strip().lower(),
            autocomplete_model=os.getenv(
                "AUTOCOMPLETE_MODEL", "gemini-2.5-flash-lite"
            ).strip(),
            autocomplete_local_fallback=os.getenv(
                "AUTOCOMPLETE_LOCAL_FALLBACK", "true"
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            chunk_size=int(os.getenv("WEEK7_CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("WEEK7_CHUNK_OVERLAP", "120")),
            top_k=int(os.getenv("WEEK7_TOP_K", "4")),
        )
