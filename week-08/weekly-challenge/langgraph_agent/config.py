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
    knowledge_dir: Path = REPO_ROOT / "RAG_chatbot" / "knowledge"
    primary_provider: str = "anthropic"
    fallback_provider: str | None = "gemini"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-haiku-4-5-20251001"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash-lite"
    chunk_size: int = 800
    chunk_overlap: int = 120
    default_top_k: int = 4

    @classmethod
    def from_env(cls) -> "Settings":
        primary = os.getenv(
            "LANGGRAPH_PRIMARY_PROVIDER",
            os.getenv("RAG_PRIMARY_PROVIDER", "anthropic"),
        ).strip().lower()
        fallback = os.getenv(
            "LANGGRAPH_FALLBACK_PROVIDER",
            os.getenv("RAG_FALLBACK_PROVIDER", "gemini"),
        ).strip().lower()
        knowledge_dir = Path(
            os.getenv("LANGGRAPH_KNOWLEDGE_DIR", "RAG_chatbot/knowledge")
        )
        if not knowledge_dir.is_absolute():
            knowledge_dir = REPO_ROOT / knowledge_dir
        return cls(
            knowledge_dir=knowledge_dir.resolve(),
            primary_provider=primary or "anthropic",
            fallback_provider=fallback or None,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            anthropic_model=(
                os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001").strip()
                or "claude-haiku-4-5-20251001"
            ),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
            gemini_model=(
                os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()
                or "gemini-2.5-flash-lite"
            ),
            chunk_size=int(os.getenv("LANGGRAPH_CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("LANGGRAPH_CHUNK_OVERLAP", "120")),
            default_top_k=int(os.getenv("LANGGRAPH_TOP_K", "4")),
        )
