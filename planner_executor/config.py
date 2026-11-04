"""Shared configuration: .env loading, model factory and safety limits.

Everything that talks to Google's Generative AI API goes through here, so the
rest of the package never has to care about API keys or model names.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load `.env` from the project root (works no matter which directory you run from).
load_dotenv(PROJECT_ROOT / ".env")


class MissingAPIKeyError(RuntimeError):
    """Raised when no Google AI Studio key is available."""


@dataclass(frozen=True)
class Settings:
    """Small, typed view over the environment variables."""

    model: str
    temperature: float
    max_steps: int
    max_revisions: int
    recursion_limit: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            temperature=float(os.getenv("GEMINI_TEMPERATURE", "0")),
            max_steps=int(os.getenv("MAX_STEPS", "8")),
            max_revisions=int(os.getenv("MAX_REVISIONS", "2")),
            recursion_limit=int(os.getenv("RECURSION_LIMIT", "60")),
        )


SETTINGS = Settings.from_env()


def get_api_key() -> str:
    """Return the Gemini key, accepting either common env-var name."""
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key or key.strip() in {"", "paste-your-key-here"}:
        raise MissingAPIKeyError(
            "No Google API key found.\n"
            "  1. Copy .env.example to .env\n"
            "  2. Set GOOGLE_API_KEY=<your key from https://aistudio.google.com/apikey>\n"
            "  3. Run the command again."
        )
    return key.strip()


def has_api_key() -> bool:
    """True when a usable-looking key is configured (used for friendly errors)."""
    try:
        get_api_key()
    except MissingAPIKeyError:
        return False
    return True


def get_llm(*, temperature: float | None = None):
    """Create a Gemini chat model.

    Imported lazily so the tutorial files can be imported/run without the
    Gemini package being configured.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=SETTINGS.model,
        api_key=get_api_key(),
        temperature=SETTINGS.temperature if temperature is None else temperature,
        max_retries=3,
        timeout=90,
    )
