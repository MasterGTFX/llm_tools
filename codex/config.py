from __future__ import annotations

import os

DEFAULT_BASE_URL = os.getenv("OPENAI_CODEX_BASE_URL", "https://api.openai.com/v1")
DEFAULT_MODEL = os.getenv("OPENAI_CODEX_MODEL", "gpt-5")
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "OPENAI_CODEX_SYSTEM_PROMPT",
    "You are a helpful coding assistant. Answer concisely.",
)
DEFAULT_MAX_RETRIES = int(os.getenv("OPENAI_CODEX_MAX_RETRIES", "2"))
DEFAULT_TIMEOUT = float(os.getenv("OPENAI_CODEX_TIMEOUT", "60"))


def get_access_token() -> str:
    return os.getenv("OPENAI_CODEX_ACCESS_TOKEN", "")
