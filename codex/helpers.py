from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from openai import AsyncOpenAI, OpenAI

from .config import DEFAULT_BASE_URL, DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT, get_access_token

MessageDict = Dict[str, str]
HistoryLike = Sequence[MessageDict]


class CodexError(RuntimeError):
    pass


class CodexAgentError(RuntimeError):
    pass


def build_input(user_prompt: str, history: Optional[HistoryLike] = None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    if history:
        for msg in history:
            role = msg.get("role")
            content = msg.get("content")
            if role not in {"user", "assistant", "system"}:
                raise ValueError(f"Unsupported history role: {role!r}")
            if not isinstance(content, str):
                raise ValueError("History message content must be a string")
            items.append(
                {
                    "role": role,
                    "content": [{"type": "input_text", "text": content}],
                }
            )

    items.append(
        {
            "role": "user",
            "content": [{"type": "input_text", "text": user_prompt}],
        }
    )
    return items


def updated_history(
    user_prompt: str,
    assistant_text: str,
    history: Optional[HistoryLike] = None,
) -> List[MessageDict]:
    new_history = list(history or [])
    new_history.append({"role": "user", "content": user_prompt})
    new_history.append({"role": "assistant", "content": assistant_text})
    return new_history


def get_openai_client(
    access_token: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: float = DEFAULT_TIMEOUT,
) -> OpenAI:
    token = access_token or get_access_token()
    if not token:
        raise CodexError("OPENAI_CODEX_ACCESS_TOKEN is required")

    return OpenAI(api_key=token, base_url=base_url, max_retries=max_retries, timeout=timeout)


def get_async_openai_client(
    access_token: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: float = DEFAULT_TIMEOUT,
) -> AsyncOpenAI:
    token = access_token or get_access_token()
    if not token:
        raise CodexAgentError("OPENAI_CODEX_ACCESS_TOKEN is required")

    return AsyncOpenAI(api_key=token, base_url=base_url, max_retries=max_retries, timeout=timeout)
