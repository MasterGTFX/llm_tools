from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import httpx
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from .config import DEFAULT_BASE_URL, DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT, get_access_token

MessageDict = Dict[str, str]
HistoryLike = Sequence[MessageDict]


class CodexError(RuntimeError):
    pass


class CodexAgentError(RuntimeError):
    pass


class RateLimitWindow(BaseModel):
    limit_window_seconds: int
    used_percent: float
    reset_at: int
    reset_after_seconds: int


class RateLimit(BaseModel):
    limit_reached: bool
    primary_window: RateLimitWindow
    secondary_window: RateLimitWindow


class Credits(BaseModel):
    balance: int


class CodexUsage(BaseModel):
    rate_limit: RateLimit
    plan_type: str
    credits: Credits


def _require_access_token(
    access_token: Optional[str],
    *,
    error_type: type[RuntimeError],
) -> str:
    token = access_token or get_access_token()
    if not token:
        raise error_type("OPENAI_CODEX_ACCESS_TOKEN is required")
    return token


def build_input(user_prompt: str, history: Optional[HistoryLike] = None) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []

    if history:
        for msg in history:
            role = msg.get("role")
            content = msg.get("content")
            if role not in {"user", "assistant", "system"}:
                raise ValueError(f"Unsupported history role: {role!r}")
            if not isinstance(content, str):
                raise ValueError("History message content must be a string")
            items.append({"role": role, "content": content})

    items.append({"role": "user", "content": user_prompt})
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
    token = _require_access_token(access_token, error_type=CodexError)
    return OpenAI(api_key=token, base_url=base_url, max_retries=max_retries, timeout=timeout)


def get_codex_usage(
    access_token: Optional[str] = None,
    account_id: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> CodexUsage:
    token = _require_access_token(access_token, error_type=CodexError)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id

    resp = httpx.get(
        "https://chatgpt.com/backend-api/wham/usage",
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()
    return CodexUsage.model_validate(resp.json())


def get_async_openai_client(
    access_token: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: float = DEFAULT_TIMEOUT,
) -> AsyncOpenAI:
    token = _require_access_token(access_token, error_type=CodexAgentError)
    return AsyncOpenAI(api_key=token, base_url=base_url, max_retries=max_retries, timeout=timeout)
