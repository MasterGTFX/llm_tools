from __future__ import annotations

import asyncio
import importlib.util
import os

import pytest

from llm_tools import AsyncLLMClient, AuthenticationError, CodexLocalAuth, Usage


def _has_codex_auth() -> bool:
    try:
        CodexLocalAuth().get_token()
    except AuthenticationError:
        return False
    return True


def _has_codex_dependencies() -> bool:
    return all(importlib.util.find_spec(package) is not None for package in ("agents", "openai", "httpx"))


pytestmark = pytest.mark.skipif(
    not _has_codex_auth() or not _has_codex_dependencies(),
    reason=(
        "Set OPENAI_CODEX_ACCESS_TOKEN or sign in with Codex local auth, "
        "and install Codex provider dependencies to run real Codex e2e tests"
    ),
)


def test_codex_text_e2e() -> None:
    async def run() -> None:
        from llm_tools import CodexProvider

        client = AsyncLLMClient(
            provider=CodexProvider(),
            model=os.getenv("OPENAI_CODEX_E2E_MODEL", "gpt-5.5"),
            system_prompt="Follow exact output instructions and do not add extra text.",
            timeout=float(os.getenv("OPENAI_CODEX_E2E_TIMEOUT", "120")),
        )
        client.reset()

        result = await client.text("Reply exactly: llm-tools-e2e-text")

        assert result.output == "llm-tools-e2e-text"
        assert "llm-tools-e2e-text" in repr(result.conversation.items)

    asyncio.run(run())


def test_codex_usage_e2e() -> None:
    async def run() -> None:
        from llm_tools import CodexProvider

        client = AsyncLLMClient(provider=CodexProvider())

        usage = await client.usage()

        assert isinstance(usage, Usage)
        assert usage.raw

    asyncio.run(run())
