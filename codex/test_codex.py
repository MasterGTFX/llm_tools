from __future__ import annotations

import asyncio
import os
from typing import Any, List

import pytest
from agents import function_tool
from pydantic import BaseModel

from codex import (
    DEFAULT_MODEL,
    CodexUsage,
    codex_agent_generate_model,
    codex_agent_generate_model_async,
    codex_agent_generate_text,
    codex_agent_generate_text_async,
    codex_generate_model,
    codex_generate_model_async,
    codex_generate_text,
    codex_generate_text_async,
    get_codex_usage,
)

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_CODEX_ACCESS_TOKEN"),
    reason="OPENAI_CODEX_ACCESS_TOKEN is required for real e2e API tests",
)

E2E_MODEL = os.getenv("OPENAI_CODEX_E2E_MODEL", DEFAULT_MODEL)
E2E_TIMEOUT = float(os.getenv("OPENAI_CODEX_E2E_TIMEOUT", "120"))
EXACT_SYSTEM_PROMPT = (
    "You are running deterministic end-to-end tests. "
    "Follow exact output instructions and do not add extra text."
)


class E2EAnswer(BaseModel):
    marker: str
    answer: str


def _access_token() -> str:
    token = os.getenv("OPENAI_CODEX_ACCESS_TOKEN")
    assert token
    return token


def _assert_contains_marker(text: str, marker: str) -> None:
    assert isinstance(text, str)
    assert marker in text


def _assert_responses_history(history: List[dict[str, str]], turns: int, marker: str) -> None:
    assert len(history) == turns * 2
    assert [item["role"] for item in history] == ["user", "assistant"] * turns
    assert marker in repr(history)


def _assert_agent_history(history: List[dict[str, Any]], marker: str) -> None:
    assert history
    assert any(item.get("role") == "user" for item in history)
    assert marker in repr(history)


@function_tool
def e2e_lookup_marker(key: str) -> str:
    """Return deterministic marker values for e2e tests."""
    markers = {
        "text": "codex-e2e-agent-text",
        "model": "codex-e2e-agent-model",
    }
    return markers[key]


def test_codex_generate_text_e2e_one_turn_then_two_turn() -> None:
    token = _access_token()
    marker = "codex-e2e-responses-text"

    first, history = codex_generate_text(
        f"Reply exactly: {marker}",
        access_token=token,
        model=E2E_MODEL,
        system_prompt=EXACT_SYSTEM_PROMPT,
        timeout=E2E_TIMEOUT,
        return_history=True,
    )
    _assert_contains_marker(first, marker)
    _assert_responses_history(history, 1, marker)

    second, history = codex_generate_text(
        "What marker did you just return? Reply exactly with that marker.",
        access_token=token,
        model=E2E_MODEL,
        system_prompt=EXACT_SYSTEM_PROMPT,
        timeout=E2E_TIMEOUT,
        history=history,
        return_history=True,
    )
    _assert_contains_marker(second, marker)
    _assert_responses_history(history, 2, marker)


def test_codex_generate_model_e2e_one_turn_then_two_turn() -> None:
    token = _access_token()
    marker = "codex-e2e-responses-model"

    first, history = codex_generate_model(
        f"Return marker={marker!r} and answer='ready'.",
        E2EAnswer,
        access_token=token,
        model=E2E_MODEL,
        system_prompt=EXACT_SYSTEM_PROMPT,
        timeout=E2E_TIMEOUT,
        return_history=True,
    )
    assert first.marker == marker
    assert first.answer == "ready"
    _assert_responses_history(history, 1, marker)

    second, history = codex_generate_model(
        "Using the previous turn, return the same marker and answer='remembered'.",
        E2EAnswer,
        access_token=token,
        model=E2E_MODEL,
        system_prompt=EXACT_SYSTEM_PROMPT,
        timeout=E2E_TIMEOUT,
        history=history,
        return_history=True,
    )
    assert second.marker == marker
    assert second.answer == "remembered"
    _assert_responses_history(history, 2, marker)


def test_codex_generate_text_async_e2e_one_turn_then_two_turn() -> None:
    async def run_test() -> None:
        token = _access_token()
        marker = "codex-e2e-responses-async-text"

        first, history = await codex_generate_text_async(
            f"Reply exactly: {marker}",
            access_token=token,
            model=E2E_MODEL,
            system_prompt=EXACT_SYSTEM_PROMPT,
            timeout=E2E_TIMEOUT,
            return_history=True,
        )
        _assert_contains_marker(first, marker)
        _assert_responses_history(history, 1, marker)

        second, history = await codex_generate_text_async(
            "What marker did you just return? Reply exactly with that marker.",
            access_token=token,
            model=E2E_MODEL,
            system_prompt=EXACT_SYSTEM_PROMPT,
            timeout=E2E_TIMEOUT,
            history=history,
            return_history=True,
        )
        _assert_contains_marker(second, marker)
        _assert_responses_history(history, 2, marker)

    asyncio.run(run_test())


def test_codex_generate_model_async_e2e_one_turn_then_two_turn() -> None:
    async def run_test() -> None:
        token = _access_token()
        marker = "codex-e2e-responses-async-model"

        first, history = await codex_generate_model_async(
            f"Return marker={marker!r} and answer='ready'.",
            E2EAnswer,
            access_token=token,
            model=E2E_MODEL,
            system_prompt=EXACT_SYSTEM_PROMPT,
            timeout=E2E_TIMEOUT,
            return_history=True,
        )
        assert first.marker == marker
        assert first.answer == "ready"
        _assert_responses_history(history, 1, marker)

        second, history = await codex_generate_model_async(
            "Using the previous turn, return the same marker and answer='remembered'.",
            E2EAnswer,
            access_token=token,
            model=E2E_MODEL,
            system_prompt=EXACT_SYSTEM_PROMPT,
            timeout=E2E_TIMEOUT,
            history=history,
            return_history=True,
        )
        assert second.marker == marker
        assert second.answer == "remembered"
        _assert_responses_history(history, 2, marker)

    asyncio.run(run_test())


def test_codex_agent_generate_text_e2e_one_turn_then_two_turn() -> None:
    token = _access_token()
    marker = "codex-e2e-agent-text"

    first, history = codex_agent_generate_text(
        "Use e2e_lookup_marker with key='text'. Reply exactly with the tool result.",
        tools=[e2e_lookup_marker],
        access_token=token,
        model=E2E_MODEL,
        system_prompt=EXACT_SYSTEM_PROMPT,
        timeout=E2E_TIMEOUT,
        return_history=True,
    )
    _assert_contains_marker(first, marker)
    _assert_agent_history(history, marker)

    second, history = codex_agent_generate_text(
        "What marker did the tool return in the previous turn? Reply exactly with that marker.",
        tools=[e2e_lookup_marker],
        access_token=token,
        model=E2E_MODEL,
        system_prompt=EXACT_SYSTEM_PROMPT,
        timeout=E2E_TIMEOUT,
        history=history,
        return_history=True,
    )
    _assert_contains_marker(second, marker)
    _assert_agent_history(history, marker)


def test_codex_agent_generate_model_e2e_one_turn_then_two_turn() -> None:
    token = _access_token()
    marker = "codex-e2e-agent-model"

    first, history = codex_agent_generate_model(
        "Use e2e_lookup_marker with key='model'. Return marker equal to the tool result and answer='tool'.",
        E2EAnswer,
        tools=[e2e_lookup_marker],
        access_token=token,
        model=E2E_MODEL,
        system_prompt=EXACT_SYSTEM_PROMPT,
        timeout=E2E_TIMEOUT,
        return_history=True,
    )
    assert first.marker == marker
    assert first.answer == "tool"
    _assert_agent_history(history, marker)

    second, history = codex_agent_generate_model(
        "Using the previous turn, return the same marker and answer='remembered'.",
        E2EAnswer,
        tools=[e2e_lookup_marker],
        access_token=token,
        model=E2E_MODEL,
        system_prompt=EXACT_SYSTEM_PROMPT,
        timeout=E2E_TIMEOUT,
        history=history,
        return_history=True,
    )
    assert second.marker == marker
    assert second.answer == "remembered"
    _assert_agent_history(history, marker)


def test_codex_agent_generate_text_async_e2e_one_turn_then_two_turn() -> None:
    async def run_test() -> None:
        token = _access_token()
        marker = "codex-e2e-agent-text"

        first, history = await codex_agent_generate_text_async(
            "Use e2e_lookup_marker with key='text'. Reply exactly with the tool result.",
            tools=[e2e_lookup_marker],
            access_token=token,
            model=E2E_MODEL,
            system_prompt=EXACT_SYSTEM_PROMPT,
            timeout=E2E_TIMEOUT,
            return_history=True,
        )
        _assert_contains_marker(first, marker)
        _assert_agent_history(history, marker)

        second, history = await codex_agent_generate_text_async(
            "What marker did the tool return in the previous turn? Reply exactly with that marker.",
            tools=[e2e_lookup_marker],
            access_token=token,
            model=E2E_MODEL,
            system_prompt=EXACT_SYSTEM_PROMPT,
            timeout=E2E_TIMEOUT,
            history=history,
            return_history=True,
        )
        _assert_contains_marker(second, marker)
        _assert_agent_history(history, marker)

    asyncio.run(run_test())


def test_codex_agent_generate_model_async_e2e_one_turn_then_two_turn() -> None:
    async def run_test() -> None:
        token = _access_token()
        marker = "codex-e2e-agent-model"

        first, history = await codex_agent_generate_model_async(
            "Use e2e_lookup_marker with key='model'. Return marker equal to the tool result and answer='tool'.",
            E2EAnswer,
            tools=[e2e_lookup_marker],
            access_token=token,
            model=E2E_MODEL,
            system_prompt=EXACT_SYSTEM_PROMPT,
            timeout=E2E_TIMEOUT,
            return_history=True,
        )
        assert first.marker == marker
        assert first.answer == "tool"
        _assert_agent_history(history, marker)

        second, history = await codex_agent_generate_model_async(
            "Using the previous turn, return the same marker and answer='remembered'.",
            E2EAnswer,
            tools=[e2e_lookup_marker],
            access_token=token,
            model=E2E_MODEL,
            system_prompt=EXACT_SYSTEM_PROMPT,
            timeout=E2E_TIMEOUT,
            history=history,
            return_history=True,
        )
        assert second.marker == marker
        assert second.answer == "remembered"
        _assert_agent_history(history, marker)

    asyncio.run(run_test())


def test_get_codex_usage_e2e() -> None:
    token = _access_token()
    usage = get_codex_usage(access_token=token)
    assert isinstance(usage, CodexUsage)
    assert isinstance(usage.rate_limit.limit_reached, bool)
    assert usage.rate_limit.primary_window.limit_window_seconds > 0
    assert 0.0 <= usage.rate_limit.primary_window.used_percent <= 100.0
    assert usage.rate_limit.secondary_window.limit_window_seconds > 0
    assert 0.0 <= usage.rate_limit.secondary_window.used_percent <= 100.0
    assert isinstance(usage.plan_type, str)
    assert isinstance(usage.credits.balance, int)
