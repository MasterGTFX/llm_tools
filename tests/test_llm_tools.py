from __future__ import annotations

import asyncio
from typing import Any, Optional, Sequence, Type, TypeVar

import pytest
from pydantic import BaseModel

from llm_tools import (
    AsyncLLMClient,
    AuthenticationError,
    CodexLocalAuth,
    Conversation,
    GenerateOptions,
    GenerateResult,
    LLMClient,
    StaticTokenAuth,
    SyncClientError,
    Usage,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


class Extracted(BaseModel):
    value: str


class FakeProvider:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[str, GenerateOptions, Conversation]] = []

    async def text(
        self,
        prompt: str,
        *,
        conversation: Conversation,
        options: GenerateOptions,
    ) -> GenerateResult[str]:
        self.calls.append(("text", options, conversation))
        return GenerateResult(
            output=f"text:{prompt}",
            raw_output=f"text:{prompt}",
            conversation=conversation.with_assistant(prompt, f"text:{prompt}"),
            metadata={"provider": self.name},
        )

    async def model(
        self,
        prompt: str,
        output_type: Type[ModelT],
        *,
        conversation: Conversation,
        options: GenerateOptions,
    ) -> GenerateResult[ModelT]:
        self.calls.append(("model", options, conversation))
        output = output_type(value=prompt)
        return GenerateResult(
            output=output,
            raw_output=output.model_dump_json(),
            conversation=conversation.with_assistant(prompt, output.model_dump_json()),
            metadata={"provider": self.name},
        )

    async def tools(
        self,
        prompt: str,
        *,
        tools: Sequence[Any],
        output_type: Optional[Type[ModelT]],
        conversation: Conversation,
        options: GenerateOptions,
    ) -> GenerateResult[Any]:
        self.calls.append(("tools", options, conversation))
        output: Any = output_type(value=prompt) if output_type else f"tools:{len(tools)}:{prompt}"
        return GenerateResult(
            output=output,
            raw_output=str(output),
            conversation=conversation.with_items(prompt, [{"role": "assistant", "content": str(output)}]),
            metadata={"provider": self.name},
        )

    async def usage(self, *, account_id: Optional[str] = None) -> Optional[Usage]:
        return Usage(total_tokens=3, raw={"account_id": account_id})


def test_static_token_auth_requires_non_empty_token() -> None:
    with pytest.raises(AuthenticationError):
        StaticTokenAuth("").get_token()

    assert StaticTokenAuth("token").get_token() == "token"


def test_codex_local_auth_prefers_env_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text('{"tokens": {"access_token": "file-token"}}', encoding="utf-8")
    monkeypatch.setenv("OPENAI_CODEX_ACCESS_TOKEN", "env-token")

    assert CodexLocalAuth(paths=[auth_file]).get_token() == "env-token"


def test_codex_local_auth_reads_local_auth_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        '{"auth_mode": "chatgpt", "tokens": {"access_token": "file-token"}}',
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_CODEX_ACCESS_TOKEN", raising=False)

    assert CodexLocalAuth(paths=[auth_file]).get_token() == "file-token"


def test_codex_local_auth_reports_missing_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text('{"tokens": {"refresh_token": "refresh-only"}}', encoding="utf-8")
    monkeypatch.delenv("OPENAI_CODEX_ACCESS_TOKEN", raising=False)

    with pytest.raises(AuthenticationError, match="no Codex access token"):
        CodexLocalAuth(paths=[auth_file]).get_token()


def test_async_client_persists_conversation_and_passes_defaults() -> None:
    async def run() -> None:
        provider = FakeProvider()
        client = AsyncLLMClient(
            provider=provider,
            model="model-a",
            system_prompt="system-a",
            timeout=12,
            max_retries=4,
        )

        first = await client.text("hello")
        second = await client.text("again")

        assert first.output == "text:hello"
        assert second.output == "text:again"
        assert len(second.conversation.items) == 4

        _, options, first_conversation = provider.calls[0]
        assert options.model == "model-a"
        assert options.system_prompt == "system-a"
        assert options.timeout == 12
        assert options.max_retries == 4
        assert first_conversation.items == []

        _, _, second_conversation = provider.calls[1]
        assert len(second_conversation.items) == 2

    asyncio.run(run())


def test_async_client_supports_structured_output_and_tools() -> None:
    async def run() -> None:
        provider = FakeProvider()
        client = AsyncLLMClient(provider=provider, persist_history=False)

        model_result = await client.model("field", Extracted)
        tools_result = await client.tools("work", tools=[object()])

        assert model_result.output == Extracted(value="field")
        assert tools_result.output == "tools:1:work"
        assert client.conversation.items == []

    asyncio.run(run())


def test_sync_client_facade() -> None:
    client = LLMClient(provider=FakeProvider(), model="sync-model")

    result = client.text("hello")

    assert result.output == "text:hello"
    assert len(client.conversation.items) == 2


def test_sync_client_rejects_running_event_loop() -> None:
    async def run() -> None:
        client = LLMClient(provider=FakeProvider())
        with pytest.raises(SyncClientError):
            client.text("hello")

    asyncio.run(run())
