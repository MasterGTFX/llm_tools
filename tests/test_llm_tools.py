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
    ImageGeneration,
    LLMClient,
    StaticTokenAuth,
    SyncClientError,
    Usage,
    UsageExhaustedError,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


class Extracted(BaseModel):
    value: str


class FakeProvider:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[str, GenerateOptions, Conversation]] = []
        self.usage_calls: list[Optional[str]] = []
        self.usage_exhausted = False

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

    async def image(
        self,
        prompt: str,
        *,
        conversation: Conversation,
        options: GenerateOptions,
    ) -> GenerateResult[ImageGeneration]:
        self.calls.append(("image", options, conversation))
        output = ImageGeneration(image_base64="aW1hZ2U=", image_bytes=b"image")
        return GenerateResult(
            output=output,
            raw_output=output.image_base64,
            conversation=conversation.with_assistant(prompt, "[image generated]"),
            metadata={"provider": self.name},
        )

    async def usage(self, *, account_id: Optional[str] = None) -> Optional[Usage]:
        self.usage_calls.append(account_id)
        return Usage(
            total_tokens=3,
            rate_limit_reached=self.usage_exhausted,
            raw={"account_id": account_id},
        )


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


def test_async_and_sync_clients_support_image_generation() -> None:
    async def run() -> None:
        provider = FakeProvider()
        client = AsyncLLMClient(provider=provider, persist_history=False)

        result = await client.image("draw a square")

        assert result.output.image_bytes == b"image"
        assert result.output.mime_type == "image/png"
        assert provider.calls[0][0] == "image"

    asyncio.run(run())

    sync_result = LLMClient(provider=FakeProvider()).image("draw a square")

    assert sync_result.output.image_base64 == "aW1hZ2U="


def test_usage_is_exhausted() -> None:
    assert Usage().is_exhausted() is False
    assert Usage(rate_limit_reached=True).is_exhausted() is True
    assert Usage(credits_balance=0).is_exhausted() is True
    assert Usage(credits_balance=-1).is_exhausted() is True
    assert Usage(credits_balance=5).is_exhausted() is False
    assert Usage(primary_used_percent=100).is_exhausted() is True
    assert Usage(primary_used_percent=99.9).is_exhausted() is False
    assert Usage(secondary_used_percent=100).is_exhausted() is True
    assert Usage(secondary_used_percent=100.1).is_exhausted() is True


def test_async_client_checks_usage_on_first_call_and_interval() -> None:
    async def run() -> None:
        provider = FakeProvider()
        client = AsyncLLMClient(provider=provider, check_usage_every=3)

        await client.text("a")
        await client.text("b")
        await client.text("c")
        await client.text("d")
        await client.text("e")

        # usage checked on first call (call 1) and after every 3 calls (call 4)
        assert provider.usage_calls == [None, None]
        assert len(provider.calls) == 5

    asyncio.run(run())


def test_async_client_checks_usage_with_account_id() -> None:
    async def run() -> None:
        provider = FakeProvider()
        client = AsyncLLMClient(provider=provider, check_usage_every=1, usage_account_id="acc-42")

        await client.text("a")

        assert provider.usage_calls == ["acc-42"]

    asyncio.run(run())


def test_async_client_raises_when_usage_exhausted() -> None:
    async def run() -> None:
        provider = FakeProvider()
        provider.usage_exhausted = True
        client = AsyncLLMClient(provider=provider, check_usage_every=1)

        with pytest.raises(UsageExhaustedError):
            await client.text("a")

        assert provider.usage_calls == [None]

    asyncio.run(run())


def test_async_client_reset_resets_usage_counter() -> None:
    async def run() -> None:
        provider = FakeProvider()
        client = AsyncLLMClient(provider=provider, check_usage_every=2)

        await client.text("a")
        client.reset()
        await client.text("b")

        # reset should have set counter back to check_usage_every, so next call triggers check again
        assert provider.usage_calls == [None, None]

    asyncio.run(run())


def test_sync_client_checks_usage() -> None:
    provider = FakeProvider()
    provider.usage_exhausted = True
    client = LLMClient(provider=provider, check_usage_every=1)

    with pytest.raises(UsageExhaustedError):
        client.text("hello")

    assert provider.usage_calls == [None]


def test_sync_client_rejects_running_event_loop() -> None:
    async def run() -> None:
        client = LLMClient(provider=FakeProvider())
        with pytest.raises(SyncClientError):
            client.text("hello")

    asyncio.run(run())
