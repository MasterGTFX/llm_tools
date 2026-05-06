from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional, Sequence, Type, TypeVar

from pydantic import BaseModel

from .errors import SyncClientError
from .models import Conversation, GenerateOptions, GenerateResult, Usage
from .providers.base import Provider

ModelT = TypeVar("ModelT", bound=BaseModel)


class AsyncLLMClient:
    """Async client that owns defaults, provider selection, and optional history."""

    def __init__(
        self,
        *,
        provider: Provider,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        conversation: Optional[Conversation] = None,
        persist_history: bool = True,
    ) -> None:
        self.provider = provider
        self.default_model = model
        self.default_system_prompt = system_prompt
        self.default_timeout = timeout
        self.default_max_retries = max_retries
        self.conversation = conversation or Conversation()
        self.persist_history = persist_history

    def reset(self, conversation: Optional[Conversation] = None) -> None:
        self.conversation = conversation or Conversation()

    def _options(
        self,
        *,
        model: Optional[str],
        system_prompt: Optional[str],
        timeout: Optional[float],
        max_retries: Optional[int],
        metadata: Optional[dict[str, Any]],
        provider_options: Optional[dict[str, Any]],
    ) -> GenerateOptions:
        return GenerateOptions(
            model=model or self.default_model,
            system_prompt=system_prompt or self.default_system_prompt,
            timeout=timeout if timeout is not None else self.default_timeout,
            max_retries=max_retries if max_retries is not None else self.default_max_retries,
            metadata=metadata or {},
            provider_options=provider_options or {},
        )

    def _conversation(self, conversation: Optional[Conversation]) -> Conversation:
        if conversation is not None:
            return conversation
        if self.persist_history:
            return self.conversation
        return Conversation()

    def _capture(self, result: GenerateResult[Any], conversation: Optional[Conversation]) -> None:
        if conversation is None and self.persist_history:
            self.conversation = result.conversation

    async def text(
        self,
        prompt: str,
        *,
        conversation: Optional[Conversation] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
        provider_options: Optional[dict[str, Any]] = None,
    ) -> GenerateResult[str]:
        result = await self.provider.text(
            prompt,
            conversation=self._conversation(conversation),
            options=self._options(
                model=model,
                system_prompt=system_prompt,
                timeout=timeout,
                max_retries=max_retries,
                metadata=metadata,
                provider_options=provider_options,
            ),
        )
        self._capture(result, conversation)
        return result

    async def model(
        self,
        prompt: str,
        output_type: Type[ModelT],
        *,
        conversation: Optional[Conversation] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
        provider_options: Optional[dict[str, Any]] = None,
    ) -> GenerateResult[ModelT]:
        result = await self.provider.model(
            prompt,
            output_type,
            conversation=self._conversation(conversation),
            options=self._options(
                model=model,
                system_prompt=system_prompt,
                timeout=timeout,
                max_retries=max_retries,
                metadata=metadata,
                provider_options=provider_options,
            ),
        )
        self._capture(result, conversation)
        return result

    async def tools(
        self,
        prompt: str,
        *,
        tools: Sequence[Any],
        output_type: Optional[Type[ModelT]] = None,
        conversation: Optional[Conversation] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
        provider_options: Optional[dict[str, Any]] = None,
    ) -> GenerateResult[Any]:
        result = await self.provider.tools(
            prompt,
            tools=tools,
            output_type=output_type,
            conversation=self._conversation(conversation),
            options=self._options(
                model=model,
                system_prompt=system_prompt,
                timeout=timeout,
                max_retries=max_retries,
                metadata=metadata,
                provider_options=provider_options,
            ),
        )
        self._capture(result, conversation)
        return result

    async def usage(self, *, account_id: Optional[str] = None) -> Optional[Usage]:
        return await self.provider.usage(account_id=account_id)


class LLMClient:
    """Sync facade for code that is not already running an event loop."""

    def __init__(self, **kwargs: Any) -> None:
        self._async_client = AsyncLLMClient(**kwargs)

    @property
    def conversation(self) -> Conversation:
        return self._async_client.conversation

    def reset(self, conversation: Optional[Conversation] = None) -> None:
        self._async_client.reset(conversation)

    def _run(self, awaitable_factory: Callable[[], Awaitable[Any]], method_name: str) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable_factory())
        raise SyncClientError(
            f"LLMClient.{method_name}() cannot be called from a running event loop. "
            f"Use AsyncLLMClient.{method_name}() instead."
        )

    def text(self, prompt: str, **kwargs: Any) -> GenerateResult[str]:
        return self._run(lambda: self._async_client.text(prompt, **kwargs), "text")

    def model(self, prompt: str, output_type: Type[ModelT], **kwargs: Any) -> GenerateResult[ModelT]:
        return self._run(lambda: self._async_client.model(prompt, output_type, **kwargs), "model")

    def tools(self, prompt: str, **kwargs: Any) -> GenerateResult[Any]:
        return self._run(lambda: self._async_client.tools(prompt, **kwargs), "tools")

    def usage(self, *, account_id: Optional[str] = None) -> Optional[Usage]:
        return self._run(lambda: self._async_client.usage(account_id=account_id), "usage")
