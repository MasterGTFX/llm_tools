from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence, Type, TypeVar

from pydantic import BaseModel

from ..models import Conversation, GenerateOptions, GenerateResult, Usage

ModelT = TypeVar("ModelT", bound=BaseModel)


class Provider(Protocol):
    """Async provider contract implemented by concrete model backends."""

    name: str

    async def text(
        self,
        prompt: str,
        *,
        conversation: Conversation,
        options: GenerateOptions,
    ) -> GenerateResult[str]:
        ...

    async def model(
        self,
        prompt: str,
        output_type: Type[ModelT],
        *,
        conversation: Conversation,
        options: GenerateOptions,
    ) -> GenerateResult[ModelT]:
        ...

    async def tools(
        self,
        prompt: str,
        *,
        tools: Sequence[Any],
        output_type: Optional[Type[ModelT]],
        conversation: Conversation,
        options: GenerateOptions,
    ) -> GenerateResult[Any]:
        ...

    async def usage(self, *, account_id: Optional[str] = None) -> Optional[Usage]:
        ...
