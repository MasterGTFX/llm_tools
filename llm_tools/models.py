from __future__ import annotations

from typing import Any, Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]
OutputT = TypeVar("OutputT")


class Message(BaseModel):
    role: Role
    content: str


class Conversation(BaseModel):
    """Replayable provider input/history.

    Items are dictionaries because tool-capable providers may need to replay
    provider-native records such as function calls and function outputs.
    """

    items: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def from_messages(cls, messages: list[Message]) -> "Conversation":
        return cls(items=[message.model_dump() for message in messages])

    def copy_items(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.items]

    def with_user(self, prompt: str) -> "Conversation":
        items = self.copy_items()
        items.append({"role": "user", "content": prompt})
        return Conversation(items=items)

    def with_assistant(self, prompt: str, output: str) -> "Conversation":
        items = self.copy_items()
        items.append({"role": "user", "content": prompt})
        items.append({"role": "assistant", "content": output})
        return Conversation(items=items)

    def with_items(self, prompt: str, new_items: list[dict[str, Any]]) -> "Conversation":
        items = self.copy_items()
        items.append({"role": "user", "content": prompt})
        items.extend(new_items)
        return Conversation(items=items)


class GenerateOptions(BaseModel):
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    timeout: Optional[float] = None
    max_retries: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    provider_options: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    rate_limit_reached: Optional[bool] = None
    primary_used_percent: Optional[float] = None
    secondary_used_percent: Optional[float] = None
    credits_balance: Optional[int] = None
    plan_type: Optional[str] = None
    raw: dict[str, Any] = Field(default_factory=dict)


class GenerateResult(BaseModel, Generic[OutputT]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    output: OutputT
    conversation: Conversation
    raw_output: Optional[str] = None
    usage: Optional[Usage] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
