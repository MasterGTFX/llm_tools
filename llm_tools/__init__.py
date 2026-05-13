from .auth import AuthProvider, CodexLocalAuth, StaticTokenAuth
from .client import AsyncLLMClient, LLMClient
from .errors import (
    AuthenticationError,
    EmptyOutputError,
    LLMToolsError,
    ProviderError,
    SyncClientError,
    UsageExhaustedError,
)
from .models import Conversation, GenerateOptions, GenerateResult, ImageGeneration, Message, Usage
from .providers import Provider

__all__ = [
    "AsyncLLMClient",
    "AuthProvider",
    "AuthenticationError",
    "CodexProvider",
    "CodexLocalAuth",
    "Conversation",
    "EmptyOutputError",
    "GenerateOptions",
    "GenerateResult",
    "ImageGeneration",
    "LLMClient",
    "LLMToolsError",
    "Message",
    "Provider",
    "ProviderError",
    "StaticTokenAuth",
    "SyncClientError",
    "Usage",
    "UsageExhaustedError",
]


def __getattr__(name: str) -> object:
    if name == "CodexProvider":
        from .providers.codex import CodexProvider

        return CodexProvider
    raise AttributeError(f"module 'llm_tools' has no attribute {name!r}")
