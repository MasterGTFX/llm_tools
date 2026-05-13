class LLMToolsError(RuntimeError):
    """Base error for llm_tools."""


class AuthenticationError(LLMToolsError):
    """Raised when provider authentication is missing or invalid."""


class ProviderError(LLMToolsError):
    """Raised when a provider returns an unusable response."""


class EmptyOutputError(ProviderError):
    """Raised when a provider completes without returning output."""


class UsageExhaustedError(LLMToolsError):
    """Raised when the provider reports that usage limits or credits have been exhausted."""


class SyncClientError(LLMToolsError):
    """Raised when the sync facade is used from an async context."""
