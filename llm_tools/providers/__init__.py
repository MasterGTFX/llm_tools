from .base import Provider

__all__ = ["CodexProvider", "Provider"]


def __getattr__(name: str) -> object:
    if name == "CodexProvider":
        from .codex import CodexProvider

        return CodexProvider
    raise AttributeError(f"module 'llm_tools.providers' has no attribute {name!r}")
