from __future__ import annotations

import asyncio
from typing import Any, Awaitable, List, Optional, Sequence, Tuple, Type, TypeVar, Union

from pydantic import BaseModel

from .agent_async import codex_agent_generate_model_async, codex_agent_generate_text_async
from .config import DEFAULT_BASE_URL, DEFAULT_MAX_RETRIES, DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT, DEFAULT_TIMEOUT
from .helpers import CodexAgentError

ModelT = TypeVar("ModelT", bound=BaseModel)
AgentHistory = List[dict[str, Any]]


def _run_sync(coro: Awaitable[Any], fn_name: str) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise CodexAgentError(
        f"{fn_name} cannot be called from a running event loop. "
        f"Use {fn_name}_async instead."
    )


def codex_agent_generate_text(
    user_prompt: str,
    *,
    tools: Sequence[Any],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
    history: Optional[AgentHistory] = None,
    return_history: bool = False,
    access_token: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: float = DEFAULT_TIMEOUT,
) -> Union[str, Tuple[str, AgentHistory]]:
    return _run_sync(
        codex_agent_generate_text_async(
            user_prompt,
            tools=tools,
            system_prompt=system_prompt,
            model=model,
            history=history,
            return_history=return_history,
            access_token=access_token,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
        ),
        "codex_agent_generate_text",
    )


def codex_agent_generate_model(
    user_prompt: str,
    response_model: Type[ModelT],
    *,
    tools: Sequence[Any],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
    history: Optional[AgentHistory] = None,
    return_history: bool = False,
    access_token: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: float = DEFAULT_TIMEOUT,
) -> Union[ModelT, Tuple[ModelT, AgentHistory]]:
    return _run_sync(
        codex_agent_generate_model_async(
            user_prompt,
            response_model,
            tools=tools,
            system_prompt=system_prompt,
            model=model,
            history=history,
            return_history=return_history,
            access_token=access_token,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
        ),
        "codex_agent_generate_model",
    )
