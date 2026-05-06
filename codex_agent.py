from __future__ import annotations

import asyncio
from typing import Any, List, Optional, Sequence, Tuple, Type, TypeVar, Union

from pydantic import BaseModel

from codex_agent_async import codex_agent_generate_model_async, codex_agent_generate_text_async
from codex_helpers import CodexAgentError
from config import DEFAULT_BASE_URL, DEFAULT_MAX_RETRIES, DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT, DEFAULT_TIMEOUT

ModelT = TypeVar("ModelT", bound=BaseModel)


def codex_agent_generate_text(
    user_prompt: str,
    *,
    tools: Sequence[Any],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
    history: Optional[List[dict[str, Any]]] = None,
    return_history: bool = False,
    access_token: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: float = DEFAULT_TIMEOUT,
) -> Union[str, Tuple[str, List[dict[str, Any]]]]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
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
            )
        )
    raise CodexAgentError("codex_agent_generate_text cannot be called from a running event loop. Use codex_agent_generate_text_async instead.")


def codex_agent_generate_model(
    user_prompt: str,
    response_model: Type[ModelT],
    *,
    tools: Sequence[Any],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
    history: Optional[List[dict[str, Any]]] = None,
    return_history: bool = False,
    access_token: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: float = DEFAULT_TIMEOUT,
) -> Union[ModelT, Tuple[ModelT, List[dict[str, Any]]]]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
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
            )
        )
    raise CodexAgentError("codex_agent_generate_model cannot be called from a running event loop. Use codex_agent_generate_model_async instead.")


__all__ = [
    "CodexAgentError",
    "codex_agent_generate_model",
    "codex_agent_generate_text",
]


if __name__ == "__main__":
    from agents import function_tool

    class StockAnswer(BaseModel):
        ticker: str
        price: float
        currency: str

    @function_tool
    def get_stock_price(ticker: str) -> str:
        """Get a fake current stock price for a ticker.

        Args:
            ticker: Stock ticker symbol, for example AAPL.
        """
        print(f"TOOL CALLED: get_stock_price({ticker})")
        return f'{ticker.upper()} is 123.45 USD'

    print("TEXT DEMO:")
    print(
        codex_agent_generate_text(
            "What is the price of AAPL? Use the tool.",
            tools=[get_stock_price],
            system_prompt="You are a concise assistant. Use tools when useful.",
        )
    )

    print()
    print("MODEL DEMO:")
    print(
        codex_agent_generate_model(
            "Return the price of AAPL as structured output. Use the tool.",
            StockAnswer,
            tools=[get_stock_price],
            system_prompt="You are a concise assistant. Use tools when useful.",
        )
    )
