from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Type, TypeVar, Union

from agents import Agent, ModelSettings, OpenAIResponsesModel, Runner, set_default_openai_client
from openai import AsyncOpenAI
from pydantic import BaseModel

from config import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TIMEOUT,
    get_access_token,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


class CodexAgentError(RuntimeError):
    pass


def _get_client(
    access_token: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: float = DEFAULT_TIMEOUT,
) -> AsyncOpenAI:
    token = access_token or get_access_token()
    if not token:
        raise CodexAgentError("OPENAI_CODEX_ACCESS_TOKEN is required")

    return AsyncOpenAI(
        api_key=token,
        base_url=base_url,
        max_retries=max_retries,
        timeout=timeout,
    )


def _build_agent(
    *,
    client: AsyncOpenAI,
    system_prompt: str,
    model: str,
    tools: Sequence[Any],
    output_type: Any | None = None,
) -> Agent[Any]:
    set_default_openai_client(client)

    return Agent(
        name="Codex agent",
        instructions=system_prompt,
        model=OpenAIResponsesModel(model=model, openai_client=client),
        model_settings=ModelSettings(store=False),
        tools=list(tools),
        output_type=output_type,
    )


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
    client = _get_client(
        access_token=access_token,
        base_url=base_url,
        max_retries=max_retries,
        timeout=timeout,
    )
    agent = _build_agent(
        client=client,
        system_prompt=system_prompt,
        model=model,
        tools=tools,
    )

    async def _consume() -> Tuple[str, List[dict[str, Any]]]:
        input_payload: Union[str, List[dict[str, Any]]] = user_prompt
        if history:
            input_payload = list(history) + [{"role": "user", "content": user_prompt}]

        result = Runner.run_streamed(agent, input_payload)
        async for _event in result.stream_events():
            pass

        new_history: List[dict[str, Any]] = list(history or [])
        if not history:
            new_history.append({"role": "user", "content": user_prompt})
        else:
            new_history.append({"role": "user", "content": user_prompt})

        for item in getattr(result, 'new_items', []) or []:
            raw_item = getattr(item, 'raw_item', None)
            if raw_item is not None:
                if hasattr(raw_item, 'model_dump'):
                    new_history.append(raw_item.model_dump())
                elif isinstance(raw_item, dict):
                    new_history.append(raw_item)

        return str(result.final_output), new_history

    import asyncio

    output, replay_history = asyncio.run(_consume())
    if not output:
        raise CodexAgentError("Codex agent returned empty text output")
    if return_history:
        return output, replay_history
    return output


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
    client = _get_client(
        access_token=access_token,
        base_url=base_url,
        max_retries=max_retries,
        timeout=timeout,
    )
    agent = _build_agent(
        client=client,
        system_prompt=system_prompt,
        model=model,
        tools=tools,
        output_type=response_model,
    )

    async def _consume() -> Tuple[Any, List[dict[str, Any]]]:
        input_payload: Union[str, List[dict[str, Any]]] = user_prompt
        if history:
            input_payload = list(history) + [{"role": "user", "content": user_prompt}]

        result = Runner.run_streamed(agent, input_payload)
        async for _event in result.stream_events():
            pass

        new_history: List[dict[str, Any]] = list(history or [])
        new_history.append({"role": "user", "content": user_prompt})

        for item in getattr(result, 'new_items', []) or []:
            raw_item = getattr(item, 'raw_item', None)
            if raw_item is not None:
                if hasattr(raw_item, 'model_dump'):
                    new_history.append(raw_item.model_dump())
                elif isinstance(raw_item, dict):
                    new_history.append(raw_item)

        return result.final_output, new_history

    import asyncio

    output, replay_history = asyncio.run(_consume())
    if output is None:
        raise CodexAgentError("Codex agent returned empty structured output")
    if not isinstance(output, response_model):
        raise CodexAgentError(f"Unexpected output type: {type(output)!r}")
    if return_history:
        return output, replay_history
    return output


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
