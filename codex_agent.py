from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Type, TypeVar, Union

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
    return_response_id: bool = False,
    access_token: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: float = DEFAULT_TIMEOUT,
) -> Union[str, Tuple[str, str]]:
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

    async def _consume() -> Tuple[str, Optional[str]]:
        result = Runner.run_streamed(agent, user_prompt)
        async for _event in result.stream_events():
            pass
        response_id = getattr(result, 'last_response_id', None)
        if not response_id:
            raw_responses = getattr(result, 'raw_responses', None) or []
            if raw_responses:
                response_id = getattr(raw_responses[-1], 'response_id', None)
        return str(result.final_output), response_id

    import asyncio

    output, response_id = asyncio.run(_consume())
    if not output:
        raise CodexAgentError("Codex agent returned empty text output")
    if return_response_id:
        if not response_id:
            raise CodexAgentError("Codex agent did not expose a response id")
        return output, response_id
    return output


def codex_agent_generate_model(
    user_prompt: str,
    response_model: Type[ModelT],
    *,
    tools: Sequence[Any],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
    return_response_id: bool = False,
    access_token: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: float = DEFAULT_TIMEOUT,
) -> Union[ModelT, Tuple[ModelT, str]]:
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

    async def _consume() -> Tuple[Any, Optional[str]]:
        result = Runner.run_streamed(agent, user_prompt)
        async for _event in result.stream_events():
            pass
        response_id = getattr(result, 'last_response_id', None)
        if not response_id:
            raw_responses = getattr(result, 'raw_responses', None) or []
            if raw_responses:
                response_id = getattr(raw_responses[-1], 'response_id', None)
        return result.final_output, response_id

    import asyncio

    output, response_id = asyncio.run(_consume())
    if output is None:
        raise CodexAgentError("Codex agent returned empty structured output")
    if not isinstance(output, response_model):
        raise CodexAgentError(f"Unexpected output type: {type(output)!r}")
    if return_response_id:
        if not response_id:
            raise CodexAgentError("Codex agent did not expose a response id")
        return output, response_id
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
