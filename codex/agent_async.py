from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Type, TypeVar, Union

from agents import Agent, ModelSettings, OpenAIResponsesModel, Runner, set_default_openai_client
from pydantic import BaseModel

from .config import DEFAULT_BASE_URL, DEFAULT_MAX_RETRIES, DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT, DEFAULT_TIMEOUT
from .helpers import CodexAgentError, get_async_openai_client

ModelT = TypeVar("ModelT", bound=BaseModel)
AgentHistory = List[dict[str, Any]]


def _build_agent(
    *,
    client: Any,
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


async def _run_agent_with_history(
    *,
    agent: Agent[Any],
    user_prompt: str,
    history: Optional[AgentHistory] = None,
) -> Tuple[Any, AgentHistory]:
    input_payload: Union[str, AgentHistory] = user_prompt
    if history:
        input_payload = list(history) + [{"role": "user", "content": user_prompt}]

    result = Runner.run_streamed(agent, input_payload)
    async for _event in result.stream_events():
        pass

    new_history: AgentHistory = list(history or [])
    new_history.append({"role": "user", "content": user_prompt})

    for item in getattr(result, "new_items", []) or []:
        raw_item = getattr(item, "raw_item", None)
        if raw_item is not None:
            if hasattr(raw_item, "model_dump"):
                new_history.append(raw_item.model_dump())
            elif isinstance(raw_item, dict):
                new_history.append(raw_item)

    return result.final_output, new_history


def _build_client_agent(
    *,
    tools: Sequence[Any],
    system_prompt: str,
    model: str,
    access_token: Optional[str],
    base_url: str,
    max_retries: int,
    timeout: float,
    output_type: Any | None = None,
) -> Agent[Any]:
    client = get_async_openai_client(
        access_token=access_token,
        base_url=base_url,
        max_retries=max_retries,
        timeout=timeout,
    )
    return _build_agent(
        client=client,
        system_prompt=system_prompt,
        model=model,
        tools=tools,
        output_type=output_type,
    )


async def codex_agent_generate_text_async(
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
    agent = _build_client_agent(
        access_token=access_token,
        base_url=base_url,
        max_retries=max_retries,
        timeout=timeout,
        system_prompt=system_prompt,
        model=model,
        tools=tools,
    )

    output, replay_history = await _run_agent_with_history(agent=agent, user_prompt=user_prompt, history=history)
    output_text = str(output)
    if not output_text:
        raise CodexAgentError("Codex agent returned empty text output")
    if return_history:
        return output_text, replay_history
    return output_text


async def codex_agent_generate_model_async(
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
    agent = _build_client_agent(
        access_token=access_token,
        base_url=base_url,
        max_retries=max_retries,
        timeout=timeout,
        system_prompt=system_prompt,
        model=model,
        tools=tools,
        output_type=response_model,
    )

    output, replay_history = await _run_agent_with_history(agent=agent, user_prompt=user_prompt, history=history)
    if output is None:
        raise CodexAgentError("Codex agent returned empty structured output")
    if not isinstance(output, response_model):
        raise CodexAgentError(f"Unexpected output type: {type(output)!r}")
    if return_history:
        return output, replay_history
    return output
