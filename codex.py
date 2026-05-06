from __future__ import annotations

from typing import Any, List, Optional, Tuple, Type, TypeVar, Union

from pydantic import BaseModel

from codex_helpers import CodexError, HistoryLike, MessageDict, build_input, get_openai_client, updated_history
from config import DEFAULT_BASE_URL, DEFAULT_MAX_RETRIES, DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT, DEFAULT_TIMEOUT

ModelT = TypeVar("ModelT", bound=BaseModel)


def codex_generate_text(
    user_prompt: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
    history: Optional[HistoryLike] = None,
    return_history: bool = False,
    access_token: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: float = DEFAULT_TIMEOUT,
) -> Union[str, Tuple[str, List[MessageDict]]]:
    client = get_openai_client(
        access_token=access_token,
        base_url=base_url,
        max_retries=max_retries,
        timeout=timeout,
    )
    stream = client.responses.create(
        model=model,
        store=False,
        stream=True,
        instructions=system_prompt,
        input=build_input(user_prompt=user_prompt, history=history),
    )

    text_parts: List[str] = []
    for event in stream:
        if getattr(event, "type", None) == "response.output_text.delta":
            delta = getattr(event, "delta", "")
            if delta:
                text_parts.append(delta)
        elif getattr(event, "type", None) == "response.output_text.done" and not text_parts:
            done_text = getattr(event, "text", "")
            if done_text:
                text_parts.append(done_text)

    result = "".join(text_parts).strip()
    if not result:
        raise CodexError("Codex returned empty text output")

    if return_history:
        return result, updated_history(user_prompt=user_prompt, assistant_text=result, history=history)
    return result


def codex_generate_model(
    user_prompt: str,
    response_model: Type[ModelT],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
    history: Optional[HistoryLike] = None,
    return_history: bool = False,
    access_token: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: float = DEFAULT_TIMEOUT,
) -> Union[ModelT, Tuple[ModelT, List[MessageDict]]]:
    client = get_openai_client(
        access_token=access_token,
        base_url=base_url,
        max_retries=max_retries,
        timeout=timeout,
    )

    parsed: Optional[ModelT] = None
    raw_text = ""

    with client.responses.stream(
        model=model,
        store=False,
        instructions=system_prompt,
        input=build_input(user_prompt=user_prompt, history=history),
        text_format=response_model,
    ) as stream:
        for event in stream:
            event_type = getattr(event, "type", None)
            if event_type == "response.output_text.done":
                raw_text = getattr(event, "text", "") or raw_text
                parsed = getattr(event, "parsed", None)

    if parsed is None:
        raise CodexError(f"Codex did not return parsed output. Raw text: {raw_text!r}")

    if return_history:
        history_text = raw_text or parsed.model_dump_json()
        return parsed, updated_history(user_prompt=user_prompt, assistant_text=history_text, history=history)
    return parsed


__all__ = [
    "CodexError",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_SYSTEM_PROMPT",
    "codex_generate_model",
    "codex_generate_text",
]


if __name__ == "__main__":
    class DemoExtraction(BaseModel):
        company: str
        score: int

    print("TEXT DEMO:")
    print(codex_generate_text("Say hello in one short sentence."))
    print()
    print("MODEL DEMO:")
    print(codex_generate_model("Extract company and score from: FutureApps AI scored 91.", DemoExtraction))
