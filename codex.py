from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, TypeVar, Union

from openai import OpenAI
from pydantic import BaseModel

from config import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TIMEOUT,
    get_access_token,
)

MessageDict = Dict[str, str]
HistoryLike = Sequence[MessageDict]
ModelT = TypeVar("ModelT", bound=BaseModel)


class CodexError(RuntimeError):
    pass


def _build_input(user_prompt: str, history: Optional[HistoryLike] = None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    if history:
        for msg in history:
            role = msg.get("role")
            content = msg.get("content")
            if role not in {"user", "assistant", "system"}:
                raise ValueError(f"Unsupported history role: {role!r}")
            if not isinstance(content, str):
                raise ValueError("History message content must be a string")
            items.append(
                {
                    "role": role,
                    "content": [
                        {
                            "type": "input_text",
                            "text": content,
                        }
                    ],
                }
            )

    items.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": user_prompt,
                }
            ],
        }
    )
    return items


def _get_client(
    access_token: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: float = DEFAULT_TIMEOUT,
) -> OpenAI:
    token = access_token or get_access_token()
    if not token:
        raise CodexError("OPENAI_CODEX_ACCESS_TOKEN is required")

    return OpenAI(
        api_key=token,
        base_url=base_url,
        max_retries=max_retries,
        timeout=timeout,
    )


def _updated_history(
    user_prompt: str,
    assistant_text: str,
    history: Optional[HistoryLike] = None,
) -> List[MessageDict]:
    new_history = list(history or [])
    new_history.append({"role": "user", "content": user_prompt})
    new_history.append({"role": "assistant", "content": assistant_text})
    return new_history


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
    client = _get_client(
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
        input=_build_input(user_prompt=user_prompt, history=history),
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
        return result, _updated_history(user_prompt=user_prompt, assistant_text=result, history=history)
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
    client = _get_client(
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
        input=_build_input(user_prompt=user_prompt, history=history),
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
        return parsed, _updated_history(user_prompt=user_prompt, assistant_text=history_text, history=history)
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
