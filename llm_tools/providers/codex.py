from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Type, TypeVar

import httpx
from agents import Agent, ModelSettings, OpenAIResponsesModel, Runner, set_default_openai_client
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

from ..auth import AuthProvider, CodexLocalAuth
from ..errors import EmptyOutputError, ProviderError
from ..models import Conversation, GenerateOptions, GenerateResult, ImageGeneration, Usage

load_dotenv()

ModelT = TypeVar("ModelT", bound=BaseModel)

DEFAULT_CODEX_BASE_URL = os.getenv("OPENAI_CODEX_BASE_URL", "https://chatgpt.com/backend-api/codex")
DEFAULT_CODEX_MODEL = os.getenv("OPENAI_CODEX_MODEL", "gpt-5.5")
DEFAULT_CODEX_IMAGE_MODEL = os.getenv("OPENAI_CODEX_IMAGE_MODEL", "gpt-image-2")
DEFAULT_CODEX_SYSTEM_PROMPT = os.getenv(
    "OPENAI_CODEX_SYSTEM_PROMPT",
    "You are a helpful coding assistant. Answer concisely.",
)
DEFAULT_CODEX_MAX_RETRIES = int(os.getenv("OPENAI_CODEX_MAX_RETRIES", "2"))
DEFAULT_CODEX_TIMEOUT = float(os.getenv("OPENAI_CODEX_TIMEOUT", "60"))


@dataclass
class CodexProvider:
    """Codex backend provider using Responses API and Agents SDK semantics."""

    auth: AuthProvider = field(default_factory=CodexLocalAuth)
    base_url: str = DEFAULT_CODEX_BASE_URL
    default_model: str = DEFAULT_CODEX_MODEL
    default_image_model: str = DEFAULT_CODEX_IMAGE_MODEL
    default_system_prompt: str = DEFAULT_CODEX_SYSTEM_PROMPT
    default_timeout: float = DEFAULT_CODEX_TIMEOUT
    default_max_retries: int = DEFAULT_CODEX_MAX_RETRIES

    name: str = "codex"

    def _model(self, options: GenerateOptions) -> str:
        return options.model or self.default_model

    def _system_prompt(self, options: GenerateOptions) -> str:
        return options.system_prompt or self.default_system_prompt

    def _timeout(self, options: GenerateOptions) -> float:
        return options.timeout if options.timeout is not None else self.default_timeout

    def _max_retries(self, options: GenerateOptions) -> int:
        return options.max_retries if options.max_retries is not None else self.default_max_retries

    def _image_tool(self, options: GenerateOptions) -> tuple[dict[str, Any], dict[str, Any]]:
        provider_options = dict(options.provider_options)
        image_model = provider_options.pop("image_model", self.default_image_model)
        image_tool_options = provider_options.pop("image_tool_options", {})
        if not isinstance(image_tool_options, dict):
            raise ProviderError("image_tool_options must be a dictionary")
        image_tool = {"type": "image_generation", "model": image_model}
        image_tool.update(image_tool_options)
        return image_tool, provider_options

    def _client(self, options: GenerateOptions) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.auth.get_token(),
            base_url=self.base_url,
            max_retries=self._max_retries(options),
            timeout=self._timeout(options),
        )

    @staticmethod
    def _input(prompt: str, conversation: Conversation) -> list[dict[str, Any]]:
        items = conversation.copy_items()
        items.append({"role": "user", "content": prompt})
        return items

    async def text(
        self,
        prompt: str,
        *,
        conversation: Conversation,
        options: GenerateOptions,
    ) -> GenerateResult[str]:
        client = self._client(options)
        stream = await client.responses.create(
            model=self._model(options),
            store=False,
            stream=True,
            instructions=self._system_prompt(options),
            input=self._input(prompt, conversation),
            **options.provider_options,
        )

        text_parts: list[str] = []
        async for event in stream:
            event_type = getattr(event, "type", None)
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    text_parts.append(delta)
            elif event_type == "response.output_text.done" and not text_parts:
                done_text = getattr(event, "text", "")
                if done_text:
                    text_parts.append(done_text)

        output = "".join(text_parts).strip()
        if not output:
            raise EmptyOutputError("Codex returned empty text output")

        return GenerateResult(
            output=output,
            raw_output=output,
            conversation=conversation.with_assistant(prompt, output),
            metadata={"provider": self.name, "model": self._model(options)},
        )

    async def model(
        self,
        prompt: str,
        output_type: Type[ModelT],
        *,
        conversation: Conversation,
        options: GenerateOptions,
    ) -> GenerateResult[ModelT]:
        client = self._client(options)
        parsed: Optional[ModelT] = None
        raw_text = ""

        async with client.responses.stream(
            model=self._model(options),
            store=False,
            instructions=self._system_prompt(options),
            input=self._input(prompt, conversation),
            text_format=output_type,
            **options.provider_options,
        ) as stream:
            async for event in stream:
                if getattr(event, "type", None) == "response.output_text.done":
                    raw_text = getattr(event, "text", "") or raw_text
                    parsed = getattr(event, "parsed", None)

        if parsed is None:
            raise ProviderError(f"Codex did not return parsed output. Raw text: {raw_text!r}")

        history_text = raw_text or parsed.model_dump_json()
        return GenerateResult(
            output=parsed,
            raw_output=raw_text or history_text,
            conversation=conversation.with_assistant(prompt, history_text),
            metadata={"provider": self.name, "model": self._model(options)},
        )

    async def tools(
        self,
        prompt: str,
        *,
        tools: Sequence[Any],
        output_type: Optional[Type[ModelT]],
        conversation: Conversation,
        options: GenerateOptions,
    ) -> GenerateResult[Any]:
        client = self._client(options)
        set_default_openai_client(client)
        agent = Agent(
            name="Codex agent",
            instructions=self._system_prompt(options),
            model=OpenAIResponsesModel(model=self._model(options), openai_client=client),
            model_settings=ModelSettings(store=False),
            tools=list(tools),
            output_type=output_type,
        )

        input_payload: str | list[dict[str, Any]] = prompt
        if conversation.items:
            input_payload = self._input(prompt, conversation)

        result = Runner.run_streamed(agent, input_payload)
        async for _event in result.stream_events():
            pass

        output = result.final_output
        if output is None:
            raise EmptyOutputError("Codex agent returned empty output")
        if output_type is not None and not isinstance(output, output_type):
            raise ProviderError(f"Unexpected Codex agent output type: {type(output)!r}")

        new_items: list[dict[str, Any]] = []
        for item in getattr(result, "new_items", []) or []:
            raw_item = getattr(item, "raw_item", None)
            if raw_item is None:
                continue
            if hasattr(raw_item, "model_dump"):
                new_items.append(raw_item.model_dump())
            elif isinstance(raw_item, dict):
                new_items.append(raw_item)

        return GenerateResult(
            output=output,
            raw_output=str(output),
            conversation=conversation.with_items(prompt, new_items),
            metadata={"provider": self.name, "model": self._model(options)},
        )

    async def image(
        self,
        prompt: str,
        *,
        conversation: Conversation,
        options: GenerateOptions,
    ) -> GenerateResult[ImageGeneration]:
        client = self._client(options)
        image_tool, provider_options = self._image_tool(options)
        stream = await client.responses.create(
            model=self._model(options),
            store=False,
            stream=True,
            instructions=self._system_prompt(options),
            input=self._input(prompt, conversation),
            tools=[image_tool],
            **provider_options,
        )

        image_base64: Optional[str] = None
        revised_prompt: Optional[str] = None
        async for event in stream:
            event_type = getattr(event, "type", None)
            if event_type == "response.image_generation_call.partial_image" and image_base64 is None:
                image_base64 = getattr(event, "partial_image_b64", None)
            elif event_type == "response.output_item.done":
                item = getattr(event, "item", None)
                if getattr(item, "type", None) == "image_generation_call":
                    image_base64 = getattr(item, "result", None) or image_base64
                    revised_prompt = getattr(item, "revised_prompt", None) or revised_prompt

        if image_base64:
            image = ImageGeneration(
                image_base64=image_base64,
                image_bytes=base64.b64decode(image_base64),
                revised_prompt=revised_prompt,
            )
            return GenerateResult(
                output=image,
                raw_output=image_base64,
                conversation=conversation.with_assistant(prompt, "[image generated]"),
                metadata={
                    "provider": self.name,
                    "model": self._model(options),
                    "image_model": image_tool.get("model"),
                },
            )

        raise EmptyOutputError("Codex returned no image output")

    async def usage(self, *, account_id: Optional[str] = None) -> Optional[Usage]:
        headers = {
            "Authorization": f"Bearer {self.auth.get_token()}",
            "Accept": "application/json",
        }
        if account_id:
            headers["ChatGPT-Account-Id"] = account_id

        async with httpx.AsyncClient(timeout=self.default_timeout) as client:
            response = await client.get("https://chatgpt.com/backend-api/wham/usage", headers=headers)
        response.raise_for_status()
        data = response.json()
        rate_limit = data.get("rate_limit") or {}
        primary = rate_limit.get("primary_window") or {}
        secondary = rate_limit.get("secondary_window") or {}
        credits = data.get("credits") or {}
        return Usage(
            rate_limit_reached=rate_limit.get("limit_reached"),
            primary_used_percent=primary.get("used_percent"),
            secondary_used_percent=secondary.get("used_percent"),
            credits_balance=credits.get("balance"),
            plan_type=data.get("plan_type"),
            raw=data,
        )
