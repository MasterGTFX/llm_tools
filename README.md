# llm_tools

Small, copyable Python helpers for LLM workflows.

This repo starts with two lightweight modules for using the ChatGPT Codex backend from the OpenAI Python SDK:

- `codex.py` - synchronous helpers
- `codex_async.py` - asynchronous helpers

The goal is to stay simple, readable, and easy to copy into other projects.

## What is included

### `codex.py`
Sync helpers:

- `codex_generate_text(...)`
- `codex_generate_model(...)`

### `codex_async.py`
Async helpers:

- `codex_generate_text_async(...)`
- `codex_generate_model_async(...)`

Both modules:
- work as importable Python modules
- support plain text generation
- support structured output with Pydantic models
- support optional conversation history
- can optionally return updated history

## Requirements

- Python 3.10+
- `openai`
- `pydantic`

Install:

```bash
pip install openai pydantic
```

## Access token

These helpers expect:

```bash
export OPENAI_CODEX_ACCESS_TOKEN="your_token_here"
```

You can also pass `access_token=` directly to each function.

## How to fetch the access token

If you use OpenClaw / Codex CLI locally, the token may already exist in an auth profile file similar to:

```bash
~/.openclaw/agents/main/agent/auth-profiles.json
```

Look for an entry like:

- `openai-codex:default`
- or `openai-codex:<your-email>`

and use the `access` value as `OPENAI_CODEX_ACCESS_TOKEN`.

Important:
- treat this token like a secret
- do not commit it
- do not paste it into public repos or logs

## Usage

### Sync text generation

```python
from codex import codex_generate_text

text = codex_generate_text(
    "Say hello in one short sentence.",
    system_prompt="You are a helpful coding assistant. Answer concisely.",
)

print(text)
```

### Sync structured output

```python
from pydantic import BaseModel
from codex import codex_generate_model

class Extraction(BaseModel):
    company: str
    score: int

result = codex_generate_model(
    "Extract company and score from: FutureApps AI scored 91.",
    Extraction,
)

print(result)
print(result.company, result.score)
```

### Async text generation

```python
import asyncio
from codex_async import codex_generate_text_async

async def main():
    text = await codex_generate_text_async(
        "Say hello in one short sentence.",
    )
    print(text)

asyncio.run(main())
```

### Async structured output

```python
import asyncio
from pydantic import BaseModel
from codex_async import codex_generate_model_async

class Extraction(BaseModel):
    company: str
    score: int

async def main():
    result = await codex_generate_model_async(
        "Extract company and score from: FutureApps AI scored 91.",
        Extraction,
    )
    print(result)

asyncio.run(main())
```

## Notes

These wrappers are intentionally minimal.

They handle a few backend quirks required by the ChatGPT Codex endpoint, including:
- `store=False`
- streaming responses
- structured output via streamed parsed events

That makes them convenient for direct reuse, but they are not meant to hide everything behind a huge abstraction layer.

## Future direction

This repo is meant to grow into a small collection of practical LLM utilities, not a framework.
