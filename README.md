# llm_tools

Small, copyable Python helpers for LLM workflows.

This repo starts with a few lightweight modules for using the ChatGPT Codex backend from the OpenAI Python SDK:

- `config.py` - shared config with env-first defaults
- `codex.py` - synchronous helpers
- `codex_async.py` - asynchronous helpers
- `codex_agent.py` - synchronous OpenAI Agents SDK helpers with tools

The goal is to stay simple, readable, and easy to copy into other projects.

## What is included

### `config.py`
Shared config values.

Each default first checks environment variables, then falls back to a built-in value.

Supported env vars:

- `OPENAI_CODEX_ACCESS_TOKEN`
- `OPENAI_CODEX_BASE_URL`
- `OPENAI_CODEX_MODEL`
- `OPENAI_CODEX_SYSTEM_PROMPT`
- `OPENAI_CODEX_MAX_RETRIES`
- `OPENAI_CODEX_TIMEOUT`

### `codex.py`
Sync helpers:

- `codex_generate_text(...)`
- `codex_generate_model(...)`

### `codex_async.py`
Async helpers:

- `codex_generate_text_async(...)`
- `codex_generate_model_async(...)`

### `codex_agent.py`
Sync tool-enabled helpers built on the OpenAI Agents SDK:

- `codex_agent_generate_text(...)`
- `codex_agent_generate_model(...)`

These can optionally return the final backend response id with:
- `return_response_id=True`

Both modules:
- work as importable Python modules
- support plain text generation
- support structured output with Pydantic models
- support optional conversation history
- can optionally return updated history
- expose SDK retry and timeout settings

## Requirements

- Python 3.10+
- `openai`
- `pydantic`
- `openai-agents`

Install:

```bash
pip install -r requirements.txt
```

or:

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
    max_retries=3,
    timeout=90,
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

### Agent tool calling

```python
from agents import function_tool
from codex_agent import codex_agent_generate_text

@function_tool
def get_stock_price(ticker: str) -> str:
    """Get a fake current stock price for a ticker."""
    return f"{ticker.upper()} is 123.45 USD"

text = codex_agent_generate_text(
    "What is the price of AAPL? Use the tool.",
    tools=[get_stock_price],
    system_prompt="You are a concise assistant. Use tools when useful.",
)

print(text)
```

### Agent response ids

```python
from agents import function_tool
from codex_agent import codex_agent_generate_text

@function_tool
def get_stock_price(ticker: str) -> str:
    return f"{ticker.upper()} is 123.45 USD"

text, response_id = codex_agent_generate_text(
    "What is the price of AAPL? Use the tool.",
    tools=[get_stock_price],
    return_response_id=True,
)
```

Note: this Codex backend returns a response id, but currently rejects `previous_response_id`, so native multi-turn response chaining is not supported here.

## Retry behavior

The wrappers use the OpenAI SDK's built-in retry support.

You can control it with:
- `max_retries` (default: `2`)
- `timeout` (default: `60` seconds)

Example:

```python
text = codex_generate_text(
    "Explain retry handling briefly.",
    max_retries=4,
    timeout=120,
)
```

This is cleaner than adding a custom backoff decorator unless you later need very specific retry rules.

## Notes

These wrappers are intentionally minimal.

They handle a few backend quirks required by the ChatGPT Codex endpoint, including:
- `store=False`
- streaming responses
- structured output via streamed parsed events

That makes them convenient for direct reuse, but they are not meant to hide everything behind a huge abstraction layer.

## Future direction

This repo is meant to grow into a small collection of practical LLM utilities, not a framework.
