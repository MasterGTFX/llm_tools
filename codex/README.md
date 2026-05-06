# codex

Detailed docs for the `codex` package.

## Modules

### `codex.config`
Configuration defaults loaded from environment variables.

Exposes:
- `DEFAULT_BASE_URL`
- `DEFAULT_MODEL`
- `DEFAULT_SYSTEM_PROMPT`
- `DEFAULT_MAX_RETRIES`
- `DEFAULT_TIMEOUT`
- `get_access_token()`

Supported env vars:
- `OPENAI_CODEX_ACCESS_TOKEN`
- `OPENAI_CODEX_BASE_URL`
- `OPENAI_CODEX_MODEL`
- `OPENAI_CODEX_SYSTEM_PROMPT`
- `OPENAI_CODEX_MAX_RETRIES`
- `OPENAI_CODEX_TIMEOUT`

### `codex.helpers`
Shared internal helpers.

Exposes:
- `CodexError`
- `CodexAgentError`
- `build_input(...)`
- `updated_history(...)`
- `get_openai_client(...)`
- `get_async_openai_client(...)`

### `codex.sync`
Synchronous wrappers over the OpenAI Responses API.

Exposes:
- `codex_generate_text(...)`
- `codex_generate_model(...)`

Features:
- plain text generation
- structured output with Pydantic models
- optional plain chat history replay
- optional `return_history=True`

### `codex.async_client`
Asynchronous wrappers over the OpenAI Responses API.

Exposes:
- `codex_generate_text_async(...)`
- `codex_generate_model_async(...)`

Features match `codex.sync`, but async.

### `codex.agent`
Synchronous tool-enabled wrappers built on the OpenAI Agents SDK.

Exposes:
- `codex_agent_generate_text(...)`
- `codex_agent_generate_model(...)`

Notes:
- this is a sync facade over the async implementation
- it raises a clear error if called from an already-running event loop

### `codex.agent_async`
Asynchronous tool-enabled wrappers built on the OpenAI Agents SDK.

Exposes:
- `codex_agent_generate_text_async(...)`
- `codex_agent_generate_model_async(...)`

Features:
- tool calling via `tools=[...]`
- structured output with Pydantic models
- replayable multi-turn tool history
- optional `return_history=True`

### `codex.__init__`
Convenience re-exports for package-level imports.

Example:

```python
from codex import codex_generate_text, codex_agent_generate_text_async
```

## Installation

```bash
pip install -r requirements.txt
```

Dependencies:
- `openai`
- `pydantic`
- `openai-agents`

## Access token

Set:

```bash
export OPENAI_CODEX_ACCESS_TOKEN="your_token_here"
```

Or pass `access_token=` directly.

## Usage

### Sync text generation

```python
from codex.sync import codex_generate_text

text = codex_generate_text(
    "Say hello in one short sentence.",
    system_prompt="You are a helpful coding assistant. Answer concisely.",
)

print(text)
```

### Async structured output

```python
import asyncio
from pydantic import BaseModel
from codex.async_client import codex_generate_model_async

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

### Sync agent tool calling

```python
from agents import function_tool
from codex.agent import codex_agent_generate_text

@function_tool
def get_stock_price(ticker: str) -> str:
    return f"{ticker.upper()} is 123.45 USD"

text = codex_agent_generate_text(
    "What is the price of AAPL? Use the tool.",
    tools=[get_stock_price],
    system_prompt="You are a concise assistant. Use tools when useful.",
)

print(text)
```

### Async agent multi-turn history

```python
import asyncio
from agents import function_tool
from codex.agent_async import codex_agent_generate_text_async

@function_tool
def get_stock_price(ticker: str) -> str:
    return f"{ticker.upper()} is 123.45 USD"

async def main():
    text, history = await codex_agent_generate_text_async(
        "What is the price of AAPL? Use the tool.",
        tools=[get_stock_price],
        return_history=True,
    )

    follow_up, history = await codex_agent_generate_text_async(
        "And what ticker did you just check?",
        tools=[get_stock_price],
        history=history,
        return_history=True,
    )

    print(text)
    print(follow_up)

asyncio.run(main())
```

Returned agent history is replayable for this backend. It includes the items needed for multi-turn continuation, including user messages, assistant `function_call` items, `function_call_output` items, and assistant message items.

## Backend note

The Codex backend returns valid `response.id` values, but currently rejects:

```text
Unsupported parameter: previous_response_id
```

So multi-turn continuation should use replayed history, not native response-id chaining.
