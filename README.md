# llm_tools

Small, copyable Python helpers for provider-based LLM workflows.

The public API is client-oriented:

- `AsyncLLMClient` is the primary async client.
- `LLMClient` is a sync facade for scripts.
- Providers implement the backend-specific details.
- `CodexProvider` is the first concrete provider.

## Requirements

- Python 3.10+
- `openai`
- `pydantic`
- `openai-agents`

## Install

From this repo checkout:

```bash
pip install .
```

From another project by Git URL:

```bash
pip install "git+https://github.com/MasterGTFX/llm_tools.git"
```

For editable local development:

```bash
pip install -e "D:\llm_tools[dev]"
```

## Usage

### Async text

```python
from llm_tools import AsyncLLMClient, CodexProvider

client = AsyncLLMClient(
    provider=CodexProvider(),
    model="gpt-5.5",
)

result = await client.text("Write a small Python function that adds two numbers.")
print(result.output)
```

### Structured output

```python
from pydantic import BaseModel


class Extraction(BaseModel):
    company: str
    score: int


result = await client.model(
    "Extract company and score from: FutureApps AI scored 91.",
    Extraction,
)
print(result.output)
```

### Tools

```python
from agents import function_tool


@function_tool
def get_stock_price(ticker: str) -> str:
    return f"{ticker.upper()} is 123.45 USD"


result = await client.tools(
    "What is the price of AAPL? Use the tool.",
    tools=[get_stock_price],
)
print(result.output)
```

### Image generation

```python
result = await client.image("Draw a simple red square icon on a white background.")

with open("red-square.png", "wb") as f:
    f.write(result.output.image_bytes)
```

### Sync facade

```python
from llm_tools import LLMClient, CodexProvider

client = LLMClient(provider=CodexProvider())
result = client.text("Say hello.")
print(result.output)
```

`LLMClient` cannot be used from an already-running event loop. Use
`AsyncLLMClient` in async applications.

### Auth

`CodexProvider()` uses `CodexLocalAuth` by default. It checks
`OPENAI_CODEX_ACCESS_TOKEN` first, then local Codex auth files:

- `~/.codex/auth.json`
- `~/.config/codex/auth.json`
- `%APPDATA%\codex\auth.json` on Windows, when available

For an explicit token, pass `StaticTokenAuth`:

```python
from llm_tools import CodexProvider, StaticTokenAuth

provider = CodexProvider(auth=StaticTokenAuth("your-token"))
```
