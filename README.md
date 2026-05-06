# llm_tools

Small, copyable Python helpers for LLM workflows.

## Included

### `codex/`
Helpers for the ChatGPT Codex backend.

Modules:
- `codex.config` - env-based defaults
- `codex.helpers` - shared internal helpers and errors
- `codex.sync` - sync Responses API wrappers
- `codex.async_client` - async Responses API wrappers
- `codex.agent` - sync Agents SDK wrappers with tools
- `codex.agent_async` - async Agents SDK wrappers with tools
- `codex.__init__` - convenience re-exports

Package-level exports include:
- `codex_generate_text(...)`
- `codex_generate_model(...)`
- `codex_generate_text_async(...)`
- `codex_generate_model_async(...)`
- `codex_agent_generate_text(...)`
- `codex_agent_generate_model(...)`
- `codex_agent_generate_text_async(...)`
- `codex_agent_generate_model_async(...)`

For detailed usage and examples, see:
- `codex/README.md`

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
pip install "git+https://github.com/<owner>/<repo>.git"
```

For editable local development:

```bash
pip install -e "D:\llm_tools[dev]"
```

## Usage

After install, import the package as `codex`:

```python
from codex import codex_generate_text

result = codex_generate_text(
    "Write a small Python function that adds two numbers.",
    access_token="your-token",
)
print(result)
```
