# AGENTS

## What This Repo Is
`llm_tools` is a small Python helper library for OpenAI/Codex-style LLM workflows. It provides thin, copyable wrappers for:
- plain Responses API usage (sync + async)
- tool-enabled Agents SDK usage (sync facade + async)
- optional replayable chat history across calls

## Main Concepts
- `codex/config.py`: environment-based defaults (`OPENAI_CODEX_*`)
- `codex/helpers.py`: shared types, input/history helpers, client builders, errors
- `codex/sync.py` and `codex/async_client.py`: Responses API wrappers for text and Pydantic models
- `codex/agent_async.py`: Agents SDK execution with tools and replayable history
- `codex/agent.py`: sync wrappers over async agent functions
- `codex/__init__.py`: package-level re-exports

## Expected Code/Style Approach
- Keep public APIs minimal and backwards-compatible.
- Prefer small, explicit helper functions over broad abstractions.
- Deduplicate repeated logic only when behavior stays identical.
- Keep errors clear and actionable (missing token, empty output, wrong type).
- Maintain type hints and concise doc examples.
- Validation should stay lightweight (import/compile checks first).
