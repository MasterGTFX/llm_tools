from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence

from .errors import AuthenticationError


class AuthProvider(Protocol):
    """Supplies an access token for a provider request."""

    def get_token(self) -> str:
        """Return a usable bearer token."""


@dataclass(frozen=True)
class StaticTokenAuth:
    """Manual-token auth for providers that use bearer tokens."""

    token: str

    def get_token(self) -> str:
        if not self.token:
            raise AuthenticationError("A non-empty access token is required")
        return self.token


@dataclass(frozen=True)
class CodexLocalAuth:
    """Auth provider that checks env first, then local Codex auth files."""

    env_var: str = "OPENAI_CODEX_ACCESS_TOKEN"
    paths: Optional[Sequence[Path | str]] = None

    def get_token(self) -> str:
        env_token = os.getenv(self.env_var, "").strip()
        if env_token:
            return env_token

        checked_paths: list[str] = []
        for path in self._candidate_paths():
            checked_paths.append(str(path))
            token = self._read_token(path)
            if token:
                return token

        raise AuthenticationError(
            f"{self.env_var} is not set and no Codex access token was found in: "
            f"{', '.join(checked_paths)}"
        )

    def _candidate_paths(self) -> list[Path]:
        if self.paths is not None:
            return [Path(path).expanduser() for path in self.paths]

        paths = [
            Path.home() / ".codex" / "auth.json",
            Path.home() / ".config" / "codex" / "auth.json",
        ]
        appdata = os.getenv("APPDATA")
        if appdata:
            paths.append(Path(appdata) / "codex" / "auth.json")
        return paths

    def _read_token(self, path: Path) -> Optional[str]:
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AuthenticationError(f"Could not read Codex auth file {path}: {exc}") from exc
        if not isinstance(data, Mapping):
            raise AuthenticationError(f"Codex auth file {path} must contain a JSON object")
        return self._extract_token(data)

    def _extract_token(self, data: Mapping[str, Any]) -> Optional[str]:
        for key in (self.env_var, "access_token", "OPENAI_API_KEY"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        tokens = data.get("tokens")
        if isinstance(tokens, Mapping):
            value = tokens.get("access_token")
            if isinstance(value, str) and value.strip():
                return value.strip()

        return self._find_nested_access_token(data)

    def _find_nested_access_token(self, value: Any) -> Optional[str]:
        if isinstance(value, Mapping):
            token = value.get("access_token")
            if isinstance(token, str) and token.strip():
                return token.strip()
            for nested in value.values():
                found = self._find_nested_access_token(nested)
                if found:
                    return found
        elif isinstance(value, list):
            for nested in value:
                found = self._find_nested_access_token(nested)
                if found:
                    return found
        return None
