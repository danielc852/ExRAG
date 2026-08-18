"""Shared provider connection settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


OPENROUTER_API_KEY_ENV_VAR = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL_ENV_VAR = "OPENROUTER_BASE_URL"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_TIMEOUT_SECONDS = 120.0
DEFAULT_OPENROUTER_HEADERS = {"X-Title": "ExRAG"}


@dataclass(frozen=True)
class OpenRouterConfig:
    """Connection settings shared by the chat and embedding adapters."""

    api_key: str
    base_url: str = DEFAULT_OPENROUTER_BASE_URL
    timeout_seconds: float = DEFAULT_OPENROUTER_TIMEOUT_SECONDS
    headers: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_OPENROUTER_HEADERS)
    )

    def __post_init__(self) -> None:
        api_key = self.api_key.strip()
        base_url = self.base_url.rstrip("/")
        if not api_key:
            raise ValueError("OpenRouter API key must not be empty")
        if not base_url:
            raise ValueError("OpenRouter base URL must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("OpenRouter timeout must be positive")
        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "base_url", base_url)

    @classmethod
    def from_env(cls) -> "OpenRouterConfig":
        api_key = os.getenv(OPENROUTER_API_KEY_ENV_VAR, "").strip()
        if not api_key:
            raise RuntimeError(
                f"{OPENROUTER_API_KEY_ENV_VAR} is required for OpenRouter"
            )
        return cls(
            api_key=api_key,
            base_url=os.getenv(
                OPENROUTER_BASE_URL_ENV_VAR, DEFAULT_OPENROUTER_BASE_URL
            ),
        )

    def request_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.headers,
        }
