"""Chat-model provider selection for ExRAG."""

from __future__ import annotations

from typing import Literal

from .config import OpenRouterConfig
from .ollama import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL, OllamaProvider
from .openrouter import OpenRouterProvider


LLMProviderName = Literal["ollama", "openrouter"]


def get_provider(name: str) -> OllamaProvider | OpenRouterProvider:
    """Return the selected provider adapter, not a concrete chat model."""
    if name == "ollama":
        return OllamaProvider()
    if name == "openrouter":
        return OpenRouterProvider()
    raise ValueError(f"Unsupported LLM provider: {name!r}")


__all__ = [
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_OLLAMA_URL",
    "LLMProviderName",
    "OllamaProvider",
    "OpenRouterConfig",
    "OpenRouterProvider",
    "get_provider",
]
