"""OpenRouter embedding API integration."""

from .client import OpenRouterAPIError, OpenRouterClient
from .config import DEFAULT_MODEL
from .embedder import OpenRouterEmbedder

__all__ = [
    "DEFAULT_MODEL",
    "OpenRouterAPIError",
    "OpenRouterClient",
    "OpenRouterEmbedder",
]
