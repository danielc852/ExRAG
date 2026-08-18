"""OpenRouter embedding API integration."""

from .client import OpenRouterAPIError, OpenRouterClient
from .embedder import OpenRouterEmbedder

__all__ = ["OpenRouterAPIError", "OpenRouterClient", "OpenRouterEmbedder"]
