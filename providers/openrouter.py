"""OpenRouter chat-model provider."""

from __future__ import annotations

from .config import OpenRouterConfig


class OpenRouterProvider:
    """Construct LangChain-compatible OpenRouter chat models."""

    def __init__(self, config: OpenRouterConfig | None = None) -> None:
        self._config = config

    def create_chat_model(
        self,
        model_name: str,
        *,
        base_url: str | None = None,
        temperature: float = 0,
    ):
        if not model_name.strip():
            raise ValueError("OpenRouter chat model must not be empty")
        from langchain_openai import ChatOpenAI

        config = self._config or OpenRouterConfig.from_env()
        return ChatOpenAI(
            model=model_name,
            api_key=config.api_key,
            base_url=base_url or config.base_url,
            timeout=config.timeout_seconds,
            temperature=temperature,
            default_headers=config.headers,
        )
