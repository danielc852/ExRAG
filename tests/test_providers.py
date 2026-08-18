from __future__ import annotations

import pytest

from providers import OllamaProvider, OpenRouterConfig, OpenRouterProvider, get_provider


def test_get_provider_returns_provider_objects_and_rejects_unknown():
    assert isinstance(get_provider("ollama"), OllamaProvider)
    assert isinstance(get_provider("openrouter"), OpenRouterProvider)
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_provider("unknown")


def test_openrouter_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterProvider().create_chat_model("openai/gpt-4.1-mini")


def test_openrouter_constructs_tool_compatible_chat_model():
    model = OpenRouterProvider(
        OpenRouterConfig(
            api_key="secret",
            base_url="https://openrouter.test/api/v1/",
            timeout_seconds=17,
        )
    ).create_chat_model("openai/gpt-4.1-mini")

    assert model.model_name == "openai/gpt-4.1-mini"
    assert model.openai_api_base == "https://openrouter.test/api/v1"
    assert model.request_timeout == 17
    assert model.default_headers == {"X-Title": "ExRAG"}
    assert callable(model.bind_tools)


def test_openrouter_config_is_shared_with_embedding_client(monkeypatch):
    from data.processing.embeddings.openrouter import OpenRouterClient

    monkeypatch.setenv("OPENROUTER_API_KEY", "shared-secret")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.test/api/v1/")

    client = OpenRouterClient.from_env()

    assert client._base_url == "https://openrouter.test/api/v1"
    assert client._timeout_seconds == 120
    assert client._headers["Authorization"] == "Bearer shared-secret"
    assert client._headers["X-Title"] == "ExRAG"
