"""Ollama chat-model provider."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


DEFAULT_OLLAMA_MODEL = "hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaProvider:
    """Validate and construct LangChain-compatible Ollama chat models."""

    def validate(self, base_url: str, model_name: str) -> None:
        request = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {base_url}. Start Ollama and try again. ({exc})"
            ) from exc
        installed = {
            str(item.get("name") or item.get("model"))
            for item in payload.get("models", [])
            if isinstance(item, dict)
        }
        requested = (
            model_name if ":" in model_name else f"{model_name}:latest"
        ).casefold()
        normalized = {
            (name if ":" in name else f"{name}:latest").casefold()
            for name in installed
        }
        if requested not in normalized:
            raise RuntimeError(
                f"Ollama model {model_name!r} is missing. Run: ollama pull {model_name}"
            )

    def create_chat_model(
        self,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        *,
        base_url: str | None = None,
        temperature: float = 0,
    ):
        from langchain_ollama import ChatOllama

        resolved_url = base_url or DEFAULT_OLLAMA_URL
        self.validate(resolved_url, model_name)
        return ChatOllama(
            model=model_name,
            base_url=resolved_url,
            temperature=temperature,
        )
