"""Language model construction for ExRAG agents."""

from __future__ import annotations


DEFAULT_OLLAMA_MODEL = "hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M"


def create_ollama_model(
    model_name: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = "http://localhost:11434",
    temperature: float = 0,
):
    """Create the Ollama chat model used by an agent."""
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model_name, base_url=base_url, temperature=temperature)
