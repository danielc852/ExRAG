"""Language model construction for Enterprise RAG agents."""

from __future__ import annotations


def create_ollama_model(
    model_name: str = "qwen3:8b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0,
):
    """Create the Ollama chat model used by an agent."""
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model_name, base_url=base_url, temperature=temperature)
