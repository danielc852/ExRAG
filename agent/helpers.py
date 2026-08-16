"""Private helpers for agent runtime result processing."""

from __future__ import annotations

from typing import Any


def _message_content(message: Any) -> str:
    """Normalize message content into text for use in ``runtime.py``."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                parts.append(str(block.get("text", "")))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip() if content is not None else ""


def _artifact_payload(artifact: Any) -> dict[str, Any] | None:
    """Normalize a tool artifact into a dictionary for use in ``runtime.py``."""
    if artifact is None:
        return None
    if isinstance(artifact, dict):
        return artifact
    if hasattr(artifact, "model_dump"):
        return artifact.model_dump()
    return None
