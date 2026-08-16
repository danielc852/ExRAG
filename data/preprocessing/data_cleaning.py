"""Apply deterministic text cleaning before document chunking."""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    normalized = (
        unicodedata.normalize("NFKC", text)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = re.sub(r"\n{4,}", "\n\n\n", normalized)
    return normalized.strip()
