"""Rules for the complete frozen held-out evaluation dataset."""

from __future__ import annotations

from data import BenchmarkQuestion

def get_full_questions(
    questions: list[BenchmarkQuestion],
) -> list[BenchmarkQuestion]:
    """Return every frozen question in the held-out evaluation dataset."""
    return questions
