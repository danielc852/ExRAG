"""Minimal synchronous client for OpenRouter's embeddings endpoint."""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any

from providers.config import OpenRouterConfig

from .config import DEFAULT_BASE_URL, DEFAULT_TIMEOUT_SECONDS


class OpenRouterAPIError(RuntimeError):
    """An OpenRouter request failed or returned an invalid response."""


class OpenRouterClient:
    """Call the OpenRouter embeddings API without an additional SDK dependency."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OpenRouter API key must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("OpenRouter timeout must be positive")
        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._opener = opener
        self._headers = OpenRouterConfig(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout_seconds=timeout_seconds,
        ).request_headers()

    @classmethod
    def from_env(cls) -> "OpenRouterClient":
        config = OpenRouterConfig.from_env()
        return cls(
            config.api_key,
            base_url=config.base_url,
            timeout_seconds=config.timeout_seconds,
        )

    def embed(
        self,
        *,
        model: str,
        texts: Sequence[str],
        input_type: str,
    ) -> list[list[float]]:
        if not model.strip():
            raise ValueError("OpenRouter embedding model must not be empty")
        if input_type not in {"query", "document"}:
            raise ValueError("OpenRouter input_type must be 'query' or 'document'")
        if not texts:
            return []

        request = urllib.request.Request(
            f"{self._base_url}/embeddings",
            data=json.dumps(
                {
                    "model": model,
                    "input": list(texts),
                    "input_type": input_type,
                    "encoding_format": "float",
                }
            ).encode("utf-8"),
            headers=self._headers,
            method="POST",
        )
        try:
            with self._opener(request, timeout=self._timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = self._http_error_detail(exc)
            raise OpenRouterAPIError(
                f"OpenRouter embeddings request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise OpenRouterAPIError(
                f"Cannot reach OpenRouter embeddings API: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise OpenRouterAPIError("OpenRouter embeddings request timed out") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OpenRouterAPIError(
                "OpenRouter embeddings API returned invalid JSON"
            ) from exc

        return self._parse_embeddings(payload, expected_count=len(texts))

    @staticmethod
    def _http_error_detail(exc: urllib.error.HTTPError) -> str:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return exc.reason or "unknown error"
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
        return exc.reason or "unknown error"

    @staticmethod
    def _parse_embeddings(
        payload: Any, *, expected_count: int
    ) -> list[list[float]]:
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise OpenRouterAPIError(
                "OpenRouter embeddings response does not contain a data list"
            )
        data = payload["data"]
        if len(data) != expected_count:
            raise OpenRouterAPIError(
                "OpenRouter embeddings response count does not match the request"
            )

        ordered: list[list[float] | None] = [None] * expected_count
        dimension: int | None = None
        for fallback_index, item in enumerate(data):
            if not isinstance(item, dict):
                raise OpenRouterAPIError("OpenRouter returned an invalid embedding item")
            index = item.get("index", fallback_index)
            embedding = item.get("embedding")
            if (
                not isinstance(index, int)
                or index < 0
                or index >= expected_count
                or ordered[index] is not None
                or not isinstance(embedding, list)
                or not embedding
            ):
                raise OpenRouterAPIError("OpenRouter returned an invalid embedding item")
            try:
                vector = [float(value) for value in embedding]
            except (TypeError, ValueError) as exc:
                raise OpenRouterAPIError(
                    "OpenRouter returned a non-numeric embedding"
                ) from exc
            if not all(math.isfinite(value) for value in vector):
                raise OpenRouterAPIError(
                    "OpenRouter returned a non-finite embedding"
                )
            if dimension is not None and len(vector) != dimension:
                raise OpenRouterAPIError(
                    "OpenRouter returned inconsistent embedding dimensions"
                )
            dimension = len(vector)
            ordered[index] = vector

        if any(vector is None for vector in ordered):
            raise OpenRouterAPIError("OpenRouter omitted an embedding from its response")
        return [vector for vector in ordered if vector is not None]
