"""Configuration defaults for OpenRouter embedding requests."""

DEFAULT_MODEL = "nvidia/nemotron-3-embed-1b:free"
MODEL_DIMENSIONS = {DEFAULT_MODEL: 2048}

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT_SECONDS = 120.0

API_KEY_ENV_VAR = "OPENROUTER_API_KEY"
BASE_URL_ENV_VAR = "OPENROUTER_BASE_URL"
