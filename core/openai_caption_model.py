"""
Resolve the OpenAI model used for caption / title / hashtag generation.

Settings UI saves ``trillOpenaiModel`` → ``user_preferences.trill_openai_model``.
Worker may also set ``_openai_model_override``. This helper is the single
allowlisted resolver for caption_stage, trill_content, and worker override write.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Tuple

ALLOWED_OPENAI_CAPTION_MODELS: Tuple[str, ...] = (
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
)

DEFAULT_OPENAI_CAPTION_MODEL = "gpt-4o"


def _env_caption_model_default() -> str:
    raw = (os.environ.get("OPENAI_CAPTION_MODEL") or "").strip()
    if raw in ALLOWED_OPENAI_CAPTION_MODELS:
        return raw
    return DEFAULT_OPENAI_CAPTION_MODEL


def normalize_openai_caption_model(
    value: Any,
    *,
    default: Optional[str] = None,
) -> str:
    """Allowlist a model id; unknown/empty → default (gpt-4o unless overridden)."""
    fallback = (
        default
        if default in ALLOWED_OPENAI_CAPTION_MODELS
        else DEFAULT_OPENAI_CAPTION_MODEL
    )
    v = str(value or "").strip()
    if v in ALLOWED_OPENAI_CAPTION_MODELS:
        return v
    return fallback


def resolve_openai_caption_model(
    user_settings: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Priority:
      1. ``_openai_model_override`` (worker / simulate)
      2. ``trillOpenaiModel`` / ``trill_openai_model``
      3. ``openai_model`` (legacy alias)
      4. ``OPENAI_CAPTION_MODEL`` env / ``gpt-4o``
    """
    us = user_settings or {}
    for key in (
        "_openai_model_override",
        "trillOpenaiModel",
        "trill_openai_model",
        "openai_model",
    ):
        raw = us.get(key)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            continue
        return normalize_openai_caption_model(raw)
    return _env_caption_model_default()


__all__ = [
    "ALLOWED_OPENAI_CAPTION_MODELS",
    "DEFAULT_OPENAI_CAPTION_MODEL",
    "normalize_openai_caption_model",
    "resolve_openai_caption_model",
]
