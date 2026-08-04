"""Unit tests for OpenAI caption model resolution (Settings → M8)."""

from __future__ import annotations

import os

import pytest

from core.openai_caption_model import (
    ALLOWED_OPENAI_CAPTION_MODELS,
    DEFAULT_OPENAI_CAPTION_MODEL,
    normalize_openai_caption_model,
    resolve_openai_caption_model,
)


def test_default_is_gpt4o():
    assert DEFAULT_OPENAI_CAPTION_MODEL == "gpt-4o"
    assert "gpt-4o" in ALLOWED_OPENAI_CAPTION_MODELS


def test_normalize_allowlist_and_unknown():
    assert normalize_openai_caption_model("gpt-4o-mini") == "gpt-4o-mini"
    assert normalize_openai_caption_model("gpt-4o") == "gpt-4o"
    assert normalize_openai_caption_model("gpt-4-turbo") == "gpt-4-turbo"
    assert normalize_openai_caption_model("bogus-model") == "gpt-4o"
    assert normalize_openai_caption_model("") == "gpt-4o"
    assert normalize_openai_caption_model(None) == "gpt-4o"


def test_resolve_priority_override_wins():
    us = {
        "_openai_model_override": "gpt-4o-mini",
        "trillOpenaiModel": "gpt-4o",
        "trill_openai_model": "gpt-4-turbo",
        "openai_model": "gpt-4o",
    }
    assert resolve_openai_caption_model(us) == "gpt-4o-mini"


def test_resolve_camel_before_snake():
    us = {
        "trillOpenaiModel": "gpt-4-turbo",
        "trill_openai_model": "gpt-4o-mini",
    }
    assert resolve_openai_caption_model(us) == "gpt-4-turbo"


def test_resolve_snake_when_camel_missing():
    assert resolve_openai_caption_model({"trill_openai_model": "gpt-4o-mini"}) == "gpt-4o-mini"


def test_resolve_legacy_openai_model():
    assert resolve_openai_caption_model({"openai_model": "gpt-4-turbo"}) == "gpt-4-turbo"


def test_resolve_empty_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("OPENAI_CAPTION_MODEL", raising=False)
    assert resolve_openai_caption_model({}) == "gpt-4o"
    assert resolve_openai_caption_model(None) == "gpt-4o"


def test_resolve_env_when_prefs_empty(monkeypatch):
    monkeypatch.setenv("OPENAI_CAPTION_MODEL", "gpt-4o-mini")
    assert resolve_openai_caption_model({}) == "gpt-4o-mini"
    monkeypatch.setenv("OPENAI_CAPTION_MODEL", "not-a-real-model")
    assert resolve_openai_caption_model({}) == "gpt-4o"


def test_baseline_defaults_use_gpt4o():
    from core.upload_baseline_defaults import UNIVERSAL_UPLOAD_BASELINE

    assert UNIVERSAL_UPLOAD_BASELINE.get("trillOpenaiModel") == "gpt-4o"
    assert UNIVERSAL_UPLOAD_BASELINE.get("trill_openai_model") == "gpt-4o"
