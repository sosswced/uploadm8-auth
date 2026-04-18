"""Tests for services.meta_oauth scope modes and permission helpers."""

from __future__ import annotations

import os

import pytest

from services import meta_oauth as mo


def test_minimal_mode_uses_only_three_scopes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("META_OAUTH_MODE", "minimal")
    assert "pages_show_list" in mo.meta_facebook_oauth_scope()
    assert "publish_video" not in mo.meta_facebook_oauth_scope()
    assert "instagram_basic" not in mo.meta_instagram_oauth_scope()


def test_full_mode_includes_publish_and_insights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("META_OAUTH_MODE", "full")
    assert "publish_video" in mo.meta_facebook_oauth_scope()
    assert "instagram_content_publish" in mo.meta_instagram_oauth_scope()


def test_permission_granted_from_blob() -> None:
    td = {
        "meta_permissions": [
            {"permission": "instagram_content_publish", "status": "declined"},
        ]
    }
    assert mo.meta_permission_granted_from_blob(td, "instagram_content_publish") is False
    assert mo.require_instagram_publish(td) is not None


def test_unknown_blob_allows_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("META_OAUTH_MODE", raising=False)
    assert mo.require_instagram_publish({}) is None
