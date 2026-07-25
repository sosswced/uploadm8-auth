"""OpenAI image-edit thumbnail fallback (Phase 3)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from services.openai_thumbnail_edit import (
    MAX_EDIT_CALLS_PER_UPLOAD,
    build_openai_edit_prompt,
    finalize_platform_cover,
    openai_thumbnail_edit_eligible,
    openai_thumbnail_edit_enabled,
)
from stages.thumbnail_stage import _thumbnail_styled_render_order


def test_kill_switch_off_is_legacy_behavior(monkeypatch):
    monkeypatch.delenv("OPENAI_THUMBNAIL_EDIT_ENABLED", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert openai_thumbnail_edit_enabled() is False
    us = {}
    ents = SimpleNamespace(can_ai_thumbnail_styling=True)
    assert openai_thumbnail_edit_eligible(us, ents) is False
    order = _thumbnail_styled_render_order("auto", studio_ok=True, ai_edit_ok=False)
    assert order == ["studio", "template"]
    assert "ai_edit" not in order


def test_eligible_when_enabled_and_tiered(monkeypatch):
    monkeypatch.setenv("OPENAI_THUMBNAIL_EDIT_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    ents = SimpleNamespace(can_ai_thumbnail_styling=True)
    assert openai_thumbnail_edit_eligible({}, ents) is True
    order = _thumbnail_styled_render_order("auto", studio_ok=True, ai_edit_ok=True)
    assert order == ["studio", "ai_edit", "template"]


def test_tier_gate_blocks_without_entitlement(monkeypatch):
    monkeypatch.setenv("OPENAI_THUMBNAIL_EDIT_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    ents = SimpleNamespace(can_ai_thumbnail_styling=False)
    assert openai_thumbnail_edit_eligible({}, ents) is False


def test_build_prompt_carries_hero_facts_and_do_not_invent():
    prompt = build_openai_edit_prompt(
        {"selected_headline": "TOMATO HARVEST", "color_mood": "blue_white"},
        {
            "subject": "raised-bed tomato harvest",
            "hero_facts": [
                {"text": "ripe roma tomatoes"},
                {"text": "first harvest of season"},
            ],
            "do_not_invent": ["no verified speed data — never state a speed"],
        },
        platform="youtube",
    )
    assert "TOMATO HARVEST" in prompt
    assert "ripe roma tomatoes" in prompt
    assert "never state a speed" in prompt
    assert "do not add people" in prompt.lower()


def test_call_cap_is_two():
    assert MAX_EDIT_CALLS_PER_UPLOAD == 2


def test_finalize_platform_cover_youtube(tmp_path):
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")
    raw = tmp_path / "raw.png"
    Image.new("RGB", (1536, 1024), color=(40, 80, 120)).save(raw)
    out = tmp_path / "yt.jpg"
    assert finalize_platform_cover(raw, "youtube", out) is True
    assert out.exists() and out.stat().st_size >= 2048
    with Image.open(out) as img:
        assert img.size == (1280, 720)


def test_generate_openai_edited_cover_fail_soft_no_key(tmp_path, monkeypatch):
    import asyncio

    from services import openai_thumbnail_edit as mod

    monkeypatch.setattr(mod, "OPENAI_API_KEY", "")
    base = tmp_path / "frame.jpg"
    base.write_bytes(b"\xff\xd8\xff" + b"0" * 3000)
    out = tmp_path / "out.png"
    ok = asyncio.run(
        mod.generate_openai_edited_cover(base, "prompt", "16:9", out, upload_id="u1")
    )
    assert ok is False
    assert not out.exists()
