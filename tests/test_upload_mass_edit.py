"""Mass-edit + writing-mix merge for pending upload metadata."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from core.models import UploadMassEditBody, UploadUpdate
from services.uploads_api import (
    merge_caption_creative_prefs,
    normalize_platforms_list,
)


def test_normalize_platforms_ok():
    assert normalize_platforms_list(["YouTube", "tiktok", "youtube"]) == ["youtube", "tiktok"]


def test_normalize_platforms_rejects_empty():
    with pytest.raises(HTTPException):
        normalize_platforms_list([])


def test_merge_caption_creative_fixed_knobs():
    prefs, applied = merge_caption_creative_prefs(
        {"captionStyle": "story"},
        caption_style="freestyle",
        caption_tone="hype",
        caption_voice="teacher",
    )
    assert prefs["captionStyle"] == "freestyle"
    assert prefs["caption_style"] == "freestyle"
    assert prefs["captionTone"] == "hype"
    assert prefs["captionVoice"] == "teacher"
    assert prefs["randomizeCaptionCreative"] is False
    assert prefs["captionCreativePickMode"] == "off"
    assert applied["captionStyle"] == "freestyle"
    assert applied["captionVoice"] == "teacher"


def test_merge_caption_creative_randomize():
    prefs, applied = merge_caption_creative_prefs({}, randomize_writing_mix=True)
    assert prefs["randomizeCaptionCreative"] is True
    assert prefs["captionCreativePickMode"] == "random"
    assert applied["captionStyle"]
    assert applied["captionTone"]
    assert applied["captionVoice"]


def test_upload_update_aliases():
    u = UploadUpdate.model_validate(
        {
            "captionStyle": "punchy",
            "captionTone": "hype",
            "captionVoice": "coach",
            "randomizeWritingMix": False,
            "platforms": ["facebook"],
        }
    )
    assert u.caption_style == "punchy"
    assert u.caption_tone == "hype"
    assert u.caption_voice == "coach"
    assert u.platforms == ["facebook"]


def test_mass_edit_body_aliases():
    b = UploadMassEditBody.model_validate(
        {
            "uploadIds": ["11111111-1111-1111-1111-111111111111"],
            "randomizeWritingMix": True,
            "shiftMinutes": 15,
        }
    )
    assert b.randomize_writing_mix is True
    assert b.shift_minutes == 15
    assert len(b.upload_ids) == 1


def test_update_upload_metadata_writes_mix():
    import asyncio
    from services import uploads_api as ua

    captured = {}

    class FakeConn:
        async def fetchrow(self, *_a, **_k):
            return {
                "id": "u1",
                "status": "pending",
                "platforms": ["facebook"],
                "schedule_metadata": {},
                "scheduled_time": None,
                "user_preferences": json.dumps({"captionStyle": "story"}),
            }

        async def execute(self, sql, *params):
            captured["sql"] = sql
            captured["params"] = params

    update = UploadUpdate(
        title="Hello",
        caption_style="freestyle",
        caption_tone="hype",
        caption_voice="teacher",
    )
    applied = asyncio.run(ua.update_upload_metadata(FakeConn(), "u1", "user-1", update))
    assert applied["title"] == "Hello"
    assert applied["writing_mix"]["captionStyle"] == "freestyle"
    assert "user_preferences" in captured["sql"]
    prefs_blob = [p for p in captured["params"] if isinstance(p, str) and "freestyle" in p]
    assert prefs_blob
