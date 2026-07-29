"""Transcode stage detail + FFmpeg stderr progress helpers for live UI."""

from __future__ import annotations

import asyncio
import inspect

import worker
from stages.ffmpeg_progress import iter_ffmpeg_stderr_text, parse_ffmpeg_time_seconds
from stages.transcode_status import (
    build_transcode_status_plan,
    group_split_why,
    patch_group_status,
    stage_detail_from_artifacts,
    summarize_transcode_status,
)
from services.upload.list_detail import (
    _stage_detail_for_upload_row,
    slim_output_artifacts_for_ui,
)


def test_parse_ffmpeg_time_seconds_handles_cr_style_progress():
    assert parse_ffmpeg_time_seconds("frame=10 time=00:00:12.50 bitrate=1000kbits/s") == 12.5
    assert parse_ffmpeg_time_seconds("no time here") is None


def test_iter_ffmpeg_stderr_splits_on_carriage_return():
    class _FakeStderr:
        def __init__(self, chunks):
            self._chunks = list(chunks)

        async def read(self, n):
            if not self._chunks:
                return b""
            return self._chunks.pop(0)

    async def _run():
        stderr = _FakeStderr(
            [
                b"frame=1 time=00:00:01.00\rframe=2 time=00:00:02.00\r",
                b"frame=3 time=00:00:03.00\n",
            ]
        )
        parts = [p async for p in iter_ffmpeg_stderr_text(stderr)]
        assert any("time=00:00:01.00" in p for p in parts)
        assert any("time=00:00:02.00" in p for p in parts)
        assert any("time=00:00:03.00" in p for p in parts)

    asyncio.run(_run())


def test_transcode_status_summary_explains_split_encodes():
    plan = build_transcode_status_plan(
        source={"width": 1920, "height": 1080, "duration_sec": 60, "tier": "1080p"},
        groups=[
            {
                "platforms": ["youtube"],
                "canonical": "youtube",
                "target": "1080x1920",
                "status": "encoding",
                "encode_pct": 40,
            },
            {
                "platforms": ["tiktok", "instagram"],
                "canonical": "tiktok",
                "target": "1080x1920",
                "status": "pending",
            },
        ],
    )
    assert plan["groups_total"] == 2
    assert "YouTube" in plan["summary"] or "youtube" in plan["summary"].lower()
    assert "1080p" in plan["summary"]
    assert "copyright" in group_split_why(["youtube"], total_groups=2).lower()

    updated = patch_group_status(plan, canonical="youtube", group_status="done", encode_pct=100)
    assert updated["groups_done"] >= 1
    done_summary = summarize_transcode_status({**updated, "phase": "done"})
    assert "ready" in done_summary.lower() or "encode" in done_summary.lower()


def test_stage_detail_from_artifacts_and_list_row():
    arts = {
        "transcode_status": {
            "summary": "3 separate encodes · encoding Youtube 1080x1920 40%",
            "groups": [],
        },
        "hydration_blob": {"x": 1},
    }
    assert "separate encodes" in (stage_detail_from_artifacts(arts) or "")
    slim = slim_output_artifacts_for_ui(arts)
    assert "transcode_status" in slim
    assert "hydration_blob" not in slim
    row = {
        "status": "processing",
        "processing_stage": "transcode",
        "output_artifacts": arts,
    }
    assert "separate encodes" in (_stage_detail_for_upload_row(row) or "")
    done = {"status": "succeeded", "output_artifacts": arts}
    assert _stage_detail_for_upload_row(done) is None


def test_deduplicated_transcode_persists_status_and_avoids_readline():
    src = inspect.getsource(worker._run_deduplicated_transcode)
    assert "transcode_status" in src
    assert "iter_ffmpeg_stderr_text" in src
    assert "parse_ffmpeg_time_seconds" in src
    assert ".readline" not in src
    assert "build_transcode_status_plan" in src
