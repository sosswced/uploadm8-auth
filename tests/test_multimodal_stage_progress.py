"""Multimodal parallel work must leave the transcode UI label immediately."""

from __future__ import annotations

import inspect

import worker
from stages.transcode_status import stage_detail_from_artifacts


def test_worker_marks_multimodal_stages_before_gather():
    src = inspect.getsource(worker.run_processing_pipeline)
    assert "_mark_mm_stage" in src
    assert 'detail="Analyzing audio + AI scene scan' in src
    assert '"video_intelligence"' in src
    assert "Leave \"Building platform formats\"" in src or "Building platform formats" in src


def test_stage_detail_prefers_live_stage_status_over_transcode_plan():
    arts = {
        "transcode_status": {"summary": "3 separate encodes · encoding Youtube 40%"},
        "stage_status": {
            "summary": "AI scene scan — deep video analysis (proxy + tracks)",
            "stage": "video_intelligence",
        },
    }
    detail = stage_detail_from_artifacts(arts) or ""
    assert "deep video analysis" in detail
    assert "separate encodes" not in detail
