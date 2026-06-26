"""Unit tests for upload list/detail UI payload helpers."""

from services.upload.list_detail import (
    failure_diag_from_upload_row,
    slim_output_artifacts_for_ui,
)


def test_slim_output_artifacts_for_ui_keeps_queue_fields_only():
    raw = {
        "publish_quality_notice": "Low bitrate",
        "content_hotness": {"band": "warm", "score": 0.72},
        "coach_hints": ["Try shorter hook"],
        "failure_phase": "caption",
        "hydration_blob": {"frames": [1, 2, 3]},
        "scene_story": "Should not ship on list cards",
    }
    slim = slim_output_artifacts_for_ui(raw)
    assert slim["publish_quality_notice"] == "Low bitrate"
    assert slim["content_hotness"]["band"] == "warm"
    assert slim["coach_hints"] == ["Try shorter hook"]
    assert slim["failure_phase"] == "caption"
    assert "hydration_blob" not in slim
    assert "scene_story" not in slim


def test_failure_diag_from_explicit_artifact():
    row = {
        "output_artifacts": {
            "failure_diag": {"stage": "watermark", "code": "timeout"},
        }
    }
    fd = failure_diag_from_upload_row(row)
    assert fd == {"stage": "watermark", "code": "timeout"}


def test_failure_diag_from_pipeline_manifest_failed_steps():
    row = {
        "output_artifacts": {},
        "pipeline_manifest": {
            "terminal": "failed",
            "steps": [
                {"name": "download", "status": "ok"},
                {"name": "caption", "status": "failed", "error": "LLM timeout"},
            ],
        },
    }
    fd = failure_diag_from_upload_row(row)
    assert fd["terminal"] == "failed"
    assert len(fd["pipeline_failed_steps"]) == 1
    assert fd["pipeline_failed_steps"][0]["name"] == "caption"


def test_slim_output_artifacts_normalizes_coach_hints_json_string():
    raw = {
        "coach_hints": '[{"id":"geo_signals_unused","message":"Try scenic boost"}]',
        "content_hotness": '{"band":"warm","score":0.6}',
    }
    slim = slim_output_artifacts_for_ui(raw)
    assert isinstance(slim["coach_hints"], list)
    assert slim["coach_hints"][0]["id"] == "geo_signals_unused"
    assert isinstance(slim["content_hotness"], dict)
    assert slim["content_hotness"]["band"] == "warm"


def test_failure_diag_fallback_from_error_code():
    row = {
        "output_artifacts": {"failure_phase": "publish"},
        "error_code": "publish_failed",
        "error_detail": "TikTok rejected media",
    }
    fd = failure_diag_from_upload_row(row)
    assert fd["failure_phase"] == "publish"
    assert fd["error_code"] == "publish_failed"
    assert "TikTok" in fd["error"]
