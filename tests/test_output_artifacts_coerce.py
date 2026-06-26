"""Tests for output_artifacts coercion and pikzels warning helper."""

import json

from core.helpers import coerce_output_artifacts_dict
from services.upload.thumbnails import pikzels_template_thumbnail_warning


def test_coerce_output_artifacts_dict_list_legacy_row():
    legacy = [json.dumps({"thumbnail_render_method": "template", "studio_render_report": "{}"})]
    assert coerce_output_artifacts_dict(legacy) == {}


def test_coerce_output_artifacts_dict_normal_dict():
    raw = {"thumbnail_render_method": "studio_renderer"}
    assert coerce_output_artifacts_dict(raw) == raw


def test_pikzels_warning_tolerates_list_artifacts(monkeypatch):
    monkeypatch.setenv("PIKZELS_API_KEY", "test-key")
    legacy = [json.dumps({"thumbnail_render_method": "template"})]
    # Should not raise AttributeError on list artifacts.
    result = pikzels_template_thumbnail_warning(legacy)
    assert result is None or isinstance(result, dict)
