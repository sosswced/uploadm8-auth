"""Unit tests for upload stage labels."""

from services.upload.stage_labels import STAGE_LABELS, stage_label_for


def test_stage_label_for_known_stages():
    assert stage_label_for("download") == "Copying your video"
    assert stage_label_for("publish") == "Publishing to platforms"
    assert stage_label_for("PUBLISH") == "Publishing to platforms"


def test_stage_label_for_unknown_stage_title_cases():
    assert stage_label_for("custom_step") == "Custom Step"


def test_stage_label_for_empty():
    assert stage_label_for(None) == "Processing"
    assert stage_label_for("") == "Processing"


def test_stage_labels_has_core_pipeline_keys():
    for key in ("transcode", "thumbnail", "caption", "publish"):
        assert key in STAGE_LABELS
