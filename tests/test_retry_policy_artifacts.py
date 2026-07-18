"""Retry metadata must tolerate list-shaped output_artifacts (UPLOADM8-7W)."""

from services.retry_policy import bump_retry_metadata, get_retry_count


def test_bump_retry_metadata_list_artifacts():
    legacy = [{"ai_pipeline_trace_v1": '{"upload_id": "x"}'}]
    out = bump_retry_metadata(
        legacy,
        actor_user_id="user-1",
        prior_error_code="INTERNAL",
        mode="full",
    )
    assert out["ai_pipeline_trace_v1"] == '{"upload_id": "x"}'
    assert out["retry"]["count"] == 1
    assert out["retry"]["last_mode"] == "full"


def test_get_retry_count_list_with_retry_section():
    legacy = [{"retry": {"count": 3}}]
    assert get_retry_count(legacy) == 3


def test_bump_retry_metadata_none():
    out = bump_retry_metadata(
        None,
        actor_user_id="user-1",
        prior_error_code=None,
        mode="partial",
        retry_platforms=["youtube"],
    )
    assert out["retry"]["count"] == 1
    assert out["retry"]["history"][0]["platforms"] == ["youtube"]
