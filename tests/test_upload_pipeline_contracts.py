"""Cross-layer upload pipeline contracts (queue / scheduled / worker)."""

from services.upload.r2_storage_guard import ERROR_SOURCE_NOT_IN_R2
from services.upload.schedule_guard import ERROR_SCHEDULE_INCOMPLETE, UPLOAD_ERROR_MESSAGES
from services.upload.status import (
    CANCELLABLE_STATUSES,
    REQUEUEABLE_ERROR_CODES,
    SCHEDULED_PIPELINE_STATUSES,
    is_requeueable_upload,
)


def test_requeueable_error_codes_have_user_messages():
    for code in REQUEUEABLE_ERROR_CODES:
        assert code in UPLOAD_ERROR_MESSAGES, f"missing friendly message for {code}"


def test_cancellable_narrower_than_scheduled_pipeline():
    for st in CANCELLABLE_STATUSES:
        assert st in SCHEDULED_PIPELINE_STATUSES
    for st in ("staged", "ready_to_publish", "processing"):
        assert st not in CANCELLABLE_STATUSES


def test_is_requeueable_only_pending_with_allowed_codes():
    assert is_requeueable_upload("pending", "ENQUEUE_FAILED")
    assert is_requeueable_upload("pending", "SCHEDULE_INCOMPLETE")
    assert is_requeueable_upload("pending", "PUBLISH_SLOT_MISSING")
    assert not is_requeueable_upload("staged", "ENQUEUE_FAILED")
    assert not is_requeueable_upload("pending", "PLATFORM_AUTH_FAILED")


def test_schedule_attention_codes_are_requeueable_or_messaged():
    from services.upload.status import SCHEDULE_ATTENTION_ERROR_CODES

    for code in SCHEDULE_ATTENTION_ERROR_CODES:
        assert code in UPLOAD_ERROR_MESSAGES, f"missing friendly message for {code}"
    assert "PUBLISH_SLOT_MISSING" in REQUEUEABLE_ERROR_CODES
    assert "PUBLISH_SLOT_MISSING" in SCHEDULE_ATTENTION_ERROR_CODES


def test_three_requeue_retry_ux_paths():
    """Lock the three user-facing recovery paths for schedule/enqueue failures."""
    from services.upload.status import (
        SCHEDULE_ATTENTION_ERROR_CODES,
        is_retryable_upload,
    )

    # 1) Pending + schedule/enqueue errors → Re-queue (Scheduled + Queue)
    for code in (
        "ENQUEUE_FAILED",
        "QUEUE_UNAVAILABLE",
        "SCHEDULE_INCOMPLETE",
        "PUBLISH_SLOT_MISSING",
    ):
        assert is_requeueable_upload("pending", code), code
        assert not is_retryable_upload("pending", error_code=code), code

    # 2) Failed PUBLISH_SLOT_MISSING (and related) → Retry (slots rebuild on retry)
    for code in SCHEDULE_ATTENTION_ERROR_CODES:
        assert is_retryable_upload("failed", error_code=code), code
        assert not is_requeueable_upload("failed", code), code

    # 3) Queue Retry path for generic failed still works
    assert is_retryable_upload("failed", error_code=None)
    assert is_retryable_upload("failed", error_code="INTERNAL")
    assert is_retryable_upload("partial", has_failed_platform=True)
    assert not is_retryable_upload("failed", error_code="PLATFORM_AUTH_FAILED")
    assert not is_retryable_upload("failed", error_code="SOURCE_NOT_IN_R2")


def test_schedule_incomplete_message_present():
    assert ERROR_SCHEDULE_INCOMPLETE in UPLOAD_ERROR_MESSAGES


def test_source_not_in_r2_message_present():
    assert ERROR_SOURCE_NOT_IN_R2 in UPLOAD_ERROR_MESSAGES
