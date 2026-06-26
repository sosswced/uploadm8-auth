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
    assert not is_requeueable_upload("staged", "ENQUEUE_FAILED")
    assert not is_requeueable_upload("pending", "PLATFORM_AUTH_FAILED")


def test_schedule_incomplete_message_present():
    assert ERROR_SCHEDULE_INCOMPLETE in UPLOAD_ERROR_MESSAGES


def test_source_not_in_r2_message_present():
    assert ERROR_SOURCE_NOT_IN_R2 in UPLOAD_ERROR_MESSAGES
