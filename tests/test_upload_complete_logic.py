"""Tests for complete-upload idempotency status sets."""

from services.upload.status import (
    COMPLETE_IDEMPOTENT_STATUSES,
    COMPLETED_STATUSES,
    PARTIAL_STATUSES,
)


def test_idempotent_includes_terminal_success_buckets():
    for st in COMPLETED_STATUSES:
        assert st in COMPLETE_IDEMPOTENT_STATUSES
    for st in PARTIAL_STATUSES:
        assert st in COMPLETE_IDEMPOTENT_STATUSES


def test_pending_is_only_mutable_complete_entry():
    """Only pending should transition on /complete; others are no-op success."""
    mutable = {"pending"}
    idempotent = set(COMPLETE_IDEMPOTENT_STATUSES)
    assert mutable.isdisjoint(idempotent)
