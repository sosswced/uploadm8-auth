"""Canonical upload status buckets (single source of truth for queue/dashboard)."""

from __future__ import annotations

from typing import Any, Dict

# Status view groupings for queue/dashboard.
#
# UI contract (see frontend queue.html / dashboard.html):
#   processing — anything not yet finalised: pending, scheduled, smart-scheduled,
#                future uploads, staged, queued, ready_to_publish, currently processing.
#   completed  — fully successful uploads only (every platform succeeded).
#   partial    — at least one platform succeeded AND at least one failed.
#   failed     — every platform failed (or upload errored before publish).
#
# `pending` / `staged` are kept as aliases so legacy callers (older clients,
# admin tooling, scheduled.html stats) keep working.
# ── Canonical status buckets (single source of truth) ──────────────────────────
#
# These tuples are imported by routers, services, and serialized into shared
# constants for the frontend (frontend/js/scheduled-status.js) so dashboard,
# queue, and scheduled pages all agree on what counts as "scheduled" / etc.
#
# DO NOT inline these literals in SQL elsewhere. Import from here.

# "Scheduled" / "in the pipeline, not yet started publishing" — the canonical
# definition shared by dashboard.scheduled, scheduled.html list+stats, queue
# pending tab, and edit-permission checks.
SCHEDULED_PIPELINE_STATUSES: tuple[str, ...] = (
    "pending",
    "scheduled",
    "queued",
    "staged",
    "ready_to_publish",
)

# Currently doing publish work (one bucket past scheduled).
PROCESSING_STATUSES: tuple[str, ...] = ("processing",)

# "In the queue / processing tab" = scheduled + actively processing.
QUEUE_VIEW_STATUSES: tuple[str, ...] = SCHEDULED_PIPELINE_STATUSES + PROCESSING_STATUSES

# Terminal-success and terminal-non-success buckets.
COMPLETED_STATUSES: tuple[str, ...] = ("completed", "succeeded")
PARTIAL_STATUSES: tuple[str, ...] = ("partial",)
FAILED_STATUSES: tuple[str, ...] = ("failed",)

# Narrower than SCHEDULED_PIPELINE_STATUSES — worker may have claimed staged rows.
# Matches routers/scheduled.py cancel policy and queue UI cancel buttons.
CANCELLABLE_STATUSES: tuple[str, ...] = ("pending", "scheduled", "queued")

# Pending uploads eligible for POST /api/uploads/{id}/requeue (transient pipeline errors).
# Keep in sync with frontend/js/scheduled-status.js isRequeueable().
REQUEUEABLE_ERROR_CODES: frozenset[str] = frozenset(
    {
        "ENQUEUE_FAILED",
        "QUEUE_UNAVAILABLE",
        "SCHEDULE_INCOMPLETE",
        "PUBLISH_SLOT_MISSING",
        "STUCK_PENDING",
        "ABANDONED_PENDING",
    }
)

# Failed / cancelled / partial (and stale processing) → POST /api/uploads/{id}/retry.
RETRYABLE_STATUSES: frozenset[str] = frozenset({"failed", "cancelled", "canceled", "partial"})

# Terminal schedule failures still shown on scheduled.html so users can Retry there.
SCHEDULE_ATTENTION_ERROR_CODES: frozenset[str] = frozenset(
    {
        "PUBLISH_SLOT_MISSING",
        "SCHEDULE_INCOMPLETE",
        "STUCK_READY_TO_PUBLISH",
        "STUCK_PENDING",
        "ABANDONED_PENDING",
    }
)


def is_requeueable_upload(status: str, error_code: str | None = None) -> bool:
    if (status or "").lower() != "pending":
        return False
    ec = (error_code or "").strip().upper()
    if not ec:
        return True
    return ec in REQUEUEABLE_ERROR_CODES


def is_retryable_upload(
    status: str,
    *,
    error_code: str | None = None,
    has_failed_platform: bool = False,
    stale_processing: bool = False,
) -> bool:
    """Whether POST /api/uploads/{id}/retry is appropriate (mirrors queue UI).

    Hard-block ``error_code`` values match ``classify_retry_error`` so list/detail
    ``is_retryable`` never advertises a retry the API will reject with 409.
    """
    from services.retry_policy import classify_retry_error

    s = (status or "").lower()
    if s in RETRYABLE_STATUSES:
        if s == "partial" and not has_failed_platform:
            return False
        if not classify_retry_error(error_code).allowed:
            return False
        return True
    if s == "processing" and stale_processing:
        if not classify_retry_error(error_code).allowed:
            return False
        return True
    return False

# /complete is a no-op (success, no re-enqueue) when status is already in-flight or terminal.
COMPLETE_IDEMPOTENT_STATUSES: tuple[str, ...] = (
    "queued",
    "staged",
    "processing",
    "ready_to_publish",
    *COMPLETED_STATUSES,
    *PARTIAL_STATUSES,
)


def scheduled_in_clause(start_param_idx: int) -> tuple[str, list[str]]:
    """Return ``("$2,$3,...", [statuses])`` for inlining the canonical
    SCHEDULED_PIPELINE_STATUSES into a parametrized SQL ``IN (...)`` clause.

    ``start_param_idx`` is the asyncpg ``$N`` index of the first status param.
    """
    statuses = list(SCHEDULED_PIPELINE_STATUSES)
    placeholders = ", ".join(f"${i}" for i in range(start_param_idx, start_param_idx + len(statuses)))
    return placeholders, statuses


UPLOAD_VIEW_STATUS: Dict[str, Any] = {
    "processing": QUEUE_VIEW_STATUSES,
    "completed": COMPLETED_STATUSES,
    "partial": PARTIAL_STATUSES,
    "failed": FAILED_STATUSES,
    # Legacy aliases — same canonical scheduled bucket. Do not remove without
    # sweeping callers (front-end and admin tooling still read these keys).
    "pending": SCHEDULED_PIPELINE_STATUSES,
    "staged": SCHEDULED_PIPELINE_STATUSES,
    "scheduled": SCHEDULED_PIPELINE_STATUSES,
    "smart_schedule": None,
}

UPLOAD_STATUS_LABEL = {
    "pending": "Pending",
    "staged": "Scheduled",
    "queued": "Queued",
    "scheduled": "Scheduled",
    "ready_to_publish": "Ready to publish",
    "processing": "Processing",
    "completed": "Completed",
    "succeeded": "Succeeded",
    "partial": "Partial",
    "failed": "Failed",
    "cancelled": "Cancelled",
}

ALLOWED_VIDEO_TYPES = frozenset(
    {
        "video/mp4",
        "video/quicktime",
        "video/x-msvideo",
        "video/webm",
        "video/x-matroska",
    }
)
