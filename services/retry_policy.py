"""Retry policy for user-initiated upload re-runs.

Single source of truth for:

* Which `error_code` values block a retry until the user fixes something
  upstream (token reconnect, top up wallet, upgrade tier).
* The soft retry cap that stops a user from infinitely re-running a deterministic
  failure that costs tokens each pass.
* The ``output_artifacts.retry`` counter shape so multiple modules read/write
  the same field.

The wallet/billing semantics of retries are intentionally not covered here —
that's owned by ``core/wallet.py`` and the worker's pipeline cost-attribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

# Re-running these will deterministically fail again until the user fixes
# something. Block at the API layer with HTTP 409 + a hint instead of burning
# tokens on a guaranteed-fail pipeline run.
_HARD_BLOCK: Dict[str, str] = {
    # Wallet / entitlements -- needs top-up or upgrade
    "INSUFFICIENT_TOKENS":  "Wallet is empty. Top up your tokens before retrying.",
    "QUOTA_EXCEEDED":       "You've hit a plan quota. Upgrade or wait for the next refill window before retrying.",
    "TIER_BLOCKED":         "Your current plan tier doesn't allow this upload. Upgrade and retry.",
    "FEATURE_DISABLED":     "A required feature is disabled on your plan. Upgrade and retry.",

    # OAuth / platform connectivity -- needs reconnect
    "PLATFORM_AUTH_FAILED": "A platform connection has expired. Reconnect the account in Settings, then retry.",
    "AUTH_FAILED":          "Authentication with a platform failed. Reconnect the account, then retry.",
    "TOKEN_EXPIRED":        "A platform token has expired. Reconnect the account, then retry.",
    "NOT_CONNECTED":        "A target platform isn't connected. Connect it in Settings, then retry.",
    "NO_TOKEN":             "A target platform has no usable token. Reconnect it in Settings, then retry.",

    # Source data is broken -- retrying won't help
    "VALIDATION":           "The upload data failed validation. Edit the upload (or re-upload the source), then retry.",
    "SOURCE_NOT_IN_R2":     "The video file never finished uploading to storage. Start a new upload with the video file.",
}

# Codes that are inherently transient -- retrying often succeeds.
# Currently used only for documentation / future smart-cap logic; the soft cap
# applies regardless.
_TRANSIENT: Tuple[str, ...] = (
    "INTERNAL", "UPSTREAM", "NETWORK_ERROR", "TIMEOUT", "ENQUEUE_FAILED",
    "RATE_LIMIT", "OPENAI_RATE_LIMIT", "PLATFORM_RATE_LIMIT",
    "DOWNLOAD_FAILED", "UPLOAD_FAILED", "STORAGE",
    "TRANSCODE_FAILED", "FFMPEG_FAILED",
    "AI_CAPTION_FAILED", "OPENAI_ERROR",
    "DB_ERROR",
    "PUBLISH_EXCEPTION", "PUBLISH_FAILED", "PLATFORM_UPLOAD_FAILED",
    "CONTAINER_FAILED", "CONTAINER_TIMEOUT", "CONTAINER_ERROR",
)

# Soft cap on user-initiated retries per upload. Each press still costs tokens
# at the worker, so an unbounded retry loop on a deterministic failure is just
# burning the user's wallet.
MAX_USER_RETRIES_DEFAULT = 5

# Idempotency window: a second click within this many seconds is a no-op
# (returns the in-flight retry instead of double-enqueueing).
RETRY_IDEMPOTENCY_TTL_SEC = 5


@dataclass(frozen=True)
class RetryDecision:
    """Outcome of evaluating whether a retry should be enqueued."""

    allowed: bool
    http_status: int = 200
    code: str = "ok"
    message: str = ""
    hint: Optional[str] = None


def classify_retry_error(error_code: Optional[str]) -> RetryDecision:
    """Return a RetryDecision based on the upload's last error_code.

    Unknown / NULL error codes are allowed (we don't know enough to block).
    """
    if not error_code:
        return RetryDecision(allowed=True)
    code = str(error_code).strip().upper()
    if code in _HARD_BLOCK:
        return RetryDecision(
            allowed=False,
            http_status=409,
            code="retry_blocked",
            message=_HARD_BLOCK[code],
            hint=code,
        )
    return RetryDecision(allowed=True)


def is_transient_error(error_code: Optional[str]) -> bool:
    if not error_code:
        return False
    return str(error_code).strip().upper() in _TRANSIENT


# ---------------------------------------------------------------------------
# Retry counter (stored in uploads.output_artifacts JSONB)
# ---------------------------------------------------------------------------

def get_retry_count(output_artifacts: Optional[Any]) -> int:
    """Return the number of user-initiated retries already attempted."""
    from core.helpers import coerce_output_artifacts_dict

    arts = coerce_output_artifacts_dict(output_artifacts)
    section = arts.get("retry")
    if isinstance(section, dict):
        try:
            return int(section.get("count") or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def bump_retry_metadata(
    output_artifacts: Optional[Any],
    *,
    actor_user_id: str,
    prior_error_code: Optional[str],
    mode: str,                     # "full" | "partial"
    retry_platforms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return a new output_artifacts dict with the retry section incremented.

    Caller is responsible for persisting it. Never mutates the input.
    Tolerates list-shaped legacy ``output_artifacts`` (UPLOADM8-7W).
    """
    from core.helpers import coerce_output_artifacts_dict

    base: Dict[str, Any] = dict(coerce_output_artifacts_dict(output_artifacts))
    section = base.get("retry") if isinstance(base.get("retry"), dict) else {}
    history = list(section.get("history") or [])


    new_count = int(section.get("count") or 0) + 1
    now_iso = datetime.now(timezone.utc).isoformat()

    history.append({
        "at": now_iso,
        "by": actor_user_id,
        "mode": mode,
        "prior_error_code": prior_error_code,
        "platforms": list(retry_platforms or []),
    })
    # Keep history bounded so the JSONB blob doesn't grow without limit.
    if len(history) > 25:
        history = history[-25:]

    base["retry"] = {
        "count": new_count,
        "last_at": now_iso,
        "last_by": actor_user_id,
        "last_mode": mode,
        "last_prior_error_code": prior_error_code,
        "history": history,
    }
    return base


# ---------------------------------------------------------------------------
# Partial-status helpers
# ---------------------------------------------------------------------------

def split_platform_results(
    platform_results: Optional[List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return (succeeded_entries, failed_platform_names) from a platform_results blob.

    Tolerant of None / non-list / per-entry shape drift.
    """
    succeeded: List[Dict[str, Any]] = []
    failed_names: List[str] = []
    if not isinstance(platform_results, list):
        return succeeded, failed_names
    for entry in platform_results:
        if not isinstance(entry, dict):
            continue
        if entry.get("success") is True:
            succeeded.append(entry)
        else:
            name = entry.get("platform")
            if isinstance(name, str) and name:
                failed_names.append(name)
    # De-dup while preserving order
    seen = set()
    deduped: List[str] = []
    for n in failed_names:
        if n in seen:
            continue
        seen.add(n)
        deduped.append(n)
    return succeeded, deduped


STALE_PROCESSING_MINUTES_DEFAULT = 20

# ready_to_publish past due by this many minutes → user may Retry (republish).
# Worker auto-redispatch uses a longer window (STUCK_READY_REDISPATCH_MINUTES).
OVERDUE_READY_GRACE_MINUTES_DEFAULT = 5

# Worker-initiated auto-retries (transient failures; immediate + scheduled/smart).
MAX_AUTO_RETRIES_DEFAULT = 3
AUTO_RETRY_BACKOFF_MINUTES: Tuple[int, ...] = (2, 5, 15)

# Extra failed codes safe for worker auto-retry (scheduled publish stalls).
_AUTO_RETRY_EXTRA: Tuple[str, ...] = (
    "STUCK_READY_TO_PUBLISH",
    "PUBLISH_SLOT_MISSING",
    "SCHEDULE_INCOMPLETE",
    "STALE_PROCESSING",
    "ENQUEUE_FAILED",
    "QUEUE_UNAVAILABLE",
)


def get_auto_retry_count(output_artifacts: Optional[Any]) -> int:
    from core.helpers import coerce_output_artifacts_dict

    arts = coerce_output_artifacts_dict(output_artifacts)
    section = arts.get("auto_retry")
    if isinstance(section, dict):
        try:
            return int(section.get("count") or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def auto_retry_backoff_minutes(prior_count: int) -> int:
    idx = max(0, min(int(prior_count or 0), len(AUTO_RETRY_BACKOFF_MINUTES) - 1))
    return AUTO_RETRY_BACKOFF_MINUTES[idx]


def bump_auto_retry_metadata(
    output_artifacts: Optional[Any],
    *,
    error_code: Optional[str],
) -> Dict[str, Any]:
    from core.helpers import coerce_output_artifacts_dict

    base: Dict[str, Any] = dict(coerce_output_artifacts_dict(output_artifacts))
    section = base.get("auto_retry") if isinstance(base.get("auto_retry"), dict) else {}
    new_count = int(section.get("count") or 0) + 1
    now_iso = datetime.now(timezone.utc).isoformat()
    base["auto_retry"] = {
        "count": new_count,
        "last_at": now_iso,
        "last_error_code": error_code,
    }
    return base


def is_auto_retryable_error(error_code: Optional[str]) -> bool:
    """Transient + schedule-stall codes the worker may auto-retry."""
    if not error_code:
        return False
    code = str(error_code).strip().upper()
    return is_transient_error(code) or code in _AUTO_RETRY_EXTRA


def should_auto_retry_upload(
    upload_row: Any,
    *,
    max_retries: int = MAX_AUTO_RETRIES_DEFAULT,
) -> bool:
    """True when worker may re-queue a failed upload with a retryable error.

    Immediate, scheduled, and smart modes are eligible. Hard-block codes never
    auto-retry. Scheduled/smart use deferred process jobs so publish slots stay
    intact.
    """
    status = (
        (upload_row.get("status") if isinstance(upload_row, dict) else upload_row["status"])
        or ""
    ).lower()
    if status != "failed":
        return False
    mode = (
        (upload_row.get("schedule_mode") if isinstance(upload_row, dict) else upload_row["schedule_mode"])
        or "immediate"
    ).strip().lower()
    if mode not in ("immediate", "scheduled", "smart"):
        return False
    error_code = (
        upload_row.get("error_code") if isinstance(upload_row, dict) else upload_row["error_code"]
    )
    if not is_auto_retryable_error(error_code):
        return False
    if not classify_retry_error(error_code).allowed:
        return False
    arts = (
        upload_row.get("output_artifacts") if isinstance(upload_row, dict) else upload_row["output_artifacts"]
    )
    from core.helpers import coerce_output_artifacts_dict

    return get_auto_retry_count(coerce_output_artifacts_dict(arts)) < max_retries


def upload_is_overdue_ready_to_publish(
    upload_row: Any,
    *,
    grace_minutes: int | None = None,
    now: datetime | None = None,
) -> bool:
    """True when status=ready_to_publish and the due slot is past grace."""
    status = (
        (upload_row.get("status") if isinstance(upload_row, dict) else upload_row["status"])
        or ""
    ).lower()
    if status != "ready_to_publish":
        return False
    due = None
    if isinstance(upload_row, dict):
        due = upload_row.get("scheduled_time")
        if due is None:
            from services.deferred_publish_schedule import next_due_scheduled_time

            due = next_due_scheduled_time(upload_row)
    else:
        due = upload_row["scheduled_time"] if "scheduled_time" in upload_row.keys() else None
        if due is None:
            from services.deferred_publish_schedule import next_due_scheduled_time

            due = next_due_scheduled_time(dict(upload_row))
    if not due:
        return False
    if getattr(due, "tzinfo", None) is None:
        due = due.replace(tzinfo=timezone.utc)
    threshold = grace_minutes if grace_minutes is not None else OVERDUE_READY_GRACE_MINUTES_DEFAULT
    clock = now or datetime.now(timezone.utc)
    return clock > (due + timedelta(minutes=threshold))


# Queued with a lost Redis message — user Retry / stale recovery can re-enqueue.
# Keep short so deploy/recycle zombies are nudgeable without waiting STALE_QUEUED_MINUTES.
STUCK_QUEUED_MINUTES_DEFAULT = 2


def _aware_utc(dt: Any) -> datetime | None:
    if dt is None:
        return None
    if getattr(dt, "tzinfo", None) is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def upload_is_stale_processing(upload_row: Any, *, minutes: int | None = None) -> bool:
    """True when status=processing and last progress is older than ``minutes``.

    Uses ``updated_at``, then ``processing_started_at``, then ``created_at``.
    Rows stuck in ``processing`` with a null ``processing_started_at`` still
    qualify (worker reclaim historically skipped those).
    """
    get = upload_row.get if isinstance(upload_row, dict) else upload_row.__getitem__
    status = (get("status") or "").lower()
    if status != "processing":
        return False
    threshold = minutes if minutes is not None else STALE_PROCESSING_MINUTES_DEFAULT
    now = datetime.now(timezone.utc)

    try:
        updated = _aware_utc(get("updated_at"))
    except Exception:
        updated = None
    try:
        started = _aware_utc(get("processing_started_at"))
    except Exception:
        started = None
    try:
        created = _aware_utc(get("created_at"))
    except Exception:
        created = None

    anchor = updated or started or created
    if not anchor:
        return False
    return (now - anchor) > timedelta(minutes=threshold)


def upload_is_stuck_queued(upload_row: Any, *, minutes: int | None = None) -> bool:
    """True when status=queued and the row has sat without a claim for ``minutes``.

    Typical cause: worker recycle dropped the Redis process message while the
    DB row stayed ``queued``. User Retry and stale recovery re-enqueue.
    """
    get = upload_row.get if isinstance(upload_row, dict) else upload_row.__getitem__
    status = (get("status") or "").lower()
    if status != "queued":
        return False
    threshold = minutes if minutes is not None else STUCK_QUEUED_MINUTES_DEFAULT
    now = datetime.now(timezone.utc)
    try:
        updated = _aware_utc(get("updated_at"))
    except Exception:
        updated = None
    try:
        created = _aware_utc(get("created_at"))
    except Exception:
        created = None
    anchor = updated or created
    if not anchor:
        return False
    return (now - anchor) > timedelta(minutes=threshold)
