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

def get_retry_count(output_artifacts: Optional[Dict[str, Any]]) -> int:
    """Return the number of user-initiated retries already attempted."""
    if not isinstance(output_artifacts, dict):
        return 0
    section = output_artifacts.get("retry")
    if isinstance(section, dict):
        try:
            return int(section.get("count") or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def bump_retry_metadata(
    output_artifacts: Optional[Dict[str, Any]],
    *,
    actor_user_id: str,
    prior_error_code: Optional[str],
    mode: str,                     # "full" | "partial"
    retry_platforms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return a new output_artifacts dict with the retry section incremented.

    Caller is responsible for persisting it. Never mutates the input.
    """
    base: Dict[str, Any] = dict(output_artifacts or {})
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

# Worker-initiated auto-retries (immediate uploads with transient failures).
MAX_AUTO_RETRIES_DEFAULT = 3
AUTO_RETRY_BACKOFF_MINUTES: Tuple[int, ...] = (2, 5, 15)


def get_auto_retry_count(output_artifacts: Optional[Dict[str, Any]]) -> int:
    if not isinstance(output_artifacts, dict):
        return 0
    section = output_artifacts.get("auto_retry")
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
    output_artifacts: Optional[Dict[str, Any]],
    *,
    error_code: Optional[str],
) -> Dict[str, Any]:
    base: Dict[str, Any] = dict(output_artifacts or {})
    section = base.get("auto_retry") if isinstance(base.get("auto_retry"), dict) else {}
    new_count = int(section.get("count") or 0) + 1
    now_iso = datetime.now(timezone.utc).isoformat()
    base["auto_retry"] = {
        "count": new_count,
        "last_at": now_iso,
        "last_error_code": error_code,
    }
    return base


def should_auto_retry_upload(
    upload_row: Any,
    *,
    max_retries: int = MAX_AUTO_RETRIES_DEFAULT,
) -> bool:
    """True when worker may re-queue a failed immediate upload with a transient error."""
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
    if mode != "immediate":
        return False
    error_code = (
        upload_row.get("error_code") if isinstance(upload_row, dict) else upload_row["error_code"]
    )
    if not is_transient_error(error_code):
        return False
    arts = (
        upload_row.get("output_artifacts") if isinstance(upload_row, dict) else upload_row["output_artifacts"]
    )
    if isinstance(arts, str):
        try:
            import json

            arts = json.loads(arts)
        except Exception:
            arts = {}
    return get_auto_retry_count(arts if isinstance(arts, dict) else None) < max_retries


def upload_is_stale_processing(upload_row: Any, *, minutes: int | None = None) -> bool:
    """True when status=processing and updated_at has not moved for ``minutes``."""
    status = (
        (upload_row.get("status") if isinstance(upload_row, dict) else upload_row["status"])
        or ""
    ).lower()
    if status != "processing":
        return False
    updated = upload_row.get("updated_at") if isinstance(upload_row, dict) else upload_row["updated_at"]
    if not updated:
        return False
    threshold = minutes if minutes is not None else STALE_PROCESSING_MINUTES_DEFAULT
    now = datetime.now(timezone.utc)
    if getattr(updated, "tzinfo", None) is None:
        updated = updated.replace(tzinfo=timezone.utc)
    return (now - updated) > timedelta(minutes=threshold)
