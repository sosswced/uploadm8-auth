"""
Per-platform deferred publish scheduling for smart uploads.

Smart mode stores per-platform ISO datetimes in ``uploads.schedule_metadata``.
The worker scheduler dispatches publish batches when each platform's slot is
due, merging results into ``platform_results`` until every target is handled.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Iterable, List, Optional, Set, Tuple

from stages.context import JobContext, PlatformResult


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_iso_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _aware(value)
    s = str(value).strip()
    if not s:
        return None
    s = s.replace("Z", "+00:00").replace("z", "+00:00")
    try:
        return _aware(datetime.fromisoformat(s))
    except ValueError:
        return None


def parse_schedule_metadata(raw: Any) -> dict[str, datetime]:
    """Platform (lowercase) -> aware UTC datetime."""
    if raw is None:
        return {}
    data = raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, datetime] = {}
    for key, val in data.items():
        plat = str(key or "").strip().lower()
        if not plat:
            continue
        dt = parse_iso_datetime(val)
        if dt is not None:
            out[plat] = dt
    return out


def normalize_platform_results(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
    if isinstance(raw, dict):
        return [{"platform": k, **v} for k, v in raw.items() if isinstance(v, dict)]
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    return []


def publish_target_key(platform: str, token_row_id: Any = None) -> str:
    plat = str(platform or "").strip().lower()
    tid = str(token_row_id or "").strip()
    return f"{plat}|{tid}" if tid else plat


def expected_publish_targets(upload_record: dict[str, Any]) -> list[tuple[str, Optional[str]]]:
    """
    Return (platform, token_row_id) pairs the publish stage would attempt.
    Mirrors publish_stage target resolution at a high level.
    """
    platforms = [str(p).strip().lower() for p in (upload_record.get("platforms") or []) if str(p).strip()]
    target_accounts = upload_record.get("target_accounts") or []
    if isinstance(target_accounts, str):
        target_accounts = [target_accounts]
    token_ids = [str(t).strip() for t in target_accounts if str(t).strip()]
    if token_ids:
        # Token -> platform mapping requires DB; callers with tokens pass resolved pairs.
        return [(p, None) for p in platforms]
    return [(p, None) for p in platforms]


def expected_publish_targets_resolved(
    platforms: Iterable[str],
    target_account_platforms: Iterable[tuple[str, str]],
) -> list[tuple[str, Optional[str]]]:
    """When token rows are resolved: list of (platform, token_id)."""
    pairs = [(str(p).lower(), str(tid)) for p, tid in target_account_platforms if p and tid]
    if pairs:
        return pairs
    return [(str(p).lower(), None) for p in platforms if str(p).strip()]


def handled_target_keys(platform_results: Any) -> set[str]:
    keys: set[str] = set()
    for row in normalize_platform_results(platform_results):
        plat = str(row.get("platform") or "").strip().lower()
        if not plat:
            continue
        # Index every id form the publish stage may write so callers that pass
        # token_row_id vs account_id / open_id still match.
        for tid_field in (
            row.get("token_row_id"),
            row.get("account_id"),
            row.get("token_id"),
        ):
            if tid_field:
                keys.add(publish_target_key(plat, tid_field))
        keys.add(plat)
    return keys


def is_publish_target_handled(
    platform_results: Any,
    platform: str,
    token_row_id: Any = None,
) -> bool:
    """
    True when ``platform_results`` already covers this publish target.

    Unresolved targets ``(platform, None)`` are covered by any result for that
    platform (including account-scoped ``tiktok|<uuid>`` keys). This matches
    ``publish_target_already_done`` and prevents false ``PUBLISH_SLOT_MISSING``
    after a successful single-account immediate publish.
    """
    plat = str(platform or "").strip().lower()
    if not plat:
        return True
    tid = str(token_row_id or "").strip()
    for row in normalize_platform_results(platform_results):
        if str(row.get("platform") or "").strip().lower() != plat:
            continue
        row_ids = {
            str(x).strip()
            for x in (
                row.get("token_row_id"),
                row.get("account_id"),
                row.get("token_id"),
            )
            if x is not None and str(x).strip()
        }
        if not tid:
            return True
        if tid in row_ids:
            return True
        if not row_ids:
            return True
    return False


def slot_for_platform(upload_record: dict[str, Any], platform: str) -> Optional[datetime]:
    """
    Resolvable publish slot for one platform.

    Smart mode: only ``schedule_metadata[platform]`` (no stale ``scheduled_time``
    fallback — that time often belongs to an already-published earliest platform).
    Other modes: metadata slot, else top-level ``scheduled_time``.
    """
    plat = str(platform or "").strip().lower()
    schedule = parse_schedule_metadata(upload_record.get("schedule_metadata"))
    if plat in schedule:
        return schedule[plat]
    mode = str(upload_record.get("schedule_mode") or "").strip().lower()
    if mode == "smart":
        return None
    st = upload_record.get("scheduled_time")
    if st is not None:
        return parse_iso_datetime(st)
    return None


def platforms_due_for_publish(
    upload_record: dict[str, Any],
    now: Optional[datetime] = None,
    publish_targets: Optional[list[tuple[str, Optional[str]]]] = None,
) -> frozenset[str]:
    """
    Platform names (lowercase) that should publish in the next batch.

    Immediate: unhandled platforms (``scheduled_time`` may be null).
    Scheduled: unhandled platforms when ``scheduled_time <= now``.
    Smart: per-platform slots from ``schedule_metadata``.

    ``publish_targets``: optional resolved (platform, token_id) list; when omitted,
    uses one target per entry in ``uploads.platforms``.
    """
    now = _aware(now or _now_utc())
    mode = str(upload_record.get("schedule_mode") or "scheduled").strip().lower()

    platforms = [str(p).strip().lower() for p in (upload_record.get("platforms") or []) if str(p).strip()]
    if not platforms:
        return frozenset()

    targets = publish_targets or [(p, None) for p in platforms]
    results = upload_record.get("platform_results")

    def _unhandled_platforms() -> set[str]:
        due_plats: set[str] = set()
        for plat in platforms:
            plat_targets = [(p, tid) for p, tid in targets if str(p).strip().lower() == plat]
            if not plat_targets:
                plat_targets = [(plat, None)]
            unhandled = [
                (p, tid)
                for p, tid in plat_targets
                if not is_publish_target_handled(results, p, tid)
            ]
            if unhandled:
                due_plats.add(plat)
        return due_plats

    if mode != "smart":
        st = parse_iso_datetime(upload_record.get("scheduled_time"))
        # Immediate often stores null scheduled_time — still due until all platforms
        # are handled. Empty due here caused ready_to_publish recovery to re-dispatch
        # forever while deferred publish exited as a no-op.
        due_now = mode == "immediate" or (st is not None and st <= now)
        if not due_now:
            return frozenset()
        return frozenset(_unhandled_platforms())

    schedule = parse_schedule_metadata(upload_record.get("schedule_metadata"))
    pending = _unhandled_platforms()
    due: set[str] = set()
    for plat in pending:
        slot = schedule.get(plat)
        # No scheduled_time fallback — missing metadata means "not due" (repair needed).
        if slot is None or slot > now:
            continue
        due.add(plat)
    return frozenset(due)


def still_has_pending_publish_slots(
    upload_record: dict[str, Any],
    platform_results: Any,
    now: Optional[datetime] = None,
    publish_targets: Optional[list[tuple[str, Optional[str]]]] = None,
) -> bool:
    """True while upload has publish targets not yet attempted."""
    del now  # retained for call-site compatibility
    platforms = [str(p).strip().lower() for p in (upload_record.get("platforms") or []) if str(p).strip()]
    targets = publish_targets or [(p, None) for p in platforms]

    for p, tid in targets:
        if not is_publish_target_handled(platform_results, p, tid):
            return True
    return False


def next_due_scheduled_time(
    upload_record: dict[str, Any],
    platform_results: Any = None,
    publish_targets: Optional[list[tuple[str, Optional[str]]]] = None,
) -> Optional[datetime]:
    """
    Earliest remaining smart-slot time for unhandled publish targets.

    Used after a partial deferred publish to advance ``uploads.scheduled_time``
    so queue/scheduled countdowns and scheduler queries stay aligned with the
    next platform (not the already-published earliest slot).
    """
    mode = str(upload_record.get("schedule_mode") or "").strip().lower()
    results = platform_results if platform_results is not None else upload_record.get("platform_results")

    platforms = [str(p).strip().lower() for p in (upload_record.get("platforms") or []) if str(p).strip()]
    if not platforms:
        return None

    targets = publish_targets or [(p, None) for p in platforms]
    pending_plats: set[str] = set()
    for p, tid in targets:
        if is_publish_target_handled(results, p, tid):
            continue
        pending_plats.add(str(p).strip().lower())

    if not pending_plats:
        return None

    if mode != "smart":
        return parse_iso_datetime(upload_record.get("scheduled_time"))

    schedule = parse_schedule_metadata(upload_record.get("schedule_metadata"))
    candidates: list[datetime] = []
    for plat in pending_plats:
        slot = schedule.get(plat)
        if slot is not None:
            candidates.append(slot)
    if not candidates:
        # Smart: do not fall back to top-level scheduled_time — it often belongs to
        # an already-published platform and hides incomplete schedule_metadata.
        if mode == "smart":
            return None
        return parse_iso_datetime(upload_record.get("scheduled_time"))
    return min(candidates)


def hydrate_platform_results_into_ctx(ctx: JobContext, raw: Any) -> None:
    """Load prior partial publish results into ctx before the next batch."""
    ctx.platform_results = []
    for row in normalize_platform_results(raw):
        ctx.platform_results.append(
            PlatformResult(
                platform=str(row.get("platform") or ""),
                success=bool(row.get("success")),
                platform_video_id=row.get("platform_video_id"),
                platform_url=row.get("platform_url"),
                publish_id=row.get("publish_id"),
                token_row_id=row.get("token_row_id"),
                account_id=row.get("account_id"),
                account_username=row.get("account_username"),
                account_name=row.get("account_name"),
                account_avatar=row.get("account_avatar"),
                attempt_id=row.get("attempt_id"),
                http_status=row.get("http_status"),
                error_code=row.get("error_code"),
                error_message=row.get("error_message"),
                verify_status=str(row.get("verify_status") or "pending"),
                views=int(row.get("views") or 0),
                likes=int(row.get("likes") or 0),
            )
        )


def publish_target_already_done(ctx: JobContext, platform: str, token_id: Optional[str]) -> bool:
    plat = str(platform or "").strip().lower()
    tid = str(token_id or "").strip()
    for r in ctx.platform_results:
        if str(r.platform or "").strip().lower() != plat:
            continue
        rt = str(getattr(r, "token_row_id", None) or "").strip()
        if tid and rt:
            if rt == tid:
                return True
            continue
        if not tid and not rt:
            return True
        if not tid:
            return True
    return False


def scheduled_dates_for_blocking(
    schedule_mode: str,
    scheduled_time: Any,
    schedule_metadata: Any,
    num_days: int,
    now: Optional[datetime] = None,
) -> set[int]:
    """
    Day offsets (1..num_days) occupied by an upload — used to deconflict smart slots.
    """
    now = _aware(now or _now_utc())
    today = now.date()
    used: set[int] = set()

    def _add_dt(dt: Optional[datetime]) -> None:
        if dt is None:
            return
        diff = (_aware(dt).date() - today).days
        if 0 < diff <= num_days:
            used.add(diff)

    mode = str(schedule_mode or "").strip().lower()
    if mode == "smart":
        for dt in parse_schedule_metadata(schedule_metadata).values():
            _add_dt(dt)
    else:
        _add_dt(parse_iso_datetime(scheduled_time))
    return used
