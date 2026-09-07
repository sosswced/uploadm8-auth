"""
Connection readiness — can every scheduled post actually publish?

Smart Schedule windows reach years ahead, so "the token is fine right now" is not
the question. The question is whether each connection will still be *refreshable*
at the moment its scheduled posts are due. Access tokens are short-lived and the
keepalive renews them; the refresh grant is the real ceiling, because once it
lapses only the user can restore it.

This module answers that for every connection, and cross-references it against
pending scheduled work per user, platform, and video.

The risk math is pure so it can be tested without a database:
``evaluate_connection`` and ``assess_scheduled_risk``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from core.platform_token_expiry import REFRESH_EXPIRY_WARN_LEAD

logger = logging.getLogger("uploadm8-api")

# Connection states, worst first. Ordering drives summary rollups.
STATE_NEEDS_RECONNECT = "needs_reconnection"
STATE_EXPIRING = "expiring_soon"
STATE_DEGRADED = "degraded"
STATE_STALE = "unverified"
STATE_OK = "ok"

_STATE_SEVERITY = {
    STATE_NEEDS_RECONNECT: 4,
    STATE_EXPIRING: 3,
    STATE_DEGRADED: 2,
    STATE_STALE: 1,
    STATE_OK: 0,
}

# A connection the sweep has not confirmed in this long is treated as unproven
# rather than healthy — it is how a silently stalled worker becomes visible.
DEFAULT_STALE_AFTER = timedelta(days=2)

# Scheduled-post risk reasons.
RISK_NO_CONNECTION = "no_connection"
RISK_NEEDS_RECONNECT = "needs_reconnection"
RISK_GRANT_EXPIRES_FIRST = "grant_expires_before_publish"


def _as_utc(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return None


def evaluate_connection(
    row: Mapping[str, Any],
    *,
    now: Optional[datetime] = None,
    warn_lead: Optional[timedelta] = None,
    stale_after: Optional[timedelta] = None,
) -> Dict[str, Any]:
    """
    Health of one connection plus how far ahead it can be trusted to publish.

    ``publishable_until`` is the refresh-grant expiry (None = no known limit);
    it is the horizon beyond which scheduled posts cannot be guaranteed.
    """
    now = now or datetime.now(timezone.utc)
    warn_lead = warn_lead if warn_lead is not None else REFRESH_EXPIRY_WARN_LEAD
    stale_after = stale_after if stale_after is not None else DEFAULT_STALE_AFTER

    health = str(row.get("oauth_health") or "").strip().lower()
    fail_count = int(row.get("oauth_fail_count") or 0)
    access_expires_at = _as_utc(row.get("access_expires_at"))
    refresh_expires_at = _as_utc(row.get("refresh_expires_at"))
    last_verified_at = _as_utc(row.get("oauth_last_verified_at"))

    reasons: List[str] = []
    if health == STATE_NEEDS_RECONNECT:
        state = STATE_NEEDS_RECONNECT
        reasons.append("provider rejected the stored grant")
    elif refresh_expires_at is not None and now >= refresh_expires_at:
        state = STATE_NEEDS_RECONNECT
        reasons.append("refresh grant has expired")
    elif refresh_expires_at is not None and now + warn_lead >= refresh_expires_at:
        state = STATE_EXPIRING
        reasons.append("refresh grant expires soon")
    elif fail_count > 0:
        state = STATE_DEGRADED
        reasons.append(f"{fail_count} consecutive refresh failure(s), retrying")
    elif last_verified_at is None or now - last_verified_at > stale_after:
        # Not proof of breakage — proof that nothing has confirmed it lately.
        state = STATE_STALE
        reasons.append("not verified by the keepalive sweep recently")
    else:
        state = STATE_OK

    return {
        "token_row_id": str(row.get("id") or ""),
        "user_id": str(row.get("user_id") or ""),
        "platform": str(row.get("platform") or "").lower(),
        "account_name": row.get("account_name") or row.get("account_username") or "",
        "state": state,
        "severity": _STATE_SEVERITY[state],
        "reasons": reasons,
        "oauth_health": health or None,
        "fail_count": fail_count,
        "last_error": row.get("oauth_last_error") or None,
        "access_expires_at": access_expires_at.isoformat() if access_expires_at else None,
        "refresh_expires_at": refresh_expires_at.isoformat() if refresh_expires_at else None,
        "last_verified_at": last_verified_at.isoformat() if last_verified_at else None,
        "next_retry_at": (
            _as_utc(row.get("oauth_next_retry_at")).isoformat()
            if _as_utc(row.get("oauth_next_retry_at"))
            else None
        ),
        # None means no known ceiling; publishing is limited only by the sweep.
        "publishable_until": refresh_expires_at.isoformat() if refresh_expires_at else None,
        "publishable_days": (
            max(0, int((refresh_expires_at - now).total_seconds() // 86400))
            if refresh_expires_at
            else None
        ),
    }


def _connection_index(
    connections: Iterable[Mapping[str, Any]],
) -> tuple[Dict[str, Mapping[str, Any]], Dict[tuple, List[Mapping[str, Any]]]]:
    """Index evaluated connections by row id and by (user, platform)."""
    by_id: Dict[str, Mapping[str, Any]] = {}
    by_user_platform: Dict[tuple, List[Mapping[str, Any]]] = {}
    for c in connections:
        rid = str(c.get("token_row_id") or "")
        if rid:
            by_id[rid] = c
        key = (str(c.get("user_id") or ""), str(c.get("platform") or "").lower())
        by_user_platform.setdefault(key, []).append(c)
    return by_id, by_user_platform


def _publish_times(upload: Mapping[str, Any]) -> Dict[str, Optional[datetime]]:
    """Per-platform publish time, falling back to the upload's scheduled_time."""
    base = _as_utc(upload.get("scheduled_time"))
    per_platform = upload.get("platform_times") or {}
    out: Dict[str, Optional[datetime]] = {}
    for plat in upload.get("platforms") or []:
        p = str(plat or "").strip().lower()
        if not p:
            continue
        out[p] = _as_utc(per_platform.get(p)) or base
    if not out and base is not None:
        out[""] = base
    return out


def assess_scheduled_risk(
    uploads: Sequence[Mapping[str, Any]],
    connections: Sequence[Mapping[str, Any]],
    *,
    now: Optional[datetime] = None,
    detail_limit: int = 200,
) -> Dict[str, Any]:
    """
    Which pending scheduled posts cannot be trusted to publish, and why.

    ``uploads`` rows need: id, user_id, platforms, target_accounts,
    scheduled_time, and optionally platform_times (platform -> datetime).
    ``connections`` are outputs of :func:`evaluate_connection`.
    """
    now = now or datetime.now(timezone.utc)
    by_id, by_user_platform = _connection_index(connections)

    at_risk: List[Dict[str, Any]] = []
    counts = {RISK_NO_CONNECTION: 0, RISK_NEEDS_RECONNECT: 0, RISK_GRANT_EXPIRES_FIRST: 0}
    checked = 0
    affected_uploads: set = set()
    affected_users: set = set()
    # Exact per-account tallies, unaffected by ``detail_limit`` — reconnect
    # emails quote these, so they must not be truncated.
    risk_by_account: Dict[str, int] = {}

    for up in uploads:
        uid = str(up.get("user_id") or "")
        upload_id = str(up.get("id") or "")
        targets = [str(t) for t in (up.get("target_accounts") or []) if str(t or "").strip()]

        for plat, when in _publish_times(up).items():
            if when is None or when <= now:
                continue
            checked += 1

            # Explicit targets win; otherwise every connection the user has on
            # this platform is a candidate publisher.
            candidates: List[Mapping[str, Any]] = []
            for t in targets:
                c = by_id.get(t)
                if c and (not plat or str(c.get("platform") or "") == plat):
                    candidates.append(c)
            if not candidates and not targets:
                candidates = list(by_user_platform.get((uid, plat), []))

            if not candidates:
                reason = RISK_NO_CONNECTION
                detail = "no connected account for this platform"
            else:
                # A post is safe if any candidate publisher survives to its time.
                usable = [
                    c
                    for c in candidates
                    if c.get("state") != STATE_NEEDS_RECONNECT
                    and _survives_until(c, when)
                ]
                if usable:
                    continue
                if all(c.get("state") == STATE_NEEDS_RECONNECT for c in candidates):
                    reason = RISK_NEEDS_RECONNECT
                    detail = "connection needs reconnecting"
                else:
                    reason = RISK_GRANT_EXPIRES_FIRST
                    detail = "refresh grant expires before this post is due"

            counts[reason] += 1
            affected_uploads.add(upload_id)
            if uid:
                affected_users.add(uid)
            if plat:
                key = f"{uid}|{plat}"
                risk_by_account[key] = risk_by_account.get(key, 0) + 1
            if len(at_risk) < max(0, int(detail_limit)):
                at_risk.append(
                    {
                        "upload_id": upload_id,
                        "user_id": uid,
                        "platform": plat or None,
                        "scheduled_time": when.isoformat(),
                        "days_out": max(0, int((when - now).total_seconds() // 86400)),
                        "reason": reason,
                        "detail": detail,
                    }
                )

    return {
        "slots_checked": checked,
        "slots_at_risk": sum(counts.values()),
        "uploads_at_risk": len(affected_uploads),
        "users_affected": len(affected_users),
        "by_reason": counts,
        "by_user_platform": risk_by_account,
        "at_risk": at_risk,
        "at_risk_truncated": sum(counts.values()) > len(at_risk),
    }


def _survives_until(connection: Mapping[str, Any], when: datetime) -> bool:
    """Will this connection's refresh grant still be valid at ``when``?"""
    raw = connection.get("publishable_until")
    if not raw:
        return True
    try:
        until = datetime.fromisoformat(str(raw))
    except ValueError:
        return True
    if until.tzinfo is None:
        until = until.replace(tzinfo=timezone.utc)
    return when < until


def summarize_connections(connections: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Counts by state plus the soonest grant expiry, for dashboards and alerts."""
    by_state: Dict[str, int] = {}
    by_platform: Dict[str, Dict[str, int]] = {}
    soonest: Optional[str] = None
    for c in connections:
        state = str(c.get("state") or STATE_OK)
        by_state[state] = by_state.get(state, 0) + 1
        plat = str(c.get("platform") or "unknown")
        by_platform.setdefault(plat, {})
        by_platform[plat][state] = by_platform[plat].get(state, 0) + 1
        exp = c.get("refresh_expires_at")
        if exp and (soonest is None or str(exp) < soonest):
            soonest = str(exp)
    return {
        "total": len(connections),
        "by_state": by_state,
        "by_platform": by_platform,
        "healthy": by_state.get(STATE_OK, 0),
        "needs_reconnection": by_state.get(STATE_NEEDS_RECONNECT, 0),
        "expiring_soon": by_state.get(STATE_EXPIRING, 0),
        "degraded": by_state.get(STATE_DEGRADED, 0),
        "unverified": by_state.get(STATE_STALE, 0),
        "soonest_grant_expiry": soonest,
    }


async def fetch_connections(conn, *, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Evaluate every live connection (optionally for one user)."""
    where = ["revoked_at IS NULL"]
    params: List[Any] = []
    if user_id:
        params.append(str(user_id))
        where.append(f"user_id = ${len(params)}::uuid")

    try:
        rows = await conn.fetch(
            f"""
            SELECT id, user_id, platform, account_name, account_username,
                   oauth_health, COALESCE(oauth_fail_count, 0) AS oauth_fail_count,
                   oauth_last_error, oauth_next_retry_at, oauth_last_verified_at,
                   access_expires_at, refresh_expires_at
              FROM platform_tokens
             WHERE {' AND '.join(where)}
            """,
            *params,
        )
    except Exception as e:
        # Pre-migration install: fall back to what always exists so the report
        # still renders (states will read as unverified).
        logger.warning("[oauth-readiness] enriched fetch failed, using base columns: %s", e)
        rows = await conn.fetch(
            f"""
            SELECT id, user_id, platform, account_name, account_username, oauth_health
              FROM platform_tokens
             WHERE {' AND '.join(where)}
            """,
            *params,
        )

    return [evaluate_connection(dict(r)) for r in (rows or [])]


async def fetch_pending_scheduled(
    conn,
    *,
    user_id: Optional[str] = None,
    limit: int = 5000,
) -> List[Dict[str, Any]]:
    """Pending scheduled uploads with their per-platform publish times."""
    from services.deferred_publish_schedule import parse_schedule_metadata
    from services.upload.status import scheduled_in_clause

    params: List[Any] = []
    where = ["scheduled_time IS NOT NULL"]
    if user_id:
        params.append(str(user_id))
        where.append(f"user_id = ${len(params)}::uuid")

    placeholders, statuses = scheduled_in_clause(len(params) + 1)
    params.extend(statuses)
    where.append(f"status IN ({placeholders})")
    params.append(int(limit))

    rows = await conn.fetch(
        f"""
        SELECT id, user_id, platforms, target_accounts, scheduled_time, schedule_metadata
          FROM uploads
         WHERE {' AND '.join(where)}
         ORDER BY scheduled_time ASC
         LIMIT ${len(params)}
        """,
        *params,
    )

    out: List[Dict[str, Any]] = []
    for r in rows or []:
        d = dict(r)
        d["platform_times"] = parse_schedule_metadata(d.get("schedule_metadata"))
        d.pop("schedule_metadata", None)
        out.append(d)
    return out


async def connection_readiness_report(
    conn,
    *,
    user_id: Optional[str] = None,
    detail_limit: int = 200,
    scheduled_limit: int = 5000,
) -> Dict[str, Any]:
    """
    Full readiness picture: every connection, plus the scheduled work behind it.

    This is the answer to "can everything we have promised to publish actually
    publish", across platforms, videos, and users.
    """
    now = datetime.now(timezone.utc)
    connections = await fetch_connections(conn, user_id=user_id)
    uploads = await fetch_pending_scheduled(
        conn, user_id=user_id, limit=scheduled_limit
    )
    risk = assess_scheduled_risk(
        uploads, connections, now=now, detail_limit=detail_limit
    )
    summary = summarize_connections(connections)
    return {
        "generated_at": now.isoformat(),
        "scope": {"user_id": user_id or None, "connections": len(connections)},
        "connections_summary": summary,
        "connections": sorted(
            connections, key=lambda c: (-int(c.get("severity") or 0), c.get("platform") or "")
        ),
        "scheduled_risk": risk,
        "pending_scheduled_uploads": len(uploads),
        "ready": (
            summary.get("needs_reconnection", 0) == 0
            and risk.get("slots_at_risk", 0) == 0
        ),
    }


__all__ = [
    "DEFAULT_STALE_AFTER",
    "RISK_GRANT_EXPIRES_FIRST",
    "RISK_NEEDS_RECONNECT",
    "RISK_NO_CONNECTION",
    "STATE_DEGRADED",
    "STATE_EXPIRING",
    "STATE_NEEDS_RECONNECT",
    "STATE_OK",
    "STATE_STALE",
    "assess_scheduled_risk",
    "connection_readiness_report",
    "evaluate_connection",
    "fetch_connections",
    "fetch_pending_scheduled",
    "summarize_connections",
]
