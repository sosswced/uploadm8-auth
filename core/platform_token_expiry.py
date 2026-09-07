"""
Platform OAuth access-token expiry helpers.

Historically ``token_blob.expires_at`` stored provider ``expires_in`` (TTL seconds).
Keepalive + gated refresh require **absolute UTC** timestamps. These helpers
normalize both shapes and decide when a proactive refresh is due.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Mapping, Optional

# Refresh this far *before* access expiry (platform policy).
PLATFORM_REFRESH_LEAD: Dict[str, timedelta] = {
    "youtube": timedelta(minutes=20),
    "tiktok": timedelta(hours=3),
    "instagram": timedelta(days=7),
    "facebook": timedelta(days=7),
}

# When Meta marks page tokens non-expiring, still re-mint periodically.
_META_NON_EXPIRING_REFRESH_AFTER = timedelta(days=50)

# Refresh-token lifetime is the hard ceiling on unattended publishing: once it
# lapses no sweep can recover the account, only a human reconnect. Providers do
# not always send refresh_expires_in, so these are the documented defaults used
# when the blob carries no absolute refresh expiry.
PLATFORM_REFRESH_TOKEN_LIFETIME: Dict[str, Optional[timedelta]] = {
    # TikTok refresh tokens expire ~365d and only rotate on use.
    "tiktok": timedelta(days=365),
    # Google refresh tokens do not expire for published apps, but go stale after
    # ~6 months of no use (and are revoked outright while the app is in testing).
    "youtube": timedelta(days=180),
    # Meta long-lived user tokens are ~60d; page tokens derived from them inherit it.
    "instagram": timedelta(days=60),
    "facebook": timedelta(days=60),
}

# Warn this far ahead of refresh-token death so the user can reconnect in time.
REFRESH_EXPIRY_WARN_LEAD = timedelta(days=14)

# Provider responses that mean the grant is genuinely gone. Anything else
# (timeouts, 5xx, rate limits) is transient and must not kill the connection.
_PERMANENT_OAUTH_ERROR_MARKERS = (
    "invalid_grant",
    "invalid_request",
    "invalid_client",
    "unauthorized_client",
    "invalid_token",
    "token_revoked",
    "revoked",
    "consent_required",
    "access_denied",
    "unsupported_grant_type",
    "oauth2 parameter error",
    "refresh token is invalid",
    "refresh token not found",
    "refresh token expired",
    "session has been invalidated",
    "user has not authorized",
    "code expired",
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if isinstance(value, str) and value.strip():
        try:
            dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            return None
    return None


def parse_access_expires_at(
    blob: Optional[Mapping[str, Any]],
    *,
    now: Optional[datetime] = None,
) -> Optional[datetime]:
    """Return absolute UTC expiry for the access token, or None if unknown/non-expiring."""
    if not isinstance(blob, Mapping):
        return None
    if blob.get("access_non_expiring") is True and blob.get("expires_at") in (None, "", 0, "0"):
        return None

    raw = blob.get("expires_at")
    if raw is None or raw == "":
        return None

    as_dt = parse_dt(raw)
    if as_dt is not None:
        return as_dt

    try:
        n = float(raw)
    except (TypeError, ValueError):
        return None

    # Unix timestamp (seconds since epoch)
    if n >= 1_000_000_000:
        return datetime.fromtimestamp(n, tz=timezone.utc)

    # Legacy TTL seconds relative to access_obtained_at (or now as last resort)
    now = now or _utcnow()
    base = parse_dt(blob.get("access_obtained_at")) or now
    return base + timedelta(seconds=max(0, int(n)))


def stamp_token_expiry(
    blob: Mapping[str, Any],
    *,
    expires_in: Optional[Any] = None,
    refresh_expires_in: Optional[Any] = None,
    non_expiring: bool = False,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Return a copy of blob with absolute expiry fields set."""
    now = now or _utcnow()
    out: Dict[str, Any] = dict(blob or {})
    out["access_obtained_at"] = now.isoformat()

    if non_expiring:
        out["expires_at"] = None
        out["access_non_expiring"] = True
    elif expires_in is not None and str(expires_in).strip() != "":
        try:
            sec = max(0, int(float(expires_in)))
            out["expires_in"] = sec
            out["expires_at"] = (now + timedelta(seconds=sec)).isoformat()
            out.pop("access_non_expiring", None)
        except (TypeError, ValueError):
            pass

    if refresh_expires_in is not None and str(refresh_expires_in).strip() != "":
        try:
            rsec = max(0, int(float(refresh_expires_in)))
            out["refresh_expires_in"] = rsec
            out["refresh_expires_at"] = (now + timedelta(seconds=rsec)).isoformat()
        except (TypeError, ValueError):
            pass

    return out


def normalize_connect_expires_at(
    expires_in_or_at: Any,
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build expiry fields for a fresh OAuth connect blob."""
    now = now or _utcnow()
    stub = stamp_token_expiry({}, expires_in=expires_in_or_at, now=now)
    # If provider already sent an absolute ISO/unix, prefer that.
    as_dt = parse_dt(expires_in_or_at)
    if as_dt is None:
        try:
            n = float(expires_in_or_at)
            if n >= 1_000_000_000:
                as_dt = datetime.fromtimestamp(n, tz=timezone.utc)
        except (TypeError, ValueError):
            as_dt = None
    if as_dt is not None:
        stub["expires_at"] = as_dt.isoformat()
        stub["expires_in"] = max(0, int((as_dt - now).total_seconds()))
    return {
        "expires_at": stub.get("expires_at"),
        "expires_in": stub.get("expires_in"),
        "access_obtained_at": stub.get("access_obtained_at"),
    }


def refresh_lead_for_platform(platform: str) -> timedelta:
    return PLATFORM_REFRESH_LEAD.get(str(platform or "").lower(), timedelta(hours=1))


def parse_refresh_expires_at(
    blob: Optional[Mapping[str, Any]],
    *,
    platform: str = "",
    now: Optional[datetime] = None,
) -> Optional[datetime]:
    """
    Absolute UTC expiry of the *refresh* token, or None when it never expires.

    Prefers an absolute value from the provider, then ``refresh_expires_in``
    relative to when the token was obtained, then the platform default lifetime.
    """
    if not isinstance(blob, Mapping):
        return None
    if blob.get("refresh_non_expiring") is True:
        return None

    absolute = parse_dt(blob.get("refresh_expires_at"))
    if absolute is not None:
        return absolute

    now = now or _utcnow()
    obtained = (
        parse_dt(blob.get("refresh_obtained_at"))
        or parse_dt(blob.get("access_obtained_at"))
        or parse_dt(blob.get("connected_at"))
    )

    raw = blob.get("refresh_expires_in")
    if raw is not None and str(raw).strip() != "":
        try:
            secs = int(float(raw))
        except (TypeError, ValueError):
            secs = 0
        if secs > 0:
            return (obtained or now) + timedelta(seconds=secs)

    if not str(blob.get("refresh_token") or "").strip():
        return None

    lifetime = PLATFORM_REFRESH_TOKEN_LIFETIME.get(str(platform or "").lower())
    if lifetime is None or obtained is None:
        return None
    return obtained + lifetime


def refresh_token_alive_at(
    blob: Optional[Mapping[str, Any]],
    when: datetime,
    *,
    platform: str = "",
    now: Optional[datetime] = None,
) -> bool:
    """
    Can this connection still be refreshed at ``when``?

    Drives the scheduling lookahead: a post scheduled past refresh-token death
    cannot publish without the user reconnecting first.
    """
    expiry = parse_refresh_expires_at(blob, platform=platform, now=now)
    if expiry is None:
        return True
    return when < expiry


def refresh_expiry_state(
    blob: Optional[Mapping[str, Any]],
    *,
    platform: str = "",
    now: Optional[datetime] = None,
    warn_lead: Optional[timedelta] = None,
) -> str:
    """``expired`` / ``expiring_soon`` / ``ok`` for the refresh token."""
    now = now or _utcnow()
    expiry = parse_refresh_expires_at(blob, platform=platform, now=now)
    if expiry is None:
        return "ok"
    if now >= expiry:
        return "expired"
    if now + (warn_lead or REFRESH_EXPIRY_WARN_LEAD) >= expiry:
        return "expiring_soon"
    return "ok"


def is_permanent_oauth_error(error: Any) -> bool:
    """
    True only when the provider said the grant is gone.

    Transient faults (timeout, 5xx, rate limit, connection reset) return False
    so the keepalive retries instead of declaring the account dead.
    """
    if error is None:
        return False
    text = str(error).strip().lower()
    if not text:
        return False
    return any(marker in text for marker in _PERMANENT_OAUTH_ERROR_MARKERS)


def should_refresh_access_token(
    platform: str,
    blob: Optional[Mapping[str, Any]],
    *,
    now: Optional[datetime] = None,
    force: bool = False,
) -> bool:
    """
    True when keepalive / gated refresh should call the provider.

    - ``force``: always
    - Absolute expiry within platform lead (or already past)
    - Legacy/unknown expiry: refresh YouTube/TikTok; Meta non-expiring uses 50d cadence
    """
    if force:
        return True
    plat = str(platform or "").lower()
    now = now or _utcnow()
    blob = blob or {}

    exp = parse_access_expires_at(blob, now=now)
    if exp is not None:
        return now + refresh_lead_for_platform(plat) >= exp

    # Non-expiring Meta page tokens — still refresh periodically from obtained_at.
    if plat in ("instagram", "facebook") and (
        blob.get("access_non_expiring") is True or blob.get("expires_at") in (None, "")
    ):
        obtained = parse_dt(blob.get("access_obtained_at"))
        if obtained is None:
            return True  # unknown age → refresh once to stamp
        return now >= obtained + _META_NON_EXPIRING_REFRESH_AFTER - refresh_lead_for_platform(plat)

    # YouTube / TikTok / unknown with no expiry → refresh (safe; short-lived).
    return plat in ("youtube", "tiktok") or plat not in ("instagram", "facebook")


__all__ = [
    "PLATFORM_REFRESH_LEAD",
    "PLATFORM_REFRESH_TOKEN_LIFETIME",
    "REFRESH_EXPIRY_WARN_LEAD",
    "is_permanent_oauth_error",
    "normalize_connect_expires_at",
    "parse_access_expires_at",
    "parse_dt",
    "parse_refresh_expires_at",
    "refresh_expiry_state",
    "refresh_lead_for_platform",
    "refresh_token_alive_at",
    "should_refresh_access_token",
    "stamp_token_expiry",
]
