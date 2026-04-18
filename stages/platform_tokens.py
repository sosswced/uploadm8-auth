"""
Canonical `platform_tokens.platform` values used in the database.

OAuth callbacks persist tiktok / youtube / instagram / facebook (see app.py).
Publish and verify stages must query the same slugs or token load silently fails.
"""

from __future__ import annotations

_KNOWN = frozenset({"tiktok", "youtube", "instagram", "facebook"})


def platform_tokens_db_key(platform: str) -> str:
    """Return the `platform_tokens.platform` column value for a publish/API platform slug."""
    p = (platform or "").strip().lower()
    if p in _KNOWN:
        return p
    # Legacy alias (some older paths used the Google OAuth provider name)
    if p == "google":
        return "youtube"
    return p
