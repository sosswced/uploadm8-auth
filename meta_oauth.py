"""
Meta (Facebook / Instagram) OAuth scope selection and permission checks.

Use META_OAUTH_MODE=minimal during App Review demos when only a subset of permissions
is approved. Production uses full scopes after Meta approves publishing and insights.

Environment:
  META_OAUTH_MODE   full | minimal | custom (default: full)
  META_INSTAGRAM_OAUTH_SCOPE   optional override when META_OAUTH_MODE=custom
  META_FACEBOOK_OAUTH_SCOPE    optional override when META_OAUTH_MODE=custom
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

# Approved for your app (demo-safe): list Pages, Business Manager linkage, Page engagement.
_SCOPES_MINIMAL = "pages_show_list,pages_read_engagement,business_management"

# Production: publishing + insights + listing (requires Meta App Review per permission).
_SCOPES_INSTAGRAM_FULL = (
    "instagram_basic,"
    "instagram_content_publish,"
    "instagram_manage_insights,"
    "pages_show_list,"
    "pages_read_engagement,"
    "business_management"
)

_SCOPES_FACEBOOK_FULL = (
    "pages_manage_posts,"
    "pages_read_engagement,"
    "pages_read_user_content,"
    "pages_show_list,"
    "publish_video,"
    "read_insights"
)


def meta_oauth_mode() -> str:
    m = (os.environ.get("META_OAUTH_MODE") or "full").strip().lower()
    if m in ("full", "minimal", "custom"):
        return m
    return "full"


def meta_instagram_oauth_scope() -> str:
    if meta_oauth_mode() == "minimal":
        return _SCOPES_MINIMAL
    if meta_oauth_mode() == "custom":
        return (os.environ.get("META_INSTAGRAM_OAUTH_SCOPE") or _SCOPES_INSTAGRAM_FULL).strip()
    return _SCOPES_INSTAGRAM_FULL


def meta_facebook_oauth_scope() -> str:
    if meta_oauth_mode() == "minimal":
        return _SCOPES_MINIMAL
    if meta_oauth_mode() == "custom":
        return (os.environ.get("META_FACEBOOK_OAUTH_SCOPE") or _SCOPES_FACEBOOK_FULL).strip()
    return _SCOPES_FACEBOOK_FULL


async def fetch_granted_permissions(
    client: httpx.AsyncClient,
    user_access_token: str,
    *,
    graph_version: str = "v18.0",
) -> List[Dict[str, Any]]:
    """
    GET /me/permissions — returns [{"permission": "...", "status": "granted"|"declined"}, ...]
    """
    if not user_access_token:
        return []
    try:
        resp = await client.get(
            f"https://graph.facebook.com/{graph_version}/me/permissions",
            params={"access_token": user_access_token},
            timeout=20.0,
        )
        if resp.status_code != 200:
            return []
        data = resp.json() or {}
        raw = data.get("data") or []
        return raw if isinstance(raw, list) else []
    except Exception:
        return []


def permission_status(granted: List[Dict[str, Any]], permission: str) -> Optional[str]:
    """Return 'granted', 'declined', or None if not listed."""
    p = (permission or "").strip().lower()
    for row in granted:
        if str(row.get("permission") or "").lower() == p:
            return str(row.get("status") or "").lower() or None
    return None


def is_permission_granted(granted: List[Dict[str, Any]], permission: str) -> bool:
    return permission_status(granted, permission) == "granted"


def meta_permission_granted_from_blob(token_data: dict, permission: str) -> Optional[bool]:
    """
    True / False if we have a stored permission snapshot; None if unknown (legacy tokens).
    When None, callers may attempt the API call and surface Graph errors.
    """
    raw = token_data.get("meta_permissions")
    if not isinstance(raw, list) or not raw:
        return None
    st = permission_status(raw, permission)
    if st == "granted":
        return True
    if st == "declined":
        return False
    return None


def require_instagram_publish(token_data: dict) -> Optional[str]:
    """Return error message if publish cannot proceed; None if allowed or unknown."""
    g = meta_permission_granted_from_blob(token_data, "instagram_content_publish")
    if g is False:
        return (
            "Instagram publishing requires instagram_content_publish (not granted on this connection). "
            "Approve the permission in Meta App Review and reconnect with META_OAUTH_MODE=full, "
            "or use a developer/tester account with advanced access."
        )
    return None


def require_facebook_publish(token_data: dict) -> Optional[str]:
    g = meta_permission_granted_from_blob(token_data, "publish_video")
    if g is False:
        return (
            "Facebook publishing requires publish_video (not granted on this connection). "
            "Approve the permission in Meta App Review and reconnect with META_OAUTH_MODE=full."
        )
    return None
