"""
White-label branding settings for Agency tier accounts.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from fastapi import HTTPException

from stages.entitlements import get_entitlements_from_user

_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
_HTTPS_URL = re.compile(r"^https://", re.I)
_R2_LOGO_PREFIX = re.compile(r"^white-label/", re.I)


def resolve_white_label_logo_url(raw: Optional[str], *, presign: bool = True) -> str:
    """Return browser-ready logo URL (HTTPS or presigned R2 key)."""
    url = (raw or "").strip()
    if not url:
        return ""
    if _R2_LOGO_PREFIX.match(url) or url.startswith("avatars/"):
        try:
            from core.r2 import r2_presign_get_url, resolve_user_profile_avatar_url

            if url.startswith("avatars/"):
                return resolve_user_profile_avatar_url(url, presign=presign) or url
            if presign:
                return r2_presign_get_url(url) or url
        except Exception:
            return url
    return url


def _validate_logo_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if _R2_LOGO_PREFIX.match(url):
        if len(url) > 512:
            raise HTTPException(400, "logo_url must be 512 characters or fewer")
        return url
    if not _HTTPS_URL.match(url):
        raise HTTPException(400, "logo_url must be an https:// URL or uploaded logo key")
    if len(url) > 512:
        raise HTTPException(400, "logo_url must be 512 characters or fewer")
    return url


def _public_settings_row(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return dict(DEFAULTS)
    out = {**DEFAULTS, **row}
    out["logo_url"] = resolve_white_label_logo_url(out.get("logo_url"))
    return out

DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "company_name": "",
    "logo_url": "",
    "primary_color": "#f97316",
}


def white_label_owner_user_id(user: dict) -> str:
    """Workspace owner (billing account) stores white-label settings."""
    return str(user.get("billing_user_id") or user["id"])


def can_manage_white_label(user: dict) -> bool:
    """Only workspace owners/admins (or solo accounts) may edit branding."""
    ws = user.get("workspace")
    if not isinstance(ws, dict):
        return True
    role = str(ws.get("role") or "").lower()
    return role in ("owner", "admin")


def _require_white_label(user: dict) -> None:
    ent = get_entitlements_from_user(user)
    if not ent.can_white_label:
        raise HTTPException(
            403,
            {
                "code": "feature_white_label",
                "message": "White label requires Agency plan.",
            },
        )


def _require_white_label_manage(user: dict) -> None:
    _require_white_label(user)
    if not can_manage_white_label(user):
        raise HTTPException(
            403,
            {
                "code": "workspace_role_forbidden",
                "message": "Only workspace owners and admins can change white label settings.",
            },
        )


def _validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "enabled" in payload:
        out["enabled"] = bool(payload["enabled"])
    if "company_name" in payload:
        name = (payload["company_name"] or "").strip()
        if len(name) > 255:
            raise HTTPException(400, "company_name must be 255 characters or fewer")
        out["company_name"] = name
    if "logo_url" in payload:
        out["logo_url"] = _validate_logo_url(payload.get("logo_url") or "")
    if "primary_color" in payload:
        color = (payload["primary_color"] or "").strip()
        if color and not _HEX_COLOR.match(color):
            raise HTTPException(400, "primary_color must be a #RRGGBB hex color")
        out["primary_color"] = color or DEFAULTS["primary_color"]
    return out


async def get_white_label_settings(conn, user_id: str) -> Optional[Dict[str, Any]]:
    row = await conn.fetchrow(
        """
        SELECT enabled, logo_url, company_name, primary_color, created_at
        FROM white_label_settings
        WHERE user_id = $1
        """,
        user_id,
    )
    if not row:
        return None
    raw_logo = row["logo_url"] or ""
    return {
        "enabled": bool(row["enabled"]),
        "logo_url": resolve_white_label_logo_url(raw_logo),
        "logo_r2_key": raw_logo if _R2_LOGO_PREFIX.match(raw_logo or "") else "",
        "company_name": row["company_name"] or "",
        "primary_color": row["primary_color"] or DEFAULTS["primary_color"],
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
    }


async def get_white_label_settings_or_defaults(conn, user_id: str) -> Dict[str, Any]:
    stored = await get_white_label_settings(conn, user_id)
    if not stored:
        return dict(DEFAULTS)
    return {**DEFAULTS, **stored}


async def upsert_white_label_settings(
    conn,
    user: dict,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    _require_white_label_manage(user)
    validated = _validate_payload(payload)
    user_id = white_label_owner_user_id(user)

    existing = await get_white_label_settings(conn, user_id) or dict(DEFAULTS)
    merged = {**existing, **validated}

    if merged.get("enabled") and not (merged.get("company_name") or "").strip():
        raise HTTPException(400, "company_name is required when white label is enabled")

    await conn.execute(
        """
        INSERT INTO white_label_settings (user_id, enabled, logo_url, company_name, primary_color, created_at)
        VALUES ($1, $2, $3, $4, $5, NOW())
        ON CONFLICT (user_id) DO UPDATE SET
            enabled = EXCLUDED.enabled,
            logo_url = EXCLUDED.logo_url,
            company_name = EXCLUDED.company_name,
            primary_color = EXCLUDED.primary_color
        """,
        user_id,
        bool(merged.get("enabled")),
        merged.get("logo_url") or None,
        merged.get("company_name") or None,
        merged.get("primary_color") or DEFAULTS["primary_color"],
    )
    return _public_settings_row(await get_white_label_settings_or_defaults(conn, user_id))


async def branding_payload_for_user(conn, user: dict) -> Optional[Dict[str, Any]]:
    """Effective in-app branding for the authenticated user (workspace owner settings)."""
    ent = get_entitlements_from_user(user)
    if not ent.can_white_label:
        return None
    owner_id = white_label_owner_user_id(user)
    settings = await get_white_label_settings(conn, owner_id)
    if not settings or not settings.get("enabled"):
        return None
    return {
        "enabled": True,
        "company_name": settings.get("company_name") or "",
        "logo_url": resolve_white_label_logo_url(settings.get("logo_url")),
        "primary_color": settings.get("primary_color") or DEFAULTS["primary_color"],
        "can_manage": can_manage_white_label(user),
    }


async def load_effective_brand_context(conn, user_id: str, user: Optional[dict] = None):
    """Return BrandContext for emails/exports when white label is enabled."""
    from stages.emails.base import BrandContext
    from stages.entitlements import get_entitlements_from_user

    u = user
    if u is None:
        u = await conn.fetchrow(
            "SELECT id, subscription_tier, role, flex_enabled FROM users WHERE id = $1",
            user_id,
        )
        u = dict(u) if u else {}
    ent = get_entitlements_from_user(u)
    if not ent.can_white_label:
        return None

    owner_id = white_label_owner_user_id(u) if u else str(user_id)
    settings = await get_white_label_settings(conn, owner_id)
    if not settings or not settings.get("enabled"):
        return None

    name = (settings.get("company_name") or "").strip() or "Agency"
    raw_logo = (settings.get("logo_url") or "").strip()
    logo = resolve_white_label_logo_url(raw_logo) or BrandContext.uploadm8_default().logo_url
    color = settings.get("primary_color") or DEFAULTS["primary_color"]
    return BrandContext(
        logo_url=logo,
        company_name=name,
        primary_color=color,
        footer_name=name,
        product_name=name,
    )


def company_slug(name: str) -> str:
    """Filesystem-safe slug from company name."""
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return s[:48] or "agency"

