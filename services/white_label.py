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

DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "company_name": "",
    "logo_url": "",
    "primary_color": "#f97316",
}


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
        url = (payload["logo_url"] or "").strip()
        if url and not _HTTPS_URL.match(url):
            raise HTTPException(400, "logo_url must be an https:// URL")
        if len(url) > 512:
            raise HTTPException(400, "logo_url must be 512 characters or fewer")
        out["logo_url"] = url
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
    return {
        "enabled": bool(row["enabled"]),
        "logo_url": row["logo_url"] or "",
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
    _require_white_label(user)
    validated = _validate_payload(payload)
    user_id = str(user["id"])

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
    return await get_white_label_settings_or_defaults(conn, user_id)


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

    settings = await get_white_label_settings(conn, user_id)
    if not settings or not settings.get("enabled"):
        return None

    name = (settings.get("company_name") or "").strip() or "Agency"
    logo = (settings.get("logo_url") or "").strip() or BrandContext.uploadm8_default().logo_url
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

