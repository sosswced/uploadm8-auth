"""
Creator personas for the signed-in user — shared by /api/me, settings, and Thumbnail Studio.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import asyncpg

logger = logging.getLogger("uploadm8-api")


async def list_thumbnail_studio_personas(conn: asyncpg.Connection, user_id: Any) -> List[Dict[str, Any]]:
    """
    Return the same shape as ``GET /api/thumbnail-studio/personas`` rows
    (``id``, ``name``, ``image_count``, ``created_at``, ``pikzels_linked``, ``link_status``, ``link_error``).
    """
    try:
        rows = await conn.fetch(
            """
            SELECT id, name, image_count, created_at, profile_json, link_status, link_error
            FROM creator_personas
            WHERE user_id = $1
            ORDER BY created_at DESC
            """,
            user_id,
        )
        asset_rows = await conn.fetch(
            """
            SELECT local_persona_id, pikzels_pikzonality_id
            FROM pikzels_user_assets
            WHERE user_id = $1::uuid
              AND kind = 'persona'
              AND status = 'linked'
              AND local_persona_id IS NOT NULL
            ORDER BY updated_at DESC NULLS LAST, created_at DESC
            """,
            str(user_id),
        )
    except asyncpg.exceptions.UndefinedTableError:
        return []
    except Exception:
        logger.debug("list_thumbnail_studio_personas query failed", exc_info=True)
        return []

    linked_assets = {
        str(r["local_persona_id"]): str(r["pikzels_pikzonality_id"])
        for r in asset_rows
        if r.get("local_persona_id") and r.get("pikzels_pikzonality_id")
    }
    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        prof = r.get("profile_json")
        if isinstance(prof, str):
            try:
                prof = json.loads(prof)
            except Exception:
                prof = {}
        if not isinstance(prof, dict):
            prof = {}
        pkz = str(prof.get("pikzels_pikzonality_id") or "").strip()
        if not pkz:
            pkz = linked_assets.get(str(r["id"]), "")
        link_status = str(r.get("link_status") or "").lower()
        if pkz:
            link_status = "linked"
        out_rows.append(
            {
                "id": str(r["id"]),
                "name": r["name"],
                "image_count": int(r["image_count"] or 0),
                "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
                "pikzels_linked": bool(pkz),
                "link_status": link_status or "local",
                "link_error": str(r.get("link_error") or "")[:500],
            }
        )
    return out_rows
