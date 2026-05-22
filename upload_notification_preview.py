"""
Presigned HTTPS preview URLs for upload completion (email + user Discord webhook).

Copies the best stored thumbnail (or processed still) into ``notifications/upload-previews/``
so Mailgun/Discord can fetch a long-lived GET URL without relying on transient pipeline keys.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from core.r2 import copy_r2_object_within_bucket, generate_presigned_download_url

logger = logging.getLogger("uploadm8-worker")


def thumbnail_quality_summary_text(ctx: Any) -> str:
    """Short human-readable thumbnail pipeline summary for notifications (not Pikzels API score)."""
    art: Dict[str, Any] = getattr(ctx, "output_artifacts", None) or {}
    if not isinstance(art, dict):
        art = {}
    bits: list[str] = []
    raw = art.get("thumbnail_scores")
    scores: list[float] = []
    if isinstance(raw, str) and raw.strip():
        try:
            d = json.loads(raw)
            if isinstance(d, dict):
                for v in d.values():
                    try:
                        scores.append(float(v))
                    except (TypeError, ValueError):
                        continue
        except Exception:
            pass
    ts = getattr(ctx, "thumbnail_scores", None) or {}
    if isinstance(ts, dict) and ts:
        for v in ts.values():
            try:
                scores.append(float(v))
            except (TypeError, ValueError):
                continue
    if scores:
        bits.append(f"Best frame sharpness (internal): {max(scores):.3f}")
    rm = str(art.get("thumbnail_render_method") or "").strip()
    if rm:
        bits.append(f"Render: {rm}")
    cat = str(art.get("thumbnail_category") or "").strip()
    if cat:
        bits.append(f"Category: {cat}")
    return " · ".join(bits)[:500]


def _pick_source_r2_key(ctx: Any, row: Optional[Dict[str, Any]]) -> str:
    for k in ("thumbnail_r2_key", "processed_r2_key"):
        v = getattr(ctx, k, None)
        if v and str(v).strip():
            return str(v).strip()
    if row:
        for k in ("thumbnail_r2_key", "processed_r2_key"):
            v = row.get(k)
            if v and str(v).strip():
                return str(v).strip()
    return ""


async def resolve_upload_notification_preview_https_url(db_pool: Any, ctx: Any) -> Optional[str]:
    """
    Return an ``https://`` presigned URL for the upload's published thumbnail preview, or ``None``.

    When R2 copy succeeds, the object lives at
    ``notifications/upload-previews/{user_id}/{upload_id}/preview.jpg`` (overwritten per completion).
    """
    uid = str(getattr(ctx, "user_id", "") or "").strip()
    upid = str(getattr(ctx, "upload_id", "") or "").strip()
    if not upid:
        return None

    row: Optional[Dict[str, Any]] = None
    if db_pool and uid:
        try:
            async with db_pool.acquire() as conn:
                r = await conn.fetchrow(
                    """
                    SELECT thumbnail_r2_key, processed_r2_key
                    FROM uploads
                    WHERE id = $1::uuid AND user_id = $2::uuid
                    """,
                    upid,
                    uid,
                )
            if r:
                row = dict(r)
        except Exception as e:
            logger.debug("notification preview DB lookup failed upload=%s: %s", upid, e)

    src = _pick_source_r2_key(ctx, row)
    if not src:
        return None

    sign_key = src
    if uid:
        dest = f"notifications/upload-previews/{uid}/{upid}/preview.jpg"
        try:
            copy_r2_object_within_bucket(src, dest)
            sign_key = dest
        except Exception as e:
            logger.debug(
                "notification preview R2 copy skipped/failed upload=%s — presigning source key: %s",
                upid,
                e,
            )

    try:
        url = generate_presigned_download_url(sign_key, ttl=604_800)
        if isinstance(url, str) and url.startswith("https://"):
            return url[:4096]
    except Exception as e:
        logger.warning("notification preview presign failed upload=%s: %s", upid, e)
    return None
