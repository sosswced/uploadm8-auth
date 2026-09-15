"""
AV training pack retention helpers (P7).

Compact packs only — never full masters. TTL + account purge for ml-packs/.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("uploadm8-api")

# Default 90-day TTL for compact packs (env override).
DEFAULT_PACK_TTL_DAYS = int(os.environ.get("AV_TRAINING_PACK_TTL_DAYS", "90") or "90")


def pack_expired(created_at_iso: Optional[str], *, now: Optional[datetime] = None) -> bool:
    if not created_at_iso:
        return False
    try:
        created = datetime.fromisoformat(str(created_at_iso).replace("Z", "+00:00"))
    except Exception:
        return False
    now = now or datetime.now(timezone.utc)
    return created + timedelta(days=max(1, DEFAULT_PACK_TTL_DAYS)) < now


def _collect_pack_keys(meta: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    pack_key = str(meta.get("pack_r2_key") or "").strip()
    if pack_key:
        keys.append(pack_key)
    for k in meta.get("keyframe_r2_keys") or []:
        s = str(k or "").strip()
        if s and s not in keys:
            keys.append(s)
    return keys


def list_expired_pack_targets(
    rows: Sequence[Dict[str, Any]],
    *,
    now: Optional[datetime] = None,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """
    From DB-shaped rows ``{upload_id, user_id, pack_meta}``, select expired packs
    with non-empty R2 keys. ``pack_meta`` is ``output_artifacts.av_training_pack_v1``.
    """
    now = now or datetime.now(timezone.utc)
    out: List[Dict[str, Any]] = []
    for row in rows:
        if len(out) >= max(1, int(limit)):
            break
        meta = row.get("pack_meta") if isinstance(row.get("pack_meta"), dict) else {}
        if not meta:
            continue
        if meta.get("purged_at"):
            continue
        created = meta.get("created_at")
        if not pack_expired(created, now=now):
            continue
        keys = _collect_pack_keys(meta)
        if not keys:
            continue
        out.append(
            {
                "upload_id": str(row.get("upload_id") or ""),
                "user_id": str(row.get("user_id") or ""),
                "created_at": created,
                "r2_keys": keys,
                "pack_r2_key": str(meta.get("pack_r2_key") or ""),
            }
        )
    return out


async def fetch_expired_pack_targets(
    conn: Any,
    *,
    limit: int = 500,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Scan uploads for pack meta past TTL with R2 keys (asyncpg connection)."""
    sql = """
    SELECT u.id AS upload_id,
           u.user_id,
           u.output_artifacts->'av_training_pack_v1' AS pack_meta
    FROM uploads u
    WHERE u.output_artifacts ? 'av_training_pack_v1'
      AND COALESCE((u.output_artifacts->'av_training_pack_v1'->>'pack_r2_key'), '') <> ''
    ORDER BY u.created_at ASC
    LIMIT $1::int
    """
    # Fetch a wider window then filter by TTL in Python (created_at lives in JSON).
    fetch_n = max(int(limit) * 5, 500)
    rows = await conn.fetch(sql, fetch_n)
    shaped: List[Dict[str, Any]] = []
    for r in rows:
        meta = r["pack_meta"]
        if isinstance(meta, str):
            import json

            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        if not isinstance(meta, dict):
            meta = {}
        shaped.append(
            {
                "upload_id": r["upload_id"],
                "user_id": r["user_id"],
                "pack_meta": meta,
            }
        )
    return list_expired_pack_targets(shaped, now=now, limit=limit)


def purge_pack_r2_keys(keys: Sequence[str]) -> int:
    """Batch-delete R2 object keys. Returns deleted count (best-effort)."""
    clean = [str(k).strip() for k in keys if str(k or "").strip()]
    if not clean:
        return 0
    try:
        from core.config import R2_BUCKET_NAME
        from core.r2 import get_s3_client

        if not R2_BUCKET_NAME:
            return 0
        s3 = get_s3_client()
        deleted = 0
        # S3 DeleteObjects max 1000
        for i in range(0, len(clean), 1000):
            chunk = clean[i : i + 1000]
            s3.delete_objects(
                Bucket=R2_BUCKET_NAME,
                Delete={"Objects": [{"Key": k} for k in chunk], "Quiet": True},
            )
            deleted += len(chunk)
        return deleted
    except Exception as e:
        logger.warning("purge_pack_r2_keys failed: %s", e)
        return 0


async def mark_pack_purged(conn: Any, upload_id: Any) -> None:
    """Clear pack R2 keys on artifact and set purged_at (keeps coarse meta)."""
    import json

    uid = str(upload_id)
    row = await conn.fetchrow(
        "SELECT output_artifacts FROM uploads WHERE id = $1::uuid",
        uid,
    )
    if not row:
        return
    arts = row["output_artifacts"]
    if isinstance(arts, str):
        try:
            arts = json.loads(arts)
        except Exception:
            return
    if not isinstance(arts, dict):
        return
    pack = arts.get("av_training_pack_v1")
    if not isinstance(pack, dict):
        return
    pack = dict(pack)
    pack["pack_r2_key"] = ""
    pack["keyframe_r2_keys"] = []
    pack["purged_at"] = datetime.now(timezone.utc).isoformat()
    arts = dict(arts)
    arts["av_training_pack_v1"] = pack
    await conn.execute(
        "UPDATE uploads SET output_artifacts = $2::jsonb WHERE id = $1::uuid",
        uid,
        json.dumps(arts, default=str),
    )


def purge_user_ml_packs(user_id: Any) -> int:
    """
    Best-effort delete of R2 ml-packs/{user_id}/ prefix.
    Returns deleted object count (0 if R2 unavailable).
    """
    uid = str(user_id or "").strip()
    if not uid:
        return 0
    prefix = f"ml-packs/{uid}/"
    try:
        from core.config import R2_BUCKET_NAME
        from core.r2 import get_s3_client

        if not R2_BUCKET_NAME:
            return 0
        s3 = get_s3_client()
        deleted = 0
        token = None
        while True:
            kwargs = {"Bucket": R2_BUCKET_NAME, "Prefix": prefix, "MaxKeys": 1000}
            if token:
                kwargs["ContinuationToken"] = token
            resp = s3.list_objects_v2(**kwargs)
            contents = resp.get("Contents") or []
            if contents:
                s3.delete_objects(
                    Bucket=R2_BUCKET_NAME,
                    Delete={"Objects": [{"Key": o["Key"]} for o in contents], "Quiet": True},
                )
                deleted += len(contents)
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return deleted
    except Exception as e:
        logger.warning("purge_user_ml_packs failed for %s: %s", uid, e)
        return 0


__all__ = [
    "DEFAULT_PACK_TTL_DAYS",
    "pack_expired",
    "list_expired_pack_targets",
    "fetch_expired_pack_targets",
    "purge_pack_r2_keys",
    "mark_pack_purged",
    "purge_user_ml_packs",
]
