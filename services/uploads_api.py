"""
Upload HTTP helpers (platform_results enrichment, PATCH metadata) for routers/uploads.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, List

from fastapi import HTTPException

from core.helpers import _now_utc, _safe_col
from core.sql_allowlist import UPLOADS_METADATA_PATCH_COLUMNS, assert_set_fragments_columns
from core.models import UploadUpdate
from core.r2 import resolve_stored_account_avatar_url

logger = logging.getLogger("uploadm8-api")


def _parse_platform_results_items(upload_row: dict) -> list:
    raw = upload_row.get("platform_results") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = []

    if isinstance(raw, dict):
        return [{"platform": k, **v} for k, v in raw.items() if isinstance(v, dict)]
    if isinstance(raw, list):
        return list(raw)
    return []


def _platform_items_already_enriched(items: list) -> bool:
    if not items:
        return True
    successful = [e for e in items if e.get("success") is not False]
    return bool(
        successful
        and all(
            e.get("account_username") or e.get("account_name") or e.get("account_id")
            for e in successful
        )
    )


def _resolve_platform_result_avatars(items: list) -> list:
    """Presign R2 keys in account_avatar so <img src> works without same-origin redirect."""
    out: list = []
    for e in items:
        if isinstance(e, dict):
            out.append({**e, "account_avatar": resolve_stored_account_avatar_url(e.get("account_avatar"))})
        else:
            out.append(e)
    return out


def _merge_platform_entries(items: list, token_map: dict, platform_fallback: dict) -> list:
    enriched = []
    for entry in items:
        p = (entry.get("platform") or "").lower()

        stored_token_id = entry.get("token_row_id") or ""
        if stored_token_id and stored_token_id in token_map:
            acct = token_map[stored_token_id]
        elif token_map:
            acct = next((v for v in token_map.values() if v.get("platform") == p), {})
        else:
            acct = platform_fallback.get(p, {})

        merged = {**entry}
        for field in ("token_row_id", "account_id", "account_name", "account_username", "account_avatar"):
            if not merged.get(field) and acct.get(field):
                merged[field] = acct[field]

        enriched.append(merged)

    return enriched


async def enrich_platform_results_batch(conn, upload_rows: list, user_id: str) -> list:
    """
    Enrich platform_results for many uploads with at most two DB round-trips total
    (target token ids + per-platform primary fallbacks), instead of up to two per row.
    """
    n = len(upload_rows)
    if n == 0:
        return []

    out: list = [None] * n
    pending_indices: list[int] = []
    target_uuid_set: set[str] = set()
    platform_set: set[str] = set()

    for i, upload_row in enumerate(upload_rows):
        items = _parse_platform_results_items(upload_row)
        if not items:
            out[i] = []
            continue
        if _platform_items_already_enriched(items):
            out[i] = items
            continue
        pending_indices.append(i)
        for t in upload_row.get("target_accounts") or []:
            if t:
                target_uuid_set.add(str(t))
        for e in items:
            p = (e.get("platform") or "").strip().lower()
            if p:
                platform_set.add(p)

    global_by_id: dict = {}
    if target_uuid_set:
        try:
            tok_rows = await conn.fetch(
                """SELECT id, platform, account_id, account_name, account_username, account_avatar
                   FROM platform_tokens
                   WHERE user_id = $1 AND id = ANY($2::uuid[]) AND revoked_at IS NULL""",
                user_id,
                list(target_uuid_set),
            )
            for r in tok_rows:
                global_by_id[str(r["id"])] = {
                    "token_row_id": str(r["id"]),
                    "account_id": r["account_id"] or "",
                    "account_name": r["account_name"] or "",
                    "account_username": r["account_username"] or "",
                    "account_avatar": r["account_avatar"] or "",
                    "platform": r["platform"],
                }
        except Exception as e:
            logger.warning("enrich_platform_results_batch target lookup failed: %s", e)

    global_platform_fallback: dict = {}
    if platform_set:
        try:
            pf_rows = await conn.fetch(
                """SELECT DISTINCT ON (platform)
                          id, platform, account_id, account_name, account_username, account_avatar
                   FROM platform_tokens
                   WHERE user_id = $1 AND platform = ANY($2::text[]) AND revoked_at IS NULL
                   ORDER BY platform, is_primary DESC NULLS LAST, updated_at DESC""",
                user_id,
                list(platform_set),
            )
            for r in pf_rows:
                global_platform_fallback[r["platform"]] = {
                    "token_row_id": str(r["id"]),
                    "account_id": r["account_id"] or "",
                    "account_name": r["account_name"] or "",
                    "account_username": r["account_username"] or "",
                    "account_avatar": r["account_avatar"] or "",
                }
        except Exception as e:
            logger.warning("enrich_platform_results_batch fallback lookup failed: %s", e)

    for i in pending_indices:
        upload_row = upload_rows[i]
        items = _parse_platform_results_items(upload_row)
        target_ids = [str(t) for t in (upload_row.get("target_accounts") or []) if t]
        token_map = {tid: global_by_id[tid] for tid in target_ids if tid in global_by_id}
        out[i] = _merge_platform_entries(items, token_map, global_platform_fallback)

    for i in range(n):
        row = out[i]
        if isinstance(row, list):
            out[i] = _resolve_platform_result_avatars(row)

    return out


async def enrich_platform_results(conn, upload_row: dict, user_id: str) -> list:
    """
    Return platform_results as a flat list. Each entry enriched with
    account_name/username/avatar.
    """
    batch = await enrich_platform_results_batch(conn, [upload_row], user_id)
    return batch[0] if batch else []


def parse_smart_schedule(sm: dict, upload_platforms: list) -> tuple:
    """Returns (schedule_metadata_dict, scheduled_time_dt)."""
    if not isinstance(sm, dict):
        raise HTTPException(400, "smart_schedule must be a dict of platform -> ISO datetime string")
    if not sm:
        raise HTTPException(400, "smart_schedule requires per-platform times (non-empty object)")
    platforms = list(upload_platforms or [])
    for k in sm:
        if k not in platforms:
            raise HTTPException(400, f"smart_schedule platform '{k}' not in upload platforms")
    dts = []
    for v in sm.values():
        if not v:
            continue
        s = str(v).replace("Z", "+00:00").replace("z", "+00:00")
        try:
            dts.append(datetime.fromisoformat(s))
        except ValueError:
            raise HTTPException(400, "smart_schedule values must be valid ISO datetime strings")
    metadata = {k: v for k, v in sm.items() if v}
    scheduled_dt = min(dts) if dts else None
    return metadata, scheduled_dt


async def update_upload_metadata(conn, upload_id: str, user_id: str, update_data: UploadUpdate) -> None:
    """PATCH fields: title, caption, hashtags, scheduled_time, smart_schedule."""
    upload = await conn.fetchrow(
        "SELECT id, status, platforms FROM uploads WHERE id = $1 AND user_id = $2",
        upload_id,
        user_id,
    )
    if not upload:
        raise HTTPException(404, "Upload not found")
    editable = ("pending", "scheduled", "queued", "staged", "ready_to_publish")
    if upload["status"] not in editable:
        raise HTTPException(400, "Cannot edit upload that is already processing or published")

    cols = UPLOADS_METADATA_PATCH_COLUMNS
    updates: List[str] = []
    params: List[Any] = [upload_id, user_id]
    param_count = 2

    if update_data.title is not None:
        param_count += 1
        updates.append(f"{_safe_col('title', cols)} = ${param_count}")
        params.append(update_data.title)

    if update_data.caption is not None:
        param_count += 1
        updates.append(f"{_safe_col('caption', cols)} = ${param_count}")
        params.append(update_data.caption)

    if update_data.hashtags is not None:
        param_count += 1
        updates.append(f"{_safe_col('hashtags', cols)} = ${param_count}")
        params.append(update_data.hashtags)

    if update_data.scheduled_time is not None:
        param_count += 1
        updates.append(f"{_safe_col('scheduled_time', cols)} = ${param_count}")
        params.append(update_data.scheduled_time)

    if update_data.smart_schedule is not None:
        metadata, scheduled_dt = parse_smart_schedule(update_data.smart_schedule, upload["platforms"])
        param_count += 1
        updates.append(f"{_safe_col('schedule_metadata', cols)} = ${param_count}::jsonb")
        params.append(json.dumps(metadata))
        if scheduled_dt is not None:
            param_count += 1
            updates.append(f"{_safe_col('scheduled_time', cols)} = ${param_count}")
            params.append(scheduled_dt)
        param_count += 1
        updates.append(f"{_safe_col('schedule_mode', cols)} = ${param_count}")
        params.append("smart")

    if not updates:
        raise HTTPException(400, "No updates provided")

    param_count += 1
    updates.append(f"{_safe_col('updated_at', cols)} = ${param_count}")
    params.append(_now_utc())

    assert_set_fragments_columns(updates, UPLOADS_METADATA_PATCH_COLUMNS)

    await conn.execute(f"UPDATE uploads SET {', '.join(updates)} WHERE id = $1 AND user_id = $2", *params)
