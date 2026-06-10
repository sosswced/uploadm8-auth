"""
Persist and recall user visual-entity buckets across uploads.

Each processed upload upserts named entities (vehicles, food, plants, …)
into ``user_visual_entity_catalog`` so M8 and hydration can reference what
this creator has shown before — dashcam, vlog, cooking, camping, fishing, etc.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from core.visual_entity_taxonomy import (
    RECOGNITION_BUCKETS,
    narrative_bucket_labels,
    niche_bucket_order,
)

logger = logging.getLogger("uploadm8.visual_entity_memory")

_PERSIST_BUCKETS = tuple(
    b for b in RECOGNITION_BUCKETS if b not in ("web_matches", "all_entities", "people", "text_on_screen")
)


def _normalize_entity(name: str) -> str:
    return re.sub(r"\s+", " ", str(name or "").strip().lower())[:200]


_VEHICLE_TOKENS = frozenset(
    {
        "car",
        "cars",
        "truck",
        "trucks",
        "vehicle",
        "vehicles",
        "motorcycle",
        "motorcycles",
        "bus",
        "suv",
        "van",
        "boat",
        "boats",
        "ship",
        "ships",
    }
)


def _flat_from_recognition_summary_row(row: Any) -> Dict[str, List[str]]:
    """Build catalog_flat from persisted summary when recognition_flat was never stored."""
    raw = row.get("raw_summary")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = {}
    if isinstance(raw, dict):
        nested = raw.get("recognition_flat")
        if isinstance(nested, dict) and nested:
            return {k: list(v) for k, v in nested.items() if isinstance(v, list) and v}

    flat: Dict[str, List[str]] = {}
    objs = [str(x).strip() for x in (row.get("top_objects") or []) if str(x).strip()]
    if objs:
        flat["objects"] = objs[:32]
        vehicles = [o for o in objs if _normalize_entity(o) in _VEHICLE_TOKENS]
        if vehicles:
            flat["vehicles"] = vehicles[:32]
    logos = [str(x).strip() for x in (row.get("top_logos") or []) if str(x).strip()]
    if logos:
        flat["brands"] = logos[:32]
    text = [str(x).strip() for x in (row.get("top_text") or []) if str(x).strip()]
    if text:
        flat["signage"] = text[:32]
    return flat


async def backfill_catalog_from_recognition_summaries(db_pool, user_id: str) -> int:
    """
    Repair empty user_visual_entity_catalog from upload_recognition_summary rows
    (uploads processed before catalog upsert or without recognition_flat in raw_summary).
    """
    if not db_pool or not user_id:
        return 0
    try:
        async with db_pool.acquire() as conn:
            has_catalog = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM user_visual_entity_catalog WHERE user_id = $1::uuid)",
                user_id,
            )
            if has_catalog:
                return 0
            rows = await conn.fetch(
                """
                SELECT upload_id::text AS upload_id, raw_summary, top_objects, top_logos, top_text
                FROM upload_recognition_summary
                WHERE user_id = $1::uuid
                ORDER BY updated_at DESC
                LIMIT 200
                """,
                user_id,
            )
        if not rows:
            return 0
        touched = 0
        for row in rows:
            flat = _flat_from_recognition_summary_row(row)
            if not flat:
                continue
            touched += await upsert_catalog_entities(
                db_pool,
                user_id=user_id,
                upload_id=str(row["upload_id"]),
                catalog_flat=flat,
                category="general",
            )
        if touched:
            logger.info(
                "[visual_entity_memory] backfilled catalog user=%s from %d summaries (~%d entity rows)",
                user_id,
                len(rows),
                touched,
            )
        return touched
    except Exception as e:
        logger.debug("[visual_entity_memory] catalog backfill: %s", str(e)[:160])
        return 0


async def upsert_catalog_entities(
    db_pool,
    *,
    user_id: str,
    upload_id: str,
    catalog_flat: Optional[Dict[str, List[str]]],
    category: str = "general",
) -> int:
    """
    Merge this upload's recognition_flat into the user's entity catalog.
    Returns count of rows touched (best-effort).
    """
    if not db_pool or not user_id or not upload_id:
        return 0
    if not isinstance(catalog_flat, dict) or not catalog_flat:
        return 0

    cat = (category or "general").strip().lower()[:64]
    touched = 0
    try:
        async with db_pool.acquire() as conn:
            for bucket in _PERSIST_BUCKETS:
                names = catalog_flat.get(bucket) or []
                if not isinstance(names, list):
                    continue
                for raw in names[:32]:
                    name = str(raw or "").strip()
                    if len(name) < 2:
                        continue
                    norm = _normalize_entity(name)
                    if len(norm) < 2:
                        continue
                    await conn.execute(
                        """
                        INSERT INTO user_visual_entity_catalog (
                            user_id, bucket, entity_name, normalized_name,
                            seen_count, last_category, last_upload_id, last_seen_at
                        ) VALUES ($1::uuid, $2, $3, $4, 1, $5, $6::uuid, NOW())
                        ON CONFLICT (user_id, bucket, normalized_name) DO UPDATE SET
                            seen_count = user_visual_entity_catalog.seen_count + 1,
                            entity_name = EXCLUDED.entity_name,
                            last_category = EXCLUDED.last_category,
                            last_upload_id = EXCLUDED.last_upload_id,
                            last_seen_at = NOW()
                        """,
                        user_id,
                        bucket[:32],
                        name[:200],
                        norm[:200],
                        cat,
                        upload_id,
                    )
                    touched += 1
    except Exception as e:
        err = str(e)
        if "relation" in err and "does not exist" in err:
            logger.debug(
                "[visual_entity_memory] table missing — run migrations: %s",
                err[:160],
            )
        else:
            logger.warning("[visual_entity_memory] upsert failed: %s", err[:200])
        return 0
    if touched:
        logger.info(
            "[visual_entity_memory] user=%s upload=%s upserted ~%d entity rows",
            user_id,
            upload_id,
            touched,
        )
    return touched


async def fetch_user_entity_recall(
    db_pool,
    *,
    user_id: str,
    category: str = "general",
    limit_per_bucket: int = 8,
) -> Dict[str, List[str]]:
    """
    Top entities this user has shown before, ordered by recency × frequency.
    Prioritizes buckets relevant to the content niche.
    """
    out: Dict[str, List[str]] = {b: [] for b in _PERSIST_BUCKETS}
    if not db_pool or not user_id:
        return out
    order = niche_bucket_order(category)
    try:
        async with db_pool.acquire() as conn:
            for bucket in order:
                if bucket not in _PERSIST_BUCKETS:
                    continue
                rows = await conn.fetch(
                    """
                    SELECT entity_name, seen_count, last_seen_at
                      FROM user_visual_entity_catalog
                     WHERE user_id = $1::uuid AND bucket = $2
                     ORDER BY seen_count DESC, last_seen_at DESC
                     LIMIT $3
                    """,
                    user_id,
                    bucket,
                    limit_per_bucket,
                )
                out[bucket] = [str(r["entity_name"]) for r in rows if r.get("entity_name")]
            # Fill any remaining buckets not in niche priority list
            for bucket in _PERSIST_BUCKETS:
                if out.get(bucket):
                    continue
                rows = await conn.fetch(
                    """
                    SELECT entity_name
                      FROM user_visual_entity_catalog
                     WHERE user_id = $1::uuid AND bucket = $2
                     ORDER BY seen_count DESC, last_seen_at DESC
                     LIMIT $3
                    """,
                    user_id,
                    bucket,
                    min(4, limit_per_bucket),
                )
                out[bucket] = [str(r["entity_name"]) for r in rows if r.get("entity_name")]
    except Exception as e:
        err = str(e)
        if "relation" not in err or "does not exist" not in err:
            logger.debug("[visual_entity_memory] recall fetch: %s", err[:160])
    return out


async def fetch_channel_catalog_detail(
    db_pool,
    *,
    user_id: str,
    category: str = "general",
    limit_per_bucket: int = 12,
) -> Dict[str, Any]:
    """
    Rich payload for Settings UI: entities per bucket with counts and dates.
    """
    labels = narrative_bucket_labels()
    buckets_out: List[Dict[str, Any]] = []
    total_entities = 0
    last_seen_at: Optional[str] = None
    uploads_with_entities = 0

    if not db_pool or not user_id:
        return {
            "category": category,
            "entity_count": 0,
            "bucket_count": 0,
            "uploads_with_entities": 0,
            "last_seen_at": None,
            "buckets": [],
        }

    await backfill_catalog_from_recognition_summaries(db_pool, user_id)

    order = niche_bucket_order(category)
    seen_bucket_keys: set = set()

    try:
        async with db_pool.acquire() as conn:
            stats = await conn.fetchrow(
                """
                SELECT COUNT(*)::int AS entity_count,
                       COUNT(DISTINCT bucket)::int AS bucket_count,
                       COUNT(DISTINCT last_upload_id)::int AS uploads_with_entities,
                       MAX(last_seen_at) AS last_seen_at
                  FROM user_visual_entity_catalog
                 WHERE user_id = $1::uuid
                """,
                user_id,
            )
            if stats:
                total_entities = int(stats["entity_count"] or 0)
                uploads_with_entities = int(stats["uploads_with_entities"] or 0)
                ts = stats.get("last_seen_at")
                if ts:
                    last_seen_at = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

            for bucket in order + [b for b in _PERSIST_BUCKETS if b not in order]:
                if bucket in seen_bucket_keys:
                    continue
                seen_bucket_keys.add(bucket)
                rows = await conn.fetch(
                    """
                    SELECT entity_name, seen_count, last_seen_at, last_category
                      FROM user_visual_entity_catalog
                     WHERE user_id = $1::uuid AND bucket = $2
                     ORDER BY seen_count DESC, last_seen_at DESC
                     LIMIT $3
                    """,
                    user_id,
                    bucket,
                    limit_per_bucket,
                )
                entities = []
                for r in rows:
                    ts = r.get("last_seen_at")
                    entities.append(
                        {
                            "name": str(r["entity_name"]),
                            "seen_count": int(r["seen_count"] or 1),
                            "last_seen_at": ts.isoformat() if ts and hasattr(ts, "isoformat") else None,
                            "last_category": str(r.get("last_category") or ""),
                        }
                    )
                if entities:
                    buckets_out.append(
                        {
                            "key": bucket,
                            "label": labels.get(bucket, bucket.replace("_", " ").title()),
                            "entities": entities,
                        }
                    )
    except Exception as e:
        err = str(e)
        if "relation" not in err or "does not exist" not in err:
            logger.debug("[visual_entity_memory] channel catalog: %s", err[:160])

    return {
        "category": (category or "general").lower(),
        "entity_count": total_entities,
        "bucket_count": len(buckets_out),
        "uploads_with_entities": uploads_with_entities,
        "last_seen_at": last_seen_at,
        "buckets": buckets_out,
    }


async def fetch_platform_bucket_kpis(
    db_pool,
    *,
    since,
    limit: int = 15,
) -> Dict[str, Any]:
    """Admin KPIs: top entities per bucket across all users in a time window."""
    out: Dict[str, List[Dict[str, Any]]] = {b: [] for b in _PERSIST_BUCKETS}
    if not db_pool:
        return {"buckets": out, "total_rows": 0}

    try:
        async with db_pool.acquire() as conn:
            total_rows = await conn.fetchval(
                """
                SELECT COUNT(*)::int
                  FROM user_visual_entity_catalog c
                  JOIN uploads u ON u.id = c.last_upload_id
                 WHERE u.created_at >= $1
                """,
                since,
            )
            for bucket in _PERSIST_BUCKETS:
                rows = await conn.fetch(
                    """
                    SELECT c.entity_name AS name,
                           SUM(c.seen_count)::bigint AS total_seen,
                           COUNT(DISTINCT c.user_id)::int AS creator_count
                      FROM user_visual_entity_catalog c
                      JOIN uploads u ON u.id = c.last_upload_id
                     WHERE c.bucket = $1
                       AND u.created_at >= $2
                     GROUP BY lower(c.entity_name), c.entity_name
                     ORDER BY total_seen DESC
                     LIMIT $3
                    """,
                    bucket,
                    since,
                    limit,
                )
                out[bucket] = [
                    {
                        "name": str(r["name"]),
                        "total_seen": int(r["total_seen"] or 0),
                        "creator_count": int(r["creator_count"] or 0),
                    }
                    for r in rows
                ]
    except Exception as e:
        err = str(e)
        if "relation" not in err or "does not exist" not in err:
            logger.debug("[visual_entity_memory] platform KPIs: %s", err[:160])
        return {"buckets": out, "total_rows": 0, "error": err[:200]}

    return {"buckets": out, "total_rows": int(total_rows or 0)}


def catalog_rows_for_hf_export(
    catalog: Dict[str, Any],
    *,
    user_id: str,
    hub_dataset_repo: str = "",
) -> List[Dict[str, Any]]:
    """Flatten channel catalog into Hub dataset rows (training / analytics)."""
    rows: List[Dict[str, Any]] = []
    uid = str(user_id)
    for block in catalog.get("buckets") or []:
        if not isinstance(block, dict):
            continue
        bucket = str(block.get("key") or "")
        for ent in block.get("entities") or []:
            if not isinstance(ent, dict):
                continue
            name = str(ent.get("name") or "").strip()
            if not name:
                continue
            rows.append(
                {
                    "user_id": uid,
                    "bucket": bucket,
                    "entity_name": name,
                    "seen_count": int(ent.get("seen_count") or 1),
                    "last_category": str(ent.get("last_category") or catalog.get("category") or ""),
                    "last_seen_at": ent.get("last_seen_at"),
                    "source": "uploadm8_visual_entity_catalog",
                    "dataset_repo": hub_dataset_repo,
                }
            )
    return rows


def format_recall_for_prompt(recall: Dict[str, List[str]], *, max_chars: int = 1200) -> str:
    """Compact block for M8: what this creator has recognized in past uploads."""
    if not isinstance(recall, dict):
        return ""
    from core.visual_entity_taxonomy import narrative_bucket_labels

    labels = narrative_bucket_labels()
    parts: List[str] = []
    for bucket, title in labels.items():
        names = [str(n).strip() for n in (recall.get(bucket) or []) if str(n).strip()]
        if names:
            parts.append(f"{title}: {', '.join(names[:8])}.")
    if not parts:
        return ""
    text = "VISUAL ENTITY MEMORY (your past uploads — reuse only when current Scene Graph supports it): " + " ".join(parts)
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text
