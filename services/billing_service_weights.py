"""
DB-backed per-service AIC weights (billing_service_weights table).

Seeded from stages.ai_service_costs.SERVICE_WEIGHTS; admin upserts tune live pricing.

Important: ``ensure_billing_weights_seeded`` uses ON CONFLICT DO NOTHING, so a code
deploy alone does **not** overwrite existing DB rows. Use
``migrate_legacy_weights_to_code_defaults`` (startup) and/or
``sync_service_weights_from_code`` (admin reset) to push a new calibration live.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional, Tuple

from stages.ai_service_costs import SERVICE_WEIGHTS

logger = logging.getLogger(__name__)

_MAX_WEIGHT = 5000


def coerce_updated_by_uuid(value: Optional[str]) -> Optional[str]:
    """``billing_service_weights.updated_by`` is UUID — reject calibration labels."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return str(uuid.UUID(text))
    except ValueError:
        return None

# Pre–Jul-2026 ordinal scale. Startup migrates a row only when it still matches
# these exact values (admin-tuned rows are left alone).
_LEGACY_CODE_DEFAULTS: Dict[str, int] = {
    "twelvelabs": 28,
    "video_intelligence": 24,
    "caption_llm": 22,
    "audio_whisper": 18,
    "thumbnail_ai": 16,
    "vision_google": 12,
    "dashcam_osd": 8,
    "audio_gpt_classify": 7,
    "audio_acr": 5,
    "thumbnail_recreate_ai": 14,
    "persona_consistency": 10,
    "thumbnail_ctr_ranker": 6,
    "marketing_image": 2,
    "thumbnail_competitor_gap": 4,
    "audio_yamnet": 2,
    "telemetry_trill": 1,
    "trend_intel": 2,
}

# Jul-2026 v2 calibration values that should move to current code defaults
# (e.g. audio_whisper 3 → 0 AIC-exempt). Admin-tuned rows that differ are kept.
_PRIOR_CALIBRATION_DEFAULTS: Dict[str, int] = {
    "audio_whisper": 3,
}


async def fetch_service_weights_map(conn: Any) -> Dict[str, int]:
    """Return all rows as service_id -> aic_weight (may be empty before first seed)."""
    try:
        rows = await conn.fetch("SELECT service_id, aic_weight FROM billing_service_weights")
    except Exception as e:
        logger.warning("billing_service_weights fetch failed (table missing?): %s", e)
        return {}
    out: Dict[str, int] = {}
    for r in rows or []:
        sid = str(r["service_id"] or "").strip()
        if not sid:
            continue
        try:
            out[sid] = max(0, min(_MAX_WEIGHT, int(r["aic_weight"] or 0)))
        except (TypeError, ValueError):
            continue
    return out


async def ensure_billing_weights_seeded(conn: Any) -> None:
    """Insert missing service_ids with code defaults (never overwrites ops tuning)."""
    for sid, w in SERVICE_WEIGHTS.items():
        try:
            await conn.execute(
                """
                INSERT INTO billing_service_weights (service_id, aic_weight)
                VALUES ($1, $2)
                ON CONFLICT (service_id) DO NOTHING
                """,
                sid,
                max(0, min(_MAX_WEIGHT, int(w))),
            )
        except Exception as e:
            logger.warning("ensure_billing_weights_seeded skip %s: %s", sid, e)


async def migrate_legacy_weights_to_code_defaults(conn: Any) -> int:
    """
    One-shot-safe: if a DB weight still equals a known prior code default,
    update it to the current ``SERVICE_WEIGHTS`` value. Custom admin edits are kept.
    Returns number of rows updated.
    """
    n = 0
    prior_maps = (_LEGACY_CODE_DEFAULTS, _PRIOR_CALIBRATION_DEFAULTS)
    for prior in prior_maps:
        for sid, legacy_w in prior.items():
            if sid not in SERVICE_WEIGHTS:
                continue
            new_w = max(0, min(_MAX_WEIGHT, int(SERVICE_WEIGHTS[sid])))
            if new_w == legacy_w:
                continue
            try:
                result = await conn.execute(
                    """
                    UPDATE billing_service_weights
                       SET aic_weight = $2,
                           updated_at = NOW()
                     WHERE service_id = $1
                       AND aic_weight = $3
                    """,
                    sid,
                    new_w,
                    int(legacy_w),
                )
                # asyncpg returns e.g. "UPDATE 1"
                if result and str(result).endswith("1"):
                    n += 1
                elif result and "UPDATE" in str(result).upper():
                    try:
                        touched = int(str(result).split()[-1])
                        n += max(0, touched)
                    except (TypeError, ValueError):
                        pass
            except Exception as e:
                logger.warning("migrate_legacy_weights skip %s: %s", sid, e)
    if n:
        logger.info("billing_service_weights: migrated %s legacy rows to code defaults", n)
    return n


async def sync_service_weights_from_code(
    conn: Any,
    *,
    updated_by: Optional[str] = None,
) -> int:
    """Force-upsert every ``SERVICE_WEIGHTS`` key (admin reset / deploy sync)."""
    return await upsert_service_weights(
        conn,
        {sid: int(w) for sid, w in SERVICE_WEIGHTS.items()},
        updated_by=updated_by,
    )


async def upsert_service_weights(
    conn: Any,
    weights: Dict[str, int],
    *,
    updated_by: Optional[str] = None,
) -> int:
    """
    Upsert validated weights. Returns number of rows written.
    Only keys present in SERVICE_WEIGHTS are accepted.
    """
    n = 0
    for sid, raw in weights.items():
        sid = str(sid or "").strip()
        if sid not in SERVICE_WEIGHTS:
            continue
        try:
            w = max(0, min(_MAX_WEIGHT, int(raw)))
        except (TypeError, ValueError):
            continue
        await conn.execute(
            """
            INSERT INTO billing_service_weights (service_id, aic_weight, updated_at, updated_by)
            VALUES ($1, $2, NOW(), $3::uuid)
            ON CONFLICT (service_id) DO UPDATE SET
                aic_weight = EXCLUDED.aic_weight,
                updated_at = NOW(),
                updated_by = COALESCE(EXCLUDED.updated_by, billing_service_weights.updated_by)
            """,
            sid,
            w,
            coerce_updated_by_uuid(updated_by),
        )
        n += 1
    return n


def weights_drift_summary(db_map: Optional[Dict[str, int]]) -> Dict[str, Any]:
    """Compare DB rows to code defaults (for admin UI / ops)."""
    db = db_map or {}
    drifted: Dict[str, Tuple[int, int]] = {}
    for sid, code_w in SERVICE_WEIGHTS.items():
        if sid in db and int(db[sid]) != int(code_w):
            drifted[sid] = (int(db[sid]), int(code_w))
    return {
        "calibration": "2026-07-v2",
        "code_defaults": dict(SERVICE_WEIGHTS),
        "drifted_count": len(drifted),
        "drifted": {k: {"db": v[0], "code": v[1]} for k, v in drifted.items()},
    }
