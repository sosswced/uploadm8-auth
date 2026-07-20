"""
Offline-style ML hooks: outcome labels, simple baseline vs model comparison, variant leaderboard.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import asyncpg


async def record_outcome_label(
    conn: asyncpg.Connection,
    *,
    user_id: str,
    upload_id: Optional[str],
    variant_id: Optional[str],
    feature_snapshot: Dict[str, Any],
    label_json: Dict[str, Any],
) -> None:
    await conn.execute(
        """
        INSERT INTO ml_outcome_labels (user_id, upload_id, variant_id, feature_snapshot, label_json)
        VALUES ($1::uuid, $2::uuid, $3, $4::jsonb, $5::jsonb)
        """,
        user_id,
        upload_id,
        (variant_id or "")[:128] or None,
        json.dumps(feature_snapshot or {}),
        json.dumps(label_json or {}),
    )


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _simple_propensity_score(features: Dict[str, Any]) -> float:
    """Toy scorer: maps sparse features to [0,1] — swap for trained weights later."""
    u = float(features.get("uploads_window") or 0)
    ctr = float(features.get("nudge_ctr_pct") or 0)
    rev = 1.0 if float(features.get("revenue_7d") or 0) > 0 else 0.0
    z = -1.2 + 0.04 * u + 0.03 * ctr + 1.1 * rev
    return _sigmoid(z)


async def evaluate_uplift_vs_baseline(
    conn: asyncpg.Connection,
    *,
    model_key: str = "propensity_v1",
    min_samples: int = 40,
    record_audit: bool = True,
) -> Dict[str, Any]:
    """
    Compare label rate in top half of model scores vs bottom half (offline proxy).
    Only promote if lift exceeds margin.

    When ``record_audit`` is False, skips writes to ``ml_model_promotion_audit`` (dashboard GETs).
    """
    rows = await conn.fetch(
        """
        SELECT feature_snapshot, label_json
        FROM ml_outcome_labels
        WHERE created_at >= NOW() - INTERVAL '90 days'
        ORDER BY created_at DESC
        LIMIT 5000
        """
    )
    scored: List[tuple] = []
    for r in rows:
        fs = r["feature_snapshot"]
        if isinstance(fs, str):
            try:
                fs = json.loads(fs)
            except Exception:
                fs = {}
        if not isinstance(fs, dict):
            fs = {}
        lj = r["label_json"]
        if isinstance(lj, str):
            try:
                lj = json.loads(lj)
            except Exception:
                lj = {}
        if not isinstance(lj, dict):
            lj = {}
        y = 1.0 if float(lj.get("conversion") or lj.get("revenue_7d") or 0) > 0 else 0.0
        if lj.get("selected_variant"):
            y = max(y, 1.0)
        s = _simple_propensity_score(fs)
        scored.append((s, y))

    n = len(scored)
    if n < min_samples:
        out = {
            "model_key": model_key,
            "promoted": False,
            "reason": "insufficient_samples",
            "sample_n": n,
            "baseline_rate": None,
            "model_rate": None,
            "lift": None,
        }
        if record_audit:
            await conn.execute(
                """
                INSERT INTO ml_model_promotion_audit
                    (model_key, baseline_rate, model_rate, lift, promoted, sample_n, meta)
                    VALUES ($1, NULL, NULL, NULL, FALSE, $2, $3::jsonb)
                """,
                model_key[:80],
                n,
                json.dumps(out),
            )
        return out

    scored.sort(key=lambda x: x[0])
    mid = n // 2
    low = scored[:mid]
    high = scored[mid:]
    base_rate = sum(y for _, y in low) / max(len(low), 1)
    top_rate = sum(y for _, y in high) / max(len(high), 1)
    lift = top_rate - base_rate
    margin = 0.03
    promoted = lift >= margin and top_rate > base_rate

    meta = {
        "model_key": model_key,
        "sample_n": n,
        "baseline_rate": base_rate,
        "model_rate": top_rate,
        "lift": lift,
        "promoted": promoted,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    if record_audit:
        await conn.execute(
            """
            INSERT INTO ml_model_promotion_audit
                (model_key, baseline_rate, model_rate, lift, promoted, sample_n, meta)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
            """,
            model_key[:80],
            base_rate,
            top_rate,
            lift,
            promoted,
            n,
            json.dumps(meta),
        )
    return meta


async def fetch_marketing_conversion_proxy(
    conn: asyncpg.Connection, *, days: int = 90, limit_rows: int = 80
) -> Dict[str, Any]:
    """
    Roll up marketing_events rows that carry ``payload.campaign_id`` — sends, opens,
    and generic funnel events for proxying conversion analysis in the admin UI.
    """
    days = max(7, min(int(days), 365))
    limit_rows = max(10, min(int(limit_rows), 500))
    rows = await conn.fetch(
        """
        SELECT
            NULLIF(TRIM(payload->>'campaign_id'), '') AS campaign_id,
            COUNT(*) FILTER (WHERE event_type = 'campaign_email_sent')::bigint AS email_sent,
            COUNT(*) FILTER (WHERE event_type = 'campaign_discord_prompt')::bigint AS discord_sent,
            COUNT(*) FILTER (WHERE event_type IN ('campaign_in_app_prompt', 'campaign_discount_offer'))::bigint AS in_app_sent,
            COUNT(*) FILTER (WHERE event_type = 'campaign_email_open')::bigint AS email_open,
            COUNT(*) FILTER (WHERE event_type = 'shown')::bigint AS nudge_shown,
            COUNT(*) FILTER (WHERE event_type = 'clicked')::bigint AS nudge_clicked,
            COUNT(*) FILTER (WHERE event_type = 'converted')::bigint AS converted,
            COUNT(*) FILTER (WHERE event_type = 'dismissed')::bigint AS dismissed
        FROM marketing_events
        WHERE created_at >= NOW() - ($1::int * INTERVAL '1 day')
          AND NULLIF(TRIM(payload->>'campaign_id'), '') IS NOT NULL
        GROUP BY 1
        ORDER BY (
            COUNT(*) FILTER (WHERE event_type = 'campaign_email_sent')
          + COUNT(*) FILTER (WHERE event_type = 'campaign_discord_prompt')
          + COUNT(*) FILTER (WHERE event_type IN ('campaign_in_app_prompt', 'campaign_discount_offer'))
        ) DESC NULLS LAST
        LIMIT $2
        """,
        days,
        limit_rows,
    )
    by_campaign: List[Dict[str, Any]] = []
    for r in rows:
        cid = str(r["campaign_id"] or "")
        es = int(r["email_sent"] or 0)
        eo = int(r["email_open"] or 0)
        by_campaign.append(
            {
                "campaign_id": cid,
                "email_sent": es,
                "discord_sent": int(r["discord_sent"] or 0),
                "in_app_sent": int(r["in_app_sent"] or 0),
                "email_open": eo,
                "nudge_shown": int(r["nudge_shown"] or 0),
                "nudge_clicked": int(r["nudge_clicked"] or 0),
                "converted": int(r["converted"] or 0),
                "dismissed": int(r["dismissed"] or 0),
                "open_rate_proxy_pct": round(100.0 * eo / max(es, 1), 2) if es else 0.0,
            }
        )
    return {"days": days, "by_campaign": by_campaign, "truncated_to": limit_rows}


async def infer_promo_media_defaults(conn: asyncpg.Connection) -> Dict[str, Any]:
    """
    Suggest ``promo_media`` keys from aggregate wallet-pressure levers + recent ML labels.

    Callers merge over these defaults so explicit UI fields win.
    """
    from services.growth_intelligence import fetch_sales_opportunity_levers

    levers = await fetch_sales_opportunity_levers(conn)
    low_aic = int(levers.get("users_low_aic_available_0_9") or 0)
    low_put = int(levers.get("users_low_put_available_0_29") or 0)
    if low_aic >= low_put and low_aic > 0:
        card_kind, amount = "topup_aic", 100
    elif low_put > 0:
        card_kind, amount = "topup_put", 50
    else:
        card_kind, amount = "sub_upgrade", None

    variant_hint: Optional[str] = None
    row = await conn.fetchrow(
        """
        SELECT variant_id
        FROM ml_outcome_labels
        WHERE created_at >= NOW() - INTERVAL '120 days'
          AND variant_id IS NOT NULL
          AND variant_id LIKE 'v\\_%' ESCAPE '\\'
          AND LENGTH(variant_id) BETWEEN 4 AND 64
        ORDER BY created_at DESC
        LIMIT 1
        """
    )
    if row and row.get("variant_id"):
        variant_hint = str(row["variant_id"])[:128]

    return {
        "personalize_product_card": True,
        "card_kind": card_kind,
        "amount": amount,
        "variant_id": variant_hint,
        "infer_source": "wallet_levers_plus_recent_variant",
        "levers_snapshot": {"users_low_aic": low_aic, "users_low_put": low_put},
    }


async def fetch_variant_leaderboard(conn: asyncpg.Connection, limit: int = 40) -> List[Dict[str, Any]]:
    """
    Rank Thumbnail Studio variants by labeled selections + mean upload views when upload_id is linked.
    """
    rows = await conn.fetch(
        """
        SELECT
            mol.variant_id AS variant_key,
            COUNT(*)::bigint AS samples,
            COUNT(*) FILTER (WHERE COALESCE(mol.label_json->>'event', '') = 'selected')::bigint AS selections,
            COALESCE(AVG(u.views), 0)::float AS mean_views
        FROM ml_outcome_labels mol
        LEFT JOIN uploads u ON u.id = mol.upload_id
        WHERE mol.created_at >= NOW() - INTERVAL '120 days'
          AND COALESCE(mol.variant_id, '') <> ''
        GROUP BY mol.variant_id
        ORDER BY selections DESC, mean_views DESC, samples DESC
        LIMIT $1
        """,
        limit,
    )
    out = []
    for r in rows:
        out.append(
            {
                "variant_key": r["variant_key"],
                "samples": int(r["samples"] or 0),
                "selections": int(r["selections"] or 0),
                "mean_views": float(r["mean_views"] or 0),
            }
        )
    return out


async def record_thumbnail_studio_engine_ml_batch(
    conn: asyncpg.Connection,
    *,
    user_id: str,
    job_id: str,
    engine_mode: str,
    variants: List[Dict[str, Any]],
    youtube_video_id: Optional[str] = None,
    upload_id: Optional[str] = None,
) -> None:
    """
    One row per Thumbnail Studio job for offline ML / uplift evaluation
    (feature_snapshot + variant score vector). Distinct from user \"selected\" feedback.

    When ``upload_id`` is set, the row is tied to an upload job (publish outcomes)
    and ``variant_id`` uses an ``upload_studio:`` prefix instead of ``studio_job:``.
    """
    from services.wallet_marketing import _user_campaign_features

    feats = await _user_campaign_features(conn, str(user_id), "30d")
    scores: List[Dict[str, Any]] = []
    for v in variants:
        if not isinstance(v, dict):
            continue
        scores.append(
            {
                "index": v.get("index"),
                "ctr_score": v.get("ctr_score"),
                "engine_status": v.get("engine_status"),
                "pikzels_main_score": v.get("pikzels_main_score"),
                "pikzels_recreate_http_status": v.get("pikzels_recreate_http_status"),
            }
        )
    uid = (str(upload_id).strip() if upload_id else "") or None
    if uid:
        variant_key = f"upload_studio:{uid}"[:128]
        link_upload = uid
        pipeline_source = "upload_pipeline"
    else:
        variant_key = f"studio_job:{job_id}"[:128]
        link_upload = None
        pipeline_source = "thumbnail_studio_tool"
    await record_outcome_label(
        conn,
        user_id=str(user_id),
        upload_id=link_upload,
        variant_id=variant_key,
        feature_snapshot=dict(feats),
        label_json={
            "event": "thumbnail_studio_engine_batch",
            "job_id": job_id,
            "engine_mode": engine_mode,
            "youtube_video_id": youtube_video_id,
            "variant_scores": scores[:16],
            "pipeline_source": pipeline_source,
        },
    )
