"""
Pikzels thumbnail analyzer — persist scores, recommendations, fixes, and ML labels.

Powers the KPI-page analyzer: results survive refresh; recommendations can be
applied, saved, or written back to the upload thumbnail.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from core.config import R2_BUCKET_NAME
from core.r2 import _normalize_r2_key, generate_presigned_download_url, put_object_bytes
from services.ml_marketing import record_outcome_label
from services.pikzels_v2_client import pikzels_v2_post
from services.thumbnail_studio import _pikzels_extract_image_url
from services.upload_pikzels_frame import append_hydration_to_prompt, load_upload_frame_jpeg_base64
from services.wallet_marketing import _user_campaign_features

logger = logging.getLogger("uploadm8-api")

RECOMMENDATION_STATUSES = frozenset({"open", "saved", "applied", "dismissed", "done"})

_SUBSCORE_FIX_HINTS: Dict[str, str] = {
    "clarity": "Boost contrast and readability: sharpen subject, simplify background, add bold readable text.",
    "curiosity": "Add a curiosity-gap headline overlay — tease the outcome without spoiling it.",
    "emotion": "Amplify facial expression and emotional contrast; warmer grade on the subject.",
    "virality": "Increase visual punch: brighter accent color, larger subject scale, stronger hook text.",
    "idea": "Clarify the single core idea with one bold text hook and a cleaner focal point.",
}


def _json_obj(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _main_score_from_response(data: Dict[str, Any]) -> Optional[float]:
    raw = data.get("main_score")
    if raw is None:
        raw = data.get("score")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def build_targeted_fix_prompt(suggestion: str, subscores: Optional[Dict[str, Any]]) -> str:
    """Prefer Pikzels suggestion; otherwise target the weakest subscore."""
    base = str(suggestion or "").strip()
    if base:
        return base[:950]
    subs = subscores if isinstance(subscores, dict) else {}
    weakest_key = ""
    weakest_val = 999.0
    for k, v in subs.items():
        try:
            n = float(v)
        except (TypeError, ValueError):
            continue
        if n < weakest_val:
            weakest_val = n
            weakest_key = str(k).strip().lower()
    hint = _SUBSCORE_FIX_HINTS.get(weakest_key.replace(" ", "_"), "")
    if hint:
        return hint[:950]
    return (
        "One-click thumbnail fix: add bold high-contrast text overlay, "
        "sharpen the main subject, and improve click appeal."
    )[:950]


def analysis_row_public(row: Any) -> Dict[str, Any]:
    d = dict(row)
    aid = d.get("id")
    uid = d.get("upload_id")
    ca = d.get("created_at")
    ua = d.get("updated_at")
    subs = d.get("subscores_json")
    if isinstance(subs, str):
        subs = _json_obj(subs)
    fix_subs = d.get("fix_subscores_json")
    if isinstance(fix_subs, str):
        fix_subs = _json_obj(fix_subs)
    titles = d.get("generated_titles_json")
    if isinstance(titles, str):
        try:
            titles = json.loads(titles)
        except Exception:
            titles = None
    fix_r2 = str(d.get("fix_r2_key") or "").strip()
    fix_preview = ""
    if fix_r2 and R2_BUCKET_NAME:
        try:
            fix_preview = generate_presigned_download_url(fix_r2, ttl=3600) or ""
        except Exception:
            fix_preview = ""
    if not fix_preview:
        fix_preview = str(d.get("fix_image_url") or "").strip()
    return {
        "analysis_id": str(aid) if aid else "",
        "upload_id": str(uid) if uid else "",
        "upload_title": str(d.get("upload_title") or "") or None,
        "main_score": d.get("main_score"),
        "subscores": subs if isinstance(subs, dict) else {},
        "suggestion": str(d.get("suggestion") or ""),
        "recommendation_status": str(d.get("recommendation_status") or "open"),
        "frame_source": str(d.get("frame_source") or "primary_thumbnail"),
        "title": str(d.get("title") or ""),
        "fix_image_url": str(d.get("fix_image_url") or ""),
        "fix_preview_url": fix_preview,
        "fix_r2_key": fix_r2 or None,
        "fix_score": d.get("fix_score"),
        "fix_subscores": fix_subs if isinstance(fix_subs, dict) else {},
        "generated_titles": titles,
        "persona_id": str(d.get("persona_id") or "") or None,
        "parent_analysis_id": str(d.get("parent_analysis_id") or "") or None,
        "created_at": ca.isoformat() if hasattr(ca, "isoformat") else str(ca or ""),
        "updated_at": ua.isoformat() if hasattr(ua, "isoformat") else str(ua or ""),
    }


async def record_analyzer_ml_event(
    conn: Any,
    *,
    user_id: str,
    upload_id: str,
    analysis_id: str,
    event: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        feats = await _user_campaign_features(conn, str(user_id), "30d")
        label: Dict[str, Any] = {
            "event": event,
            "analysis_id": analysis_id,
            "pipeline_source": "pikzels_kpi_analyzer",
        }
        if extra:
            label.update(extra)
        await record_outcome_label(
            conn,
            user_id=str(user_id),
            upload_id=str(upload_id),
            variant_id=f"pikzels_analyzer:{analysis_id}"[:128],
            feature_snapshot=dict(feats),
            label_json=label,
        )
    except Exception:
        logger.debug("pikzels analyzer ML label failed", exc_info=True)


async def persist_score_analysis(
    conn: Any,
    *,
    user_id: str,
    upload_id: str,
    frame_source: str,
    title: str,
    response_data: Dict[str, Any],
    persona_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Insert a scored analysis row and record ML outcome."""
    subs = response_data.get("subscores")
    if not isinstance(subs, dict):
        subs = {}
    suggestion = str(response_data.get("suggestion") or "").strip()
    main_score = _main_score_from_response(response_data)
    row = await conn.fetchrow(
        """
        INSERT INTO pikzels_thumbnail_analyses (
            user_id, upload_id, main_score, subscores_json, suggestion,
            recommendation_status, frame_source, title, response_json, persona_id
        )
        VALUES ($1::uuid, $2::uuid, $3, $4::jsonb, $5, 'open', $6, $7, $8::jsonb, $9)
        RETURNING *
        """,
        str(user_id),
        str(upload_id),
        main_score,
        json.dumps(subs),
        suggestion[:4000],
        (frame_source or "primary_thumbnail")[:32],
        (title or "")[:200] or None,
        json.dumps(response_data),
        (persona_id or "")[:128] or None,
    )
    out = analysis_row_public(row)
    await record_analyzer_ml_event(
        conn,
        user_id=str(user_id),
        upload_id=str(upload_id),
        analysis_id=out["analysis_id"],
        event="pikzels_analyzer_score",
        extra={
            "main_score": main_score,
            "subscores": subs,
            "has_suggestion": bool(suggestion),
        },
    )
    return out


async def get_analysis_for_user(
    conn: Any, user_id: str, analysis_id: str
) -> Optional[Dict[str, Any]]:
    try:
        aid = uuid.UUID(str(analysis_id).strip())
    except (ValueError, TypeError, AttributeError):
        return None
    row = await conn.fetchrow(
        """
        SELECT * FROM pikzels_thumbnail_analyses
        WHERE id = $1::uuid AND user_id = $2::uuid
        """,
        aid,
        str(user_id),
    )
    return analysis_row_public(row) if row else None


async def get_latest_analysis_for_upload(
    conn: Any, user_id: str, upload_id: str
) -> Optional[Dict[str, Any]]:
    try:
        uid = uuid.UUID(str(upload_id).strip())
    except (ValueError, TypeError, AttributeError):
        return None
    row = await conn.fetchrow(
        """
        SELECT * FROM pikzels_thumbnail_analyses
        WHERE upload_id = $1::uuid AND user_id = $2::uuid
        ORDER BY created_at DESC
        LIMIT 1
        """,
        uid,
        str(user_id),
    )
    return analysis_row_public(row) if row else None


async def list_analyses(
    conn: Any,
    *,
    user_id: str,
    upload_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 30,
) -> List[Dict[str, Any]]:
    params: List[Any] = [str(user_id)]
    where = "WHERE user_id = $1::uuid"
    if upload_id:
        try:
            params.append(uuid.UUID(str(upload_id).strip()))
            where += f" AND upload_id = ${len(params)}::uuid"
        except (ValueError, TypeError, AttributeError):
            return []
    if status and status in RECOMMENDATION_STATUSES:
        params.append(status)
        where += f" AND recommendation_status = ${len(params)}"
    params.append(max(1, min(int(limit), 100)))
    where_a = where.replace("user_id = $1::uuid", "a.user_id = $1::uuid").replace(
        "upload_id =", "a.upload_id ="
    ).replace("recommendation_status =", "a.recommendation_status =")
    rows = await conn.fetch(
        f"""
        SELECT a.*, COALESCE(u.title, u.ai_title, u.filename) AS upload_title
        FROM pikzels_thumbnail_analyses a
        LEFT JOIN uploads u ON u.id = a.upload_id
        {where_a}
        ORDER BY a.created_at DESC
        LIMIT ${len(params)}
        """,
        *params,
    )
    return [analysis_row_public(r) for r in rows]


async def update_recommendation_status(
    conn: Any,
    *,
    user_id: str,
    analysis_id: str,
    status: str,
) -> Optional[Dict[str, Any]]:
    st = (status or "").strip().lower()
    if st not in RECOMMENDATION_STATUSES:
        return None
    try:
        aid = uuid.UUID(str(analysis_id).strip())
    except (ValueError, TypeError, AttributeError):
        return None
    row = await conn.fetchrow(
        """
        UPDATE pikzels_thumbnail_analyses
        SET recommendation_status = $3, updated_at = NOW()
        WHERE id = $1::uuid AND user_id = $2::uuid
        RETURNING *
        """,
        aid,
        str(user_id),
        st,
    )
    if not row:
        return None
    out = analysis_row_public(row)
    if st in ("saved", "dismissed", "done"):
        await record_analyzer_ml_event(
            conn,
            user_id=str(user_id),
            upload_id=out["upload_id"],
            analysis_id=out["analysis_id"],
            event=f"pikzels_analyzer_{st}",
            extra={"recommendation_status": st},
        )
    return out


async def _download_image_bytes(url: str) -> Tuple[bytes, str]:
    from services.pikzels_v2 import resolve_public_api_key

    headers: Dict[str, str] = {}
    if "cdn.pikzels.com" in (url or ""):
        key = resolve_public_api_key()
        if key:
            headers["X-Api-Key"] = key
    async with httpx.AsyncClient(timeout=90.0, follow_redirects=True) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        ext = "png" if "png" in (r.headers.get("content-type") or "").lower() else "jpg"
        return r.content, ext


async def _mirror_fix_to_r2(user_id: str, upload_id: str, analysis_id: str, image_url: str) -> str:
    raw, ext = await _download_image_bytes(image_url)
    if not raw or len(raw) > 12_000_000:
        raise ValueError("invalid_image_bytes")
    r2_key = _normalize_r2_key(
        f"thumbnails/pikzels-analyzer/{user_id}/{upload_id}/{analysis_id}.{ext}"
    )
    await asyncio.to_thread(
        put_object_bytes, r2_key, raw, "image/png" if ext == "png" else "image/jpeg"
    )
    return r2_key


async def apply_fix_to_analysis(
    conn: Any,
    *,
    user_id: str,
    analysis_id: str,
    persona: Optional[str] = None,
    use_targeted_prompt: bool = True,
    re_score: bool = True,
) -> Dict[str, Any]:
    """Run Pikzels edit (one-click fix) from the stored analysis frame."""
    row = await conn.fetchrow(
        """
        SELECT a.*, u.filename, u.title AS upload_title, u.caption, u.ai_title, u.ai_caption,
               u.output_artifacts, u.trill_metadata, u.processed_r2_key, u.r2_key, u.thumbnail_r2_key
        FROM pikzels_thumbnail_analyses a
        JOIN uploads u ON u.id = a.upload_id AND u.user_id = a.user_id
        WHERE a.id = $1::uuid AND a.user_id = $2::uuid
        """,
        uuid.UUID(str(analysis_id)),
        str(user_id),
    )
    if not row:
        raise ValueError("analysis_not_found")

    upload_row = dict(row)
    subs = _json_obj(row.get("subscores_json"))
    prompt = build_targeted_fix_prompt(
        str(row.get("suggestion") or ""),
        subs if use_targeted_prompt else None,
    )
    hydrated = append_hydration_to_prompt(
        prompt,
        upload_row,
        use_hydration=True,
        hydration_lane="combined",
        max_len=950,
    )
    image_b64, _frame_meta = await load_upload_frame_jpeg_base64(
        upload_row,
        str(row.get("frame_source") or "primary_thumbnail"),
        None,
    )
    payload: Dict[str, Any] = {
        "prompt": hydrated,
        "image_base64": image_b64,
        "format": "16:9",
        "upload_id": str(row["upload_id"]),
    }
    if persona:
        payload["persona"] = persona.strip()
    status, data = await pikzels_v2_post("/v2/thumbnail/edit", payload)
    if status >= 400 or not isinstance(data, dict):
        err = data if isinstance(data, dict) else {"message": str(data)}
        raise ValueError(f"pikzels_edit_failed:{err}")

    fix_url = _pikzels_extract_image_url(data)
    if not fix_url:
        raise ValueError("pikzels_edit_no_image")

    fix_score = None
    fix_subs: Dict[str, Any] = {}
    fix_score_data: Dict[str, Any] = {}
    if re_score:
        sc_status, sc_data = await pikzels_v2_post(
            "/v2/thumbnail/score",
            {
                "image_url": fix_url,
                "title": str(row.get("title") or row.get("upload_title") or "")[:200] or None,
                "upload_id": str(row["upload_id"]),
            },
        )
        if sc_status < 400 and isinstance(sc_data, dict):
            fix_score_data = sc_data
            fix_score = _main_score_from_response(sc_data)
            raw_subs = sc_data.get("subscores")
            if isinstance(raw_subs, dict):
                fix_subs = raw_subs

    fix_r2_key = ""
    try:
        fix_r2_key = await _mirror_fix_to_r2(
            str(user_id), str(row["upload_id"]), str(analysis_id), fix_url
        )
    except Exception:
        logger.warning("pikzels analyzer fix R2 mirror failed", exc_info=True)

    updated = await conn.fetchrow(
        """
        UPDATE pikzels_thumbnail_analyses
        SET fix_image_url = $3,
            fix_r2_key = $4,
            fix_score = $5,
            fix_subscores_json = $6::jsonb,
            fix_response_json = $7::jsonb,
            recommendation_status = 'applied',
            persona_id = COALESCE($8, persona_id),
            updated_at = NOW()
        WHERE id = $1::uuid AND user_id = $2::uuid
        RETURNING *
        """,
        uuid.UUID(str(analysis_id)),
        str(user_id),
        fix_url[:2048],
        fix_r2_key or None,
        fix_score,
        json.dumps(fix_subs),
        json.dumps({"edit": data, "rescore": fix_score_data}),
        (persona or "")[:128] or None,
    )
    out = analysis_row_public(updated)
    await record_analyzer_ml_event(
        conn,
        user_id=str(user_id),
        upload_id=out["upload_id"],
        analysis_id=out["analysis_id"],
        event="pikzels_analyzer_fix_applied",
        extra={
            "main_score": out.get("main_score"),
            "fix_score": fix_score,
            "score_delta": (
                (float(fix_score) - float(out["main_score"]))
                if fix_score is not None and out.get("main_score") is not None
                else None
            ),
            "persona": persona,
        },
    )
    return out


async def save_fix_as_upload_thumbnail(
    conn: Any,
    *,
    user_id: str,
    analysis_id: str,
) -> Dict[str, Any]:
    """Write the applied fix image onto uploads.thumbnail_r2_key."""
    row = await conn.fetchrow(
        """
        SELECT a.*, u.output_artifacts
        FROM pikzels_thumbnail_analyses a
        JOIN uploads u ON u.id = a.upload_id AND u.user_id = a.user_id
        WHERE a.id = $1::uuid AND a.user_id = $2::uuid
        """,
        uuid.UUID(str(analysis_id)),
        str(user_id),
    )
    if not row:
        raise ValueError("analysis_not_found")
    fix_r2 = str(row.get("fix_r2_key") or "").strip()
    fix_url = str(row.get("fix_image_url") or "").strip()
    if not fix_r2:
        if not fix_url:
            raise ValueError("no_fix_image")
        fix_r2 = await _mirror_fix_to_r2(
            str(user_id), str(row["upload_id"]), str(analysis_id), fix_url
        )
        await conn.execute(
            """
            UPDATE pikzels_thumbnail_analyses
            SET fix_r2_key = $3, updated_at = NOW()
            WHERE id = $1::uuid AND user_id = $2::uuid
            """,
            uuid.UUID(str(analysis_id)),
            str(user_id),
            fix_r2,
        )

    artifacts = _json_obj(row.get("output_artifacts"))
    artifacts["pikzels_analyzer_fix"] = {
        "analysis_id": str(analysis_id),
        "fix_r2_key": fix_r2,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "source": "pikzels_kpi_analyzer",
    }
    await conn.execute(
        """
        UPDATE uploads
        SET thumbnail_r2_key = $3,
            content_variant_meta = COALESCE(content_variant_meta, '{}'::jsonb) || $4::jsonb,
            updated_at = NOW()
        WHERE id = $1::uuid AND user_id = $2::uuid
        """,
        row["upload_id"],
        str(user_id),
        fix_r2,
        json.dumps(
            {
                "pikzels_analyzer_analysis_id": str(analysis_id),
                "pikzels_analyzer_fix_r2_key": fix_r2,
            }
        ),
    )
    await conn.execute(
        """
        UPDATE pikzels_thumbnail_analyses
        SET recommendation_status = 'done', updated_at = NOW()
        WHERE id = $1::uuid AND user_id = $2::uuid
        """,
        uuid.UUID(str(analysis_id)),
        str(user_id),
    )
    preview = ""
    try:
        preview = generate_presigned_download_url(fix_r2, ttl=3600) or ""
    except Exception:
        preview = fix_url

    out = await get_analysis_for_user(conn, str(user_id), str(analysis_id))
    await record_analyzer_ml_event(
        conn,
        user_id=str(user_id),
        upload_id=str(row["upload_id"]),
        analysis_id=str(analysis_id),
        event="pikzels_analyzer_thumbnail_saved",
        extra={"fix_r2_key": fix_r2},
    )
    return {
        "analysis": out,
        "thumbnail_r2_key": fix_r2,
        "thumbnail_url": preview,
        "upload_id": str(row["upload_id"]),
    }


async def generate_titles_for_analysis(
    conn: Any,
    *,
    user_id: str,
    analysis_id: str,
) -> Dict[str, Any]:
    row = await conn.fetchrow(
        """
        SELECT a.*, u.caption, u.ai_caption, u.title AS upload_title, u.ai_title,
               u.output_artifacts, u.trill_metadata
        FROM pikzels_thumbnail_analyses a
        JOIN uploads u ON u.id = a.upload_id AND u.user_id = a.user_id
        WHERE a.id = $1::uuid AND a.user_id = $2::uuid
        """,
        uuid.UUID(str(analysis_id)),
        str(user_id),
    )
    if not row:
        raise ValueError("analysis_not_found")

    base = str(row.get("title") or row.get("upload_title") or row.get("ai_title") or "").strip()
    suggestion = str(row.get("suggestion") or "").strip()
    prompt = (base + ". " + suggestion).strip() or "High-retention short-form video title"
    hydrated = append_hydration_to_prompt(
        prompt[:1800],
        dict(row),
        use_hydration=True,
        hydration_lane="combined",
        max_len=1900,
    )
    image_url = str(row.get("fix_image_url") or "").strip()
    payload: Dict[str, Any] = {"prompt": hydrated[:2000]}
    if image_url:
        payload["support_image_url"] = image_url

    status, data = await pikzels_v2_post("/v2/title/text", payload)
    if status >= 400 or not isinstance(data, dict):
        raise ValueError("pikzels_titles_failed")

    updated = await conn.fetchrow(
        """
        UPDATE pikzels_thumbnail_analyses
        SET generated_titles_json = $3::jsonb, updated_at = NOW()
        WHERE id = $1::uuid AND user_id = $2::uuid
        RETURNING *
        """,
        uuid.UUID(str(analysis_id)),
        str(user_id),
        json.dumps(data),
    )
    out = analysis_row_public(updated)
    await record_analyzer_ml_event(
        conn,
        user_id=str(user_id),
        upload_id=out["upload_id"],
        analysis_id=out["analysis_id"],
        event="pikzels_analyzer_titles_generated",
        extra={"title_count": len(data.get("titles") or data.get("title") or [])},
    )
    return out


async def fetch_analyzer_summary(
    conn: Any,
    *,
    user_id: str,
    days: int = 90,
) -> Dict[str, Any]:
    """Compact rollup for coach / AI insights surfaces."""
    trend = await fetch_score_trend(conn, user_id=str(user_id), days=days, limit=60)
    pending = await conn.fetchval(
        """
        SELECT COUNT(*)::int FROM pikzels_thumbnail_analyses
        WHERE user_id = $1::uuid
          AND recommendation_status = 'saved'
          AND created_at >= NOW() - ($2::int || ' days')::interval
        """,
        str(user_id),
        max(7, min(days, 365)),
    )
    applied = await conn.fetchval(
        """
        SELECT COUNT(*)::int FROM pikzels_thumbnail_analyses
        WHERE user_id = $1::uuid
          AND recommendation_status IN ('applied', 'done')
          AND fix_score IS NOT NULL
          AND created_at >= NOW() - ($2::int || ' days')::interval
        """,
        str(user_id),
        max(7, min(days, 365)),
    )
    recs = await fetch_actionable_recommendations(conn, user_id=str(user_id), limit=5)
    return {
        "average_score": trend.get("average_score"),
        "sample_count": trend.get("sample_count"),
        "pending_saved_recommendations": int(pending or 0),
        "fixes_applied_count": int(applied or 0),
        "score_trend": trend.get("points") or [],
        "actionable_recommendations": recs.get("items") or [],
        "total_actionable": recs.get("total_actionable") or 0,
        "low_score_count": recs.get("low_score_count") or 0,
    }


LOW_SCORE_THRESHOLD = 40.0


async def fetch_actionable_recommendations(
    conn: Any,
    *,
    user_id: str,
    limit: int = 12,
    low_score_threshold: float = LOW_SCORE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Cross-surface feed of recommendations users should act on:

    - All ``saved`` recommendations (explicit to-dos), plus
    - ``open`` analyses scoring below ``low_score_threshold`` with a real suggestion.

    Deduped to the most recent analysis per upload so coach/insights/queue do not
    repeat the same video. Returns deep-link CTAs into the KPI analyzer.
    """
    rows = await conn.fetch(
        """
        SELECT a.id, a.upload_id, a.main_score, a.suggestion, a.recommendation_status,
               a.created_at, a.fix_score,
               COALESCE(u.title, u.ai_title, u.filename) AS upload_title
        FROM pikzels_thumbnail_analyses a
        LEFT JOIN uploads u ON u.id = a.upload_id
        WHERE a.user_id = $1::uuid
          AND (
            a.recommendation_status = 'saved'
            OR (
              a.recommendation_status = 'open'
              AND a.main_score IS NOT NULL
              AND a.main_score < $2
              AND NULLIF(TRIM(a.suggestion), '') IS NOT NULL
            )
          )
        ORDER BY a.created_at DESC
        """,
        str(user_id),
        float(low_score_threshold),
    )

    items: List[Dict[str, Any]] = []
    seen_uploads: set[str] = set()
    pending_saved = 0
    low_score = 0
    for r in rows:
        up = str(r["upload_id"])
        status = str(r["recommendation_status"] or "open")
        if status == "saved":
            pending_saved += 1
        else:
            low_score += 1
        if up in seen_uploads:
            continue
        seen_uploads.add(up)
        if len(items) >= max(1, min(int(limit), 50)):
            continue
        aid = str(r["id"])
        items.append(
            {
                "analysis_id": aid,
                "upload_id": up,
                "upload_title": str(r["upload_title"] or "") or None,
                "main_score": float(r["main_score"]) if r["main_score"] is not None else None,
                "fix_score": float(r["fix_score"]) if r["fix_score"] is not None else None,
                "suggestion": str(r["suggestion"] or ""),
                "status": status,
                "reason": "saved" if status == "saved" else "low_score",
                "created_at": r["created_at"].isoformat() if r["created_at"] else "",
                "cta_href": f"kpi.html#pkz-analyzer&analysis={aid}",
            }
        )
    return {
        "items": items,
        "pending_saved_count": pending_saved,
        "low_score_count": low_score,
        "total_actionable": pending_saved + low_score,
    }


async def fetch_score_trend(
    conn: Any,
    *,
    user_id: str,
    days: int = 90,
    limit: int = 60,
) -> Dict[str, Any]:
    since = datetime.now(timezone.utc) - timedelta(days=max(7, min(days, 365)))
    rows = await conn.fetch(
        """
        SELECT created_at, main_score, fix_score, upload_id, recommendation_status
        FROM pikzels_thumbnail_analyses
        WHERE user_id = $1::uuid
          AND created_at >= $2
          AND main_score IS NOT NULL
        ORDER BY created_at ASC
        LIMIT $3
        """,
        str(user_id),
        since,
        max(1, min(limit, 200)),
    )
    points = []
    scores: List[float] = []
    for r in rows:
        ms = float(r["main_score"])
        scores.append(ms)
        points.append(
            {
                "at": r["created_at"].isoformat() if r["created_at"] else "",
                "main_score": ms,
                "fix_score": float(r["fix_score"]) if r["fix_score"] is not None else None,
                "upload_id": str(r["upload_id"]),
                "status": str(r["recommendation_status"] or ""),
            }
        )
    avg = round(sum(scores) / len(scores), 1) if scores else None
    return {
        "points": points,
        "average_score": avg,
        "sample_count": len(points),
        "since": since.isoformat(),
    }


def _frame_source_for_upload(upload_row: Dict[str, Any]) -> str:
    if upload_row.get("thumbnail_r2_key"):
        return "primary_thumbnail"
    return "video_best"


async def score_upload_thumbnail(
    conn: Any,
    *,
    user_id: str,
    upload_row: Dict[str, Any],
    frame_source: Optional[str] = None,
    title: Optional[str] = None,
    persona_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Score one upload thumbnail via Pikzels and persist the analysis row."""
    fs = (frame_source or _frame_source_for_upload(upload_row)).strip() or "video_best"
    sc_title = (title or "").strip() or str(
        upload_row.get("title") or upload_row.get("ai_title") or ""
    )[:200]
    image_b64, _frame_meta = await load_upload_frame_jpeg_base64(upload_row, fs, None)
    payload: Dict[str, Any] = {
        "image_base64": image_b64,
        "upload_id": str(upload_row["id"]),
    }
    if sc_title:
        payload["title"] = sc_title
    status, data = await pikzels_v2_post("/v2/thumbnail/score", payload)
    if status >= 400 or not isinstance(data, dict):
        err = data if isinstance(data, dict) else {"message": str(data)}
        raise ValueError(f"pikzels_score_failed:{err}")
    return await persist_score_analysis(
        conn,
        user_id=str(user_id),
        upload_id=str(upload_row["id"]),
        frame_source=fs,
        title=sc_title,
        response_data=data,
        persona_id=persona_id,
    )


async def batch_score_user_uploads(
    conn: Any,
    *,
    user_id: str,
    limit: int = 15,
    persona_id: Optional[str] = None,
    rescore_recent: bool = False,
    on_before_score: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """
    Score up to ``limit`` recent uploads and return them ranked weakest-first.

    ``on_before_score`` is awaited before each Pikzels call (e.g. wallet debit).
    Skips uploads with no scorable frame unless ``rescore_recent`` is False and a
    score exists from the last 24 hours.
    """
    cap = max(1, min(int(limit), 30))
    rows = await conn.fetch(
        """
        SELECT id, user_id, filename, title, caption, ai_title, ai_caption,
               output_artifacts, trill_metadata, created_at, updated_at,
               processed_r2_key, r2_key, thumbnail_r2_key
        FROM uploads
        WHERE user_id = $1::uuid
          AND (thumbnail_r2_key IS NOT NULL OR processed_r2_key IS NOT NULL OR r2_key IS NOT NULL)
        ORDER BY created_at DESC
        LIMIT $2
        """,
        str(user_id),
        cap,
    )
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    ranked: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for row in rows:
        upload_row = dict(row)
        up_id = str(upload_row["id"])
        upload_title = str(
            upload_row.get("title") or upload_row.get("ai_title") or upload_row.get("filename") or ""
        ).strip()

        if not rescore_recent:
            recent = await conn.fetchrow(
                """
                SELECT id, main_score, created_at
                FROM pikzels_thumbnail_analyses
                WHERE upload_id = $1::uuid AND user_id = $2::uuid
                  AND main_score IS NOT NULL AND created_at >= $3
                ORDER BY created_at DESC
                LIMIT 1
                """,
                uuid.UUID(up_id),
                str(user_id),
                since,
            )
            if recent:
                analysis = analysis_row_public(recent)
                analysis["upload_title"] = upload_title or None
                skipped.append(
                    {
                        "upload_id": up_id,
                        "upload_title": upload_title or None,
                        "reason": "recent_score",
                        "analysis_id": analysis.get("analysis_id"),
                        "main_score": analysis.get("main_score"),
                    }
                )
                ranked.append(analysis)
                continue

        try:
            if on_before_score:
                maybe = on_before_score()
                if asyncio.iscoroutine(maybe):
                    await maybe
            analysis = await score_upload_thumbnail(
                conn,
                user_id=str(user_id),
                upload_row=upload_row,
                persona_id=persona_id,
            )
            analysis["upload_title"] = upload_title or analysis.get("upload_title")
            ranked.append(analysis)
        except ValueError as e:
            errors.append({"upload_id": up_id, "upload_title": upload_title or None, "error": str(e)})
        except Exception as e:
            logger.warning("batch score failed upload=%s: %s", up_id, e)
            errors.append({"upload_id": up_id, "upload_title": upload_title or None, "error": str(e)})

    def _score_key(item: Dict[str, Any]) -> float:
        try:
            return float(item.get("main_score"))
        except (TypeError, ValueError):
            return 9999.0

    ranked.sort(key=_score_key)
    scores = [
        float(x["main_score"])
        for x in ranked
        if x.get("main_score") is not None
    ]
    return {
        "ranked": ranked,
        "skipped": skipped,
        "errors": errors,
        "scored_count": len(ranked) - len(skipped),
        "skipped_count": len(skipped),
        "error_count": len(errors),
        "weakest_score": min(scores) if scores else None,
        "average_score": round(sum(scores) / len(scores), 1) if scores else None,
    }


def build_pikzels_analyzer_ab_export_zip(analysis: Dict[str, Any]) -> bytes:
    """ZIP pack for ML / A/B workflows: before/after scores + metadata."""
    buf = io.BytesIO()
    aid = str(analysis.get("analysis_id") or "")
    main = analysis.get("main_score")
    fix = analysis.get("fix_score")
    delta = None
    if main is not None and fix is not None:
        try:
            delta = round(float(fix) - float(main), 1)
        except (TypeError, ValueError):
            delta = None
    meta = {
        "analysis_id": aid,
        "upload_id": analysis.get("upload_id"),
        "upload_title": analysis.get("upload_title") or analysis.get("title"),
        "main_score": main,
        "fix_score": fix,
        "score_delta": delta,
        "recommendation_status": analysis.get("recommendation_status"),
        "suggestion": analysis.get("suggestion"),
        "subscores": analysis.get("subscores") or {},
        "fix_subscores": analysis.get("fix_subscores") or {},
        "frame_source": analysis.get("frame_source"),
        "persona_id": analysis.get("persona_id"),
        "fix_preview_url": analysis.get("fix_preview_url") or analysis.get("fix_image_url"),
        "generated_titles": analysis.get("generated_titles"),
        "created_at": analysis.get("created_at"),
        "updated_at": analysis.get("updated_at"),
    }
    before_after = {
        "before": {"main_score": main, "subscores": analysis.get("subscores") or {}},
        "after": {
            "fix_score": fix,
            "subscores": analysis.get("fix_subscores") or {},
            "fix_preview_url": meta["fix_preview_url"],
        },
        "delta": delta,
    }
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("analysis.json", json.dumps(meta, indent=2, default=str))
        zf.writestr("before_after.json", json.dumps(before_after, indent=2, default=str))
        sio = io.StringIO()
        w = csv.writer(sio)
        w.writerow(
            [
                "analysis_id",
                "upload_id",
                "upload_title",
                "main_score",
                "fix_score",
                "score_delta",
                "status",
                "suggestion",
                "fix_preview_url",
            ]
        )
        w.writerow(
            [
                aid,
                analysis.get("upload_id"),
                meta["upload_title"],
                main,
                fix,
                delta,
                analysis.get("recommendation_status"),
                (str(analysis.get("suggestion") or ""))[:500],
                meta["fix_preview_url"],
            ]
        )
        zf.writestr("summary.csv", sio.getvalue())
        zf.writestr(
            "README.txt",
            "UploadM8 Pikzels Analyzer — A/B comparison pack\n"
            "================================================\n\n"
            "analysis.json — full analysis metadata (scores, suggestion, titles).\n"
            "before_after.json — structured before/after score comparison.\n"
            "summary.csv — one-row spreadsheet summary for ML pipelines.\n\n"
            "Preview URLs in this pack are signed links (about 1 hour). Re-export from\n"
            "the KPI analyzer if you need fresh links.\n",
        )
    return buf.getvalue()


async def export_analysis_ab_pack(
    conn: Any,
    *,
    user_id: str,
    analysis_id: str,
) -> Dict[str, Any]:
    """Build analyzer A/B ZIP, upload to R2, return signed download URL."""
    if not (R2_BUCKET_NAME or "").strip():
        raise ValueError("r2_not_configured")
    row = await conn.fetchrow(
        """
        SELECT a.*, COALESCE(u.title, u.ai_title, u.filename) AS upload_title
        FROM pikzels_thumbnail_analyses a
        LEFT JOIN uploads u ON u.id = a.upload_id
        WHERE a.id = $1::uuid AND a.user_id = $2::uuid
        """,
        uuid.UUID(str(analysis_id)),
        str(user_id),
    )
    if not row:
        raise ValueError("analysis_not_found")
    analysis = analysis_row_public(row)
    zip_bytes = build_pikzels_analyzer_ab_export_zip(analysis)
    r2_key = _normalize_r2_key(
        f"thumbnail-studio/ab-packs/{user_id}/pikzels-analyzer/{analysis_id}.zip"
    )
    await asyncio.to_thread(put_object_bytes, r2_key, zip_bytes, "application/zip")
    download_url = generate_presigned_download_url(r2_key, ttl=3600) or ""
    fname = f"pikzels_ab_{str(analysis_id)[:8]}.zip"
    return {
        "exports": [
            {
                "label": "Pikzels analyzer comparison (ZIP)",
                "filename": fname,
                "download_url": download_url,
                "r2_key": r2_key,
            }
        ],
        "download_url": download_url,
        "filename": fname,
        "analysis_id": analysis_id,
        "note": "Signed download link expires in about 1 hour. Contains JSON + CSV for ML / A/B workflows.",
    }
