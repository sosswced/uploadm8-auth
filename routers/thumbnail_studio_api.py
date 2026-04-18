"""
Thumbnail Studio HTTP API — DB-backed jobs, personas, and Pikzels v2 proxy.

``routers/thumbnail_studio_routes.register_thumbnail_studio_routes`` only
includes this module's router; it does not reference ``app.api_*``.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

import core.state
from core.config import R2_BUCKET_NAME
from core.deps import get_current_user, get_current_user_readonly
from core.helpers import _now_utc
from core.r2 import generate_presigned_download_url
from core.wallet import atomic_debit_tokens
from api.schemas.pikzels_v2 import (
    PikzelsV2EditBody,
    PikzelsV2FaceswapBody,
    PikzelsV2PikzonalityBody,
    PikzelsV2PromptBody,
    PikzelsV2RecreateBody,
    PikzelsV2ScoreBody,
    PikzelsV2TitlesBody,
)
from services.pikzels_v2 import PIKZELS_FEATURE_MAP, V2_PIKZONALITY_BY_ID, resolve_public_api_key
from services.pikzels_v2_client import (
    normalize_url_or_base64,
    pikzels_v2_get,
    pikzels_v2_post,
    trim_pikzonality_images,
)
from services.thumbnail_studio import (
    _pikzels_extract_image_url,
    attach_preview_urls_to_variants,
    build_thumbnail_ab_export_zip,
    enrich_variants_with_uploadm8_engine,
    estimate_pikzels_v2_call_cost,
    estimate_studio_cost,
    extract_youtube_video_id,
    fetch_youtube_title,
    generate_recreate_variants,
    register_creator_persona_with_pikzels,
    upload_ab_export_zip_to_r2,
)
from services.growth_intelligence import m8_engine_identity_payload, record_studio_usage_event
from services.ml_marketing import record_outcome_label, record_thumbnail_studio_engine_ml_batch
from services.wallet_marketing import _user_campaign_features

logger = logging.getLogger("uploadm8-api")

router = APIRouter(tags=["thumbnail-studio"])


def _studio_job_public(row: Any) -> Dict[str, Any]:
    """Serialize a thumbnail_recreate_jobs row for API clients (no internal-only fields)."""
    d = dict(row)
    jid = d.get("id")
    ca = d.get("created_at")
    pid = d.get("persona_id")
    return {
        "job_id": str(jid) if jid else "",
        "youtube_url": str(d.get("youtube_url") or ""),
        "youtube_video_id": str(d.get("youtube_video_id") or ""),
        "source_title": str(d.get("source_title") or ""),
        "topic": str(d.get("topic") or ""),
        "niche": str(d.get("niche") or "general"),
        "closeness": int(d.get("closeness") or 55),
        "variant_count": int(d.get("variant_count") or 6),
        "competitor_gap_mode": bool(d.get("competitor_gap_mode")),
        "put_cost": int(d.get("put_cost") or 0),
        "aic_cost": int(d.get("aic_cost") or 0),
        "persona_id": str(pid) if pid else None,
        "created_at": ca.isoformat() if hasattr(ca, "isoformat") else str(ca or ""),
    }


@router.get("/api/thumbnail-studio/cdn-preview")
async def thumbnail_studio_cdn_preview(
    variant_id: str = Query(..., min_length=30, max_length=48),
    user: dict = Depends(get_current_user_readonly),
):
    """
    Fetch a Pikzels CDN image for **this user's** stored variant only (variant_json
    must contain ``pikzels_cdn_url`` or an extractable CDN URL). Uses the server
    API key — browsers cannot send ``X-Api-Key`` to cdn.pikzels.com.
    """
    try:
        vid = uuid.UUID(variant_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid variant_id")

    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT variant_json FROM thumbnail_recreate_variants
            WHERE id = $1 AND user_id = $2
            """,
            vid,
            user["id"],
        )
    if not row:
        raise HTTPException(status_code=404, detail="variant not found")

    j = row["variant_json"]
    if isinstance(j, str):
        try:
            j = json.loads(j)
        except Exception:
            j = {}
    if not isinstance(j, dict):
        j = {}

    raw = str(j.get("pikzels_cdn_url") or "").strip()
    if not raw.startswith("https://"):
        raw = _pikzels_extract_image_url(j)
    raw = (raw or "").strip()
    if not raw.startswith("https://"):
        raise HTTPException(status_code=404, detail="no cdn preview for this variant")

    host = (urlparse(raw).hostname or "").lower()
    if host != "cdn.pikzels.com":
        raise HTTPException(status_code=400, detail="host not allowed")

    key = resolve_public_api_key()
    if not key:
        raise HTTPException(status_code=503, detail="pikzels not configured")
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            r = await client.get(raw, headers={"X-Api-Key": key})
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    if r.status_code != 200 or not r.content:
        raise HTTPException(status_code=502, detail="upstream image failed")
    ct = (r.headers.get("content-type") or "").split(";")[0].strip() or "image/jpeg"
    if not ct.startswith("image/"):
        ct = "image/jpeg"
    return Response(content=r.content, media_type=ct)


class StudioEstimateBody(BaseModel):
    variant_count: int = 6
    has_persona: bool = False
    competitor_gap_mode: bool = False
    has_channel_memory: bool = True


class StudioRecreateBody(BaseModel):
    youtube_url: str = Field(min_length=4, max_length=2048)
    topic: str = ""
    niche: str = "general"
    closeness: int = Field(default=55, ge=0, le=100)
    variant_count: int = Field(default=6, ge=4, le=8)
    persona_id: Optional[str] = None
    format_key: Optional[str] = None
    competitor_gap_mode: bool = False


class StudioPersonaCreateBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    image_urls: List[str] = Field(default_factory=list, min_length=3, max_length=20)


class StudioFeedbackBody(BaseModel):
    job_id: str
    variant_id: Optional[str] = None
    upload_id: Optional[str] = None
    event_type: str = "selected"
    metadata: Dict[str, Any] = Field(default_factory=dict)


@router.post("/api/thumbnail-studio/estimate")
async def ts_estimate(body: StudioEstimateBody, user: dict = Depends(get_current_user)):
    put, aic, breakdown = estimate_studio_cost(
        variant_count=body.variant_count,
        has_persona=body.has_persona,
        competitor_gap_mode=body.competitor_gap_mode,
        has_channel_memory=body.has_channel_memory,
    )
    return {"put_cost": put, "aic_cost": aic, "breakdown": breakdown}


@router.get("/api/thumbnail-studio/personas")
async def ts_list_personas(user: dict = Depends(get_current_user)):
    async with core.state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, name, image_count, created_at, profile_json
            FROM creator_personas
            WHERE user_id = $1
            ORDER BY created_at DESC
            """,
            user["id"],
        )
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
        out_rows.append(
            {
                "id": str(r["id"]),
                "name": r["name"],
                "image_count": int(r["image_count"] or 0),
                "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
                "pikzels_linked": bool(pkz),
            }
        )
    return {"personas": out_rows}


@router.post("/api/thumbnail-studio/personas")
async def ts_create_persona(body: StudioPersonaCreateBody, user: dict = Depends(get_current_user)):
    pid = uuid.uuid4()
    async with core.state.db_pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO creator_personas (id, user_id, name, profile_json, image_count)
                VALUES ($1, $2, $3, $4::jsonb, $5)
                """,
                pid,
                user["id"],
                body.name.strip(),
                json.dumps({"source": "thumbnail_studio_ui"}),
                len(body.image_urls),
            )
            for url in body.image_urls:
                await conn.execute(
                    """
                    INSERT INTO creator_persona_images (persona_id, user_id, image_url, quality_json)
                    VALUES ($1, $2, $3, '{}'::jsonb)
                    """,
                    pid,
                    user["id"],
                    url[:8000],
                )

    resp: Dict[str, Any] = {
        "id": str(pid),
        "status": "created",
        "pikzels_pikzonality_linked": False,
    }
    pkz_id, pkz_err = await register_creator_persona_with_pikzels(
        name=body.name.strip(),
        image_refs=list(body.image_urls or []),
    )
    if pkz_id:
        merge = {
            "pikzels_pikzonality_id": pkz_id,
            "pikzels_linked_at": _now_utc().isoformat(),
        }
        async with core.state.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE creator_personas
                SET profile_json = COALESCE(profile_json, '{}'::jsonb) || $2::jsonb
                WHERE id = $1 AND user_id = $3
                """,
                pid,
                json.dumps(merge),
                user["id"],
            )
        resp["pikzels_pikzonality_linked"] = True
        resp["pikzels_pikzonality_id"] = pkz_id
    elif pkz_err:
        resp["pikzels_warning"] = pkz_err
    return resp


@router.post("/api/thumbnail-studio/personas/{persona_id}/link-pikzels")
async def ts_link_persona_pikzels(persona_id: str, user: dict = Depends(get_current_user)):
    """
    For personas created before Pikzels registration ran (or when linking failed),
    register again using the saved ``creator_persona_images`` rows (needs ≥3 photos).
    """
    try:
        pid = uuid.UUID(str(persona_id).strip())
    except ValueError:
        raise HTTPException(400, "Invalid persona_id")
    if not resolve_public_api_key():
        raise HTTPException(503, "Pikzels not configured on the server")

    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT name, profile_json FROM creator_personas
            WHERE id = $1 AND user_id = $2
            """,
            pid,
            user["id"],
        )
    if not row:
        raise HTTPException(404, "Persona not found")

    name = str(row["name"] or "").strip() or "Persona"
    prof = row.get("profile_json")
    if isinstance(prof, str):
        try:
            prof = json.loads(prof)
        except Exception:
            prof = {}
    if not isinstance(prof, dict):
        prof = {}
    existing = str(prof.get("pikzels_pikzonality_id") or "").strip()
    if existing:
        return {
            "pikzels_linked": True,
            "pikzels_pikzonality_id": existing,
            "already_linked": True,
        }

    async with core.state.db_pool.acquire() as conn:
        img_rows = await conn.fetch(
            """
            SELECT image_url FROM creator_persona_images
            WHERE persona_id = $1 AND user_id = $2
            ORDER BY id ASC
            """,
            pid,
            user["id"],
        )
    refs = [str(r["image_url"]) for r in img_rows if r.get("image_url")]
    if len(refs) < 3:
        raise HTTPException(
            400,
            "This persona needs at least three saved reference photos in your account. "
            "Add photos and use Save Persona, or pick another persona.",
        )

    pkz_id, pkz_err = await register_creator_persona_with_pikzels(
        name=name,
        image_refs=refs,
    )
    if pkz_id:
        merge = {
            "pikzels_pikzonality_id": pkz_id,
            "pikzels_linked_at": _now_utc().isoformat(),
        }
        async with core.state.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE creator_personas
                SET profile_json = COALESCE(profile_json, '{}'::jsonb) || $2::jsonb
                WHERE id = $1 AND user_id = $3
                """,
                pid,
                json.dumps(merge),
                user["id"],
            )
        return {
            "pikzels_linked": True,
            "pikzels_pikzonality_id": pkz_id,
            "already_linked": False,
        }

    return {
        "pikzels_linked": False,
        "pikzels_pikzonality_id": None,
        "pikzels_warning": pkz_err or "pikzels_registration_failed",
        "already_linked": False,
    }


@router.post("/api/thumbnail-studio/recreate")
async def ts_recreate(body: StudioRecreateBody, user: dict = Depends(get_current_user)):
    title = await fetch_youtube_title(body.youtube_url)
    vid = extract_youtube_video_id(body.youtube_url)
    persona_name = ""
    pikzels_persona_pid: str = ""
    persona_uuid: Optional[uuid.UUID] = None
    if body.persona_id:
        try:
            persona_uuid = uuid.UUID(str(body.persona_id))
        except ValueError:
            raise HTTPException(400, "Invalid persona_id")
        async with core.state.db_pool.acquire() as conn:
            prow = await conn.fetchrow(
                "SELECT name, profile_json FROM creator_personas WHERE id = $1 AND user_id = $2",
                persona_uuid,
                user["id"],
            )
        if not prow:
            raise HTTPException(404, "Persona not found")
        persona_name = str(prow["name"] or "")
        prof = prow.get("profile_json")
        if isinstance(prof, str):
            try:
                prof = json.loads(prof)
            except Exception:
                prof = {}
        if isinstance(prof, dict):
            pikzels_persona_pid = str(prof.get("pikzels_pikzonality_id") or "").strip()

    put, aic, breakdown = estimate_studio_cost(
        variant_count=body.variant_count,
        has_persona=bool(persona_uuid),
        competitor_gap_mode=body.competitor_gap_mode,
        has_channel_memory=True,
    )

    job_id = uuid.uuid4()
    variants_out: List[Dict[str, Any]] = []

    async with core.state.db_pool.acquire() as conn:
        async with conn.transaction():
            ok = await atomic_debit_tokens(
                conn,
                str(user["id"]),
                put,
                aic,
                str(job_id),
                reason="thumbnail_studio_recreate",
            )
            if not ok:
                raise HTTPException(
                    429,
                    {
                        "code": "insufficient_tokens",
                        "message": f"Need {put} PUT and {aic} AIC for this run.",
                        "topup_url": "/settings.html#billing",
                    },
                )

            await conn.execute(
                """
                INSERT INTO thumbnail_recreate_jobs (
                    id, user_id, youtube_url, youtube_video_id, source_title, topic, niche,
                    closeness, variant_count, persona_id, competitor_gap_mode,
                    put_cost, aic_cost, breakdown_json
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb)
                """,
                job_id,
                user["id"],
                body.youtube_url.strip()[:2048],
                vid or None,
                (title or "")[:512],
                (body.topic or "")[:512],
                (body.niche or "general")[:120],
                int(body.closeness),
                int(body.variant_count),
                persona_uuid,
                body.competitor_gap_mode,
                put,
                aic,
                json.dumps(breakdown),
            )

    raw_variants = generate_recreate_variants(
        youtube_title=title or body.topic or "Video",
        topic=body.topic or "",
        niche=body.niche or "general",
        closeness=body.closeness,
        variant_count=body.variant_count,
        persona_name=persona_name,
        competitor_gap_mode=body.competitor_gap_mode,
        channel_memory_hint="",
        format_key=body.format_key,
    )

    raw_variants = await enrich_variants_with_uploadm8_engine(
        raw_variants,
        youtube_video_id=vid or "",
        source_title=title or body.topic or "",
        niche=body.niche or "general",
        topic=body.topic or "",
        persona_name=persona_name,
        user_id=str(user["id"]),
        job_id=str(job_id),
        pikzels_persona_pikzonality_id=pikzels_persona_pid or None,
    )

    async with core.state.db_pool.acquire() as conn:
        async with conn.transaction():
            for v in raw_variants:
                vid_row = uuid.uuid4()
                await conn.execute(
                    """
                    INSERT INTO thumbnail_recreate_variants (id, job_id, user_id, rank_idx, variant_json)
                    VALUES ($1, $2, $3, $4, $5::jsonb)
                    """,
                    vid_row,
                    job_id,
                    user["id"],
                    int(v.get("index") or 1),
                    json.dumps(v),
                )
                vo = dict(v)
                vo["variant_id"] = str(vid_row)
                variants_out.append(vo)

    attach_preview_urls_to_variants(variants_out, ttl=3600)

    engine_mode = "uploadm8_heuristic"
    if vid and resolve_public_api_key():
        engine_mode = "uploadm8_pikzels_v2_r2"
    elif vid:
        engine_mode = "uploadm8_heuristic_youtube_ref_only"

    m8_engine = m8_engine_identity_payload()
    try:
        async with core.state.db_pool.acquire() as conn:
            meta = {
                "job_id": str(job_id),
                "engine_mode": engine_mode,
                "variant_count": len(variants_out),
                "engine_ok_count": sum(
                    1 for v in variants_out if isinstance(v, dict) and v.get("engine_status") == "ok"
                ),
                "pikzels_configured": bool(resolve_public_api_key()),
                "m8_engine_ai_slug": m8_engine.get("ai_slug"),
            }
            await record_studio_usage_event(
                conn, str(user["id"]), "thumbnail_studio_recreate_job", 200, meta
            )
            await record_thumbnail_studio_engine_ml_batch(
                conn,
                user_id=str(user["id"]),
                job_id=str(job_id),
                engine_mode=engine_mode,
                variants=variants_out,
                youtube_video_id=vid or None,
            )
    except Exception:
        logger.debug("thumbnail studio engine telemetry failed", exc_info=True)

    return {
        "job_id": str(job_id),
        "variants": variants_out,
        "put_cost": put,
        "aic_cost": aic,
        "engine_mode": engine_mode,
        "m8_engine": m8_engine,
    }


@router.get("/api/thumbnail-studio/jobs")
async def ts_list_jobs(
    limit: int = Query(30, ge=1, le=100),
    offset: int = Query(0, ge=0, le=5000),
    user: dict = Depends(get_current_user_readonly),
):
    """
    Past Thumbnail Studio runs for this account (newest first).
    Use ``GET /api/thumbnail-studio/jobs/{job_id}`` to load variants + refresh preview URLs.
    """
    async with core.state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, created_at, youtube_url, youtube_video_id, source_title, topic, niche,
                   closeness, variant_count, persona_id, competitor_gap_mode, put_cost, aic_cost
            FROM thumbnail_recreate_jobs
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
            """,
            user["id"],
            limit,
            offset,
        )
    return {
        "jobs": [_studio_job_public(r) for r in rows],
        "limit": limit,
        "offset": offset,
    }


@router.get("/api/thumbnail-studio/jobs/{job_id}")
async def ts_get_job(job_id: str, user: dict = Depends(get_current_user)):
    try:
        jid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(400, "Invalid job id")
    async with core.state.db_pool.acquire() as conn:
        job = await conn.fetchrow(
            "SELECT * FROM thumbnail_recreate_jobs WHERE id = $1 AND user_id = $2",
            jid,
            user["id"],
        )
        if not job:
            raise HTTPException(404, "Job not found")
        vrows = await conn.fetch(
            """
            SELECT id, rank_idx, variant_json, selected
            FROM thumbnail_recreate_variants
            WHERE job_id = $1 AND user_id = $2
            ORDER BY rank_idx ASC
            """,
            jid,
            user["id"],
        )
    variants: List[Dict[str, Any]] = []
    for r in vrows:
        j = r["variant_json"]
        if isinstance(j, str):
            try:
                j = json.loads(j)
            except Exception:
                j = {}
        if not isinstance(j, dict):
            j = {}
        j = dict(j)
        j["variant_id"] = str(r["id"])
        variants.append(j)
    attach_preview_urls_to_variants(variants, ttl=3600)
    job_meta = _studio_job_public(job)
    return {"job": job_meta, "job_id": str(job["id"]), "variants": variants}


@router.post("/api/thumbnail-studio/feedback")
async def ts_feedback(body: StudioFeedbackBody, user: dict = Depends(get_current_user)):
    try:
        jid = uuid.UUID(body.job_id)
    except ValueError:
        raise HTTPException(400, "Invalid job id")
    var_uuid = None
    if body.variant_id:
        try:
            var_uuid = uuid.UUID(body.variant_id)
        except ValueError:
            raise HTTPException(400, "Invalid variant_id")
    async with core.state.db_pool.acquire() as conn:
        job = await conn.fetchrow(
            "SELECT 1 FROM thumbnail_recreate_jobs WHERE id = $1 AND user_id = $2",
            jid,
            user["id"],
        )
        if not job:
            raise HTTPException(404, "Job not found")
        await conn.execute(
            """
            INSERT INTO thumbnail_recreate_feedback (user_id, job_id, variant_id, event_type, metadata)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            """,
            user["id"],
            jid,
            var_uuid,
            (body.event_type or "event")[:64],
            json.dumps(body.metadata or {}),
        )
        if var_uuid:
            await conn.execute(
                """
                UPDATE thumbnail_recreate_variants SET selected = TRUE
                WHERE id = $1 AND job_id = $2 AND user_id = $3
                """,
                var_uuid,
                jid,
                user["id"],
            )
        vid_str = str(var_uuid) if var_uuid else (body.variant_id or "")[:128]
        upload_uid = None
        if body.upload_id:
            try:
                upload_uid = uuid.UUID(body.upload_id)
            except ValueError:
                upload_uid = None
        if (body.event_type or "").lower() == "selected":
            feats = await _user_campaign_features(conn, str(user["id"]), "30d")
            await record_outcome_label(
                conn,
                user_id=str(user["id"]),
                upload_id=str(upload_uid) if upload_uid else None,
                variant_id=vid_str or None,
                feature_snapshot=dict(feats),
                label_json={
                    "event": "selected",
                    "selected_variant": bool(var_uuid or body.variant_id),
                    "job_id": str(jid),
                    "metadata": body.metadata or {},
                },
            )
        if upload_uid and vid_str and (body.event_type or "").lower() == "selected":
            try:
                await conn.execute(
                    """
                    UPDATE uploads SET studio_content_variant_id = $2,
                        content_variant_meta = content_variant_meta || $3::jsonb,
                        updated_at = NOW()
                    WHERE id = $1::uuid AND user_id = $4::uuid
                    """,
                    upload_uid,
                    vid_str[:128],
                    json.dumps({"thumbnail_feedback_job": str(jid)}),
                    user["id"],
                )
            except Exception:
                pass
    return {"status": "ok"}


@router.get("/api/thumbnail-studio/ab-export/{job_id}")
async def ts_ab_export(job_id: str, user: dict = Depends(get_current_user)):
    try:
        jid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(400, "Invalid job id")
    if not (R2_BUCKET_NAME or "").strip():
        raise HTTPException(503, "Object storage is not configured; cannot build export pack.")
    async with core.state.db_pool.acquire() as conn:
        job = await conn.fetchrow(
            "SELECT * FROM thumbnail_recreate_jobs WHERE id = $1 AND user_id = $2",
            jid,
            user["id"],
        )
        if not job:
            raise HTTPException(404, "Job not found")
        vrows = await conn.fetch(
            """
            SELECT id, rank_idx, variant_json
            FROM thumbnail_recreate_variants
            WHERE job_id = $1 AND user_id = $2
            ORDER BY rank_idx ASC
            """,
            jid,
            user["id"],
        )
    if not vrows:
        raise HTTPException(404, "No variants for job")
    variants: List[Dict[str, Any]] = []
    for r in vrows:
        j = r["variant_json"]
        if isinstance(j, str):
            try:
                j = json.loads(j)
            except Exception:
                j = {}
        if not isinstance(j, dict):
            j = {}
        j = dict(j)
        j["variant_id"] = str(r["id"])
        variants.append(j)

    job_dict = {k: job[k] for k in job.keys()}
    zip_bytes = build_thumbnail_ab_export_zip(job_id=str(jid), job_row=job_dict, variants=variants)
    try:
        r2_key = await upload_ab_export_zip_to_r2(zip_bytes, str(user["id"]), str(jid))
        download_url = generate_presigned_download_url(r2_key, ttl=3600)
    except Exception as e:
        logger.warning("thumbnail ab-export R2 failed job=%s: %s", job_id, e)
        raise HTTPException(503, "Could not upload comparison pack to storage.") from e

    fname = f"thumbnail_ab_{str(jid)[:8]}.zip"
    return {
        "exports": [
            {
                "label": "Comparison pack (ZIP)",
                "filename": fname,
                "download_url": download_url,
                "r2_key": r2_key,
            }
        ],
        "download_url": download_url,
        "filename": fname,
        "note": "Signed download link expires in about 1 hour. Contains JSON + CSV summaries for ML / A/B workflows.",
    }


@router.get("/api/thumbnail-studio/weekly-digest")
async def ts_weekly_digest(user: dict = Depends(get_current_user)):
    return {
        "generated_at": _now_utc().isoformat(),
        "lines": [],
        "note": "Weekly email digest is sent separately; this endpoint is a UI placeholder.",
    }


@router.get("/api/thumbnail-studio/pikzels-v2-map")
async def ts_pikzels_map(user: dict = Depends(get_current_user)):
    out = []
    for name, key, path, desc in PIKZELS_FEATURE_MAP:
        out.append({"feature": name, "key": key, "path": path, "description": desc})
    return {"features": out}


def _pikzels_telemetry_meta(body: Any) -> Dict[str, Any]:
    try:
        raw = body.model_dump(exclude_none=True) if hasattr(body, "model_dump") else {}
    except Exception:
        raw = {}
    out: Dict[str, Any] = {}
    if raw.get("studio_variant_id"):
        out["studio_variant_id"] = str(raw["studio_variant_id"])[:128]
    if raw.get("upload_id"):
        out["upload_id"] = str(raw["upload_id"])[:64]
    return out


async def _studio_usage_log(user_id, operation: str, http_status: int, meta: Optional[Dict[str, Any]] = None) -> None:
    try:
        async with core.state.db_pool.acquire() as conn:
            await record_studio_usage_event(conn, user_id, operation, int(http_status), meta or {})
    except Exception:
        pass


async def _pikzels_debit(user_id: str, op: str) -> str:
    put, aic, meta = estimate_pikzels_v2_call_cost(op)
    ref = f"pikzels:{uuid.uuid4()}"
    async with core.state.db_pool.acquire() as conn:
        ok = await atomic_debit_tokens(
            conn, str(user_id), put, aic, ref, reason=f"pikzels_v2_{op}"
        )
    if not ok:
        raise HTTPException(
            429,
            {
                "code": "insufficient_tokens",
                "message": f"Pikzels {op} needs {put} PUT and {aic} AIC.",
                "meta": meta,
            },
        )
    return ref


@router.post("/api/thumbnail-studio/pikzels-v2/prompt")
async def ts_pikzels_prompt(body: PikzelsV2PromptBody, user: dict = Depends(get_current_user)):
    await _pikzels_debit(user["id"], "prompt")
    payload = body.model_dump(exclude_none=True)
    normalize_url_or_base64(payload, "support_image_url", "support_image_base64")
    status, data = await pikzels_v2_post("/v2/thumbnail/text", payload)
    await _studio_usage_log(user["id"], "prompt", status, _pikzels_telemetry_meta(body))
    return data if status < 400 else data | {"http_status": status}


@router.post("/api/thumbnail-studio/pikzels-v2/recreate")
async def ts_pikzels_recreate(body: PikzelsV2RecreateBody, user: dict = Depends(get_current_user)):
    await _pikzels_debit(user["id"], "recreate")
    payload = body.model_dump(exclude_none=True)
    normalize_url_or_base64(payload, "image_url", "image_base64")
    normalize_url_or_base64(payload, "support_image_url", "support_image_base64")
    status, data = await pikzels_v2_post("/v2/thumbnail/image", payload)
    await _studio_usage_log(user["id"], "recreate", status, _pikzels_telemetry_meta(body))
    return data if status < 400 else data | {"http_status": status}


@router.post("/api/thumbnail-studio/pikzels-v2/edit")
async def ts_pikzels_edit(body: PikzelsV2EditBody, user: dict = Depends(get_current_user)):
    await _pikzels_debit(user["id"], "edit")
    payload = body.model_dump(exclude_none=True)
    normalize_url_or_base64(payload, "image_url", "image_base64")
    normalize_url_or_base64(payload, "mask_url", "mask_base64")
    normalize_url_or_base64(payload, "support_image_url", "support_image_base64")
    status, data = await pikzels_v2_post("/v2/thumbnail/edit", payload)
    await _studio_usage_log(user["id"], "edit", status, _pikzels_telemetry_meta(body))
    return data if status < 400 else data | {"http_status": status}


@router.post("/api/thumbnail-studio/pikzels-v2/one-click-fix")
async def ts_pikzels_one_click(body: PikzelsV2EditBody, user: dict = Depends(get_current_user)):
    await _pikzels_debit(user["id"], "one_click_fix")
    payload = body.model_dump(exclude_none=True)
    normalize_url_or_base64(payload, "image_url", "image_base64")
    normalize_url_or_base64(payload, "mask_url", "mask_base64")
    normalize_url_or_base64(payload, "support_image_url", "support_image_base64")
    status, data = await pikzels_v2_post("/v2/thumbnail/edit", payload)
    await _studio_usage_log(user["id"], "one_click_fix", status, _pikzels_telemetry_meta(body))
    return data if status < 400 else data | {"http_status": status}


@router.post("/api/thumbnail-studio/pikzels-v2/faceswap")
async def ts_pikzels_faceswap(body: PikzelsV2FaceswapBody, user: dict = Depends(get_current_user)):
    await _pikzels_debit(user["id"], "faceswap")
    payload = body.model_dump(exclude_none=True)
    normalize_url_or_base64(payload, "image_url", "image_base64")
    normalize_url_or_base64(payload, "mask_url", "mask_base64")
    if "face_image" in payload or "face_image_base64" in payload:
        u = str(payload.get("face_image") or "").strip()
        b = str(payload.get("face_image_base64") or "").strip()
        if u and b:
            payload["face_image"], payload["face_image_base64"] = u, ""
        elif u:
            payload["face_image"], payload["face_image_base64"] = u, ""
        elif b:
            payload["face_image"], payload["face_image_base64"] = "", b
        else:
            payload["face_image"], payload["face_image_base64"] = "", ""
    status, data = await pikzels_v2_post("/v2/thumbnail/faceswap", payload)
    await _studio_usage_log(user["id"], "faceswap", status, _pikzels_telemetry_meta(body))
    return data if status < 400 else data | {"http_status": status}


@router.post("/api/thumbnail-studio/pikzels-v2/score")
async def ts_pikzels_score(body: PikzelsV2ScoreBody, user: dict = Depends(get_current_user)):
    await _pikzels_debit(user["id"], "score")
    payload = body.model_dump(exclude_none=True)
    normalize_url_or_base64(payload, "image_url", "image_base64")
    status, data = await pikzels_v2_post("/v2/thumbnail/score", payload)
    await _studio_usage_log(user["id"], "score", status, _pikzels_telemetry_meta(body))
    return data if status < 400 else data | {"http_status": status}


@router.post("/api/thumbnail-studio/pikzels-v2/titles")
async def ts_pikzels_titles(body: PikzelsV2TitlesBody, user: dict = Depends(get_current_user)):
    await _pikzels_debit(user["id"], "titles")
    payload = body.model_dump(exclude_none=True)
    normalize_url_or_base64(payload, "support_image_url", "support_image_base64")
    status, data = await pikzels_v2_post("/v2/title/text", payload)
    await _studio_usage_log(user["id"], "titles", status, _pikzels_telemetry_meta(body))
    return data if status < 400 else data | {"http_status": status}


@router.post("/api/thumbnail-studio/pikzels-v2/persona")
async def ts_pikzels_persona(body: PikzelsV2PikzonalityBody, user: dict = Depends(get_current_user)):
    await _pikzels_debit(user["id"], "persona")
    payload = body.model_dump(exclude_none=True)
    trim_pikzonality_images(payload)
    status, data = await pikzels_v2_post("/v2/pikzonality/persona", payload)
    await _studio_usage_log(user["id"], "persona", status, _pikzels_telemetry_meta(body))
    return data if status < 400 else data | {"http_status": status}


@router.post("/api/thumbnail-studio/pikzels-v2/style")
async def ts_pikzels_style(body: PikzelsV2PikzonalityBody, user: dict = Depends(get_current_user)):
    await _pikzels_debit(user["id"], "style")
    payload = body.model_dump(exclude_none=True)
    trim_pikzonality_images(payload)
    status, data = await pikzels_v2_post("/v2/pikzonality/style", payload)
    await _studio_usage_log(user["id"], "style", status, _pikzels_telemetry_meta(body))
    return data if status < 400 else data | {"http_status": status}


@router.get("/api/thumbnail-studio/pikzels-v2/pikzonality/{pikzonality_id}")
async def ts_pikzels_poll(pikzonality_id: str, user: dict = Depends(get_current_user)):
    # Polling is read-only at Pikzels; do not debit PUT/AIC on each poll.
    path = V2_PIKZONALITY_BY_ID.replace("{id}", pikzonality_id)
    status, data = await pikzels_v2_get(path)
    if status >= 400:
        raise HTTPException(status_code=min(status, 599), detail=data)
    return data
