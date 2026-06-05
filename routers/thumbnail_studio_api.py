"""
Thumbnail Studio HTTP API — DB-backed jobs, personas, and Pikzels v2 proxy.

``routers/thumbnail_studio_routes.register_thumbnail_studio_routes`` only
includes this module's router; it does not reference ``app.api_*``.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import uuid
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

import core.state
from core.config import R2_BUCKET_NAME
from core.deps import get_current_user, get_current_user_readonly
from core.helpers import _now_utc
from core.r2 import _delete_r2_objects, generate_presigned_download_url, get_s3_client, put_object_bytes
from core.wallet import atomic_debit_tokens
from api.schemas.pikzels_v2 import (
    PikzelsV2EditBody,
    PikzelsV2FaceswapBody,
    PikzelsV2PikzonalityBody,
    PikzelsV2PikzonalityUpdateBody,
    PikzelsV2PromptBody,
    PikzelsV2RecreateBody,
    PikzelsV2ScoreBody,
    PikzelsV2TitlesBody,
)
from services.thumbnail_personas_list import list_thumbnail_studio_personas
from services.pikzels_v2 import PIKZELS_FEATURE_MAP, V2_PIKZONALITY_BY_ID, resolve_public_api_key
from services.pikzels_v2_client import (
    normalize_url_or_base64,
    pikzels_v2_delete,
    pikzels_v2_get,
    pikzels_v2_patch,
    pikzels_v2_post,
    resolve_pikzels_persona_style_xor,
    trim_pikzonality_images,
)
from services.thumbnail_studio import (
    _pikzels_extract_image_url,
    attach_preview_urls_to_variants,
    build_thumbnail_ab_export_zip,
    enrich_variants_with_uploadm8_engine,
    estimate_pikzels_v2_call_cost,
    estimate_studio_cost,
    extract_cdn_pikzels_rest_url,
    extract_youtube_video_id,
    fetch_youtube_title,
    generate_recreate_variants,
    clamp_studio_variant_count,
    STUDIO_VARIANT_COUNT_DEFAULT,
    STUDIO_VARIANT_COUNT_MAX,
    STUDIO_VARIANT_COUNT_MIN,
    normalize_hydration_context,
    normalize_persona_face_ref_for_pikzels,
    register_creator_persona_with_pikzels,
    truncate_persona_image_url_for_storage,
    upload_ab_export_zip_to_r2,
    _youtube_reference_thumbnail_as_data_url,
    youtube_reference_thumbnail_url,
)
from services.pikzels_errors import format_pikzels_error_message
from services.growth_intelligence import m8_engine_identity_payload, record_studio_usage_event
from stages.notify_stage import notify_user_pikzels_generation
from services.hydration_from_upload_row import hydration_context_from_upload_row
from services.upload_pikzels_frame import append_hydration_to_prompt, load_upload_frame_jpeg_base64
from services.ml_marketing import record_outcome_label, record_thumbnail_studio_engine_ml_batch
from services.pikzels_analyzer import (
    RECOMMENDATION_STATUSES,
    apply_fix_to_analysis,
    batch_score_user_uploads,
    export_analysis_ab_pack,
    fetch_actionable_recommendations,
    fetch_score_trend,
    generate_titles_for_analysis,
    get_analysis_for_user,
    get_latest_analysis_for_upload,
    list_analyses,
    persist_score_analysis,
    save_fix_as_upload_thumbnail,
    update_recommendation_status,
)
from services.wallet_marketing import _user_campaign_features

logger = logging.getLogger("uploadm8-api")

router = APIRouter(tags=["thumbnail-studio"])


class YoutubeReferenceItem(BaseModel):
    id: Optional[str] = None
    label: str = ""
    youtube_url: str = ""
    youtube_video_id: Optional[str] = None
    is_default: bool = False


class YoutubeReferencesBody(BaseModel):
    references: List[YoutubeReferenceItem] = Field(default_factory=list)


@router.get("/api/thumbnail-studio/youtube-references")
async def get_thumbnail_studio_youtube_references(
    user: dict = Depends(get_current_user_readonly),
):
    """Saved YouTube style references + default URL for Studio and upload pipeline."""
    from services.thumbnail_youtube_refs import list_youtube_references

    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT preferences FROM users WHERE id = $1", user["id"])
    prefs = _json_obj(row["preferences"] if row else None)
    refs = list_youtube_references(prefs)
    default = next((r for r in refs if r.get("is_default")), refs[0] if refs else None)
    return {
        "references": refs,
        "default": default,
        "default_youtube_url": (default or {}).get("youtube_url") or "",
    }


@router.put("/api/thumbnail-studio/youtube-references")
async def put_thumbnail_studio_youtube_references(
    body: YoutubeReferencesBody,
    user: dict = Depends(get_current_user),
):
    from services.thumbnail_youtube_refs import merge_references_into_prefs

    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT preferences FROM users WHERE id = $1", user["id"])
        prefs = _json_obj(row["preferences"] if row else None)
        refs_in = [r.model_dump() for r in body.references]
        merged = merge_references_into_prefs(prefs, refs_in)
        await conn.execute(
            """
            UPDATE users SET preferences = $1::jsonb, updated_at = NOW()
            WHERE id = $2
            """,
            json.dumps(merged),
            user["id"],
        )
    from services.thumbnail_youtube_refs import list_youtube_references

    refs = list_youtube_references(merged)
    default = next((r for r in refs if r.get("is_default")), refs[0] if refs else None)
    return {
        "references": refs,
        "default": default,
        "default_youtube_url": (default or {}).get("youtube_url") or "",
    }


async def _resolve_linked_studio_persona_db(
    conn: Any,
    user_id: str,
    raw_persona_id: str,
) -> tuple[Optional[uuid.UUID], str, Optional[str]]:
    """
    For guided YouTube recreate: return ``(creator_personas.id, display_name, pikzels_pikzonality_id)``
    only when the persona belongs to the user and has a linked Pikzels pikzonality id.
    """
    try:
        pid = uuid.UUID(str(raw_persona_id).strip())
    except (ValueError, TypeError, AttributeError):
        return None, "", None
    row = await conn.fetchrow(
        """
        SELECT cp.id, cp.name, cp.profile_json
        FROM creator_personas cp
        WHERE cp.user_id = $1::uuid AND cp.id = $2::uuid
        """,
        user_id,
        pid,
    )
    if not row:
        return None, "", None
    prof = row["profile_json"]
    if isinstance(prof, str):
        try:
            prof = json.loads(prof)
        except Exception:
            prof = {}
    if not isinstance(prof, dict):
        prof = {}
    pkz = str(prof.get("pikzels_pikzonality_id") or "").strip()
    if not pkz:
        asset = await conn.fetchrow(
            """
            SELECT pikzels_pikzonality_id::text AS pid
            FROM pikzels_user_assets
            WHERE user_id = $1::uuid AND kind = 'persona' AND status = 'linked'
              AND local_persona_id = $2::uuid
            LIMIT 1
            """,
            user_id,
            pid,
        )
        if asset and asset["pid"]:
            pkz = str(asset["pid"]).strip()
    if not pkz:
        return None, "", None
    name = str(row["name"] or "").strip() or "Persona"
    return row["id"], name, pkz[:200]


async def _first_persona_face_ref_db(conn: Any, user_id: str, persona_id: uuid.UUID) -> str:
    """Return one saved persona face ref suitable for Pikzels FaceSwap, or empty."""
    row = await conn.fetchrow(
        """
        SELECT image_url
        FROM creator_persona_images
        WHERE persona_id = $1::uuid AND user_id = $2::uuid
        ORDER BY id ASC
        LIMIT 1
        """,
        persona_id,
        user_id,
    )
    if not row or not row.get("image_url"):
        return ""
    return _saved_persona_ref_for_pikzels(str(row["image_url"]))


def _studio_job_public(row: Any) -> Dict[str, Any]:
    """Serialize a thumbnail_recreate_jobs row for API clients (no internal-only fields)."""
    d = dict(row)
    jid = d.get("id")
    ca = d.get("created_at")
    pid = d.get("persona_id")
    breakdown = d.get("breakdown_json")
    if isinstance(breakdown, str):
        try:
            breakdown = json.loads(breakdown)
        except Exception:
            breakdown = {}
    if not isinstance(breakdown, dict):
        breakdown = {}
    hydration_context = breakdown.get("hydration_context") if isinstance(breakdown, dict) else {}
    if not isinstance(hydration_context, dict):
        hydration_context = {}
    return {
        "job_id": str(jid) if jid else "",
        "youtube_url": str(d.get("youtube_url") or ""),
        "youtube_video_id": str(d.get("youtube_video_id") or ""),
        "source_title": str(d.get("source_title") or ""),
        "topic": str(d.get("topic") or ""),
        "niche": str(d.get("niche") or "general"),
        "closeness": int(d.get("closeness") or 55),
        "variant_count": clamp_studio_variant_count(d.get("variant_count")),
        "competitor_gap_mode": bool(d.get("competitor_gap_mode")),
        "put_cost": int(d.get("put_cost") or 0),
        "aic_cost": int(d.get("aic_cost") or 0),
        "persona_id": str(pid) if pid else None,
        "format_key": str(breakdown.get("format_key") or ""),
        "hydration_context": hydration_context,
        "created_at": ca.isoformat() if hasattr(ca, "isoformat") else str(ca or ""),
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


def _truthy_meta(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "default")
    return False


def _thumbnail_strategy_from_variant(
    *,
    job_row: Any,
    variant_id: str,
    variant_json: Dict[str, Any],
) -> Dict[str, Any]:
    """Persist a selected Studio variation as upload-time prompt strategy, not fixed copy."""
    job = dict(job_row)
    v = dict(variant_json or {})
    from services.thumbnail_niches import normalize_niche

    job_yt_url = str(job.get("youtube_url") or "")[:500]
    job_yt_vid = str(job.get("youtube_video_id") or "")[:32]
    strategy = {
        "version": 1,
        "source": "thumbnail_studio_selected_variant",
        "job_id": str(job.get("id") or ""),
        "variant_id": str(variant_id or ""),
        "format_key": str(v.get("format_key") or "")[:80],
        "layout_name": str(v.get("name") or "")[:120],
        "layout_pattern": str(v.get("layout_pattern") or "")[:240],
        "audience_niche": normalize_niche(str(job.get("niche") or ""))[:120],
        "reference_youtube_url": job_yt_url,
        "youtube_url": job_yt_url,
        "reference_youtube_video_id": job_yt_vid,
        "youtube_video_id": job_yt_vid,
        "reference_topic": str(job.get("topic") or "")[:200],
        "reference_strength": int(job.get("closeness") or 55),
        "competitor_gap_mode": bool(job.get("competitor_gap_mode")),
        "persona_id": str(job.get("persona_id") or "")[:80],
        "hydration_focus": str(v.get("hydration_focus") or "")[:120],
        "selected_headline_style": str(v.get("headline") or "")[:120],
        "text_position": str(v.get("text_position") or "")[:40],
        "contrast_profile": str(v.get("contrast_profile") or "")[:40],
        "emotion": str(v.get("emotion") or "")[:60],
        "face_scale": float(v.get("face_scale") or 0) if v.get("face_scale") is not None else 0,
    }
    return {k: val for k, val in strategy.items() if val not in ("", None)}


@router.get("/api/thumbnail-studio/cdn-preview")
async def thumbnail_studio_cdn_preview(
    variant_id: str = Query(..., min_length=30, max_length=48),
    user: dict = Depends(get_current_user_readonly),
):
    """
    Fetch a Pikzels CDN image for **this user's** stored variant only (variant_json
    must contain ``pikzels_cdn_url`` or an extractable CDN URL). Uses the server
    API key — browsers cannot send ``X-Api-Key`` to cdn.pikzels.com.

    **Upstream mapping** (so Sentry and clients do not treat stale URLs as generic 502s):

    - CDN **404** → HTTP **404**, ``detail.code`` = ``upstream_image_not_found``.
    - CDN **403** → **403**, ``upstream_forbidden``.
    - CDN **401** → **502**, ``upstream_unauthorized`` (server key misconfiguration).
    - CDN **429** → **429**, ``upstream_rate_limited``.
    - CDN **5xx** → **502**, ``upstream_server_error`` + ``upstream_status``.
    - Other non-200, empty body, or non-image 200 → **502**, ``upstream_bad_response``.
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
        for k in ("subhead", "engine_text_brief", "engine_error"):
            raw = extract_cdn_pikzels_rest_url(str(j.get(k) or ""))
            if raw.startswith("https://"):
                break
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

    # Map CDN status explicitly — stale variant rows often point at expired Pikzels URLs (404).
    # A generic 502 made Sentry treat client/data issues as server faults.
    if r.status_code == 404:
        logger.warning("cdn-preview upstream 404 variant_id=%s", variant_id)
        raise HTTPException(
            status_code=404,
            detail={
                "code": "upstream_image_not_found",
                "message": "The preview image is no longer available on the CDN.",
            },
        )
    if r.status_code == 403:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "upstream_forbidden",
                "message": "The CDN rejected this preview request.",
            },
        )
    if r.status_code == 401:
        logger.error(
            "cdn-preview upstream 401 variant_id=%s — likely bad server PIKZELS / CDN API key",
            variant_id,
        )
        raise HTTPException(
            status_code=502,
            detail={
                "code": "upstream_unauthorized",
                "message": "CDN authentication failed (server configuration).",
            },
        )
    if r.status_code == 429:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "upstream_rate_limited",
                "message": (
                    "CDN rate limit (Pikzels defaults to 10 concurrent requests per API key). "
                    "Retry with backoff — see https://docs.pikzels.com/rate-limits"
                ),
                "pikzels_doc": "https://docs.pikzels.com/rate-limits",
            },
        )
    if r.status_code >= 500:
        logger.warning(
            "cdn-preview upstream %s variant_id=%s body_len=%s",
            r.status_code,
            variant_id,
            len(r.content or b""),
        )
        raise HTTPException(
            status_code=502,
            detail={
                "code": "upstream_server_error",
                "upstream_status": r.status_code,
                "message": "CDN returned an error.",
            },
        )
    if r.status_code != 200 or not r.content:
        logger.warning(
            "cdn-preview upstream_bad_response status=%s body_len=%s variant_id=%s",
            r.status_code,
            len(r.content or b""),
            variant_id,
        )
        raise HTTPException(
            status_code=502,
            detail={
                "code": "upstream_bad_response",
                "upstream_status": r.status_code,
                "message": "CDN returned a non-image response.",
            },
        )
    ct = (r.headers.get("content-type") or "").split(";")[0].strip() or "image/jpeg"
    if not ct.startswith("image/"):
        logger.warning(
            "cdn-preview upstream_bad_response status=200 content_type=%s body_len=%s variant_id=%s",
            ct,
            len(r.content or b""),
            variant_id,
        )
        raise HTTPException(
            status_code=502,
            detail={
                "code": "upstream_bad_response",
                "upstream_status": 200,
                "message": "CDN body was not an image content-type.",
            },
        )
    return Response(content=r.content, media_type=ct)


class StudioEstimateBody(BaseModel):
    variant_count: int = Field(
        default=STUDIO_VARIANT_COUNT_DEFAULT,
        ge=STUDIO_VARIANT_COUNT_MIN,
        le=STUDIO_VARIANT_COUNT_MAX,
    )
    has_persona: bool = False
    persona_id: Optional[str] = Field(
        default=None,
        description="When set, server checks creator_personas + Pikzels link and includes persona AIC in the estimate.",
    )
    competitor_gap_mode: bool = False
    has_channel_memory: bool = True


class StudioRecreateBody(BaseModel):
    youtube_url: str = Field(min_length=4, max_length=2048)
    topic: str = ""
    niche: str = "general"
    closeness: int = Field(default=55, ge=0, le=100)
    variant_count: int = Field(
        default=STUDIO_VARIANT_COUNT_DEFAULT,
        ge=STUDIO_VARIANT_COUNT_MIN,
        le=STUDIO_VARIANT_COUNT_MAX,
    )
    persona_id: Optional[str] = None
    format_key: Optional[str] = None
    competitor_gap_mode: bool = False
    hydration_context: Dict[str, Any] = Field(default_factory=dict)


class StudioPersonaCreateBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    image_urls: List[str] = Field(default_factory=list, min_length=3, max_length=20)


class StudioFeedbackBody(BaseModel):
    job_id: str
    variant_id: Optional[str] = None
    upload_id: Optional[str] = None
    event_type: str = "selected"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StudioUploadFramePikzelsBody(BaseModel):
    """
    Pikzels recreate / edit / score using a frame from the user's own upload (not a YouTube URL).

    ``persona`` / ``style`` are Pikzels pikzonality UUIDs (same as other v2 proxy routes).
    """

    operation: Literal["recreate", "edit", "score"] = "score"
    frame_source: Literal["video_best", "video_offset", "primary_thumbnail"] = "video_best"
    offset_seconds: Optional[float] = Field(default=None, ge=0, le=86400)
    use_hydration: bool = True
    hydration_lane: str = Field(default="combined", max_length=32)
    studio_variant_id: Optional[str] = Field(default=None, max_length=128)
    prompt: Optional[str] = Field(default=None, max_length=1000)
    format: str = Field(default="16:9", max_length=16)
    model: str = Field(default="pkz_4", max_length=32)
    image_weight: Optional[str] = Field(default="medium", max_length=32)
    persona: Optional[str] = None
    style: Optional[str] = None
    title: Optional[str] = Field(default=None, max_length=200)
    mask_url: Optional[str] = None
    mask_base64: Optional[str] = None
    support_image_url: Optional[str] = None
    support_image_base64: Optional[str] = None


@router.post("/api/thumbnail-studio/estimate")
async def ts_estimate(body: StudioEstimateBody, user: dict = Depends(get_current_user)):
    has_persona = bool(body.has_persona)
    if (body.persona_id or "").strip() and core.state.db_pool:
        async with core.state.db_pool.acquire() as conn:
            _pu, _pn, pkz = await _resolve_linked_studio_persona_db(
                conn, str(user["id"]), body.persona_id.strip()
            )
        has_persona = bool(pkz)
    put, aic, breakdown = estimate_studio_cost(
        variant_count=body.variant_count,
        has_persona=has_persona,
        competitor_gap_mode=body.competitor_gap_mode,
        has_channel_memory=body.has_channel_memory,
    )
    return {"put_cost": put, "aic_cost": aic, "breakdown": breakdown}


@router.get("/api/thumbnail-studio/hydration-context")
async def ts_hydration_context(user: dict = Depends(get_current_user_readonly)):
    """
    Latest backend-collected evidence used to hydrate style-recreation variants.

    Pulls from completed upload captions, hydration reports, telemetry metadata, and
    recognized music. The studio page displays this context; users do not type it.
    """
    rows: List[Any] = []
    async with core.state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, filename, title, caption, ai_title, ai_caption,
                   output_artifacts, trill_metadata, created_at, updated_at
            FROM uploads
            WHERE user_id = $1
              AND (
                NULLIF(TRIM(COALESCE(caption, '')), '') IS NOT NULL
                OR NULLIF(TRIM(COALESCE(ai_caption, '')), '') IS NOT NULL
                OR COALESCE(output_artifacts, '{}'::jsonb) <> '{}'::jsonb
                OR COALESCE(trill_metadata, '{}'::jsonb) <> '{}'::jsonb
              )
            ORDER BY updated_at DESC NULLS LAST, created_at DESC
            LIMIT 20
            """,
            user["id"],
        )

    candidates: List[Dict[str, Any]] = []
    for row in rows:
        ctx = hydration_context_from_upload_row(dict(row))
        score = sum(1 for k in ("caption", "geo", "latitude", "longitude", "artist", "track") if ctx.get(k))
        if score:
            ctx["_score"] = score
            candidates.append(ctx)
    candidates.sort(key=lambda x: int(x.get("_score") or 0), reverse=True)
    context = candidates[0] if candidates else {}
    if context:
        context = {k: v for k, v in context.items() if k != "_score"}
    return {"hydration_context": context, "candidates": candidates[:5]}


@router.get("/api/thumbnail-studio/personas")
async def ts_list_personas(user: dict = Depends(get_current_user_readonly)):
    async with core.state.db_pool.acquire() as conn:
        out_rows = await list_thumbnail_studio_personas(conn, user["id"])
    return {"personas": out_rows}


@router.post("/api/thumbnail-studio/personas")
async def ts_create_persona(body: StudioPersonaCreateBody, user: dict = Depends(get_current_user)):
    pid = uuid.uuid4()
    stored_refs: List[str] = []
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
            for idx, url in enumerate(body.image_urls):
                stored_url, quality = await _store_persona_image_ref(str(user["id"]), pid, idx, url)
                if not stored_url:
                    continue
                stored_refs.append(_saved_persona_ref_for_pikzels(stored_url))
                await conn.execute(
                    """
                    INSERT INTO creator_persona_images (persona_id, user_id, image_url, quality_json)
                    VALUES ($1, $2, $3, $4::jsonb)
                    """,
                    pid,
                    user["id"],
                    stored_url,
                    json.dumps(quality),
                )

    resp: Dict[str, Any] = {
        "id": str(pid),
        "status": "created",
        "pikzels_pikzonality_linked": False,
    }
    pkz_id, pkz_err = await register_creator_persona_with_pikzels(
        name=body.name.strip(),
        image_refs=stored_refs or [
            truncate_persona_image_url_for_storage(u) for u in (body.image_urls or [])
        ],
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
                SET profile_json = COALESCE(profile_json, '{}'::jsonb) || $2::jsonb,
                    link_status = 'linked',
                    link_error = NULL,
                    link_completed_at = NOW(),
                    updated_at = NOW()
                WHERE id = $1 AND user_id = $3
                """,
                pid,
                json.dumps(merge),
                user["id"],
            )
            await _record_pikzels_user_asset(
                conn,
                user_id=str(user["id"]),
                kind="persona",
                pikzels_pikzonality_id=pkz_id,
                name=body.name.strip(),
                local_persona_id=pid,
                metadata={"source": "thumbnail_studio_ui"},
            )
        logger.info(
            "thumbnail persona saved and linked to Pikzels user=%s persona=%s pikzels_id=%s",
            str(user["id"]),
            str(pid),
            pkz_id[:12],
        )
        resp["pikzels_pikzonality_linked"] = True
        resp["pikzels_pikzonality_id"] = pkz_id
    elif pkz_err:
        async with core.state.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE creator_personas
                SET link_status = 'failed', link_error = $2, updated_at = NOW()
                WHERE id = $1 AND user_id = $3
                """,
                pid,
                str(pkz_err)[:2000],
                user["id"],
            )
        logger.warning(
            "thumbnail persona saved but Pikzels link failed user=%s persona=%s err=%s",
            str(user["id"]),
            str(pid),
            str(pkz_err)[:240],
        )
        resp["pikzels_warning"] = pkz_err
    return resp


def _preferences_dict_from_users_row(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            return json.loads(raw) if raw.strip() else {}
        except Exception:
            return {}
    return {}


def _decode_data_url_bytes(data_url: str) -> tuple[bytes, str]:
    s = str(data_url or "").strip()
    comma = s.find(",")
    if comma < 0:
        raise ValueError("not a data URL")
    header = s[:comma].lower()
    if ";base64" not in header:
        raise ValueError("data URL is not base64")
    ct = "image/jpeg"
    if header.startswith("data:"):
        ct = header[5:].split(";", 1)[0] or ct
    try:
        return base64.b64decode(s[comma + 1 :], validate=False), ct
    except (binascii.Error, ValueError) as e:
        raise ValueError("invalid base64 data URL") from e


async def _store_persona_image_ref(user_id: str, persona_id: uuid.UUID, idx: int, raw_ref: str) -> tuple[str, Dict[str, Any]]:
    """
    Store reference images privately in R2 when configured; otherwise fall back to DB TEXT.

    ``creator_persona_images.image_url`` stores either:
    - ``r2://<key>`` for private R2-backed images, or
    - an HTTPS/data URL fallback when object storage is unavailable.
    """
    raw = str(raw_ref or "").strip()
    if not raw:
        return "", {"storage": "empty"}
    if R2_BUCKET_NAME and raw.lower().startswith("data:image"):
        normalized = normalize_persona_face_ref_for_pikzels(raw)
        if normalized:
            try:
                img_bytes, content_type = _decode_data_url_bytes(normalized)
                r2_key = f"thumbnail-studio/personas/{user_id}/{persona_id}/ref_{idx:02d}.jpg"
                await asyncio.to_thread(put_object_bytes, r2_key, img_bytes, content_type or "image/jpeg")
                return f"r2://{r2_key}", {
                    "storage": "r2",
                    "r2_key": r2_key,
                    "content_type": content_type or "image/jpeg",
                    "bytes": len(img_bytes),
                }
            except Exception as e:
                logger.warning("persona image R2 store failed; falling back to DB data URL: %s", e)
    return truncate_persona_image_url_for_storage(raw), {"storage": "db_text"}


def _saved_persona_ref_for_pikzels(stored: str) -> str:
    s = str(stored or "").strip()
    if s.startswith("r2://"):
        key = s[5:]
        try:
            # Prefer inline bytes for Pikzels persona registration. Some upstream
            # fetch paths reject signed/private URLs and return INVALID_IMAGE.
            if R2_BUCKET_NAME:
                obj = get_s3_client().get_object(Bucket=R2_BUCKET_NAME, Key=key)
                body = obj.get("Body")
                raw = body.read() if body else b""
                ct = str(obj.get("ContentType") or "image/jpeg").strip().lower()
                if raw:
                    if "png" in ct:
                        mime = "image/png"
                    elif "webp" in ct:
                        mime = "image/webp"
                    else:
                        mime = "image/jpeg"
                    return f"data:{mime};base64,{base64.standard_b64encode(raw).decode('ascii')}"
            return generate_presigned_download_url(key, ttl=3600) or ""
        except Exception:
            logger.warning(
                "persona Pikzels presign failed for r2 key prefix=%r",
                key[:120],
                exc_info=True,
            )
            return ""
    return s


def _pikzels_error_code(data: Any) -> str:
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            return str(err.get("code") or "").strip()
        return str(data.get("code") or "").strip()
    return ""


@router.delete("/api/thumbnail-studio/personas/{persona_id}")
async def ts_delete_persona(persona_id: str, user: dict = Depends(get_current_user)):
    """
    Remove a saved persona and its reference images. Past thumbnail jobs keep their
    ``persona_id`` as null (FK SET NULL). Clears ``thumbnailDefaultPersonaId`` in
    ``users.preferences`` when it pointed at this row.
    """
    try:
        pid = uuid.UUID(str(persona_id).strip())
    except ValueError:
        raise HTTPException(400, "Invalid persona_id")

    async with core.state.db_pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                "SELECT id, profile_json FROM creator_personas WHERE id = $1 AND user_id = $2",
                pid,
                user["id"],
            )
            if not row:
                raise HTTPException(404, "Persona not found")
            prof = row.get("profile_json")
            if isinstance(prof, str):
                try:
                    prof = json.loads(prof)
                except Exception:
                    prof = {}
            if not isinstance(prof, dict):
                prof = {}
            pkz_id = str(prof.get("pikzels_pikzonality_id") or "").strip()
            image_rows = await conn.fetch(
                """
                SELECT image_url, quality_json FROM creator_persona_images
                WHERE persona_id = $1 AND user_id = $2
                """,
                pid,
                user["id"],
            )

            urow = await conn.fetchrow("SELECT preferences FROM users WHERE id = $1", user["id"])
            existing = _preferences_dict_from_users_row(urow["preferences"] if urow else None)
            pid_s = str(pid).strip().lower()

            def _matches(v: Any) -> bool:
                if v is None:
                    return False
                try:
                    return str(uuid.UUID(str(v).strip())).lower() == pid_s
                except (ValueError, TypeError, AttributeError):
                    return str(v).strip().lower() == pid_s

            changed = False
            for k in ("thumbnailDefaultPersonaId", "thumbnail_default_persona_id"):
                if k in existing and _matches(existing.get(k)):
                    existing.pop(k, None)
                    changed = True
            if changed:
                await conn.execute(
                    "UPDATE users SET preferences = $1::jsonb, updated_at = NOW() WHERE id = $2",
                    existing,
                    user["id"],
                )

            remote_status = None
            remote_error = ""
            if pkz_id:
                status, data = await pikzels_v2_delete(V2_PIKZONALITY_BY_ID.replace("{id}", pkz_id))
                if status < 400 or status == 404:
                    remote_status = "deleted"
                else:
                    code = _pikzels_error_code(data)
                    remote_status = "pending" if code == "CANNOT_DELETE_PROCESSING" else "failed"
                    remote_error = format_pikzels_error_message(data, max_len=800)
                    await conn.execute(
                        """
                        INSERT INTO pikzels_remote_deletes (
                            user_id, local_persona_id, pikzels_pikzonality_id, kind,
                            status, http_status, error_code, error_message
                        )
                        VALUES ($1::uuid, $2::uuid, $3::uuid, 'persona', $4, $5, $6, $7)
                        """,
                        str(user["id"]),
                        str(pid),
                        pkz_id,
                        remote_status,
                        int(status),
                        code[:120] or None,
                        remote_error[:2000] or None,
                    )

            await conn.execute(
                "DELETE FROM creator_personas WHERE id = $1 AND user_id = $2",
                pid,
                user["id"],
            )
            if pkz_id:
                await conn.execute(
                    """
                    DELETE FROM pikzels_user_assets
                    WHERE user_id = $1::uuid AND pikzels_pikzonality_id = $2::uuid
                    """,
                    str(user["id"]),
                    pkz_id,
                )

    r2_keys: List[str] = []
    for r in image_rows:
        s = str(r.get("image_url") or "")
        if s.startswith("r2://"):
            r2_keys.append(s[5:])
        q = r.get("quality_json")
        if isinstance(q, str):
            try:
                q = json.loads(q)
            except Exception:
                q = {}
        if isinstance(q, dict) and q.get("r2_key"):
            r2_keys.append(str(q.get("r2_key")))
    if r2_keys:
        try:
            await _delete_r2_objects(sorted(set(r2_keys)))
        except Exception:
            logger.debug("persona R2 cleanup failed", exc_info=True)

    return {
        "deleted": True,
        "persona_id": str(pid),
        "remote_delete_status": remote_status,
        "remote_delete_error": remote_error or None,
    }


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
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                SELECT name, profile_json, link_status, link_started_at FROM creator_personas
                WHERE id = $1 AND user_id = $2
                FOR UPDATE
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
            if str(row.get("link_status") or "").lower() == "linking":
                raise HTTPException(
                    409,
                    {
                        "code": "pikzels_persona_link_in_progress",
                        "message": "This persona is already being linked to Pikzels. Try again in a minute.",
                    },
                )
            await conn.execute(
                """
                UPDATE creator_personas
                SET link_status = 'linking',
                    link_error = NULL,
                    link_started_at = NOW(),
                    updated_at = NOW()
                WHERE id = $1 AND user_id = $2
                """,
                pid,
                user["id"],
            )
            img_rows = await conn.fetch(
                """
                SELECT image_url FROM creator_persona_images
                WHERE persona_id = $1 AND user_id = $2
                ORDER BY id ASC
                """,
                pid,
                user["id"],
            )
    refs = [_saved_persona_ref_for_pikzels(str(r["image_url"])) for r in img_rows if r.get("image_url")]
    if len(refs) < 3:
        async with core.state.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE creator_personas
                SET link_status = 'failed', link_error = $2, updated_at = NOW()
                WHERE id = $1 AND user_id = $3
                """,
                pid,
                "This persona needs at least three saved reference photos in your account.",
                user["id"],
            )
        raise HTTPException(
            400,
            "This persona needs at least three saved reference photos in your account. "
            "Add photos and use Save Persona, or pick another persona.",
        )

    try:
        pkz_id, pkz_err = await register_creator_persona_with_pikzels(
            name=name,
            image_refs=refs,
        )
    except Exception as e:
        async with core.state.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE creator_personas
                SET link_status = 'failed', link_error = $2, updated_at = NOW()
                WHERE id = $1 AND user_id = $3
                """,
                pid,
                str(e)[:2000],
                user["id"],
            )
        raise HTTPException(502, "Pikzels persona link failed unexpectedly.") from e
    if pkz_id:
        merge = {
            "pikzels_pikzonality_id": pkz_id,
            "pikzels_linked_at": _now_utc().isoformat(),
        }
        async with core.state.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE creator_personas
                SET profile_json = COALESCE(profile_json, '{}'::jsonb) || $2::jsonb,
                    link_status = 'linked',
                    link_error = NULL,
                    link_completed_at = NOW(),
                    updated_at = NOW()
                WHERE id = $1 AND user_id = $3
                """,
                pid,
                json.dumps(merge),
                user["id"],
            )
            await _record_pikzels_user_asset(
                conn,
                user_id=str(user["id"]),
                kind="persona",
                pikzels_pikzonality_id=pkz_id,
                name=name,
                local_persona_id=pid,
                metadata={"source": "thumbnail_studio_link"},
            )
        logger.info(
            "thumbnail persona linked to Pikzels user=%s persona=%s pikzels_id=%s",
            str(user["id"]),
            str(pid),
            pkz_id[:12],
        )
        return {
            "pikzels_linked": True,
            "pikzels_pikzonality_id": pkz_id,
            "already_linked": False,
        }

    async with core.state.db_pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE creator_personas
            SET link_status = 'failed', link_error = $2, updated_at = NOW()
            WHERE id = $1 AND user_id = $3
            """,
            pid,
            str(pkz_err or "pikzels_registration_failed")[:2000],
            user["id"],
        )
    logger.warning(
        "thumbnail persona Pikzels link failed user=%s persona=%s err=%s",
        str(user["id"]),
        str(pid),
        str(pkz_err or "pikzels_registration_failed")[:240],
    )

    return {
        "pikzels_linked": False,
        "pikzels_pikzonality_id": None,
        "pikzels_warning": pkz_err or "pikzels_registration_failed",
        "already_linked": False,
    }


@router.post("/api/thumbnail-studio/recreate")
async def ts_recreate(body: StudioRecreateBody, user: dict = Depends(get_current_user)):
    from services.thumbnail_niches import normalize_niche

    title = await fetch_youtube_title(body.youtube_url)
    vid = extract_youtube_video_id(body.youtube_url)
    if not vid:
        raise HTTPException(
            400,
            {
                "code": "invalid_youtube_url",
                "message": "Paste a normal public YouTube video URL so UploadM8 can use its thumbnail as the reference.",
            },
        )

    reference_image_data_url = ""
    if resolve_public_api_key():
        reference_image_data_url = await _youtube_reference_thumbnail_as_data_url(vid)
        if not reference_image_data_url:
            raise HTTPException(
                400,
                {
                    "code": "youtube_reference_thumbnail_unavailable",
                    "message": (
                        "UploadM8 could not download a usable public thumbnail for that YouTube video. "
                        "YouTube is returning placeholder/404 thumbnail images, so Pikzels cannot recreate it. "
                        "Try a different public video or use a reference image from an upload once that path is available."
                    ),
                },
            )
    persona_name = ""
    persona_uuid: Optional[uuid.UUID] = None
    pikzels_persona_pikzonality_id: Optional[str] = None
    persona_face_ref = ""
    if (body.persona_id or "").strip():
        if core.state.db_pool is None:
            raise HTTPException(503, "Database unavailable")
        async with core.state.db_pool.acquire() as conn:
            pu, pname, pkz = await _resolve_linked_studio_persona_db(
                conn, str(user["id"]), body.persona_id.strip()
            )
            if pu and pkz:
                persona_face_ref = await _first_persona_face_ref_db(conn, str(user["id"]), pu)
        if not pkz:
            raise HTTPException(
                400,
                {
                    "code": "persona_not_linked",
                    "message": (
                        "That persona is not linked to Pikzels yet, or does not belong to your account. "
                        "Use Persona Library → Link to Pikzels, or choose “None” for reference-only variants."
                    ),
                },
            )
        persona_uuid, persona_name, pikzels_persona_pikzonality_id = pu, pname, pkz

    hydration_context = normalize_hydration_context(body.hydration_context)
    if not hydration_context:
        try:
            recent = await ts_hydration_context(user)
            hydration_context = normalize_hydration_context(
                (recent or {}).get("hydration_context") if isinstance(recent, dict) else {}
            )
        except Exception:
            logger.debug("thumbnail studio backend hydration lookup failed", exc_info=True)

    put, aic, breakdown = estimate_studio_cost(
        variant_count=body.variant_count,
        has_persona=bool(pikzels_persona_pikzonality_id),
        competitor_gap_mode=body.competitor_gap_mode,
        has_channel_memory=True,
    )
    if hydration_context:
        breakdown["hydration_context"] = hydration_context
    if (body.format_key or "").strip():
        breakdown["format_key"] = str(body.format_key).strip()[:80]

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
                normalize_niche(body.niche or "general")[:120],
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
        niche=normalize_niche(body.niche or "general"),
        closeness=body.closeness,
        variant_count=body.variant_count,
        persona_name=persona_name,
        competitor_gap_mode=body.competitor_gap_mode,
        channel_memory_hint="",
        format_key=body.format_key,
        hydration_context=hydration_context,
    )

    raw_variants = await enrich_variants_with_uploadm8_engine(
        raw_variants,
        youtube_video_id=vid or "",
        source_title=title or body.topic or "",
        niche=normalize_niche(body.niche or "general"),
        topic=body.topic or "",
        persona_name=persona_name,
        user_id=str(user["id"]),
        job_id=str(job_id),
        pikzels_persona_pikzonality_id=pikzels_persona_pikzonality_id,
        closeness=int(body.closeness),
        persona_face_ref=persona_face_ref,
        reference_image_data_url=reference_image_data_url,
        hydration_context=hydration_context,
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


@router.delete("/api/thumbnail-studio/jobs/recent")
async def ts_delete_recent_jobs(
    count: int = Query(10, ge=1, le=40),
    user: dict = Depends(get_current_user_readonly),
):
    """
    Remove the N most recent Thumbnail Studio jobs for this account (newest first).
    Variants and feedback rows cascade; does not touch personas.
    """
    async with core.state.db_pool.acquire() as conn:
        status = await conn.execute(
            """
            DELETE FROM thumbnail_recreate_jobs
            WHERE user_id = $1
              AND id IN (
                SELECT id FROM (
                    SELECT id FROM thumbnail_recreate_jobs
                    WHERE user_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                ) sub
              )
            """,
            user["id"],
            count,
        )
    n = 0
    try:
        n = int(str(status).split()[-1])
    except (ValueError, IndexError):
        n = 0
    return {"deleted_count": n}


@router.delete("/api/thumbnail-studio/jobs/{job_id}")
async def ts_delete_job(job_id: str, user: dict = Depends(get_current_user_readonly)):
    try:
        jid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(400, "Invalid job id")
    async with core.state.db_pool.acquire() as conn:
        status = await conn.execute(
            "DELETE FROM thumbnail_recreate_jobs WHERE id = $1 AND user_id = $2",
            jid,
            user["id"],
        )
    if str(status).strip() == "DELETE 0":
        raise HTTPException(404, "Job not found")
    return {"deleted": True, "job_id": str(jid)}


@router.get("/api/thumbnail-studio/jobs/{job_id}")
async def ts_get_job(job_id: str, user: dict = Depends(get_current_user_readonly)):
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
            "SELECT * FROM thumbnail_recreate_jobs WHERE id = $1 AND user_id = $2",
            jid,
            user["id"],
        )
        if not job:
            raise HTTPException(404, "Job not found")
        variant_json: Dict[str, Any] = {}
        if var_uuid:
            vrow = await conn.fetchrow(
                """
                SELECT variant_json
                FROM thumbnail_recreate_variants
                WHERE id = $1 AND job_id = $2 AND user_id = $3
                """,
                var_uuid,
                jid,
                user["id"],
            )
            if vrow:
                variant_json = _json_obj(vrow["variant_json"])
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
        default_saved = False
        if var_uuid and _truthy_meta((body.metadata or {}).get("make_default")):
            try:
                strategy = _thumbnail_strategy_from_variant(
                    job_row=job,
                    variant_id=str(var_uuid),
                    variant_json=variant_json,
                )
                urow = await conn.fetchrow("SELECT preferences FROM users WHERE id = $1", user["id"])
                prefs = _json_obj(urow["preferences"] if urow else None)
                prefs["thumbnailStudioDefaultStrategy"] = strategy
                prefs["thumbnail_studio_default_strategy"] = strategy
                prefs["thumbnailStudioEnabled"] = prefs["thumbnail_studio_enabled"] = True
                prefs["thumbnailStudioEngineEnabled"] = prefs["thumbnail_studio_engine_enabled"] = True
                prefs["thumbnailPikzelsEnabled"] = prefs["thumbnail_pikzels_enabled"] = True
                if strategy.get("persona_id"):
                    prefs["thumbnailDefaultPersonaId"] = prefs["thumbnail_default_persona_id"] = strategy["persona_id"]
                    prefs["thumbnailPersonaEnabled"] = prefs["thumbnail_persona_enabled"] = True
                yt_url = str(strategy.get("reference_youtube_url") or strategy.get("youtube_url") or "").strip()
                if yt_url:
                    from services.thumbnail_youtube_refs import (
                        list_youtube_references,
                        merge_references_into_prefs,
                        normalize_reference_entry,
                    )

                    existing = list_youtube_references(prefs)
                    vid = str(strategy.get("reference_youtube_video_id") or "").strip()
                    new_ref = normalize_reference_entry(
                        {
                            "id": "studio-default",
                            "label": "Studio default",
                            "youtube_url": yt_url,
                            "youtube_video_id": vid,
                            "is_default": True,
                        }
                    )
                    if new_ref:
                        kept = [r for r in existing if r.get("id") != "studio-default"]
                        for r in kept:
                            r["is_default"] = False
                        kept.insert(0, new_ref)
                        prefs = merge_references_into_prefs(prefs, kept)
                await conn.execute(
                    """
                    UPDATE users
                    SET preferences = $1::jsonb, updated_at = NOW()
                    WHERE id = $2
                    """,
                    json.dumps(prefs),
                    user["id"],
                )
                default_saved = True
            except Exception:
                logger.warning("save thumbnail default strategy failed", exc_info=True)
    return {"status": "ok", "default_saved": default_saved if 'default_saved' in locals() else False}


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
async def ts_weekly_digest(user: dict = Depends(get_current_user_readonly)):
    return {
        "generated_at": _now_utc().isoformat(),
        "lines": [],
        "note": "Weekly email digest is sent separately; this endpoint is a UI placeholder.",
    }


@router.get("/api/thumbnail-studio/pikzels-v2-map")
async def ts_pikzels_map(user: dict = Depends(get_current_user_readonly)):
    out = []
    for name, key, path, desc in PIKZELS_FEATURE_MAP:
        out.append({"feature": name, "key": key, "path": path, "description": desc})
    return {"features": out}


@router.get("/api/thumbnail-studio/pikzels-v2/assets")
async def ts_pikzels_assets(
    kind: Optional[str] = Query(default=None, pattern="^(persona|style)$"),
    user: dict = Depends(get_current_user_readonly),
):
    """List saved Pikzels pikzonality IDs for **this user only** (aligned with Saved Personas).

    Excludes orphan persona rows that are not tied to one of the user's ``creator_personas``
    (by ``local_persona_id`` or ``profile_json.pikzels_pikzonality_id``). Styles are listed
    when ``kind`` matches or is omitted.
    """
    params: List[Any] = [str(user["id"])]
    where = """WHERE pua.user_id = $1::uuid AND pua.status = 'linked'
      AND (
        pua.kind = 'style'
        OR EXISTS (
          SELECT 1 FROM creator_personas cp
          WHERE cp.user_id = pua.user_id
            AND (
              cp.id = pua.local_persona_id
              OR (
                pua.local_persona_id IS NULL
                AND NULLIF(trim(cp.profile_json->>'pikzels_pikzonality_id'), '') = pua.pikzels_pikzonality_id::text
              )
            )
        )
      )"""
    if kind:
        params.append(kind)
        where += " AND pua.kind = $2"
    async with core.state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT pua.kind, pua.local_persona_id, pua.pikzels_pikzonality_id, pua.name,
                   pua.metadata_json, pua.created_at, pua.updated_at
            FROM pikzels_user_assets pua
            {where}
            ORDER BY pua.updated_at DESC NULLS LAST, pua.created_at DESC
            LIMIT 100
            """,
            *params,
        )
    assets: List[Dict[str, Any]] = []
    for r in rows:
        meta = r["metadata_json"]
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        if not isinstance(meta, dict):
            meta = {}
        assets.append(
            {
                "kind": str(r["kind"] or ""),
                "local_persona_id": str(r["local_persona_id"]) if r["local_persona_id"] else None,
                "pikzels_pikzonality_id": str(r["pikzels_pikzonality_id"] or ""),
                "name": str(r["name"] or "Pikzonality"),
                "metadata": meta,
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
            }
        )
    return JSONResponse(
        content={"assets": assets},
        headers={"Cache-Control": "private, no-store, max-age=0"},
    )


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


def _pikzels_response_pikzonality_id(data: Any) -> Optional[str]:
    if not isinstance(data, dict):
        return None
    raw = str(data.get("id") or data.get("request_id") or data.get("pikzonality_id") or "").strip()
    if not raw:
        nested = data.get("data")
        if isinstance(nested, dict):
            raw = str(nested.get("id") or nested.get("request_id") or nested.get("pikzonality_id") or "").strip()
    if not raw:
        return None
    try:
        return str(uuid.UUID(raw))
    except (ValueError, TypeError, AttributeError):
        return None


async def _record_pikzels_user_asset(
    conn: Any,
    *,
    user_id: str,
    kind: str,
    pikzels_pikzonality_id: str,
    name: str = "",
    local_persona_id: Optional[uuid.UUID] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        pkz_uuid = uuid.UUID(str(pikzels_pikzonality_id).strip())
    except (ValueError, TypeError, AttributeError):
        return
    await conn.execute(
        """
        INSERT INTO pikzels_user_assets (
            user_id, kind, local_persona_id, pikzels_pikzonality_id, name, status, metadata_json
        )
        VALUES ($1::uuid, $2, $3::uuid, $4::uuid, $5, 'linked', $6::jsonb)
        ON CONFLICT (user_id, pikzels_pikzonality_id) DO UPDATE SET
            kind = EXCLUDED.kind,
            local_persona_id = COALESCE(EXCLUDED.local_persona_id, pikzels_user_assets.local_persona_id),
            name = COALESCE(NULLIF(EXCLUDED.name, ''), pikzels_user_assets.name),
            status = 'linked',
            metadata_json = pikzels_user_assets.metadata_json || EXCLUDED.metadata_json,
            updated_at = NOW()
        """,
        str(user_id),
        kind,
        str(local_persona_id) if local_persona_id else None,
        str(pkz_uuid),
        (name or "")[:255],
        json.dumps(metadata or {}),
    )


async def _user_owns_pikzels_pikzonality(conn: Any, user_id: str, kind: str, pikzels_id: str) -> bool:
    try:
        pkz_uuid = uuid.UUID(str(pikzels_id).strip())
    except (ValueError, TypeError, AttributeError):
        return False
    k = (kind or "").strip().lower()
    if k == "persona":
        row = await conn.fetchrow(
            """
            SELECT 1 FROM creator_personas
            WHERE user_id = $1::uuid
              AND profile_json->>'pikzels_pikzonality_id' = $2
            LIMIT 1
            """,
            str(user_id),
            str(pkz_uuid),
        )
        if row:
            return True
    row = await conn.fetchrow(
        """
        SELECT 1 FROM pikzels_user_assets
        WHERE user_id = $1::uuid
          AND kind = $2
          AND pikzels_pikzonality_id = $3::uuid
        LIMIT 1
        """,
        str(user_id),
        k,
        str(pkz_uuid),
    )
    return bool(row)


async def _enforce_pikzels_payload_ownership(user_id: str, payload: Dict[str, Any]) -> None:
    """Reject raw persona/style UUIDs that are not registered to the current UploadM8 user."""
    if not isinstance(payload, dict):
        return
    resolve_pikzels_persona_style_xor(payload)
    checks = []
    for kind in ("persona", "style"):
        val = str(payload.get(kind) or "").strip()
        if val:
            checks.append((kind, val))
    if not checks:
        return
    async with core.state.db_pool.acquire() as conn:
        for kind, val in checks:
            if not await _user_owns_pikzels_pikzonality(conn, str(user_id), kind, val):
                raise HTTPException(
                    403,
                    {
                        "code": "pikzels_pikzonality_not_owned",
                        "message": f"That Pikzels {kind} is not registered to your UploadM8 account.",
                    },
                )


async def _studio_usage_log(user_id, operation: str, http_status: int, meta: Optional[Dict[str, Any]] = None) -> None:
    try:
        async with core.state.db_pool.acquire() as conn:
            await record_studio_usage_event(conn, user_id, operation, int(http_status), meta or {})
    except Exception:
        pass


async def _maybe_notify_pikzels_discord(
    user_id: str,
    operation: str,
    status: int,
    data: Any,
    *,
    upload_id: Optional[str] = None,
    source_image_url: Optional[str] = None,
) -> None:
    """Fire-and-forget style: never raises; posts image embed to the user's saved webhook when configured."""
    if status >= 400 or not isinstance(data, dict):
        return
    pool = core.state.db_pool
    if pool is None:
        return
    try:
        await notify_user_pikzels_generation(
            pool,
            str(user_id),
            operation=operation,
            response_data=data,
            upload_id=upload_id,
            source_image_url=source_image_url,
        )
    except Exception:
        logger.debug("pikzels user discord notify failed", exc_info=True)


async def _pikzels_debit(user_id: str, op: str) -> str:
    put, aic, meta = estimate_pikzels_v2_call_cost(op)
    # token_ledger.upload_id is UUID — plain id only (reason carries pikzels_v2_*).
    ref = str(uuid.uuid4())
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
    await _enforce_pikzels_payload_ownership(str(user["id"]), payload)
    status, data = await pikzels_v2_post("/v2/thumbnail/text", payload)
    await _studio_usage_log(user["id"], "prompt", status, _pikzels_telemetry_meta(body))
    return data if status < 400 else data | {"http_status": status}


@router.post("/api/thumbnail-studio/pikzels-v2/recreate")
async def ts_pikzels_recreate(body: PikzelsV2RecreateBody, user: dict = Depends(get_current_user)):
    await _pikzels_debit(user["id"], "recreate")
    payload = body.model_dump(exclude_none=True)
    image_url = str(payload.get("image_url") or "").strip()
    if image_url and not image_url.lower().startswith(("data:", "http://i.ytimg.com", "https://i.ytimg.com")):
        vid = extract_youtube_video_id(image_url)
        if vid:
            payload["image_url"] = youtube_reference_thumbnail_url(vid)
    normalize_url_or_base64(payload, "image_url", "image_base64")
    normalize_url_or_base64(payload, "support_image_url", "support_image_base64")
    await _enforce_pikzels_payload_ownership(str(user["id"]), payload)
    status, data = await pikzels_v2_post("/v2/thumbnail/image", payload)
    await _studio_usage_log(user["id"], "recreate", status, _pikzels_telemetry_meta(body))
    await _maybe_notify_pikzels_discord(
        user["id"], "recreate", status, data, upload_id=str(payload.get("upload_id") or "") or None
    )
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
    await _maybe_notify_pikzels_discord(
        user["id"], "edit", status, data, upload_id=str(payload.get("upload_id") or "") or None
    )
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
    await _maybe_notify_pikzels_discord(
        user["id"], "one_click_fix", status, data, upload_id=str(payload.get("upload_id") or "") or None
    )
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
    await _maybe_notify_pikzels_discord(
        user["id"], "faceswap", status, data, upload_id=str(payload.get("upload_id") or "") or None
    )
    return data if status < 400 else data | {"http_status": status}


@router.post("/api/thumbnail-studio/pikzels-v2/score")
async def ts_pikzels_score(body: PikzelsV2ScoreBody, user: dict = Depends(get_current_user)):
    await _pikzels_debit(user["id"], "score")
    payload = body.model_dump(exclude_none=True)
    normalize_url_or_base64(payload, "image_url", "image_base64")
    status, data = await pikzels_v2_post("/v2/thumbnail/score", payload)
    await _studio_usage_log(user["id"], "score", status, _pikzels_telemetry_meta(body))
    src = str(payload.get("image_url") or "").strip()
    score_src = src if src.startswith("https://") else None
    await _maybe_notify_pikzels_discord(
        user["id"],
        "score",
        status,
        data,
        upload_id=str(payload.get("upload_id") or "") or None,
        source_image_url=score_src,
    )
    return data if status < 400 else data | {"http_status": status}


@router.post("/api/thumbnail-studio/from-upload/{upload_id}/pikzels-v2")
async def ts_pikzels_from_upload_frame(
    upload_id: str,
    body: StudioUploadFramePikzelsBody,
    user: dict = Depends(get_current_user),
):
    """
    Run Pikzels **recreate**, **edit**, or **score** on a JPEG from this user's upload
    (processed video frame or ``thumbnail_r2_key``), optionally folding backend hydration
    into recreate/edit prompts for the same evidence as Studio hydration context.
    """
    try:
        uid = uuid.UUID(str(upload_id).strip())
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(400, detail="invalid upload_id")

    if core.state.db_pool is None:
        raise HTTPException(503, detail="Database unavailable")

    async with core.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, user_id, filename, title, caption, ai_title, ai_caption,
                   output_artifacts, trill_metadata, created_at, updated_at,
                   processed_r2_key, r2_key, thumbnail_r2_key
            FROM uploads
            WHERE id = $1::uuid AND user_id = $2::uuid
            """,
            uid,
            user["id"],
        )
    if not row:
        raise HTTPException(404, detail="upload not found")

    upload_row = dict(row)
    op = (body.operation or "score").strip().lower()

    if op == "edit":
        if not (body.prompt or "").strip():
            raise HTTPException(400, detail="edit requires prompt")
        await _pikzels_debit(user["id"], "edit")
    elif op == "recreate":
        await _pikzels_debit(user["id"], "recreate")
    elif op == "score":
        await _pikzels_debit(user["id"], "score")
    else:
        raise HTTPException(400, detail="invalid operation")

    offset_for_loader: Optional[float] = None
    if body.frame_source == "video_offset":
        offset_for_loader = body.offset_seconds

    try:
        image_b64, frame_meta = await load_upload_frame_jpeg_base64(
            upload_row,
            body.frame_source,
            offset_for_loader,
        )
    except ValueError as e:
        code = str(e)
        if code == "no_primary_thumbnail":
            raise HTTPException(
                400,
                detail="No primary thumbnail for this upload yet. Use a video frame or finish processing.",
            ) from e
        if code == "no_video_key":
            raise HTTPException(400, detail="No video object on this upload.") from e
        if code == "offset_required":
            raise HTTPException(
                400,
                detail="offset_seconds is required when frame_source is video_offset.",
            ) from e
        if code == "bad_frame_source":
            raise HTTPException(400, detail="Invalid frame_source.") from e
        raise HTTPException(400, detail=code) from e
    except RuntimeError:
        raise HTTPException(502, detail="Could not extract a frame from the video.") from None

    def _usage_meta() -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "upload_id": str(uid),
            "from_upload_frame": True,
            "frame_source": body.frame_source,
            "frame_meta": frame_meta,
            "operation": op,
            "use_hydration": body.use_hydration,
            "hydration_lane": body.hydration_lane,
        }
        if body.studio_variant_id:
            out["studio_variant_id"] = str(body.studio_variant_id)[:128]
        return out

    if op == "recreate":
        raw_prompt = (body.prompt or "").strip() or (
            "High-retention thumbnail redesign that fits this frame; bold readable text; on-brand polish."
        )
        hydrated = append_hydration_to_prompt(
            raw_prompt,
            upload_row,
            use_hydration=body.use_hydration,
            hydration_lane=body.hydration_lane,
            max_len=950,
        )
        payload: Dict[str, Any] = {
            "prompt": hydrated,
            "image_base64": image_b64,
            "model": body.model,
            "format": body.format,
            "upload_id": str(uid),
        }
        if body.image_weight:
            payload["image_weight"] = body.image_weight
        if body.persona:
            payload["persona"] = body.persona
        if body.style:
            payload["style"] = body.style
        if body.support_image_url:
            payload["support_image_url"] = body.support_image_url
        if body.support_image_base64:
            payload["support_image_base64"] = body.support_image_base64
        if body.studio_variant_id:
            payload["studio_variant_id"] = body.studio_variant_id
        normalize_url_or_base64(payload, "support_image_url", "support_image_base64")
        await _enforce_pikzels_payload_ownership(str(user["id"]), payload)
        status, data = await pikzels_v2_post("/v2/thumbnail/image", payload)
        await _studio_usage_log(user["id"], "recreate", status, _usage_meta())
        await _maybe_notify_pikzels_discord(user["id"], "recreate_from_upload", status, data, upload_id=str(uid))
        return data if status < 400 else data | {"http_status": status}

    if op == "edit":
        hydrated = append_hydration_to_prompt(
            str(body.prompt or "").strip(),
            upload_row,
            use_hydration=body.use_hydration,
            hydration_lane=body.hydration_lane,
            max_len=950,
        )
        payload = {
            "prompt": hydrated,
            "image_base64": image_b64,
            "format": body.format,
            "upload_id": str(uid),
        }
        if body.mask_url:
            payload["mask_url"] = body.mask_url
        if body.mask_base64:
            payload["mask_base64"] = body.mask_base64
        if body.support_image_url:
            payload["support_image_url"] = body.support_image_url
        if body.support_image_base64:
            payload["support_image_base64"] = body.support_image_base64
        if body.studio_variant_id:
            payload["studio_variant_id"] = body.studio_variant_id
        normalize_url_or_base64(payload, "mask_url", "mask_base64")
        normalize_url_or_base64(payload, "support_image_url", "support_image_base64")
        status, data = await pikzels_v2_post("/v2/thumbnail/edit", payload)
        await _studio_usage_log(user["id"], "edit", status, _usage_meta())
        await _maybe_notify_pikzels_discord(user["id"], "edit_from_upload", status, data, upload_id=str(uid))
        return data if status < 400 else data | {"http_status": status}

    sc_title = (body.title or "").strip() or str(upload_row.get("title") or upload_row.get("ai_title") or "")[:200]
    payload = {
        "image_base64": image_b64,
        "upload_id": str(uid),
    }
    if sc_title:
        payload["title"] = sc_title
    if body.studio_variant_id:
        payload["studio_variant_id"] = body.studio_variant_id
    status, data = await pikzels_v2_post("/v2/thumbnail/score", payload)
    await _studio_usage_log(user["id"], "score", status, _usage_meta())
    score_src = None
    for rk in ("thumbnail_r2_key", "processed_r2_key"):
        raw_key = upload_row.get(rk)
        if not raw_key:
            continue
        try:
            u = generate_presigned_download_url(str(raw_key).strip(), ttl=3600)
        except Exception:
            u = None
        if isinstance(u, str) and u.startswith("https://"):
            score_src = u[:2048]
            break
    await _maybe_notify_pikzels_discord(
        user["id"],
        "score_from_upload",
        status,
        data,
        upload_id=str(uid),
        source_image_url=score_src,
    )
    out = data if status < 400 else data | {"http_status": status}
    if status < 400 and isinstance(data, dict):
        try:
            async with core.state.db_pool.acquire() as conn:
                analysis = await persist_score_analysis(
                    conn,
                    user_id=str(user["id"]),
                    upload_id=str(uid),
                    frame_source=body.frame_source,
                    title=sc_title,
                    response_data=data,
                    persona_id=body.persona,
                )
            if isinstance(out, dict):
                out = dict(out)
                out["analysis"] = analysis
                out["analysis_id"] = analysis.get("analysis_id")
        except Exception:
            logger.warning("pikzels analyzer persist score failed", exc_info=True)
    return out


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
    pkz_id = _pikzels_response_pikzonality_id(data)
    if status < 400 and pkz_id:
        try:
            async with core.state.db_pool.acquire() as conn:
                await _record_pikzels_user_asset(
                    conn,
                    user_id=str(user["id"]),
                    kind="persona",
                    pikzels_pikzonality_id=pkz_id,
                    name=payload.get("name") or "",
                    metadata={"source": "pikzels_v2_proxy"},
                )
        except Exception:
            logger.debug("record generic Pikzels persona asset failed", exc_info=True)
    return data if status < 400 else data | {"http_status": status}


@router.post("/api/thumbnail-studio/pikzels-v2/style")
async def ts_pikzels_style(body: PikzelsV2PikzonalityBody, user: dict = Depends(get_current_user)):
    await _pikzels_debit(user["id"], "style")
    payload = body.model_dump(exclude_none=True)
    trim_pikzonality_images(payload)
    status, data = await pikzels_v2_post("/v2/pikzonality/style", payload)
    await _studio_usage_log(user["id"], "style", status, _pikzels_telemetry_meta(body))
    pkz_id = _pikzels_response_pikzonality_id(data)
    if status < 400 and pkz_id:
        try:
            async with core.state.db_pool.acquire() as conn:
                await _record_pikzels_user_asset(
                    conn,
                    user_id=str(user["id"]),
                    kind="style",
                    pikzels_pikzonality_id=pkz_id,
                    name=payload.get("name") or "",
                    metadata={"source": "pikzels_v2_proxy"},
                )
        except Exception:
            logger.debug("record generic Pikzels style asset failed", exc_info=True)
    return data if status < 400 else data | {"http_status": status}


@router.get("/api/thumbnail-studio/pikzels-v2/pikzonality/{pikzonality_id}")
async def ts_pikzels_poll(pikzonality_id: str, user: dict = Depends(get_current_user_readonly)):
    # Polling is read-only at Pikzels; do not debit PUT/AIC on each poll.
    path = V2_PIKZONALITY_BY_ID.replace("{id}", pikzonality_id)
    status, data = await pikzels_v2_get(path)
    if status >= 400:
        raise HTTPException(status_code=min(status, 599), detail=data)
    return data


@router.patch("/api/thumbnail-studio/pikzels-v2/pikzonality/{pikzonality_id}")
async def ts_pikzels_update_pikzonality(
    pikzonality_id: str,
    body: PikzelsV2PikzonalityUpdateBody,
    user: dict = Depends(get_current_user),
):
    # Updating special instructions is management, not generation; do not debit thumbnail credits.
    payload = body.model_dump(exclude_none=False)
    status, data = await pikzels_v2_patch(V2_PIKZONALITY_BY_ID.replace("{id}", pikzonality_id), payload)
    await _studio_usage_log(user["id"], "pikzonality_update", status, _pikzels_telemetry_meta(body))
    return data if status < 400 else data | {"http_status": status}


class PikzelsAnalyzerStatusBody(BaseModel):
    status: Literal["open", "saved", "applied", "dismissed", "done"] = "saved"


class PikzelsAnalyzerApplyFixBody(BaseModel):
    persona: Optional[str] = None
    use_targeted_prompt: bool = True
    re_score: bool = True


class PikzelsAnalyzerBatchScoreBody(BaseModel):
    limit: int = Field(default=15, ge=1, le=30)
    persona: Optional[str] = None
    rescore_recent: bool = False


@router.get("/api/thumbnail-studio/pikzels-analyzer/analyses")
async def ts_pikzels_analyzer_list(
    upload_id: Optional[str] = Query(default=None, max_length=64),
    status: Optional[str] = Query(default=None, max_length=24),
    limit: int = Query(30, ge=1, le=100),
    user: dict = Depends(get_current_user_readonly),
):
    """List persisted Pikzels analyzer runs (newest first)."""
    if status and status not in RECOMMENDATION_STATUSES:
        raise HTTPException(400, detail="invalid status")
    async with core.state.db_pool.acquire() as conn:
        rows = await list_analyses(
            conn,
            user_id=str(user["id"]),
            upload_id=upload_id,
            status=status,
            limit=limit,
        )
    return {"analyses": rows, "limit": limit}


@router.get("/api/thumbnail-studio/pikzels-analyzer/analyses/{analysis_id}")
async def ts_pikzels_analyzer_get(
    analysis_id: str,
    user: dict = Depends(get_current_user_readonly),
):
    async with core.state.db_pool.acquire() as conn:
        row = await get_analysis_for_user(conn, str(user["id"]), analysis_id)
    if not row:
        raise HTTPException(404, detail="analysis not found")
    return row


@router.get("/api/thumbnail-studio/pikzels-analyzer/uploads/{upload_id}/latest")
async def ts_pikzels_analyzer_latest_for_upload(
    upload_id: str,
    user: dict = Depends(get_current_user_readonly),
):
    """Restore the most recent analyzer result for an upload after page refresh."""
    async with core.state.db_pool.acquire() as conn:
        row = await get_latest_analysis_for_upload(conn, str(user["id"]), upload_id)
    if not row:
        return {"analysis": None}
    return {"analysis": row}


@router.patch("/api/thumbnail-studio/pikzels-analyzer/analyses/{analysis_id}")
async def ts_pikzels_analyzer_patch_status(
    analysis_id: str,
    body: PikzelsAnalyzerStatusBody,
    user: dict = Depends(get_current_user),
):
    async with core.state.db_pool.acquire() as conn:
        row = await update_recommendation_status(
            conn,
            user_id=str(user["id"]),
            analysis_id=analysis_id,
            status=body.status,
        )
    if not row:
        raise HTTPException(404, detail="analysis not found")
    return row


@router.post("/api/thumbnail-studio/pikzels-analyzer/analyses/{analysis_id}/save-recommendation")
async def ts_pikzels_analyzer_save_recommendation(
    analysis_id: str,
    user: dict = Depends(get_current_user),
):
    async with core.state.db_pool.acquire() as conn:
        row = await update_recommendation_status(
            conn,
            user_id=str(user["id"]),
            analysis_id=analysis_id,
            status="saved",
        )
    if not row:
        raise HTTPException(404, detail="analysis not found")
    return row


@router.post("/api/thumbnail-studio/pikzels-analyzer/analyses/{analysis_id}/apply-fix")
async def ts_pikzels_analyzer_apply_fix(
    analysis_id: str,
    body: PikzelsAnalyzerApplyFixBody,
    user: dict = Depends(get_current_user),
):
    await _pikzels_debit(user["id"], "one_click_fix")
    if body.persona:
        async with core.state.db_pool.acquire() as conn:
            if not await _user_owns_pikzels_pikzonality(conn, str(user["id"]), "persona", body.persona):
                raise HTTPException(
                    403,
                    detail={
                        "code": "pikzels_pikzonality_not_owned",
                        "message": "That Pikzels persona is not registered to your account.",
                    },
                )
    try:
        async with core.state.db_pool.acquire() as conn:
            row = await apply_fix_to_analysis(
                conn,
                user_id=str(user["id"]),
                analysis_id=analysis_id,
                persona=body.persona,
                use_targeted_prompt=body.use_targeted_prompt,
                re_score=body.re_score,
            )
    except ValueError as e:
        code = str(e)
        if code == "analysis_not_found":
            raise HTTPException(404, detail=code) from e
        raise HTTPException(502, detail=code) from e
    await _studio_usage_log(
        user["id"],
        "pikzels_analyzer_apply_fix",
        200,
        {"analysis_id": analysis_id, "persona": body.persona},
    )
    return row


@router.post("/api/thumbnail-studio/pikzels-analyzer/analyses/{analysis_id}/save-as-thumbnail")
async def ts_pikzels_analyzer_save_thumbnail(
    analysis_id: str,
    user: dict = Depends(get_current_user),
):
    try:
        async with core.state.db_pool.acquire() as conn:
            out = await save_fix_as_upload_thumbnail(
                conn,
                user_id=str(user["id"]),
                analysis_id=analysis_id,
            )
    except ValueError as e:
        code = str(e)
        if code == "analysis_not_found":
            raise HTTPException(404, detail=code) from e
        raise HTTPException(400, detail=code) from e
    await _studio_usage_log(
        user["id"],
        "pikzels_analyzer_save_thumbnail",
        200,
        {"analysis_id": analysis_id, "upload_id": out.get("upload_id")},
    )
    return out


@router.post("/api/thumbnail-studio/pikzels-analyzer/analyses/{analysis_id}/generate-titles")
async def ts_pikzels_analyzer_generate_titles(
    analysis_id: str,
    user: dict = Depends(get_current_user),
):
    await _pikzels_debit(user["id"], "titles")
    try:
        async with core.state.db_pool.acquire() as conn:
            row = await generate_titles_for_analysis(
                conn,
                user_id=str(user["id"]),
                analysis_id=analysis_id,
            )
    except ValueError as e:
        code = str(e)
        if code == "analysis_not_found":
            raise HTTPException(404, detail=code) from e
        raise HTTPException(502, detail=code) from e
    await _studio_usage_log(user["id"], "pikzels_analyzer_titles", 200, {"analysis_id": analysis_id})
    return row


@router.get("/api/thumbnail-studio/pikzels-analyzer/score-trend")
async def ts_pikzels_analyzer_score_trend(
    days: int = Query(90, ge=7, le=365),
    limit: int = Query(60, ge=1, le=200),
    user: dict = Depends(get_current_user_readonly),
):
    async with core.state.db_pool.acquire() as conn:
        trend = await fetch_score_trend(conn, user_id=str(user["id"]), days=days, limit=limit)
    return trend


@router.get("/api/thumbnail-studio/pikzels-analyzer/recommendations")
async def ts_pikzels_analyzer_recommendations(
    limit: int = Query(12, ge=1, le=50),
    user: dict = Depends(get_current_user_readonly),
):
    """Cross-surface actionable recommendations (saved to-dos + low-score open analyses)."""
    async with core.state.db_pool.acquire() as conn:
        feed = await fetch_actionable_recommendations(conn, user_id=str(user["id"]), limit=limit)
    return feed


@router.post("/api/thumbnail-studio/pikzels-analyzer/batch-score")
async def ts_pikzels_analyzer_batch_score(
    body: PikzelsAnalyzerBatchScoreBody,
    user: dict = Depends(get_current_user),
):
    """Score recent uploads and return them ranked weakest-first."""

    async def _debit_score() -> None:
        await _pikzels_debit(user["id"], "score")

    async with core.state.db_pool.acquire() as conn:
        result = await batch_score_user_uploads(
            conn,
            user_id=str(user["id"]),
            limit=body.limit,
            persona_id=body.persona,
            rescore_recent=body.rescore_recent,
            on_before_score=_debit_score,
        )
    return result


@router.get("/api/thumbnail-studio/pikzels-analyzer/analyses/{analysis_id}/ab-export")
async def ts_pikzels_analyzer_ab_export(
    analysis_id: str,
    user: dict = Depends(get_current_user),
):
    """ZIP comparison pack for one analyzer run (before/after scores + metadata)."""
    try:
        async with core.state.db_pool.acquire() as conn:
            out = await export_analysis_ab_pack(
                conn,
                user_id=str(user["id"]),
                analysis_id=analysis_id,
            )
    except ValueError as e:
        code = str(e)
        if code == "analysis_not_found":
            raise HTTPException(404, detail="analysis not found") from e
        if code == "r2_not_configured":
            raise HTTPException(503, detail="Object storage is not configured; cannot build export pack.") from e
        raise HTTPException(400, detail=code) from e
    except Exception as e:
        logger.warning("pikzels analyzer ab-export failed analysis=%s: %s", analysis_id, e)
        raise HTTPException(503, detail="Could not build comparison pack.") from e
    return out


@router.delete("/api/thumbnail-studio/pikzels-v2/pikzonality/{pikzonality_id}")
async def ts_pikzels_delete_pikzonality(pikzonality_id: str, user: dict = Depends(get_current_user)):
    # Deleting a saved Pikzels asset is management, not generation; do not debit thumbnail credits.
    status, data = await pikzels_v2_delete(V2_PIKZONALITY_BY_ID.replace("{id}", pikzonality_id))
    await _studio_usage_log(user["id"], "pikzonality_delete", status, {"pikzonality_id": pikzonality_id})
    if status < 400:
        try:
            async with core.state.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE pikzels_user_assets
                    SET status = 'deleted', updated_at = NOW()
                    WHERE user_id = $1::uuid AND pikzels_pikzonality_id = $2
                    """,
                    str(user["id"]),
                    pikzonality_id,
                )
        except Exception:
            logger.debug("mark Pikzels asset deleted failed", exc_info=True)
    return data if status < 400 else data | {"http_status": status}
