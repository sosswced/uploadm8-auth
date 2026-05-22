"""
On-demand thumbnail regeneration for the uploads API.

Extracts a base frame with FFmpeg, then when the account is allowed to use styled
thumbnails and Pikzels is configured, runs the same **Pikzels-only** studio stack
as ``stages.thumbnail_stage`` (optional Pikzels edit pass — no OpenAI image-edit
or PIL template for styled output).

Reuses persisted ``output_artifacts`` (``thumbnail_brief_json``, ``hydration_payload``,
``scene_story``, ``timeline_story``) plus caption voice from settings so Pikzels
prompts match the narrative shown in user/admin UI. Merges the user's default
thumbnail strategy and canonical hydration into the brief, then persists the
final brief JSON back to ``output_artifacts``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncpg
from botocore.exceptions import ClientError

from core.config import R2_BUCKET_NAME
from core.content_attribution import normalize_thumbnail_render_pipeline
from core.helpers import coerce_jsonb_dict
from core.r2 import _normalize_r2_key, generate_presigned_download_url, get_s3_client, r2_object_exists
from stages import db as db_stage
from stages.context import JobContext
from stages.entitlements import get_entitlements_from_user
from stages.pikzels_api import refine_thumbnail_with_pikzels_edit, render_thumbnail_with_studio_renderer
from stages.thumbnail_stage import (
    _detect_category,
    _hydration_pikzels_edit_enabled,
    _render_template_thumbnail,
    _sanitize_thumbnail_brief,
    _studio_persona_for_request,
    _thumbnail_hydration_edit_prompt,
    _thumbnail_styled_render_order,
    pikzels_studio_eligible_for_styled_thumbnail,
    styled_thumbnail_platform_targets,
)
from services.thumbnail_brief_pipeline import (
    attach_youtube_support_image_from_ctx,
    copy_brief_for_persistence,
    finalize_styled_thumbnail_brief,
    merge_story_voice_youtube_into_brief,
    minimal_thumbnail_brief,
)
from services.thumbnail_sticker_pack import (
    build_sticker_pack,
    sticker_pack_from_json,
    sticker_pack_to_json,
)
from services.thumbnail_sticker_render import render_sticker_composite, sticker_composite_enabled

logger = logging.getLogger(__name__)

# Terminal-ish uploads: safe to re-extract a frame from stored video for thumbnail repair.
THUMB_REPAIR_STATUSES = frozenset({"completed", "partial", "succeeded"})


def _overlay_upload_user_preferences(settings: Dict[str, Any], raw: Any) -> None:
    if raw is None:
        return
    if isinstance(raw, str):
        snap = coerce_jsonb_dict(raw, default={})
    elif isinstance(raw, dict):
        snap = dict(raw)
    else:
        return
    for k, v in snap.items():
        if v is not None:
            settings[k] = v


def _output_artifacts_dict(upload_row: Dict[str, Any]) -> Dict[str, Any]:
    raw = upload_row.get("output_artifacts")
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            j = json.loads(raw)
            return dict(j) if isinstance(j, dict) else {}
        except Exception:
            return {}
    return {}


def _json_artifact_value(arts: Dict[str, Any], key: str) -> Any:
    v = arts.get(key)
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return v
    if isinstance(v, str) and v.strip():
        try:
            return json.loads(v)
        except Exception:
            return None
    return None


def _extract_base_frame_sync(video_path: Path, thumb_path: Path) -> tuple[float, bool]:
    """Decode one JPEG via imageio (no PATH ffmpeg). Returns (offset_seconds, success)."""
    from services.thumbnail_frame_extract import extract_frame_default_offset, extract_jpeg_at_offset, video_duration_seconds

    dur = video_duration_seconds(video_path)
    offset = max(0.5, dur * 0.30)
    if extract_jpeg_at_offset(video_path, thumb_path, offset):
        return offset, True
    return extract_frame_default_offset(video_path, thumb_path)


async def regenerate_upload_thumbnail(
    *,
    db_pool: asyncpg.Pool,
    upload_id: str,
    user_id: str,
    upload_row: Dict[str, Any],
    user_row: Dict[str, Any],
    force: bool = False,
) -> Dict[str, Any]:
    """
    Download source video, extract a frame, optionally run Pikzels + template styled stack,
    upload to R2, update ``thumbnail_r2_key`` and ``thumb_*`` processed_assets keys.

    Returns a dict suitable for JSON: thumbnail_url, r2_key, generated, method, offset_seconds.
    """
    keys_to_try: List[str] = []
    for cand in (upload_row.get("processed_r2_key"), upload_row.get("r2_key")):
        if not cand:
            continue
        s = str(cand).strip()
        if s and s not in keys_to_try:
            keys_to_try.append(s)
    if not keys_to_try:
        raise ValueError("no_video_key")

    settings = await db_stage.load_user_settings(db_pool, str(user_id))
    _overlay_upload_user_preferences(settings, upload_row.get("user_preferences"))
    await db_stage.merge_pikzels_thumbnail_persona_id(db_pool, str(user_id), settings)

    overrides = await db_stage.load_user_entitlement_overrides(db_pool, str(user_id))
    ent = get_entitlements_from_user(user_row, overrides)

    title = str(
        upload_row.get("title")
        or upload_row.get("ai_generated_title")
        or upload_row.get("ai_title")
        or ""
    ).strip()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video_path = tmp_path / "video.mp4"
        base_frame = tmp_path / "base_frame.jpg"

        s3 = get_s3_client()
        last_miss: Optional[ClientError] = None
        downloaded = False
        for cand in keys_to_try:
            norm = _normalize_r2_key(str(cand))
            try:
                await asyncio.to_thread(s3.download_file, R2_BUCKET_NAME, norm, str(video_path))
                downloaded = True
                break
            except ClientError as e:
                err = (e.response or {}).get("Error") or {}
                code = str(err.get("Code") or "")
                http = (e.response or {}).get("ResponseMetadata", {}) or {}
                status = http.get("HTTPStatusCode")
                if status == 404 or code in ("404", "NoSuchKey", "NotFound"):
                    last_miss = e
                    continue
                raise
        if not downloaded:
            if last_miss:
                raise ValueError("video_not_in_storage") from last_miss
            raise ValueError("video_not_in_storage")

        offset, ok = await asyncio.to_thread(_extract_base_frame_sync, video_path, base_frame)
        if not ok:
            raise ValueError("ffmpeg_failed")

        render_method = "ffmpeg"
        primary: Optional[Path] = None
        platform_files: Dict[str, Path] = {}

        arts = _output_artifacts_dict(upload_row)
        hp = _json_artifact_value(arts, "hydration_payload")
        saved_brief = _json_artifact_value(arts, "thumbnail_brief_json")
        if isinstance(saved_brief, dict) and saved_brief:
            brief_seed: Dict[str, Any] = dict(saved_brief)
            brief_note_tag = "regenerate.saved_brief_json"
        else:
            brief_seed = minimal_thumbnail_brief(title=title)
            brief_note_tag = "regenerate.minimal_plus_artifacts"

        brief_seed = merge_story_voice_youtube_into_brief(
            brief_seed,
            arts=arts,
            settings=settings,
            platform_results=upload_row.get("platform_results"),
        )

        ctx_for_brief = JobContext(
            job_id=str(upload_id),
            upload_id=str(upload_id),
            user_id=str(user_id),
            title=title or str(upload_row.get("title") or "")[:400],
            filename=str(upload_row.get("filename") or ""),
            caption=str(upload_row.get("caption") or "").strip(),
            platforms=[str(p) for p in (upload_row.get("platforms") or []) if str(p).strip()],
            entitlements=ent,
            user_settings=settings,
        )
        ctx_for_brief.ai_title = str(
            upload_row.get("ai_generated_title") or upload_row.get("ai_title") or ""
        ).strip()
        ctx_for_brief.ai_caption = str(
            upload_row.get("ai_generated_caption") or upload_row.get("ai_caption") or ""
        ).strip()
        ctx_for_brief.hydration_payload = hp if isinstance(hp, dict) else None
        ctx_for_brief.output_artifacts = dict(arts)

        category = _detect_category(ctx_for_brief)
        if isinstance(hp, dict):
            hc = str(hp.get("category") or "").strip().lower()
            if hc:
                category = hc

        brief = _sanitize_thumbnail_brief(ctx_for_brief, brief_seed, category, note=brief_note_tag)
        brief = finalize_styled_thumbnail_brief(
            ctx_for_brief,
            brief,
            category,
            settings,
            studio_render_report=None,
            evidence_anchor_fallbacks=False,
        )
        brief = attach_youtube_support_image_from_ctx(
            brief,
            ctx_for_brief,
            platform_results_override=upload_row.get("platform_results"),
        )

        render_pipeline_pref = normalize_thumbnail_render_pipeline(settings)
        can_custom = bool(getattr(ent, "can_custom_thumbnails", False))
        styled_on = bool(settings.get("styled_thumbnails", settings.get("styledThumbnails", True)))
        run_styled = bool(can_custom and styled_on and render_pipeline_pref != "none")

        studio_ok = bool(
            pikzels_studio_eligible_for_styled_thumbnail(
                settings, ent, require_auto_thumbnails=False
            )
            and isinstance(brief, dict)
        )
        ai_edit_ok = False

        effective_render_pipeline = render_pipeline_pref
        if studio_ok and render_pipeline_pref == "template":
            effective_render_pipeline = "auto"

        render_steps = _thumbnail_styled_render_order(
            effective_render_pipeline,
            studio_ok=studio_ok,
            ai_edit_ok=ai_edit_ok,
            sticker_ok=sticker_composite_enabled(),
        )

        saved_stickers = sticker_pack_from_json(arts.get("sticker_pack_json"))
        sticker_pack = saved_stickers or (
            build_sticker_pack(ctx_for_brief, float(offset)) if sticker_composite_enabled() else []
        )
        arts["sticker_pack_json"] = sticker_pack_to_json(sticker_pack)

        platforms_to_render = styled_thumbnail_platform_targets(
            upload_row.get("platforms"),
            platform_plan=brief.get("platform_plan") or {},
        )
        if run_styled and not platforms_to_render:
            platforms_to_render = ["youtube"]

        if run_styled and platforms_to_render:
            persona_api, studio_opts = _studio_persona_for_request(settings)
            for platform in platforms_to_render:
                out_path = tmp_path / f"thumb_styled_{platform}.jpg"
                step_ok = False
                last_method = render_method
                for step in render_steps:
                    if step == "sticker" and sticker_composite_enabled():
                        from services.platform_colors import platform_color_for, resolve_platform_colors

                        _plat_colors = resolve_platform_colors(settings)
                        step_ok = await render_sticker_composite(
                            base_frame,
                            brief,
                            platform,
                            out_path,
                            sticker_pack,
                            platform_color=platform_color_for(_plat_colors, platform),
                            accent_color=_plat_colors.get("accent"),
                        )
                        if step_ok:
                            last_method = "sticker_composite"
                    elif step == "studio" and studio_ok:
                        step_ok = await render_thumbnail_with_studio_renderer(
                            base_frame,
                            brief,
                            platform,
                            out_path,
                            upload_id=str(upload_id),
                            category=category,
                            persona=persona_api,
                            options=studio_opts,
                            job_context=ctx_for_brief,
                        )
                        if step_ok:
                            last_method = "pikzels"
                            hp_edit = _thumbnail_hydration_edit_prompt(brief or {})
                            skip_hydration_edit = bool((brief or {}).get("_uploadm8_dashcam_pov"))
                            if hp_edit and _hydration_pikzels_edit_enabled() and not skip_hydration_edit:
                                try:
                                    await refine_thumbnail_with_pikzels_edit(
                                        out_path,
                                        hp_edit,
                                        platform=platform,
                                        upload_id=str(upload_id),
                                    )
                                except Exception as _pe:
                                    logger.debug("regenerate pikzels edit skipped: %s", _pe)
                    elif step == "template":
                        from services.platform_colors import platform_color_for, resolve_platform_colors

                        _plat_colors = resolve_platform_colors(settings)
                        step_ok = _render_template_thumbnail(
                            base_frame,
                            brief,
                            platform,
                            out_path,
                            platform_color=platform_color_for(_plat_colors, platform),
                            accent_color=_plat_colors.get("accent"),
                        )
                        if step_ok:
                            last_method = "template"
                    if step_ok:
                        break
                if step_ok:
                    render_method = last_method
                    platform_files[platform] = out_path
                    if primary is None or platform == "youtube":
                        primary = out_path

        from stages.image_format import ensure_jpeg_file

        upload_final = primary if primary and primary.exists() else base_frame
        ensure_jpeg_file(upload_final)

        thumb_r2_key = f"thumbnails/{user_id}/{upload_id}/thumbnail.jpg"
        await asyncio.to_thread(
            s3.upload_file,
            str(upload_final),
            R2_BUCKET_NAME,
            thumb_r2_key,
            ExtraArgs={"ContentType": "image/jpeg"},
        )

        platform_r2_keys: Dict[str, str] = {}
        for plat, local in platform_files.items():
            if not local.exists():
                continue
            ensure_jpeg_file(local)
            pk = f"thumbnails/{user_id}/{upload_id}/{plat}.jpg"
            try:
                await asyncio.to_thread(
                    s3.upload_file,
                    str(local),
                    R2_BUCKET_NAME,
                    pk,
                    ExtraArgs={"ContentType": "image/jpeg"},
                )
                platform_r2_keys[plat] = pk
            except Exception as e:
                logger.debug("platform thumb upload %s: %s", plat, e)

        oa_merge: Dict[str, Any] = {}
        if platform_r2_keys:
            oa_merge["platform_thumbnail_r2_keys"] = json.dumps(platform_r2_keys, default=str)
        try:
            oa_merge["thumbnail_brief_json"] = json.dumps(copy_brief_for_persistence(brief), default=str)[:48000]
        except Exception:
            pass
        if sticker_composite_enabled() and arts.get("sticker_pack_json"):
            oa_merge["sticker_pack_json"] = arts["sticker_pack_json"]

        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE uploads
                SET thumbnail_r2_key = $1,
                    output_artifacts = COALESCE(output_artifacts, '{}'::jsonb) || $4::jsonb,
                    updated_at = NOW()
                WHERE id = $2 AND user_id = $3
                """,
                thumb_r2_key,
                upload_id,
                user_id,
                json.dumps(oa_merge) if oa_merge else "{}",
            )

        if platform_files:
            assets = await db_stage.load_processed_assets(db_pool, upload_id)
            merged = dict(assets) if assets else {}
            for plat in platform_files:
                merged[f"thumb_{plat}"] = f"thumbnails/{user_id}/{upload_id}/{plat}.jpg"
            await db_stage.save_processed_assets(db_pool, upload_id, merged)

        url: Optional[str] = None
        try:
            url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": R2_BUCKET_NAME, "Key": thumb_r2_key},
                ExpiresIn=3600,
            )
        except Exception:
            pass

        return {
            "thumbnail_url": url,
            "r2_key": thumb_r2_key,
            "generated": True,
            "method": render_method,
            "offset_seconds": offset,
            "platforms_rendered": list(platform_files.keys()),
            "brief_source": "saved_thumbnail_brief_json"
            if isinstance(saved_brief, dict) and saved_brief
            else "minimal_plus_artifacts",
        }


async def ensure_upload_thumbnail_resident(
    *,
    db_pool: asyncpg.Pool,
    user_id: str,
    upload_row: Dict[str, Any],
    user_row: Dict[str, Any],
) -> tuple[Optional[str], Optional[str]]:
    """
    Return ``(presigned_thumbnail_url, thumbnail_r2_key)`` for dashboard/detail views.

    When ``thumbnail_r2_key`` is set but the object was removed from R2, regenerates from
    the stored video for uploads in a terminal state (completed / partial / succeeded).
    """
    upload_id = str(upload_row.get("id") or "").strip()
    if not upload_id:
        return None, None

    status = str(upload_row.get("status") or "")
    thumb_key = upload_row.get("thumbnail_r2_key")
    sk = str(thumb_key).strip() if thumb_key else ""

    if sk:
        exists = await asyncio.to_thread(r2_object_exists, sk)
        if exists:
            try:
                url = await asyncio.to_thread(generate_presigned_download_url, sk, 3600)
                return (url or None, sk)
            except Exception:
                return None, sk

    if status not in THUMB_REPAIR_STATUSES:
        return None, sk or None

    try:
        out = await regenerate_upload_thumbnail(
            db_pool=db_pool,
            upload_id=upload_id,
            user_id=user_id,
            upload_row=upload_row,
            user_row=user_row,
            force=bool(sk),
        )
        rk = out.get("r2_key")
        return out.get("thumbnail_url"), (str(rk) if rk else None)
    except Exception as e:
        logger.warning("ensure_upload_thumbnail_resident failed upload_id=%s: %s", upload_id, e)
        return None, sk or None


def should_skip_regenerate(*, thumbnail_r2_key: Optional[str], force: bool) -> bool:
    return bool(thumbnail_r2_key) and not force
