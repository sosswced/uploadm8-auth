"""
On-demand thumbnail regeneration for the uploads API.

Extracts a base frame with FFmpeg, then when the account is allowed to use styled
thumbnails and Pikzels is configured, runs the same studio → template stack as
``stages.thumbnail_stage`` (without GPT brief / image-edit, using a minimal brief).
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncpg
from botocore.exceptions import ClientError

from core.config import R2_BUCKET_NAME
from core.content_attribution import normalize_thumbnail_render_pipeline
from core.helpers import coerce_jsonb_dict
from core.thumbnail_text import clean_thumbnail_headline, is_generic_thumbnail_headline
from core.r2 import _normalize_r2_key, get_s3_client
from stages import db as db_stage
from stages.ai_service_costs import user_pref_ai_service_enabled
from stages.context import JobContext
from stages.entitlements import get_entitlements_from_user
from stages.pikzels_api import render_thumbnail_with_studio_renderer

logger = logging.getLogger(__name__)


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


def _minimal_brief(*, title: str) -> Dict[str, Any]:
    headline = (
        clean_thumbnail_headline((title or "").strip(), max_words=5, max_chars=24) or "VIDEO HIGHLIGHT"
    )
    if is_generic_thumbnail_headline(headline):
        headline = "VIDEO HIGHLIGHT"
    return {
        "selected_headline": headline,
        "headline_options": [],
        "badge_text": "",
        "badge_style": "red",
        "directional_element": "circle",
        "props": [],
        "emotion_cue": "excited",
        "color_mood": "red_black",
        "platform_plan": {
            "youtube": {"enabled": True, "canvas": "16:9"},
            "instagram": {"enabled": True, "canvas": "9:16", "safe_center_pct": 60},
            "facebook": {"enabled": True, "canvas": "9:16", "safe_center_pct": 60},
            "tiktok": {"enabled": True, "canvas": "9:16", "thumb_offset_seconds": 1.5},
        },
        "notes": "API regenerate brief",
    }


def _ffmpeg_frame(video_path: Path, thumb_path: Path) -> tuple[float, bool]:
    """Extract one JPEG at ~30% duration. Returns (offset_seconds, success)."""
    duration = 10.0
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(video_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if probe.returncode == 0:
            data = json.loads(probe.stdout or "{}")
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    duration = float(stream.get("duration", 10) or 10)
                    break
    except Exception:
        duration = 10.0
    offset = max(0.5, duration * 0.30)
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{offset:.3f}",
                "-i",
                str(video_path),
                "-vframes",
                "1",
                "-q:v",
                "2",
                "-vf",
                "scale=1080:-2",
                str(thumb_path),
            ],
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0 or not thumb_path.exists():
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    "1",
                    "-i",
                    str(video_path),
                    "-vframes",
                    "1",
                    "-q:v",
                    "2",
                    "-vf",
                    "scale=1080:-2",
                    str(thumb_path),
                ],
                capture_output=True,
                timeout=30,
            )
    except Exception as e:
        logger.warning("ffmpeg thumbnail extract failed: %s", e)
        return offset, False
    return offset, thumb_path.exists()


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
    r2_key = upload_row.get("processed_r2_key") or upload_row.get("r2_key")
    if not r2_key:
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
        norm = _normalize_r2_key(str(r2_key))
        try:
            await asyncio.to_thread(s3.download_file, R2_BUCKET_NAME, norm, str(video_path))
        except ClientError as e:
            err = (e.response or {}).get("Error") or {}
            code = str(err.get("Code") or "")
            http = (e.response or {}).get("ResponseMetadata", {}) or {}
            status = http.get("HTTPStatusCode")
            if status == 404 or code in ("404", "NoSuchKey", "NotFound"):
                raise ValueError("video_not_in_storage") from e
            raise

        offset, ok = await asyncio.to_thread(_ffmpeg_frame, video_path, base_frame)
        if not ok:
            raise ValueError("ffmpeg_failed")

        render_method = "ffmpeg"
        primary: Optional[Path] = None
        platform_files: Dict[str, Path] = {}

        ctx_for_brief = JobContext(
            job_id=str(upload_id),
            upload_id=str(upload_id),
            user_id=str(user_id),
            title=title or "",
            filename=str(upload_row.get("filename") or ""),
            platforms=[str(p) for p in (upload_row.get("platforms") or []) if str(p).strip()],
            entitlements=ent,
            user_settings=settings,
        )
        category = _detect_category(ctx_for_brief)
        brief = _sanitize_thumbnail_brief(
            ctx_for_brief,
            _minimal_brief(title=title),
            category,
            note="API regenerate brief",
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
        designer_on = user_pref_ai_service_enabled(settings, "thumbnail_ai", default=True)
        can_ai = bool(getattr(ent, "can_ai", False))
        can_ai_designer = bool(OPENAI_API_KEY and can_ai and designer_on)
        can_ai_style = bool(getattr(ent, "can_ai_thumbnail_styling", False))
        ai_edit_ok = bool(can_ai_designer and can_ai_style and OPENAI_API_KEY)

        effective_render_pipeline = render_pipeline_pref
        if studio_ok and render_pipeline_pref == "template":
            effective_render_pipeline = "auto"

        render_steps = _thumbnail_styled_render_order(
            effective_render_pipeline,
            studio_ok=studio_ok,
            ai_edit_ok=ai_edit_ok,
        )

        plat_lower = [str(p).strip().lower() for p in (upload_row.get("platforms") or []) if str(p).strip()]
        platforms_to_render: List[str] = [
            p
            for p in ("youtube", "instagram", "facebook", "tiktok")
            if (brief.get("platform_plan", {}).get(p, {}).get("enabled", True)) and p in plat_lower
        ]
        if run_styled and not platforms_to_render:
            platforms_to_render = ["youtube"]

        if run_styled and platforms_to_render:
            persona_api, studio_opts = _studio_persona_for_request(settings)
            for platform in platforms_to_render:
                out_path = tmp_path / f"thumb_styled_{platform}.jpg"
                step_ok = False
                for step in render_steps:
                    if step == "studio" and studio_ok:
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
                            render_method = "pikzels"
                    elif step == "template":
                        step_ok = _render_template_thumbnail(base_frame, brief, platform, out_path)
                        if step_ok:
                            render_method = "template"
                    if step_ok:
                        break
                if step_ok:
                    platform_files[platform] = out_path
                    if primary is None or platform == "youtube":
                        primary = out_path

        upload_final = primary if primary and primary.exists() else base_frame

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
                json.dumps({"platform_thumbnail_r2_keys": platform_r2_keys}) if platform_r2_keys else "{}",
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
        }


def should_skip_regenerate(*, thumbnail_r2_key: Optional[str], force: bool) -> bool:
    return bool(thumbnail_r2_key) and not force
