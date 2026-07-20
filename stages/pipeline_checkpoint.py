"""
Pipeline checkpoint / resume for the upload worker.

Persists durable stages to R2 + uploads.output_artifacts (``pipeline_resume``)
so a failed or stale-recovered job can continue without re-downloading the
source or re-running FFmpeg transcode. Stages:

- ``post_telemetry`` — telemetry/trill snapshot (+ optional source re-download).
- ``post_transcode`` — per-platform MP4s under ``checkpoints/{user}/{upload}/``.
- ``post_audio`` — compact ``audio_context`` (+ same transcode R2 keys) to
  resume before Google Vision / downstream multimodal stages.
- ``post_caption`` — thumbnail keys on R2 (optional promotion from worker).

Resume is triggered by ``job_data["action"] == "retry"`` or
``job_data["resume_from_checkpoint"]`` (used by stale processing recovery).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from stages.context import JobContext, TelemetryData, TrillScore
from stages import r2 as r2_stage

logger = logging.getLogger("uploadm8-worker")

CHECKPOINT_VERSION = 1
RESUME_KEY = "pipeline_resume"
# JSON blob size cap for audio_context inside pipeline_resume (UTF-8 bytes)
AUDIO_CONTEXT_CHECKPOINT_MAX_BYTES = int(
    os.environ.get("AUDIO_CONTEXT_CHECKPOINT_MAX_BYTES", "150000")
)


def norm_platforms(platforms: Optional[List[str]]) -> List[str]:
    return sorted({str(p).strip().lower() for p in (platforms or []) if str(p).strip()})


def _output_artifacts_dict(upload_record: dict) -> dict:
    raw = upload_record.get("output_artifacts")
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            d = json.loads(raw)
            return dict(d) if isinstance(d, dict) else {}
        except Exception:
            return {}
    return {}


def load_resume(upload_record: dict) -> Optional[dict]:
    out = _output_artifacts_dict(upload_record)
    blob = out.get(RESUME_KEY)
    if isinstance(blob, str):
        try:
            blob = json.loads(blob)
        except Exception:
            return None
    if not isinstance(blob, dict):
        return None
    return blob


def checkpoint_matches_upload(cp: dict, upload_record: dict) -> bool:
    if int(cp.get("v", 0) or 0) != CHECKPOINT_VERSION:
        return False
    if (cp.get("source_r2_key") or "") != (upload_record.get("r2_key") or ""):
        return False
    cur = norm_platforms(upload_record.get("platforms") or [])
    saved = sorted(str(x).strip().lower() for x in (cp.get("platforms_norm") or []) if str(x).strip())
    if saved != cur:
        return False
    return True


def telemetry_to_dict(t: Optional[TelemetryData]) -> Optional[dict]:
    if t is None:
        return None
    from dataclasses import asdict

    return asdict(t)


def trill_to_dict(t: Optional[TrillScore]) -> Optional[dict]:
    if t is None:
        return None
    from dataclasses import asdict

    return asdict(t)


def restore_telemetry_from_dict(ctx: JobContext, telemetry: Optional[dict], trill: Optional[dict]) -> None:
    if telemetry and isinstance(telemetry, dict):
        try:
            fields = {k: telemetry[k] for k in TelemetryData.__dataclass_fields__ if k in telemetry}
            ctx.telemetry_data = TelemetryData(**fields)
            ctx.telemetry = ctx.telemetry_data
        except Exception as e:
            logger.debug(f"restore telemetry snapshot failed: {e}")
    if trill and isinstance(trill, dict):
        try:
            fields = {k: trill[k] for k in TrillScore.__dataclass_fields__ if k in trill}
            ctx.trill_score = TrillScore(**fields)
            ctx.trill = ctx.trill_score
        except Exception as e:
            logger.debug(f"restore trill snapshot failed: {e}")


def clip_audio_context_for_checkpoint(ac: Any) -> Dict[str, Any]:
    """Return a JSON-serializable audio_context dict capped for DB storage."""
    if not isinstance(ac, dict) or not ac:
        return {}
    try:
        slim: Dict[str, Any] = json.loads(json.dumps(ac, default=str))
    except Exception:
        return {}
    ts = slim.get("transcript_segments")
    if isinstance(ts, list) and len(ts) > 48:
        slim["transcript_segments"] = ts[:48]
    yn = slim.get("yamnet_events")
    if isinstance(yn, list) and len(yn) > 120:
        slim["yamnet_events"] = yn[:120]
    slim.pop("yamnet_scoreboard", None)
    t_struct = slim.get("transcript_structured")
    if isinstance(t_struct, dict) and len(json.dumps(t_struct, default=str)) > 40_000:
        slim.pop("transcript_structured", None)
    raw = json.dumps(slim, default=str).encode("utf-8")
    if len(raw) <= AUDIO_CONTEXT_CHECKPOINT_MAX_BYTES:
        return slim
    slim2 = {
        "transcript": str(slim.get("transcript") or "")[:8000],
        "fusion_narrative": str(slim.get("fusion_narrative") or "")[:1900],
        "gpt_audio_summary": str(slim.get("gpt_audio_summary") or "")[:2400],
        "music_title": str(slim.get("music_title") or "")[:400],
        "music_artist": str(slim.get("music_artist") or "")[:400],
        "music_genre": str(slim.get("music_genre") or "")[:200],
        "top_sound_class": str(slim.get("top_sound_class") or "")[:200],
        "sound_profile": str(slim.get("sound_profile") or "")[:1200],
        "transcript_language": slim.get("transcript_language"),
        "language": slim.get("language"),
        "music_detected": slim.get("music_detected"),
        "transcript_chars": slim.get("transcript_chars"),
        "suggested_keywords": (slim.get("suggested_keywords") or [])[:24]
        if isinstance(slim.get("suggested_keywords"), list)
        else [],
        "content_signals": (slim.get("content_signals") or [])[:16]
        if isinstance(slim.get("content_signals"), list)
        else [],
    }
    raw2 = json.dumps(slim2, default=str).encode("utf-8")
    if len(raw2) > AUDIO_CONTEXT_CHECKPOINT_MAX_BYTES:
        slim2["transcript"] = str(slim2.get("transcript") or "")[:4000]
    return slim2


def restore_audio_context_from_dict(ctx: JobContext, data: Any) -> None:
    if not data:
        ctx.audio_context = dict(getattr(ctx, "audio_context", None) or {})
        return
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            ctx.audio_context = {}
            return
    if not isinstance(data, dict):
        ctx.audio_context = {}
        return
    ctx.audio_context = dict(data)
    tr = str(data.get("transcript") or "").strip()
    if tr:
        ctx.ai_transcript = tr[:9500]


async def verify_transcode_r2_keys(transcode_r2: Dict[str, str]) -> bool:
    if not transcode_r2:
        return False
    for plat, key in transcode_r2.items():
        if str(plat).startswith("thumb_") or plat == "default":
            continue
        if not key or not await r2_stage.object_exists(str(key)):
            logger.info(f"checkpoint missing R2 object: {key}")
            return False
    return True


def output_artifacts_as_object_sql(column: str = "output_artifacts") -> str:
    """SQL expr: coerce jsonb array-of-objects → object before ``||`` merge.

    Postgres ``jsonb ||`` concatenates into an *array* when the left side is already
    an array. That permanently corrupts ``uploads.output_artifacts`` into a list of
    one-key objects (TUP upload ed5f5e17…).
    """
    col = column if column.replace("_", "").isalnum() else "output_artifacts"
    return f"""(
        CASE jsonb_typeof(COALESCE({col}, '{{}}'::jsonb))
            WHEN 'object' THEN COALESCE({col}, '{{}}'::jsonb)
            WHEN 'array' THEN COALESCE((
                SELECT jsonb_object_agg(kv.key, kv.value)
                  FROM jsonb_array_elements(COALESCE({col}, '[]'::jsonb)) AS elem
                  CROSS JOIN LATERAL jsonb_each(
                      CASE
                          WHEN jsonb_typeof(elem) = 'object' THEN elem
                          ELSE '{{}}'::jsonb
                      END
                  ) AS kv(key, value)
            ), '{{}}'::jsonb)
            ELSE '{{}}'::jsonb
        END
    )"""


async def merge_output_artifacts_patch(pool, upload_id: str, patch: dict) -> None:
    """Merge patch into uploads.output_artifacts (jsonb object)."""
    import asyncpg

    if not patch:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                UPDATE uploads
                SET output_artifacts = {output_artifacts_as_object_sql()} || $2::jsonb,
                    updated_at = NOW()
                WHERE id = $1
                """,
                upload_id,
                json.dumps(patch),
            )
    except asyncpg.exceptions.UndefinedColumnError:
        logger.warning("output_artifacts column missing — pipeline checkpoint not persisted")
    except Exception as e:
        logger.warning(f"merge_output_artifacts_patch failed: {e}")


async def clear_checkpoint(pool, upload_id: str) -> None:
    """Remove pipeline_resume from output_artifacts after a successful run."""
    import asyncpg

    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE uploads
                SET output_artifacts = COALESCE(output_artifacts, '{}'::jsonb) - $2,
                    updated_at = NOW()
                WHERE id = $1
                """,
                upload_id,
                RESUME_KEY,
            )
    except asyncpg.exceptions.UndefinedColumnError:
        pass
    except Exception as e:
        logger.debug(f"clear_checkpoint: {e}")


async def delete_checkpoint_r2_objects(transcode_r2: Optional[Dict[str, str]]) -> None:
    if not transcode_r2:
        return
    seen = set()
    for key in transcode_r2.values():
        k = str(key).strip()
        if not k or k in seen:
            continue
        seen.add(k)
        try:
            await r2_stage.delete_file(k)
        except Exception:
            pass


async def save_post_telemetry_checkpoint(pool, ctx: JobContext) -> None:
    """Persist a lightweight checkpoint right after telemetry parsing succeeds.

    Unlike the post-transcode checkpoint, this one writes nothing to R2 — the
    source video is already up there under ``r2_key``. We only stash the
    telemetry/trill snapshot and the ffprobe video_info dict in
    ``uploads.output_artifacts`` so a retry that fails between telemetry and
    transcode (watermark encode, transcode itself) doesn't have to
    re-run telemetry parsing or hit the geocoding APIs again.

    Idempotent: if a later checkpoint (post_transcode / post_audio / post_caption) already
    promoted the resume blob, we leave that one alone — it carries strictly
    more state than this one.
    """
    upload_id = ctx.upload_id
    from stages import db as db_stage

    ur = await db_stage.load_upload_record(pool, upload_id)
    existing = load_resume(ur or {})
    if existing and existing.get("stage") in ("post_transcode", "post_audio", "post_caption"):
        return

    resume = {
        "v": CHECKPOINT_VERSION,
        "stage": "post_telemetry",
        "source_r2_key": ctx.source_r2_key or "",
        "platforms_norm": norm_platforms(ctx.platforms),
        "telemetry": telemetry_to_dict(ctx.telemetry_data or ctx.telemetry),
        "trill": trill_to_dict(ctx.trill_score or ctx.trill),
        "video_info": dict(ctx.video_info) if getattr(ctx, "video_info", None) else None,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    await merge_output_artifacts_patch(pool, upload_id, {RESUME_KEY: resume})
    logger.info(f"[{upload_id}] Checkpoint saved: post_telemetry")


async def save_post_transcode_checkpoint(pool, ctx: JobContext) -> None:
    """Upload per-platform transcoded files to R2 and persist pipeline_resume."""
    upload_id = ctx.upload_id
    user_id = ctx.user_id
    if not ctx.platform_videos:
        return

    from stages import db as db_stage

    ur0 = await db_stage.load_upload_record(pool, upload_id)
    ex0 = load_resume(ur0 or {})
    if ex0 and ex0.get("stage") in ("post_audio", "post_caption"):
        return

    transcode_r2: Dict[str, str] = {}
    path_to_key: Dict[str, str] = {}

    for platform, video_path in ctx.platform_videos.items():
        pl = str(platform).strip().lower()
        if not pl or not video_path:
            continue
        p = Path(video_path)
        if not p.exists():
            continue
        path_str = str(p.resolve())
        if path_str in path_to_key:
            transcode_r2[str(platform)] = path_to_key[path_str]
            continue
        r2_key = f"checkpoints/{user_id}/{upload_id}/transcoded/{pl}.mp4"
        try:
            await r2_stage.upload_file(p, r2_key, "video/mp4")
            path_to_key[path_str] = r2_key
            transcode_r2[str(platform)] = r2_key
        except Exception as e:
            logger.warning(f"[{upload_id}] checkpoint transcode upload failed [{pl}]: {e}")
            return

    if not transcode_r2 or not _platforms_covered_transcode(transcode_r2, ctx.platforms or []):
        return

    resume = {
        "v": CHECKPOINT_VERSION,
        "stage": "post_transcode",
        "source_r2_key": ctx.source_r2_key or "",
        "platforms_norm": norm_platforms(ctx.platforms),
        "transcode_r2": transcode_r2,
        "telemetry": telemetry_to_dict(ctx.telemetry_data or ctx.telemetry),
        "trill": trill_to_dict(ctx.trill_score or ctx.trill),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    await merge_output_artifacts_patch(pool, upload_id, {RESUME_KEY: resume})
    logger.info(f"[{upload_id}] Checkpoint saved: post_transcode ({list(transcode_r2.keys())})")


async def save_post_audio_checkpoint(pool, ctx: JobContext) -> None:
    """Promote checkpoint to post_audio after Whisper/ACR/YAMNet (resume before Vision)."""
    from stages import db as db_stage

    upload_id = ctx.upload_id
    ur = await db_stage.load_upload_record(pool, upload_id)
    resume = load_resume(ur or {})
    if not resume or resume.get("stage") != "post_transcode":
        return
    resume = dict(resume)
    resume["stage"] = "post_audio"
    resume["audio_context"] = clip_audio_context_for_checkpoint(getattr(ctx, "audio_context", None) or {})
    resume["saved_at_audio"] = datetime.now(timezone.utc).isoformat()
    await merge_output_artifacts_patch(pool, upload_id, {RESUME_KEY: resume})
    logger.info(f"[{upload_id}] Checkpoint promoted: post_audio")


def _platforms_covered_transcode(transcode_r2: Dict[str, str], platforms: List[str]) -> bool:
    keys_lower = {str(k).strip().lower() for k in transcode_r2.keys()}
    for p in platforms:
        pl = str(p).strip().lower()
        if pl and pl not in keys_lower:
            return False
    return bool(platforms)


async def save_post_caption_checkpoint(pool, ctx: JobContext) -> None:
    """Promote checkpoint to post_caption (thumbnail + caption already persisted)."""
    from stages import db as db_stage

    upload_id = ctx.upload_id
    ur = await db_stage.load_upload_record(pool, upload_id)
    resume = load_resume(ur or {})
    if not resume or resume.get("stage") not in ("post_transcode", "post_audio"):
        return
    resume = dict(resume)
    resume["stage"] = "post_caption"
    resume["thumbnail_r2_key"] = ctx.thumbnail_r2_key or ""
    resume["saved_at_caption"] = datetime.now(timezone.utc).isoformat()
    # Preserve styled platform thumbs already on R2 (paths in ctx.output_artifacts)
    pm_json = ctx.output_artifacts.get("platform_thumbnail_r2_keys", "{}")
    try:
        pm = json.loads(pm_json) if isinstance(pm_json, str) else (pm_json or {})
        if isinstance(pm, dict) and pm:
            resume["platform_thumbnail_r2"] = pm
    except Exception:
        pass
    await merge_output_artifacts_patch(pool, upload_id, {RESUME_KEY: resume})
    logger.info(f"[{upload_id}] Checkpoint promoted: post_caption")


async def save_post_publish_checkpoint(pool, ctx: JobContext) -> None:
    """Persist per-platform publish snapshot for partial retry resume."""
    from stages import db as db_stage

    upload_id = ctx.upload_id
    platform_results = []
    for r in getattr(ctx, "platform_results", None) or []:
        platform_results.append(
            {
                "platform": getattr(r, "platform", None),
                "success": getattr(r, "success", None),
                "token_row_id": getattr(r, "token_row_id", None),
                "platform_video_id": getattr(r, "platform_video_id", None),
                "platform_url": getattr(r, "platform_url", None),
                "error_code": getattr(r, "error_code", None),
                "error_message": getattr(r, "error_message", None),
            }
        )
    processed_assets = {}
    arts = getattr(ctx, "output_artifacts", None) or {}
    if isinstance(arts, dict):
        raw_pa = arts.get("processed_assets")
        if isinstance(raw_pa, str):
            try:
                processed_assets = json.loads(raw_pa) or {}
            except Exception:
                processed_assets = {}
        elif isinstance(raw_pa, dict):
            processed_assets = dict(raw_pa)

    resume = {
        "v": CHECKPOINT_VERSION,
        "stage": "post_publish",
        "source_r2_key": ctx.source_r2_key or "",
        "platforms_norm": norm_platforms(ctx.platforms),
        "platform_results": platform_results,
        "processed_assets": processed_assets,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    await merge_output_artifacts_patch(pool, upload_id, {RESUME_KEY: resume})
    logger.info(f"[{upload_id}] Checkpoint saved: post_publish ({len(platform_results)} results)")


async def try_resume_from_checkpoint(
    pool,
    job_data: dict,
    upload_record: dict,
    ctx: JobContext,
) -> Tuple[Optional[str], Optional[Any]]:
    """
    If ``action=retry`` or ``resume_from_checkpoint`` and checkpoint is valid,
    prepare temp dir + ctx for resume.

    Also auto-resumes when a durable ``pipeline_resume`` checkpoint exists and
    the job did not opt out (``skip_checkpoint_resume``). That covers Redis
    stream redelivery after a mid-pipeline worker crash (e.g. killed during
    FFmpeg) without waiting for stale recovery.

    Returns:
        (mode, temp_dir_holder) where mode is
        ``post_telemetry`` | ``post_transcode`` | ``post_audio`` | ``post_caption`` | None,
        and temp_dir_holder is tempfile.TemporaryDirectory if mode is set (caller must cleanup).
    """
    import tempfile

    if job_data.get("skip_checkpoint_resume"):
        return None, None

    cp = load_resume(upload_record)
    if not cp or not checkpoint_matches_upload(cp, upload_record):
        return None, None

    stage = (cp.get("stage") or "").strip().lower()
    if stage not in ("post_telemetry", "post_transcode", "post_audio", "post_caption", "post_publish"):
        return None, None

    allow = (
        (job_data.get("action") == "retry")
        or bool(job_data.get("resume_from_checkpoint"))
        # Crash / stream reclaim: same job payload, no resume flag — still skip
        # finished stages when a matching checkpoint is on the upload row.
        or stage in ("post_telemetry", "post_transcode", "post_audio", "post_caption")
    )
    if not allow:
        return None, None

    if stage == "post_publish":
        retry_mode = str(job_data.get("retry_mode") or "").strip().lower()
        if retry_mode != "partial":
            return None, None
        holder = tempfile.TemporaryDirectory()
        temp_dir = Path(holder.name)
        ctx.temp_dir = temp_dir
        pa = cp.get("processed_assets") or {}
        if isinstance(pa, str):
            try:
                pa = json.loads(pa) or {}
            except Exception:
                pa = {}
        if isinstance(pa, dict):
            ctx.output_artifacts = dict(getattr(ctx, "output_artifacts", None) or {})
            ctx.output_artifacts["processed_assets"] = json.dumps(pa)
        pr = cp.get("platform_results") or []
        if isinstance(pr, list):
            from stages.context import PlatformResult

            ctx.platform_results = [
                PlatformResult(
                    platform=str(x.get("platform") or ""),
                    success=bool(x.get("success")),
                    token_row_id=x.get("token_row_id"),
                    platform_video_id=x.get("platform_video_id"),
                    platform_url=x.get("platform_url"),
                    error_code=x.get("error_code"),
                    error_message=x.get("error_message"),
                )
                for x in pr
                if isinstance(x, dict) and x.get("platform")
            ]
        return "post_publish", holder

    # ── post_telemetry: re-download source, restore telemetry snapshot ──
    if stage == "post_telemetry":
        source_key = (cp.get("source_r2_key") or upload_record.get("r2_key") or "").strip()
        if not source_key or not await r2_stage.object_exists(source_key):
            logger.info("post_telemetry checkpoint invalid — source r2_key missing")
            return None, None

        holder = tempfile.TemporaryDirectory()
        temp_dir = Path(holder.name)
        ctx.temp_dir = temp_dir

        # Re-download the original source video (cheap inside R2's network).
        filename = upload_record.get("filename") or "source.mp4"
        local_video = temp_dir / filename
        try:
            await r2_stage.download_file(source_key, local_video)
        except Exception as e:
            logger.warning(f"post_telemetry resume: source download failed ({e})")
            try:
                holder.cleanup()
            except Exception:
                pass
            return None, None
        ctx.local_video_path = local_video

        # Telemetry file is optional — only re-fetch when present.
        telem_key = (upload_record.get("telemetry_r2_key") or "").strip()
        if telem_key:
            try:
                telem_local = temp_dir / "telemetry.map"
                await r2_stage.download_file(telem_key, telem_local)
                ctx.local_telemetry_path = telem_local
            except Exception as e:
                logger.debug(f"post_telemetry resume: telemetry download skipped ({e})")
                ctx.local_telemetry_path = None

        # Restore the parsed telemetry/trill snapshot so the worker can skip
        # the (slow) telemetry stage entirely on this retry.
        restore_telemetry_from_dict(ctx, cp.get("telemetry"), cp.get("trill"))

        vi = cp.get("video_info") or {}
        if isinstance(vi, dict) and vi:
            try:
                ctx.video_info = dict(vi)
            except Exception:
                pass

        logger.info(f"[{ctx.upload_id}] Resuming from checkpoint: post_telemetry")
        return stage, holder

    transcode_r2: Dict[str, str] = dict(cp.get("transcode_r2") or {})
    if not await verify_transcode_r2_keys(transcode_r2):
        return None, None

    if stage == "post_caption":
        thumb = (cp.get("thumbnail_r2_key") or "") or (upload_record.get("thumbnail_r2_key") or "")
        if not thumb or not await r2_stage.object_exists(str(thumb)):
            logger.info("post_caption checkpoint invalid — thumbnail missing on R2")
            return None, None

    holder = tempfile.TemporaryDirectory()
    temp_dir = Path(holder.name)
    ctx.temp_dir = temp_dir

    restore_telemetry_from_dict(ctx, cp.get("telemetry"), cp.get("trill"))

    ctx.platform_videos.clear()
    by_r2: Dict[str, Path] = {}
    for platform, r2_key in transcode_r2.items():
        pl = str(platform).strip().lower()
        if not pl:
            continue
        rk = str(r2_key).strip()
        if not rk:
            continue
        if rk in by_r2:
            ctx.platform_videos[platform] = by_r2[rk]
            continue
        local_path = temp_dir / f"{pl}.mp4"
        try:
            await r2_stage.download_file(rk, local_path)
            ctx.platform_videos[platform] = local_path
            by_r2[rk] = local_path
        except Exception as e:
            logger.warning(f"checkpoint resume download failed [{pl}]: {e}")
            try:
                holder.cleanup()
            except Exception:
                pass
            return None, None

    first = next(iter(ctx.platform_videos.values()), None)
    ctx.local_video_path = first
    ctx.processed_video_path = first

    stage_out = stage
    if stage == "post_audio":
        ac_blob = cp.get("audio_context")
        if isinstance(ac_blob, dict) and ac_blob:
            restore_audio_context_from_dict(ctx, ac_blob)
        else:
            logger.info(
                "[%s] post_audio checkpoint missing audio_context — resuming at transcode (re-run audio)",
                ctx.upload_id,
            )
            stage_out = "post_transcode"

    if stage == "post_caption":
        ctx.thumbnail_r2_key = (cp.get("thumbnail_r2_key") or "") or upload_record.get("thumbnail_r2_key")
        ptr = cp.get("platform_thumbnail_r2") or {}
        platform_thumb_map: Dict[str, str] = {}
        if isinstance(ptr, dict) and ptr:
            ctx.output_artifacts["platform_thumbnail_r2_keys"] = json.dumps(ptr)
            for plat, r2k in ptr.items():
                if not r2k:
                    continue
                lp = temp_dir / f"resume_thumb_{plat}.jpg"
                try:
                    await r2_stage.download_file(str(r2k), lp)
                    if lp.exists():
                        platform_thumb_map[str(plat)] = str(lp)
                except Exception:
                    pass
        if platform_thumb_map:
            ctx.output_artifacts["platform_thumbnail_map"] = json.dumps(platform_thumb_map)

    logger.info(f"[{ctx.upload_id}] Resuming from checkpoint: {stage_out}")
    return stage_out, holder
