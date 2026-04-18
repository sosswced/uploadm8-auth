"""
Pipeline checkpoint / resume for the upload worker.

Persists post-transcode and post-caption state to R2 + uploads.output_artifacts
so a failed job can retry from the last durable stage without re-downloading
the source or re-running FFmpeg transcode.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from stages.context import JobContext, TelemetryData, TrillScore
from stages import r2 as r2_stage

logger = logging.getLogger("uploadm8-worker")

CHECKPOINT_VERSION = 1
RESUME_KEY = "pipeline_resume"


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


async def merge_output_artifacts_patch(pool, upload_id: str, patch: dict) -> None:
    """Merge patch into uploads.output_artifacts (jsonb)."""
    import asyncpg

    if not patch:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE uploads
                SET output_artifacts = COALESCE(output_artifacts, '{}'::jsonb) || $2::jsonb,
                    updated_at = NOW()
                WHERE id = $1
                """,
                upload_id,
                patch,
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


async def save_post_transcode_checkpoint(pool, ctx: JobContext) -> None:
    """Upload per-platform transcoded files to R2 and persist pipeline_resume."""
    upload_id = ctx.upload_id
    user_id = ctx.user_id
    if not ctx.platform_videos:
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
    if not resume or resume.get("stage") != "post_transcode":
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


async def try_resume_from_checkpoint(
    pool,
    job_data: dict,
    upload_record: dict,
    ctx: JobContext,
) -> Tuple[Optional[str], Optional[Any]]:
    """
    If action=retry and checkpoint is valid, prepare temp dir + ctx for resume.

    Returns:
        (mode, temp_dir_holder) where mode is 'post_transcode' | 'post_caption' | None,
        and temp_dir_holder is tempfile.TemporaryDirectory if mode is set (caller must cleanup).
    """
    import tempfile

    if job_data.get("action") != "retry":
        return None, None

    cp = load_resume(upload_record)
    if not cp or not checkpoint_matches_upload(cp, upload_record):
        return None, None

    stage = (cp.get("stage") or "").strip().lower()
    if stage not in ("post_transcode", "post_caption"):
        return None, None

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

    logger.info(f"[{ctx.upload_id}] Resuming from checkpoint: {stage}")
    return stage, holder
