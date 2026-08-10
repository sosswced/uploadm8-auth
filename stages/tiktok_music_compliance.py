"""TikTok music compliance after ACR (mute deliverable + compliance artifact).

Mirrors ``youtube_copyright_shorts``: run after audio stage, mutate only
``platform_videos["tiktok"]``, persist a notice for queue/UI.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from .context import JobContext

logger = logging.getLogger("uploadm8-worker.tiktok_music_compliance")

ARTIFACT_KEY = "tiktok_music_compliance"


def _acr_catalog_copyright_signal(ac: Dict[str, Any]) -> bool:
    if ac.get("copyright_risk"):
        return True
    cs = ac.get("content_signals") or []
    if not isinstance(cs, list):
        return False
    if "acr_catalog_match" not in {str(x).strip() for x in cs if x}:
        return False
    return bool(ac.get("music_detected") or ac.get("music_title") or ac.get("music_artist"))


def tiktok_acr_catalog_risk(ctx: JobContext) -> bool:
    """True when TikTok is a target and ACR flagged catalogue music (any duration)."""
    plats = [str(p).strip().lower() for p in (ctx.platforms or []) if p]
    if "tiktok" not in plats:
        return False
    ac = getattr(ctx, "audio_context", None) or {}
    if not isinstance(ac, dict):
        return False
    return _acr_catalog_copyright_signal(ac)


def _tt_settings_flags(ctx: JobContext) -> Dict[str, bool]:
    """Aggregate mute / license / inbox flags from tiktok_post_settings."""
    from services.tiktok_api import normalize_tiktok_post_settings

    raw = getattr(ctx, "tiktok_post_settings", None) or (ctx.user_settings or {}).get(
        "tiktok_post_settings"
    )
    mute = False
    keep_licensed = False
    finish_in_app = False
    if isinstance(raw, dict):
        by = raw.get("by_account")
        if isinstance(by, dict) and by:
            for ent in by.values():
                if not isinstance(ent, dict):
                    continue
                s = normalize_tiktok_post_settings(ent)
                if s.get("mute_audio_for_tiktok"):
                    mute = True
                if s.get("keep_catalog_audio_licensed"):
                    keep_licensed = True
                if s.get("finish_in_tiktok_app"):
                    finish_in_app = True
        else:
            s = normalize_tiktok_post_settings(raw)
            mute = bool(s.get("mute_audio_for_tiktok"))
            keep_licensed = bool(s.get("keep_catalog_audio_licensed"))
            finish_in_app = bool(s.get("finish_in_tiktok_app"))
    # Mute wins over keep-licensed when both set.
    if mute:
        keep_licensed = False
    return {
        "mute_audio_for_tiktok": mute,
        "keep_catalog_audio_licensed": keep_licensed,
        "finish_in_tiktok_app": finish_in_app,
    }


def get_tiktok_music_compliance_notice(ctx: JobContext) -> Optional[Dict[str, Any]]:
    raw = (getattr(ctx, "output_artifacts", None) or {}).get(ARTIFACT_KEY)
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            d = json.loads(raw)
            return d if isinstance(d, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _resolve_tiktok_source_path(ctx: JobContext) -> Optional[Path]:
    candidates: list[Path] = []
    pv = getattr(ctx, "platform_videos", None) or {}
    if isinstance(pv, dict) and pv.get("tiktok") is not None:
        candidates.append(Path(pv["tiktok"]))
    try:
        via = ctx.get_video_for_platform("tiktok")
        if via is not None:
            candidates.append(Path(via))
    except Exception:
        pass
    for attr in ("processed_video_path", "local_video_path"):
        lp = getattr(ctx, attr, None)
        if lp is not None:
            candidates.append(Path(lp))
    for p in candidates:
        try:
            if p.exists() and p.is_file() and p.stat().st_size > 0:
                return p
        except OSError:
            continue
    return None


def _ffmpeg_bin() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


def mute_tiktok_deliverable_sync(ctx: JobContext) -> Path:
    """
    Remux TikTok MP4 with video copy and no audio (``-an``).
    Replaces ``ctx.platform_videos["tiktok"]`` only.
    """
    src = _resolve_tiktok_source_path(ctx)
    if src is None:
        raise FileNotFoundError("no local TikTok deliverable to mute")
    temp_dir = getattr(ctx, "temp_dir", None)
    out_dir = Path(temp_dir) if temp_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"tiktok_muted_{ctx.upload_id}.mp4"
    cmd = [
        _ffmpeg_bin(),
        "-y",
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
        "-an",
        "-movflags",
        "+faststart",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0 or not out.exists() or out.stat().st_size <= 0:
        err = (proc.stderr or proc.stdout or "")[-400:]
        raise RuntimeError(f"ffmpeg mute failed rc={proc.returncode}: {err}")
    if not isinstance(getattr(ctx, "platform_videos", None), dict):
        ctx.platform_videos = {}
    ctx.platform_videos["tiktok"] = out
    return out


async def mute_tiktok_deliverable(ctx: JobContext) -> Path:
    return await asyncio.to_thread(mute_tiktok_deliverable_sync, ctx)


def _music_meta(ctx: JobContext) -> Dict[str, Any]:
    ac = getattr(ctx, "audio_context", None) or {}
    if not isinstance(ac, dict):
        return {}
    out: Dict[str, Any] = {}
    for k in ("music_title", "music_artist", "music_genre", "copyright_risk"):
        if ac.get(k) is not None and ac.get(k) != "":
            out[k] = ac.get(k)
    if ac.get("music_detected") is not None:
        out["music_detected"] = bool(ac.get("music_detected"))
    return out


async def apply_tiktok_music_compliance_after_audio(ctx: JobContext, db_pool) -> None:
    """
    After ACR: mute TikTok deliverable unless user attested catalog license,
    or when they explicitly asked to mute / finish in TikTok app (add Sound).
    """
    from stages.pipeline_checkpoint import merge_output_artifacts_patch

    plats = [str(p).strip().lower() for p in (ctx.platforms or []) if p]
    if "tiktok" not in plats:
        return

    flags = _tt_settings_flags(ctx)
    risk = tiktok_acr_catalog_risk(ctx)
    # Inbox/Sounds path: prefer silent file so creator adds TikTok Sound in-app.
    want_mute = bool(
        flags["mute_audio_for_tiktok"]
        or flags["finish_in_tiktok_app"]
        or (risk and not flags["keep_catalog_audio_licensed"])
    )
    keep_licensed = bool(risk and flags["keep_catalog_audio_licensed"] and not flags["mute_audio_for_tiktok"])

    if not risk and not flags["mute_audio_for_tiktok"] and not flags["finish_in_tiktok_app"]:
        return

    notice: Dict[str, Any] = {
        "version": 1,
        "acr_catalog_risk": risk,
        "mute_requested": flags["mute_audio_for_tiktok"],
        "finish_in_tiktok_app": flags["finish_in_tiktok_app"],
        "keep_catalog_audio_licensed": flags["keep_catalog_audio_licensed"],
        "muted": False,
        "kept_licensed_audio": False,
        **_music_meta(ctx),
    }

    if keep_licensed and not want_mute:
        notice["status"] = "kept_licensed"
        notice["kept_licensed_audio"] = True
        notice["message"] = (
            "UploadM8 detected recognized music (ACR). You confirmed you have a license "
            "to keep that audio on TikTok. Music Usage Confirmation still applies. "
            "TikTok Sounds / Commercial Music Library can only be added in the TikTok app."
        )
        ctx.output_artifacts[ARTIFACT_KEY] = json.dumps(notice)
        try:
            await merge_output_artifacts_patch(db_pool, str(ctx.upload_id), {ARTIFACT_KEY: notice})
        except Exception as e:
            logger.warning("[%s] tiktok_music_compliance merge failed: %s", ctx.upload_id, e)
        return

    if not want_mute:
        return

    notice["status"] = "muting"
    notice["message"] = (
        "UploadM8 is removing baked-in audio from the TikTok file "
        + (
            "because catalog music was detected (or you asked to mute / finish in TikTok). "
            if risk
            else "because you chose mute or Finish in TikTok app. "
        )
        + "Open TikTok to add a Sound or Commercial Music Library track after the draft lands."
    )

    try:
        await mute_tiktok_deliverable(ctx)
        notice["muted"] = True
        notice["status"] = "muted"
        notice["message"] = (
            "TikTok file was muted (no baked-in audio). "
            "When the post is ready, open TikTok"
            + (
                " Inbox to finish editing and add a Sound."
                if flags["finish_in_tiktok_app"]
                else " and add a Sound or Commercial Music Library track if you want music."
            )
        )
        notice["add_sound_hint"] = True
        notice["tiktok_open_url"] = "https://www.tiktok.com/"
    except Exception as e:
        logger.warning("[%s] TikTok mute failed: %s", ctx.upload_id, e)
        notice["muted"] = False
        notice["status"] = "mute_failed"
        notice["mute_error"] = str(e)[:300]
        notice["message"] = (
            "UploadM8 could not mute the TikTok file "
            f"({str(e)[:160]}). TikTok publish may fail until you mute locally "
            "or confirm a license for catalog music."
        )

    ctx.output_artifacts[ARTIFACT_KEY] = json.dumps(notice)
    try:
        await merge_output_artifacts_patch(db_pool, str(ctx.upload_id), {ARTIFACT_KEY: notice})
    except Exception as e:
        logger.warning("[%s] tiktok_music_compliance merge failed: %s", ctx.upload_id, e)

    try:
        from stages.pipeline_checkpoint import refresh_transcode_checkpoint_platform

        if notice.get("muted"):
            await refresh_transcode_checkpoint_platform(db_pool, ctx, "tiktok")
    except Exception as refresh_e:
        logger.warning(
            "[%s] TikTok mute checkpoint refresh skipped: %s",
            ctx.upload_id,
            refresh_e,
        )
