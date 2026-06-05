"""Burn styled thumbnail pixels into TikTok MP4 at the cover frame timestamp."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .ffmpeg_env import resolve_ffmpeg_executable
from .transcode_stage import get_platform_spec

logger = logging.getLogger("uploadm8-worker.tiktok_cover_burn")

_DEFAULT_W = 1080
_DEFAULT_H = 1920


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)) or default)
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


def tiktok_cover_burn_mode() -> str:
    """``segment`` (default) or ``full``."""
    raw = (os.environ.get("TIKTOK_COVER_BURN_MODE") or "segment").strip().lower()
    return raw if raw in ("segment", "full") else "segment"


def tiktok_cover_x264_preset() -> str:
    return (os.environ.get("TIKTOK_COVER_X264_PRESET") or "veryfast").strip() or "veryfast"


def tiktok_cover_x264_crf() -> str:
    return (os.environ.get("TIKTOK_COVER_X264_CRF") or "23").strip() or "23"


def tiktok_cover_x264_encoder() -> str:
    enc = (os.environ.get("TIKTOK_COVER_X264_ENCODER") or "libx264").strip() or "libx264"
    return enc


def tiktok_transcode_gop_sec() -> float:
    return max(0.5, _env_float("TIKTOK_TRANSCODE_GOP_SEC", 2.0))


def tiktok_transcode_force_keyframes_enabled() -> bool:
    return _env_bool("TIKTOK_TRANSCODE_FORCE_KEYFRAMES", True)


def tiktok_cover_flash_sec(fps: Optional[float] = None) -> float:
    if _env_bool("TIKTOK_COVER_SINGLE_FRAME", True) and fps and fps > 0:
        return max(1.0 / fps, 0.02)
    return max(0.04, _env_float("TIKTOK_COVER_BURN_FLASH_SEC", 0.12))


def _env_burn_globally_disabled() -> bool:
    """Explicit env false/off only — unset env lets tier policy decide."""
    raw = os.environ.get("TIKTOK_BURN_STYLED_COVER")
    if raw is None or not str(raw).strip():
        return False
    return str(raw).strip().lower() in ("0", "false", "no", "off")


def default_tiktok_burn_styled_cover_pref(entitlements: Any = None) -> bool:
    """
    Default for Settings UI and unset user prefs.

    Free: off (hard). Creator Lite+: off unless Pro+ styled thumbnails (``can_ai_thumbnail_styling``).
    """
    if _env_burn_globally_disabled():
        return False
    tier = str(getattr(entitlements, "tier", None) or "free").strip().lower()
    if tier == "free":
        return False
    return bool(getattr(entitlements, "can_ai_thumbnail_styling", False))


def tiktok_burn_enabled(
    user_settings: Optional[dict],
    entitlements: Any = None,
) -> bool:
    """
    Whether to re-encode the TikTok MP4 with a styled thumb at the cover timestamp.

    Policy:
      - Env ``TIKTOK_BURN_STYLED_COVER=false`` — global kill switch for all tiers.
      - Env unset — tier + user pref policy applies (Pro+ default on).
      - Free tier: always off (styled feed tile is a paid creator feature).
      - Explicit user pref ``tiktokBurnStyledCover`` wins on paid tiers.
      - Creator Pro+ (``can_ai_thumbnail_styling``): default on when pref unset.
      - Creator Lite / other paid without styled AI: default off, opt-in via Settings.
      - Caller must still have a styled TikTok thumb in ``platform_thumbnail_map``.
    """
    if _env_burn_globally_disabled():
        return False

    tier = str(getattr(entitlements, "tier", None) or "free").strip().lower()
    if tier == "free":
        return False

    us = user_settings if isinstance(user_settings, dict) else {}
    for key in ("tiktokBurnStyledCover", "tiktok_burn_styled_cover"):
        if key in us and us[key] is not None:
            return bool(us[key])

    return default_tiktok_burn_styled_cover_pref(entitlements)


def resolve_tiktok_cover_offset_sec(output_artifacts: Optional[dict]) -> float:
    arts = output_artifacts if isinstance(output_artifacts, dict) else {}
    for key in ("thumbnail_frame_offset_seconds", "tiktok_thumb_offset_seconds"):
        raw = arts.get(key)
        if raw is None or raw == "":
            continue
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            continue
    return 1.5


def tiktok_transcode_keyframe_force_value(cover_offset_sec: float, fps: float) -> str:
    """FFmpeg ``-force_key_frames`` value: pin cover timestamp (GOP via ``-g``)."""
    cover = max(0.0, float(cover_offset_sec))
    # Cover pin only — comma+expr mixes break on some production FFmpeg builds
    # (Invalid keyframe time: expr:gte(t). Regular spacing comes from -g/-keyint_min.
    if _env_bool("TIKTOK_TRANSCODE_FORCE_KEYFRAMES_EXPR", False):
        gop = tiktok_transcode_gop_sec()
        return f"{cover:.3f},expr:gte(t,n_forced*{int(gop)})"
    return f"{cover:.3f}"


def tiktok_transcode_gop_frames(fps: float) -> int:
    gop = tiktok_transcode_gop_sec()
    return max(12, int(round(max(1.0, fps) * gop)))


def extend_tiktok_transcode_x264_args(cmd: list, *, cover_offset_sec: float, fps: float) -> None:
    """Append GOP + forced keyframe flags for TikTok platform transcodes."""
    if not tiktok_transcode_force_keyframes_enabled():
        return
    gop_frames = tiktok_transcode_gop_frames(fps)
    cover = max(0.0, float(cover_offset_sec))
    insert_at = len(cmd)
    for i, tok in enumerate(cmd):
        if tok == "-c:v" and i + 1 < len(cmd) and cmd[i + 1] == "libx264":
            insert_at = i + 2
            break
    extra = [
        "-g",
        str(gop_frames),
        "-keyint_min",
        str(gop_frames),
    ]
    if cover > 0.001:
        extra.extend(["-force_key_frames", tiktok_transcode_keyframe_force_value(cover_offset_sec, fps)])
    for j, item in enumerate(extra):
        cmd.insert(insert_at + j, item)


@dataclass
class KeyframeWindow:
    prev_kf_sec: float
    next_kf_sec: float
    duration_sec: float
    fps: float

    @property
    def encode_span_sec(self) -> float:
        return max(0.0, self.next_kf_sec - self.prev_kf_sec)

    def to_artifact_dict(self, offset_sec: float) -> dict:
        return {
            "prev_kf_sec": round(self.prev_kf_sec, 4),
            "next_kf_sec": round(self.next_kf_sec, 4),
            "offset_sec": round(offset_sec, 4),
            "duration_sec": round(self.duration_sec, 4),
            "encode_span_sec": round(self.encode_span_sec, 4),
        }


@dataclass
class BurnResult:
    ok: bool
    mode: str = "skipped"
    elapsed_sec: float = 0.0
    window: Optional[dict] = None
    error: Optional[str] = None


def compute_keyframe_window(
    keyframe_times: List[float],
    offset_sec: float,
    duration_sec: float,
    *,
    hint: Optional[dict] = None,
) -> KeyframeWindow:
    """Pick [prev_kf, next_kf) GOP window bracketing *offset_sec*."""
    if hint:
        try:
            prev = float(hint.get("prev_kf_sec", hint.get("prev_kf", 0.0)))
            nxt = float(hint.get("next_kf_sec", hint.get("next_kf", duration_sec)))
            if 0 <= prev <= offset_sec < nxt <= duration_sec + 0.05:
                fps = float(hint.get("fps") or 30.0)
                return KeyframeWindow(prev, nxt, duration_sec, fps)
        except (TypeError, ValueError):
            pass

    kfs = sorted({max(0.0, float(t)) for t in keyframe_times if t is not None})
    if not kfs or kfs[0] > 0.01:
        kfs = [0.0] + kfs
    prev = max([t for t in kfs if t <= offset_sec + 1e-6], default=0.0)
    next_candidates = [t for t in kfs if t > offset_sec + 1e-6]
    nxt = min(next_candidates) if next_candidates else duration_sec
    if nxt <= prev:
        nxt = min(duration_sec, prev + tiktok_transcode_gop_sec())
    return KeyframeWindow(prev, nxt, duration_sec, 30.0)


def build_transcode_keyframe_hint(
    offset_sec: float,
    duration_sec: float,
    fps: float,
) -> dict:
    """Expected GOP window when TikTok transcode used forced keyframes at cover offset."""
    gop = tiktok_transcode_gop_sec()
    cover = max(0.0, float(offset_sec))
    kfs = {0.0, cover}
    t = 0.0
    while t <= duration_sec + gop:
        kfs.add(round(t, 4))
        t += gop
    window = compute_keyframe_window(sorted(kfs), offset_sec, duration_sec)
    window.fps = fps
    return window.to_artifact_dict(offset_sec)


def store_tiktok_transcode_keyframe_artifacts(
    output_artifacts: dict,
    *,
    offset_sec: float,
    fps: float,
    duration_sec: float,
) -> None:
    """Persist TikTok transcode GOP metadata for segment cover burn."""
    output_artifacts["tiktok_cover_keyframe_hint_sec"] = str(offset_sec)
    output_artifacts["tiktok_transcode_gop_sec"] = str(tiktok_transcode_gop_sec())
    output_artifacts["tiktok_transcode_gop_frames"] = str(tiktok_transcode_gop_frames(fps))
    if tiktok_transcode_force_keyframes_enabled():
        output_artifacts["tiktok_transcode_keyframe_window"] = json.dumps(
            build_transcode_keyframe_hint(offset_sec, duration_sec, fps)
        )


def resolve_keyframe_hint(output_artifacts: Optional[dict]) -> Optional[dict]:
    """Load keyframe window hint from transcode or prior burn artifacts."""
    arts = output_artifacts if isinstance(output_artifacts, dict) else {}
    for key in ("tiktok_transcode_keyframe_window", "tiktok_cover_keyframe_window"):
        raw = arts.get(key)
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip().startswith("{"):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                continue
    return None


async def _run_ffprobe_keyframes(video_path: Path) -> Tuple[List[float], float, float]:
    ffprobe = resolve_ffmpeg_executable("ffprobe") or "ffprobe"
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration:stream=r_frame_rate",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time,key_frame,pict_type",
        "-of",
        "json",
        str(video_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError((stderr.decode(errors="replace") or "ffprobe failed")[-400:])

    data = json.loads(stdout.decode(errors="replace") or "{}")
    duration = float((data.get("format") or {}).get("duration") or 0.0)
    streams = data.get("streams") or []
    fps = 30.0
    if streams:
        rate = str(streams[0].get("r_frame_rate") or "30/1")
        if "/" in rate:
            num, den = rate.split("/", 1)
            fps = float(num) / float(den) if float(den) > 0 else 30.0
        else:
            fps = float(rate)

    keyframes: List[float] = []
    for frame in data.get("frames") or []:
        if str(frame.get("key_frame")) == "1":
            try:
                keyframes.append(float(frame.get("best_effort_timestamp_time") or 0.0))
            except (TypeError, ValueError):
                continue
        elif str(frame.get("pict_type") or "").upper() == "I":
            try:
                keyframes.append(float(frame.get("best_effort_timestamp_time") or 0.0))
            except (TypeError, ValueError):
                continue

    if duration <= 0:
        duration = max(keyframes + [0.0])
    return keyframes, duration, fps


async def probe_keyframe_window(
    video_path: Path,
    offset_sec: float,
    *,
    hint: Optional[dict] = None,
) -> KeyframeWindow:
    keyframes, duration, fps = await _run_ffprobe_keyframes(video_path)
    window = compute_keyframe_window(keyframes, offset_sec, duration, hint=hint)
    window.fps = fps
    return window


def build_tiktok_cover_burn_command(
    video_path: Path,
    thumb_path: Path,
    output_path: Path,
    offset_sec: float,
    *,
    width: int = _DEFAULT_W,
    height: int = _DEFAULT_H,
    flash_sec: Optional[float] = None,
    fps: Optional[float] = None,
    ss_sec: Optional[float] = None,
    to_sec: Optional[float] = None,
    local_offset_sec: Optional[float] = None,
) -> list:
    """FFmpeg command: overlay scaled thumb on video for a short window at offset_sec."""
    ff = resolve_ffmpeg_executable() or "ffmpeg"
    eff_offset = local_offset_sec if local_offset_sec is not None else offset_sec
    start = max(0.0, float(eff_offset))
    flash = flash_sec if flash_sec is not None else tiktok_cover_flash_sec(fps)
    end = start + max(0.04, float(flash))
    filter_complex = (
        f"[1:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuva420p[thumb];"
        f"[0:v][thumb]overlay=0:0:enable='between(t,{start:.3f},{end:.3f})'[vout]"
    )
    spec = get_platform_spec("tiktok")
    cmd = [ff, "-y"]
    if ss_sec is not None and ss_sec > 0:
        cmd.extend(["-ss", f"{ss_sec:.3f}"])
    cmd.extend(["-i", str(video_path)])
    if to_sec is not None and to_sec > 0:
        cmd.extend(["-to", f"{to_sec:.3f}"])
    cmd.extend(
        [
            "-loop",
            "1",
            "-i",
            str(thumb_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-map",
            "0:a?",
            "-c:v",
            tiktok_cover_x264_encoder(),
            "-preset",
            tiktok_cover_x264_preset(),
            "-crf",
            tiktok_cover_x264_crf(),
            "-profile:v",
            spec.get("profile", "high"),
            "-level",
            spec.get("level", "4.1"),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    return cmd


def build_stream_copy_segment_command(
    video_path: Path,
    output_path: Path,
    *,
    ss_sec: float,
    to_sec: Optional[float] = None,
) -> list:
    ff = resolve_ffmpeg_executable() or "ffmpeg"
    cmd = [ff, "-y"]
    if ss_sec > 0:
        cmd.extend(["-ss", f"{ss_sec:.3f}"])
    cmd.extend(["-i", str(video_path)])
    if to_sec is not None:
        cmd.extend(["-to", f"{to_sec:.3f}"])
    cmd.extend(["-an", "-c:v", "copy", "-avoid_negative_ts", "make_zero", str(output_path)])
    return cmd


def build_concat_command(list_path: Path, output_path: Path) -> list:
    ff = resolve_ffmpeg_executable() or "ffmpeg"
    return [
        ff,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        str(output_path),
    ]


def build_mux_video_with_source_audio_command(
    video_path: Path,
    source_path: Path,
    output_path: Path,
) -> list:
    ff = resolve_ffmpeg_executable() or "ffmpeg"
    return [
        ff,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


async def _exec_ffmpeg(cmd: list) -> Tuple[bool, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    err = (stderr.decode(errors="replace") or "")[-600:]
    return proc.returncode == 0, err


async def _burn_full(
    video_path: Path,
    thumb_path: Path,
    output_path: Path,
    offset_sec: float,
    *,
    fps: Optional[float],
) -> BurnResult:
    cmd = build_tiktok_cover_burn_command(
        video_path,
        thumb_path,
        output_path,
        offset_sec,
        fps=fps,
    )
    t0 = time.monotonic()
    ok, err = await _exec_ffmpeg(cmd)
    elapsed = time.monotonic() - t0
    if not ok or not output_path.exists():
        return BurnResult(False, mode="full", elapsed_sec=elapsed, error=err)
    return BurnResult(True, mode="full", elapsed_sec=elapsed)


async def _burn_segmented(
    video_path: Path,
    thumb_path: Path,
    output_path: Path,
    offset_sec: float,
    *,
    work_dir: Path,
    fps: Optional[float],
    duration: Optional[float],
    keyframe_hint: Optional[dict],
) -> BurnResult:
    t0 = time.monotonic()
    try:
        window = await probe_keyframe_window(
            video_path,
            offset_sec,
            hint=keyframe_hint,
        )
    except Exception as e:
        logger.warning("TikTok cover burn: keyframe probe failed (%s) — falling back to full encode", e)
        return await _burn_full(video_path, thumb_path, output_path, offset_sec, fps=fps)

    if fps and fps > 0:
        window.fps = fps
    if duration and duration > 0:
        window.duration_sec = duration

    span = window.encode_span_sec
    # If the GOP window covers (almost) the whole clip, segmented splice won't help.
    if span >= max(0.0, window.duration_sec - 0.05) or span <= 0.04:
        logger.info(
            "TikTok cover burn: GOP window spans full clip (%.2fs) — using full encode",
            span,
        )
        full = await _burn_full(video_path, thumb_path, output_path, offset_sec, fps=window.fps)
        full.window = window.to_artifact_dict(offset_sec)
        return full

    work_dir.mkdir(parents=True, exist_ok=True)
    segments: List[Path] = []
    local_offset = offset_sec - window.prev_kf_sec

    if window.prev_kf_sec > 0.01:
        head_path = work_dir / "tiktok_burn_head.mp4"
        ok, err = await _exec_ffmpeg(
            build_stream_copy_segment_command(
                video_path,
                head_path,
                ss_sec=0.0,
                to_sec=window.prev_kf_sec,
            )
        )
        if not ok or not head_path.exists():
            logger.warning("TikTok cover burn: head copy failed (%s) — full encode", err[-200:])
            return await _burn_full(video_path, thumb_path, output_path, offset_sec, fps=window.fps)
        segments.append(head_path)

    mid_path = work_dir / "tiktok_burn_mid.mp4"
    ok, err = await _exec_ffmpeg(
        build_tiktok_cover_burn_command(
            video_path,
            thumb_path,
            mid_path,
            offset_sec,
            fps=window.fps,
            ss_sec=window.prev_kf_sec,
            to_sec=window.next_kf_sec,
            local_offset_sec=local_offset,
        )
    )
    if not ok or not mid_path.exists():
        logger.warning("TikTok cover burn: mid encode failed (%s) — full encode", err[-200:])
        return await _burn_full(video_path, thumb_path, output_path, offset_sec, fps=window.fps)
    segments.append(mid_path)

    if window.next_kf_sec < window.duration_sec - 0.05:
        tail_path = work_dir / "tiktok_burn_tail.mp4"
        ok, err = await _exec_ffmpeg(
            build_stream_copy_segment_command(
                video_path,
                tail_path,
                ss_sec=window.next_kf_sec,
                to_sec=None,
            )
        )
        if not ok or not tail_path.exists():
            logger.warning("TikTok cover burn: tail copy failed (%s) — full encode", err[-200:])
            return await _burn_full(video_path, thumb_path, output_path, offset_sec, fps=window.fps)
        segments.append(tail_path)

    list_path = work_dir / "tiktok_burn_concat.txt"
    list_path.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in segments),
        encoding="utf-8",
    )
    video_only = work_dir / "tiktok_burn_video_only.mp4"
    ok, err = await _exec_ffmpeg(build_concat_command(list_path, video_only))
    if not ok or not video_only.exists():
        logger.warning("TikTok cover burn: concat failed (%s) — full encode", err[-200:])
        return await _burn_full(video_path, thumb_path, output_path, offset_sec, fps=window.fps)

    ok, err = await _exec_ffmpeg(
        build_mux_video_with_source_audio_command(video_only, video_path, output_path)
    )
    elapsed = time.monotonic() - t0
    if not ok or not output_path.exists():
        logger.warning("TikTok cover burn: audio mux failed (%s) — full encode", err[-200:])
        full = await _burn_full(video_path, thumb_path, output_path, offset_sec, fps=window.fps)
        full.window = window.to_artifact_dict(offset_sec)
        return full

    art = window.to_artifact_dict(offset_sec)
    art["segment_count"] = len(segments)
    return BurnResult(True, mode="segment", elapsed_sec=elapsed, window=art)


async def burn_tiktok_styled_cover(
    video_path: Path,
    thumb_path: Path,
    output_path: Path,
    offset_sec: float,
    *,
    work_dir: Optional[Path] = None,
    video_fps: Optional[float] = None,
    video_duration: Optional[float] = None,
    keyframe_hint: Optional[dict] = None,
) -> BurnResult:
    """Burn styled thumb into TikTok MP4 at *offset_sec* (segment splice with full-encode fallback)."""
    if not video_path.exists() or not thumb_path.exists():
        return BurnResult(False, mode="skipped", error="missing_input")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = tiktok_cover_burn_mode()
    wd = work_dir or output_path.parent

    if mode == "full":
        return await _burn_full(
            video_path,
            thumb_path,
            output_path,
            offset_sec,
            fps=video_fps,
        )

    return await _burn_segmented(
        video_path,
        thumb_path,
        output_path,
        offset_sec,
        work_dir=wd,
        fps=video_fps,
        duration=video_duration,
        keyframe_hint=keyframe_hint,
    )
