"""
UploadM8 Transcode Stage
=======================
Transcode the uploaded video into a normalized, platform-friendly MP4.

Contract:
- Exports: run_transcode_stage(ctx)
- Input:  ctx.local_video_path must point to the downloaded source video
- Output: ctx.processed_video_path set to transcoded mp4

This stage is intentionally conservative:
- If input video path is missing, it raises StageError (pipeline should fail)
- If FFmpeg is unavailable, it raises StageError
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

from .errors import StageError, ErrorCode


def _get_ffmpeg_bin() -> str:
    # allow override for container differences
    return os.environ.get("FFMPEG_BIN", "ffmpeg")


async def _run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out_b, err_b = await proc.communicate()
    out = out_b.decode("utf-8", errors="replace") if out_b else ""
    err = err_b.decode("utf-8", errors="replace") if err_b else ""
    return proc.returncode, out, err


async def run_transcode_stage(ctx, *, target_height: Optional[int] = None, crf: str = "23"):
    """Canonical entrypoint expected by worker.py."""

    # Stage markers (support both sync and async variants)
    if hasattr(ctx, "mark_stage"):
        try:
            res = ctx.mark_stage("transcode")
            if hasattr(res, "__await__"):
                await res
        except Exception:
            pass

    in_path = getattr(ctx, "local_video_path", None)
    if not in_path:
        raise StageError(ErrorCode.VALIDATION, "Missing local_video_path for transcode", stage="transcode")

    in_path = Path(in_path)
    if not in_path.exists():
        raise StageError(ErrorCode.NOT_FOUND, f"Input video not found: {in_path}", stage="transcode")

    tmp_dir = getattr(ctx, "temp_dir", None)
    if not tmp_dir:
        # create a stage-local temp dir under /tmp if pipeline didn't supply one
        tmp_dir = Path("/tmp") / f"uploadm8_{getattr(ctx, 'upload_id', 'unknown')}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        setattr(ctx, "temp_dir", tmp_dir)

    out_path = Path(tmp_dir) / "processed.mp4"

    # Scale logic: optional knob, default keeps source dimensions.
    vf = None
    if target_height:
        # keep aspect ratio, force even dimensions
        vf = f"scale=-2:{int(target_height)}"

    cmd = [
        _get_ffmpeg_bin(),
        "-y",
        "-i",
        str(in_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
    ]
    if vf:
        cmd += ["-vf", vf]
    cmd += [str(out_path)]

    rc, out, err = await _run_cmd(cmd)
    if rc != 0 or not out_path.exists():
        # keep stderr short; worker logs capture full traceback anyway
        snippet = (err or out)[-1500:]
        raise StageError(ErrorCode.UPSTREAM, f"FFmpeg transcode failed (rc={rc}). {snippet}", stage="transcode")

    setattr(ctx, "processed_video_path", out_path)

    return ctx
