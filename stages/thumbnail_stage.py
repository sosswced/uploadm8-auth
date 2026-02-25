"""
UploadM8 Thumbnail Stage — Multi-Frame Edition
===============================================
Generate multiple candidate thumbnails using FFmpeg, score each frame for
sharpness, auto-select the best one, and store all candidates for user pick.

Flow:
  1. Probe video duration via ffprobe
  2. Distribute N frame offsets across the video (N = tier max_thumbnails)
  3. Extract each frame as a 1080px-wide JPEG
  4. Score each frame with FFmpeg blurdetect (higher blur_mean = blurrier)
  5. Set ctx.thumbnail_path  = sharpest frame
     Set ctx.thumbnail_paths = all successful candidates (chronological order)
     Set ctx.thumbnail_scores = {str(path): sharpness_score} for all candidates
  6. Store all candidates in ctx.output_artifacts for queue UI display

Fallback chain:
  - If blurdetect fails, fall back to file-size proxy (larger = more detail)
  - If extraction fails at all offsets, retry at t=0
  - If everything fails, raise SkipStage (non-fatal — pipeline continues)

Exports: run_thumbnail_stage(ctx)
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional

from .errors import SkipStage
from .context import JobContext

logger = logging.getLogger("uploadm8-worker.thumbnail")

# Absolute fallback offset when no user setting exists
DEFAULT_THUMBNAIL_OFFSET = 1.0
# Safety cap — prevents absurd values on very short clips
MAX_THUMBNAIL_OFFSET = 300.0
# Minimum thumbnail file size to be considered valid (bytes)
MIN_THUMB_SIZE = 2048


# ============================================================
# ffprobe — get video duration
# ============================================================

async def _get_video_duration(video_path: Path) -> float:
    """Return video duration in seconds via ffprobe. Returns 30.0 on failure."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(video_path),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        data = json.loads(stdout.decode())
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                dur = float(stream.get("duration", 0) or 0)
                if dur > 0:
                    return dur
    except Exception as e:
        logger.warning(f"ffprobe duration failed: {e}")
    return 30.0


# ============================================================
# Frame extraction
# ============================================================

async def _extract_frame(video_path: Path, output_path: Path, offset: float) -> bool:
    """
    Extract a single JPEG frame from video_path at `offset` seconds.
    Returns True on success.
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{offset:.3f}",
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", "2",
        "-vf", "scale=1080:-2",   # 1080px wide, proportional height (always even)
        str(output_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if (
        proc.returncode == 0
        and output_path.exists()
        and output_path.stat().st_size >= MIN_THUMB_SIZE
    ):
        return True

    logger.debug(
        f"FFmpeg thumb failed at {offset:.1f}s (rc={proc.returncode}): "
        f"{stderr.decode()[-200:]}"
    )
    return False


# ============================================================
# Sharpness scoring via FFmpeg blurdetect
# ============================================================

async def _score_sharpness(image_path: Path) -> float:
    """
    Run FFmpeg blurdetect on a JPEG and return a sharpness score.
    Higher = sharper. Returns 0.0 on failure (triggers file-size fallback).

    blurdetect outputs blur_mean: higher values mean MORE blur.
    We invert: sharpness = 1.0 - blur_mean (clamped to [0, 1]).
    """
    cmd = [
        "ffmpeg", "-i", str(image_path),
        "-vf", "blurdetect=high=0.1",
        "-f", "null", "-",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        output = stderr.decode()

        # blurdetect outputs a line like: "blur_mean:0.042 blur_std:..."
        for line in output.splitlines():
            if "blur_mean:" in line:
                for part in line.split():
                    if part.startswith("blur_mean:"):
                        blur_val = float(part.split(":")[1])
                        return max(0.0, 1.0 - blur_val)
    except Exception as e:
        logger.debug(f"blurdetect failed for {image_path.name}: {e}")
    return 0.0


# ============================================================
# Offset distribution
# ============================================================

def _distribute_offsets(duration: float, n: int, user_offset: Optional[float] = None) -> List[float]:
    """
    Generate N evenly-spaced offsets across the video duration.

    Anchors:
      - First frame always at 5% of duration (avoids black intro frames)
      - Last frame always at 90% of duration (avoids end credits / black fade)
      - Remaining N-2 frames evenly distributed between anchors

    Special cases:
      - n == 1: use user_offset if provided, else 30% of duration
      - n == 2: 5% and 90%
      - n >= 3: 5%, evenly spaced, 90%

    All offsets are clamped to [0.5, duration - 0.5] to avoid boundary frames.
    """
    if duration <= 0:
        duration = 30.0

    clamp = lambda v: max(0.5, min(v, duration - 0.5))

    if n <= 0:
        n = 1

    if n == 1:
        if user_offset is not None:
            return [clamp(user_offset)]
        return [clamp(duration * 0.30)]

    if n == 2:
        return [clamp(duration * 0.05), clamp(duration * 0.90)]

    # n >= 3: anchor at 5% and 90%, fill middle
    anchors = [clamp(duration * 0.05), clamp(duration * 0.90)]
    middle_count = n - 2
    if middle_count > 0:
        step = (duration * 0.90 - duration * 0.05) / (middle_count + 1)
        middle = [clamp(duration * 0.05 + step * (i + 1)) for i in range(middle_count)]
    else:
        middle = []

    return [anchors[0]] + middle + [anchors[1]]


# ============================================================
# Stage Entry Point
# ============================================================

async def run_thumbnail_stage(ctx: JobContext) -> JobContext:
    """
    Generate multiple candidate thumbnails, score them for sharpness,
    and set ctx.thumbnail_path to the sharpest one.

    Number of thumbnails = ctx.entitlements.max_thumbnails (tier-gated).
    All candidates stored in ctx.thumbnail_paths and ctx.output_artifacts.
    """
    ctx.mark_stage("thumbnail")

    # ── Choose source video ─────────────────────────────────────────────────
    video_path: Optional[Path] = None
    for candidate in (ctx.processed_video_path, ctx.local_video_path):
        if candidate and Path(candidate).exists():
            video_path = Path(candidate)
            break

    if not video_path:
        raise SkipStage("No video file available for thumbnail generation")

    if not ctx.temp_dir:
        raise SkipStage("No temp directory available")

    # ── Determine how many frames to generate ──────────────────────────────
    max_thumbnails = 1
    if ctx.entitlements:
        max_thumbnails = getattr(ctx.entitlements, "max_thumbnails", 1) or 1
    # Honour tier ceiling; never exceed it even if settings say more
    max_thumbnails = max(1, int(max_thumbnails))

    # Read user's manual offset preference (used only when max_thumbnails == 1)
    raw_offset = (ctx.user_settings or {}).get("thumbnail_offset", DEFAULT_THUMBNAIL_OFFSET)
    try:
        user_offset = float(raw_offset)
        user_offset = max(0.0, min(user_offset, MAX_THUMBNAIL_OFFSET))
    except (TypeError, ValueError):
        user_offset = DEFAULT_THUMBNAIL_OFFSET

    logger.info(
        f"Thumbnail stage: video={video_path.name}, "
        f"max_thumbnails={max_thumbnails}, user_offset={user_offset}s"
    )

    # ── Get video duration ──────────────────────────────────────────────────
    duration = await _get_video_duration(video_path)
    logger.debug(f"Video duration: {duration:.1f}s")

    # ── Distribute offsets ──────────────────────────────────────────────────
    offsets = _distribute_offsets(
        duration=duration,
        n=max_thumbnails,
        user_offset=user_offset if max_thumbnails == 1 else None,
    )
    logger.debug(f"Thumbnail offsets: {[f'{o:.1f}s' for o in offsets]}")

    # ── Extract frames ──────────────────────────────────────────────────────
    candidates: List[Tuple[Path, float]] = []  # (path, sharpness_score)

    for idx, offset in enumerate(offsets):
        out_path = ctx.temp_dir / f"thumb_{ctx.upload_id}_{idx:02d}.jpg"
        success = await _extract_frame(video_path, out_path, offset)

        if not success and offset > 0:
            # Short clip fallback: try frame 0
            logger.debug(f"Frame at {offset:.1f}s failed — retrying at 0s")
            success = await _extract_frame(video_path, out_path, 0.0)

        if success:
            # Score sharpness
            score = await _score_sharpness(out_path)
            if score == 0.0:
                # blurdetect unavailable — use file size as proxy (more bytes = more detail)
                score = out_path.stat().st_size / 1_000_000  # MB as float score
            candidates.append((out_path, score))
            sz_kb = out_path.stat().st_size / 1024
            logger.debug(f"  Frame {idx}: {out_path.name} @ {offset:.1f}s — "
                         f"sharpness={score:.4f}, size={sz_kb:.1f}KB")
        else:
            logger.warning(f"  Frame {idx} at {offset:.1f}s failed — skipping")

    if not candidates:
        logger.warning("Thumbnail generation failed for all offsets (non-fatal)")
        raise SkipStage("FFmpeg thumbnail extraction produced no output")

    # ── Sort by sharpness (best first), then re-sort chronologically for storage ──
    best_path, best_score = max(candidates, key=lambda x: x[1])

    # Store all candidates in chronological order (matches video narrative order)
    ctx.thumbnail_paths = [p for p, _ in candidates]
    ctx.thumbnail_scores = {str(p): s for p, s in candidates}

    # Primary thumbnail = sharpest frame
    ctx.thumbnail_path = best_path
    ctx.output_artifacts["thumbnail"] = str(best_path)

    # Store all candidates for queue UI / user pick
    ctx.output_artifacts["thumbnail_candidates"] = json.dumps(
        [str(p) for p in ctx.thumbnail_paths]
    )
    ctx.output_artifacts["thumbnail_scores"] = json.dumps(
        {str(p): round(s, 4) for p, s in candidates}
    )

    logger.info(
        f"Thumbnail stage complete: {len(candidates)} frames generated, "
        f"best={best_path.name} (sharpness={best_score:.4f})"
    )

    return ctx
