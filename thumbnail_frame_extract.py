"""
JPEG frame extraction for thumbnails.

Primary path: ``imageio`` + ``imageio-ffmpeg`` (bundled decoder, no PATH ffmpeg).
Fallback: system ffmpeg/ffprobe when imageio is not installed (local dev / slim venv).
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

MIN_THUMB_BYTES = 2048


def _ffmpeg_bin(name: str = "ffmpeg") -> str:
    try:
        from stages.ffmpeg_env import resolve_ffmpeg_executable

        return resolve_ffmpeg_executable(name) or name
    except Exception:
        return name


def _duration_via_ffprobe(video_path: Path) -> Optional[float]:
    ffprobe = _ffmpeg_bin("ffprobe")
    try:
        proc = subprocess.run(
            [
                ffprobe,
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(video_path),
            ],
            capture_output=True,
            check=False,
            timeout=30,
        )
        if proc.returncode != 0 or not proc.stdout:
            return None
        data = json.loads(proc.stdout.decode("utf-8", errors="replace"))
        d = float(data.get("format", {}).get("duration") or 0)
        return max(0.25, d) if d > 0 else None
    except Exception as e:
        logger.debug("thumbnail_frame_extract: ffprobe duration failed: %s", e)
        return None


def _extract_via_ffmpeg(
    video_path: Path,
    output_path: Path,
    offset_seconds: float,
    *,
    max_side: int = 1080,
    jpeg_quality: int = 92,
) -> bool:
    ffmpeg = _ffmpeg_bin("ffmpeg")
    t = max(0.0, float(offset_seconds))
    vf = f"scale='min({max_side},iw)':-2"
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{t:.3f}",
        "-i",
        str(video_path),
        "-vframes",
        "1",
        "-q:v",
        str(max(2, min(31, int(round((100 - jpeg_quality) / 3.5))))),
        "-vf",
        vf,
        str(output_path),
    ]
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(cmd, capture_output=True, check=False, timeout=60)
        if proc.returncode != 0:
            logger.debug(
                "thumbnail_frame_extract: ffmpeg extract failed at %.2fs: %s",
                offset_seconds,
                (proc.stderr or b"").decode("utf-8", errors="replace")[:300],
            )
            return False
        return output_path.exists() and output_path.stat().st_size >= MIN_THUMB_BYTES
    except Exception as e:
        logger.debug("thumbnail_frame_extract: ffmpeg extract error at %.2fs: %s", offset_seconds, e)
        return False


def video_duration_seconds(video_path: Path) -> float:
    """Return best-effort duration in seconds (defaults to 30 on failure)."""
    try:
        import imageio.v2 as imageio  # type: ignore[import-untyped]

        r = imageio.get_reader(str(video_path), "ffmpeg")
        try:
            meta = r.get_meta_data() or {}
            d = float(meta.get("duration") or 0)
            if d > 0:
                return max(0.25, d)
            fps = float(meta.get("fps") or 30) or 30.0
            nf = int(r.count_frames())
            if nf > 0 and fps > 0:
                return max(0.25, nf / fps)
        finally:
            r.close()
    except ImportError:
        d = _duration_via_ffprobe(video_path)
        return d if d is not None else 30.0
    except Exception as e:
        logger.warning("thumbnail_frame_extract: duration failed for %s: %s", video_path, e)
    d = _duration_via_ffprobe(video_path)
    return d if d is not None else 30.0


def extract_jpeg_at_offset(
    video_path: Path,
    output_path: Path,
    offset_seconds: float,
    *,
    max_side: int = 1080,
    jpeg_quality: int = 92,
) -> bool:
    """
    Decode one frame near ``offset_seconds`` and save a scaled JPEG.

    Returns True when the output exists and meets the minimum byte size gate.
    """
    try:
        import imageio.v2 as imageio  # type: ignore[import-untyped]
        import numpy as np
        from PIL import Image
    except ImportError as e:
        logger.warning("thumbnail_frame_extract: imageio unavailable (%s); using ffmpeg fallback", e)
        return _extract_via_ffmpeg(
            video_path,
            output_path,
            offset_seconds,
            max_side=max_side,
            jpeg_quality=jpeg_quality,
        )

    try:
        r = imageio.get_reader(str(video_path), "ffmpeg")
        try:
            meta = r.get_meta_data() or {}
            fps = float(meta.get("fps") or 30) or 30.0
            dur = float(meta.get("duration") or 0)
            nframes: Optional[int] = None
            if dur <= 0:
                nframes = int(r.count_frames())
                dur = nframes / fps if nframes and fps else 30.0
            t = max(0.0, min(float(offset_seconds), max(0.0, dur - 0.05)))
            idx = int(t * fps)
            if nframes is not None:
                idx = max(0, min(idx, max(0, nframes - 1)))
            else:
                idx = max(0, idx)
                try:
                    nframes = int(r.count_frames())
                    idx = max(0, min(idx, max(0, nframes - 1)))
                except Exception:
                    pass
            r.set_image_index(idx)
            frame = r.get_next_data()
        finally:
            r.close()

        im = Image.fromarray(frame).convert("RGB")
        w, h = im.size
        if w <= 0 or h <= 0:
            return False
        if w >= h:
            nw = max_side
            nh = max(1, int(h * (max_side / w)))
        else:
            nh = max_side
            nw = max(1, int(w * (max_side / h)))
        im = im.resize((nw, nh), Image.Resampling.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(output_path, "JPEG", quality=jpeg_quality, optimize=True)
        return output_path.exists() and output_path.stat().st_size >= MIN_THUMB_BYTES
    except Exception as e:
        logger.debug("thumbnail_frame_extract: extract failed at %.2fs: %s", offset_seconds, e)
        return _extract_via_ffmpeg(
            video_path,
            output_path,
            offset_seconds,
            max_side=max_side,
            jpeg_quality=jpeg_quality,
        )


def laplacian_variance_score(image_path: Path) -> float:
    """Cheap sharpness proxy (higher = sharper). Returns 0 on failure."""
    try:
        import numpy as np
        from PIL import Image

        g = np.asarray(Image.open(image_path).convert("L"), dtype=np.float64)
        if g.size < 4:
            return 0.0
        gx = np.diff(g, axis=1)
        gy = np.diff(g, axis=0)
        return float(np.var(gx) + np.var(gy))
    except Exception as e:
        logger.debug("thumbnail_frame_extract: sharpness failed for %s: %s", image_path, e)
        return 0.0


def extract_frame_default_offset(video_path: Path, thumb_path: Path) -> Tuple[float, bool]:
    """
    Extract one JPEG at ~30% duration (same intent as legacy ffmpeg helper).

    Returns ``(offset_seconds, success)``.
    """
    dur = video_duration_seconds(video_path)
    offset = max(0.5, dur * 0.30)
    ok = extract_jpeg_at_offset(video_path, thumb_path, offset)
    if not ok and offset > 0.5:
        offset = min(1.0, dur * 0.05)
        ok = extract_jpeg_at_offset(video_path, thumb_path, offset)
    return offset, ok
