"""Load a JPEG frame from a user's upload (R2) for Pikzels recreate / edit / score."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from core.config import R2_BUCKET_NAME
from core.r2 import _normalize_r2_key, get_s3_client

from services.hydration_from_upload_row import (
    hydration_context_from_upload_row,
    hydration_payload_from_upload_row,
)
from services.hydration_payload import hydration_brief_strings_compact
from stages.pikzels_api import fit_pikzels_prompt_to_budget
from services.thumbnail_studio import hydration_signal_lanes

logger = logging.getLogger(__name__)


def _ffmpeg_frame_at_offset(video_path: Path, thumb_path: Path, offset_seconds: Optional[float]) -> Tuple[float, bool]:
    """Decode one JPEG via imageio (no PATH ffmpeg/ffprobe)."""
    from services.thumbnail_frame_extract import extract_jpeg_at_offset, video_duration_seconds

    dur = video_duration_seconds(video_path)
    if offset_seconds is None:
        offset = max(0.5, dur * 0.30)
    else:
        offset = max(0.0, float(offset_seconds))
    if extract_jpeg_at_offset(video_path, thumb_path, offset):
        return offset, True
    fallback = min(1.0, max(0.15, dur * 0.05))
    return fallback, extract_jpeg_at_offset(video_path, thumb_path, fallback)


def append_hydration_to_prompt(
    prompt: str,
    upload_row: Dict[str, Any],
    *,
    use_hydration: bool,
    hydration_lane: str,
    max_len: int = 950,
) -> str:
    p = (prompt or "").strip()
    if not use_hydration:
        return p[:max_len]
    hp = hydration_payload_from_upload_row(upload_row)
    brief_bits: list[str] = []
    if hp.get("evidence"):
        for key in (
            "geo_context",
            "osd_context",
            "music_context",
            "trill_context",
            "speech_context",
            "vision_context",
        ):
            val = hydration_brief_strings_compact(hp).get(key)
            if val:
                label = {
                    "geo_context": "Geo",
                    "osd_context": "OSD",
                    "music_context": "Music",
                    "trill_context": "Trill",
                    "speech_context": "Speech",
                    "vision_context": "Vis",
                }.get(key, key)
                brief_bits.append(f"{label}: {val[:100]}")
    if brief_bits:
        suffix = " Hydr: " + "; ".join(brief_bits[:6]) + "."
    else:
        ctx = hydration_context_from_upload_row(upload_row)
        lanes = hydration_signal_lanes(ctx)
        if not lanes:
            return p[:max_len]
        lane_key = (hydration_lane or "combined").lower().strip()
        pick = None
        if lane_key == "first":
            pick = lanes[0]
        else:
            pick = next((x for x in lanes if x["key"] == lane_key), None)
        if pick is None:
            pick = next((x for x in lanes if x["key"] == "combined"), lanes[-1])
        suffix = (
            f" Hydration focus: {pick['directive']} using this evidence: "
            f"{str(pick.get('value') or '')[:240]}."
        )
    return fit_pikzels_prompt_to_budget((p + suffix).strip(), max_len)


async def load_upload_frame_jpeg_base64(
    upload_row: Dict[str, Any],
    frame_source: str,
    offset_seconds: Optional[float],
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns ``(base64_without_data_prefix, meta)`` for Pikzels ``image_base64``.
    """
    meta: Dict[str, Any] = {"frame_source": frame_source}
    src = (frame_source or "video_best").strip().lower()

    if src == "primary_thumbnail":
        key = str(upload_row.get("thumbnail_r2_key") or "").strip()
        if not key:
            raise ValueError("no_primary_thumbnail")
        norm = _normalize_r2_key(key)
        buf = await asyncio.to_thread(_download_r2_object_bytes, norm)
        meta["r2_key"] = norm
        meta["bytes"] = len(buf)
        return base64.b64encode(buf).decode("ascii"), meta

    if src not in ("video_best", "video_offset"):
        raise ValueError("bad_frame_source")

    r2_key = str(upload_row.get("processed_r2_key") or upload_row.get("r2_key") or "").strip()
    if not r2_key:
        raise ValueError("no_video_key")

    off_arg: Optional[float] = None
    if src == "video_offset":
        if offset_seconds is None:
            raise ValueError("offset_required")
        off_arg = float(offset_seconds)

    norm = _normalize_r2_key(r2_key)

    def _work() -> Tuple[float, bytes]:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "video.mp4"
            frame_path = tmp_path / "frame.jpg"
            s3 = get_s3_client()
            s3.download_file(R2_BUCKET_NAME, norm, str(video_path))
            used_offset, ok = _ffmpeg_frame_at_offset(video_path, frame_path, off_arg)
            if not ok or not frame_path.exists():
                raise RuntimeError("ffmpeg_failed")
            data = frame_path.read_bytes()
            return used_offset, data

    used_offset, raw = await asyncio.to_thread(_work)
    meta["r2_key"] = norm
    meta["offset_seconds"] = used_offset
    meta["bytes"] = len(raw)
    return base64.b64encode(raw).decode("ascii"), meta


def _download_r2_object_bytes(norm_key: str) -> bytes:
    s3 = get_s3_client()
    buf = io.BytesIO()
    s3.download_fileobj(R2_BUCKET_NAME, norm_key, buf)
    return buf.getvalue()
