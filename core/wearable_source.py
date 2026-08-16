"""Meta Glasses / wearable export filename detection.

Live Meta AI / glasses exports look like ``mcp_video-45_singular_display.mov``
or ``od_video-1_singular_display.mov``. These must never be treated as
dashcam filename tokens.
"""

from __future__ import annotations

from typing import Any

_META_FILENAME_MARKERS = (
    "mcp_video-",
    "mcp_video_",
    "od_video-",
    "od_video_",
    "singular_display",
)


def is_meta_glasses_filename(filename: str) -> bool:
    """True for Meta Glasses / Meta AI app export names."""
    low = str(filename or "").strip().lower()
    if not low:
        return False
    return any(m in low for m in _META_FILENAME_MARKERS)


def source_needs_h264_proxy(ctx: Any, video_path: Any = None) -> bool:
    """HEVC or Meta ``.mov`` with unknown codec — Google stills/VI need H.264."""
    hevc = frozenset({"hevc", "h265", "hev1", "hvc1", "hevc1"})
    info = getattr(ctx, "video_info", None) or {}
    codec = ""
    if isinstance(info, dict):
        codec = str(info.get("video_codec") or info.get("codec") or "").strip().lower()
    if codec in hevc:
        return True
    path = str(
        video_path
        or getattr(ctx, "local_video_path", None)
        or getattr(ctx, "filename", None)
        or ""
    )
    low = path.lower()
    meta = is_meta_glasses_filename(path) or is_meta_glasses_filename(
        str(getattr(ctx, "filename", "") or "")
    )
    if meta and (low.endswith(".mov") or codec in ("", "unknown")):
        return True
    # Unknown-codec .mov is often HEVC (Meta / iPhone). Do not union a
    # tuple with a frozenset — that TypeError was swallowed by callers.
    if low.endswith(".mov") and codec in ("", "unknown"):
        return True
    return False


__all__ = [
    "is_meta_glasses_filename",
    "source_needs_h264_proxy",
]
