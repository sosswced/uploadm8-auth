"""
Per-stage wall-clock budgets via asyncio.wait_for in worker.py. On timeout: log + pipeline diag.

Env: STAGE_TIMEOUT_<NAME>_SEC — AUDIO, VISION, TWELVELABS, VIDEO_INTELLIGENCE,
     THUMBNAIL, CAPTION, TRANSCODE (0 = unlimited).
"""

from __future__ import annotations

import os


def _sec(name: str, default: int) -> float:
    try:
        v = int(os.environ.get(f"STAGE_TIMEOUT_{name}_SEC", str(default)) or default)
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def stage_timeout_audio() -> float:
    return _sec("AUDIO", 240)


def stage_timeout_vision() -> float:
    return _sec("VISION", 120)


def stage_timeout_twelvelabs() -> float:
    return _sec("TWELVELABS", 420)


def stage_timeout_video_intelligence() -> float:
    return _sec("VIDEO_INTELLIGENCE", 360)


def stage_timeout_thumbnail() -> float:
    return _sec("THUMBNAIL", 600)


def stage_timeout_caption() -> float:
    return _sec("CAPTION", 360)


def stage_timeout_transcode() -> float:
    return _sec("TRANSCODE", 0)
