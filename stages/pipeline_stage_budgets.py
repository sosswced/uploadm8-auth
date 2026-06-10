"""
Per-stage wall-clock budgets via asyncio.wait_for in worker.py. On timeout: log + pipeline diag.

Env: STAGE_TIMEOUT_<NAME>_SEC — AUDIO, VISION, TWELVELABS, VIDEO_INTELLIGENCE,
     THUMBNAIL, CAPTION, TRANSCODE, WATERMARK, PUBLISH (0 = unlimited).

Legacy aliases: STAGE_TIMEOUT_<NAME>_SECONDS (worker historical names).
"""

from __future__ import annotations

import os
from typing import Dict


def _sec(name: str, default: int, *, legacy_suffix: str = "_SECONDS") -> float:
    primary = f"STAGE_TIMEOUT_{name}_SEC"
    legacy = f"STAGE_TIMEOUT_{name}{legacy_suffix}"
    raw = (os.environ.get(primary) or os.environ.get(legacy) or str(default)).strip()
    try:
        v = int(raw or default)
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def stage_timeout_watermark() -> float:
    return _sec("WATERMARK", 660)


def stage_timeout_transcode() -> float:
    return _sec("TRANSCODE", 1800)


def stage_timeout_thumbnail() -> float:
    return _sec("THUMBNAIL", 600)


def stage_timeout_publish() -> float:
    return _sec("PUBLISH", 900)


def stage_timeout_audio() -> float:
    return _sec("AUDIO", 240)


def stage_timeout_vision() -> float:
    return _sec("VISION", 120)


def stage_timeout_twelvelabs() -> float:
    return _sec("TWELVELABS", 420)


def stage_timeout_video_intelligence() -> float:
    return _sec("VIDEO_INTELLIGENCE", 1800)


def stage_timeout_caption() -> float:
    return _sec("CAPTION", 360)


def get_all_budgets() -> Dict[str, float]:
    return {
        "watermark": stage_timeout_watermark(),
        "transcode": stage_timeout_transcode(),
        "thumbnail": stage_timeout_thumbnail(),
        "publish": stage_timeout_publish(),
        "audio": stage_timeout_audio(),
        "vision": stage_timeout_vision(),
        "twelvelabs": stage_timeout_twelvelabs(),
        "video_intelligence": stage_timeout_video_intelligence(),
        "caption": stage_timeout_caption(),
    }
