"""User-friendly labels for worker processing_stage values."""

from __future__ import annotations

STAGE_LABELS: dict[str, str] = {
    "init": "Initialising",
    "download": "Copying your video",
    "telemetry": "Reading drive data",
    "watermark": "Applying watermark",
    "transcode": "Building platform formats",
    "audio": "Analyzing audio",
    "vision": "AI scene scan",
    "twelvelabs": "AI scene scan",
    "video_intelligence": "AI scene scan",
    "thumbnail": "Creating thumbnails",
    "caption": "Writing title and captions",
    "upload": "Saving processed files",
    "publish": "Publishing to platforms",
    "verify": "Final checks",
    "notify": "Final checks",
}


def stage_label_for(stage: str | None) -> str:
    """Map ``processing_stage`` to a user-friendly label."""
    s = str(stage or "").strip().lower()
    if not s:
        return "Processing"
    return STAGE_LABELS.get(s, s.replace("_", " ").title())
