"""
UploadM8 Job Context
====================
Carries all state through the processing pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from .entitlements import Entitlements


@dataclass
class TelemetryData:
    """Parsed telemetry data from .map file."""
    points: List[Dict[str, Any]] = field(default_factory=list)
    max_speed_mph: float = 0.0
    avg_speed_mph: float = 0.0
    total_distance_miles: float = 0.0
    duration_seconds: float = 0.0
    max_altitude_ft: float = 0.0
    speeding_seconds: float = 0.0
    euphoria_seconds: float = 0.0


@dataclass
class TrillScore:
    """Calculated Trill score from telemetry."""
    total: int = 0
    speed_score: int = 0
    distance_score: int = 0
    duration_score: int = 0
    altitude_score: int = 0
    thrill_factor: float = 1.0


@dataclass
class PlatformResult:
    """Result of uploading to a single platform."""
    platform: str
    success: bool
    platform_video_id: Optional[str] = None
    platform_url: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    views: int = 0
    likes: int = 0


@dataclass
class JobContext:
    """Processing context that flows through all pipeline stages."""

    job_id: str
    upload_id: str
    user_id: str
    idempotency_key: str = ""
    state: str = "queued"
    stage: str = "init"
    attempt_count: int = 0

    source_r2_key: str = ""
    telemetry_r2_key: Optional[str] = None
    processed_r2_key: Optional[str] = None
    thumbnail_r2_key: Optional[str] = None

    filename: str = ""
    file_size: int = 0
    platforms: List[str] = field(default_factory=list)
    target_accounts: List[str] = field(default_factory=list)

    title: str = ""
    caption: str = ""
    hashtags: List[str] = field(default_factory=list)
    privacy: str = "public"

    entitlements: Optional[Entitlements] = None
    user_settings: Dict[str, Any] = field(default_factory=dict)

    temp_dir: Optional[Path] = None
    local_video_path: Optional[Path] = None
    local_telemetry_path: Optional[Path] = None
    processed_video_path: Optional[Path] = None
    thumbnail_path: Optional[Path] = None

    # Platform-specific transcoded videos (platform -> Path)
    platform_videos: Dict[str, Path] = field(default_factory=dict)

    # Video metadata from ffprobe
    video_info: Dict[str, Any] = field(default_factory=dict)

    ai_title: Optional[str] = None
    ai_caption: Optional[str] = None
    ai_hashtags: List[str] = field(default_factory=list)

    telemetry_data: Optional[TelemetryData] = None
    trill_score: Optional[TrillScore] = None

    # ── Location from reverse geocoding of .map GPS coords ──────────────────
    # Set by telemetry_stage after reverse geocoding.
    # Used by caption_stage to inject location into AI prompts.
    # Example: "Las Vegas, Nevada" or "Los Angeles, California"
    location_name: Optional[str] = None
    # ─────────────────────────────────────────────────────────────────────────

    platform_results: List[PlatformResult] = field(default_factory=list)

    cancel_requested: bool = False
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    schedule_mode: str = "immediate"
    scheduled_time: Optional[datetime] = None

    def mark_stage(self, stage: str):
        self.stage = stage

    def mark_error(self, code: str, message: str):
        self.error_code = code
        self.error_message = message


def create_context(
    job_data: dict,
    upload_record: dict,
    user_settings: dict,
    entitlements: Entitlements,
) -> JobContext:
    """
    Build a JobContext from raw DB record and job payload.
    """
    ctx = JobContext(
        job_id=job_data.get("job_id", ""),
        upload_id=job_data.get("upload_id", ""),
        user_id=job_data.get("user_id", ""),
        idempotency_key=job_data.get("idempotency_key", ""),
    )

    ctx.source_r2_key = upload_record.get("r2_key", "")
    ctx.telemetry_r2_key = upload_record.get("telemetry_r2_key")
    ctx.filename = upload_record.get("filename", "")
    ctx.file_size = upload_record.get("file_size", 0)
    ctx.title = upload_record.get("title", "") or ""
    ctx.caption = upload_record.get("caption", "") or ""

    raw_hashtags = upload_record.get("hashtags") or []
    if isinstance(raw_hashtags, list):
        ctx.hashtags = [str(h) for h in raw_hashtags if h]
    elif isinstance(raw_hashtags, str):
        import json
        try:
            parsed = json.loads(raw_hashtags)
            ctx.hashtags = [str(h) for h in parsed if h] if isinstance(parsed, list) else [raw_hashtags]
        except Exception:
            ctx.hashtags = [raw_hashtags] if raw_hashtags else []
    else:
        ctx.hashtags = []

    raw_platforms = upload_record.get("platforms") or []
    ctx.platforms = list(raw_platforms) if raw_platforms else []

    ctx.privacy = upload_record.get("privacy", "public") or "public"
    ctx.schedule_mode = upload_record.get("schedule_mode", "immediate") or "immediate"
    ctx.scheduled_time = upload_record.get("scheduled_time")

    raw_accounts = upload_record.get("target_accounts") or []
    ctx.target_accounts = list(raw_accounts) if raw_accounts else []

    ctx.user_settings = user_settings or {}
    ctx.entitlements = entitlements

    return ctx
