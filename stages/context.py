"""
UploadM8 Job Context
====================
Carries all state through the processing pipeline.
"""

from __future__ import annotations

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

    # -------------------------
    # Back-compat aliases used by older stages
    # -------------------------
    @property
    def data_points(self) -> List[Dict[str, Any]]:
        return self.points

    @data_points.setter
    def data_points(self, v: List[Dict[str, Any]]):
        self.points = v or []

    @property
    def max_speed(self) -> float:
        return self.max_speed_mph

    @max_speed.setter
    def max_speed(self, v: float):
        self.max_speed_mph = float(v or 0)

    @property
    def avg_speed(self) -> float:
        return self.avg_speed_mph

    @avg_speed.setter
    def avg_speed(self, v: float):
        self.avg_speed_mph = float(v or 0)

    @property
    def distance_miles(self) -> float:
        return self.total_distance_miles

    @distance_miles.setter
    def distance_miles(self, v: float):
        self.total_distance_miles = float(v or 0)

    @property
    def total_duration(self) -> float:
        return self.duration_seconds

    @total_duration.setter
    def total_duration(self, v: float):
        self.duration_seconds = float(v or 0)


@dataclass
class TrillScore:
    """Calculated Trill score from telemetry."""

    # Primary fields (used by telemetry_stage)
    score: int = 0
    bucket: str = "chill"
    speed_score: float = 0.0
    speeding_score: float = 0.0
    euphoria_score: float = 0.0
    consistency_score: float = 0.0
    excessive_speed: bool = False
    title_modifier: str = ""
    hashtags: List[str] = field(default_factory=list)

    # Legacy aliases
    @property
    def total(self) -> int:
        return self.score

    @total.setter
    def total(self, v: int):
        self.score = v

    @property
    def thrill_factor(self) -> float:
        return self.score / 100.0 if self.score else 0.0


@dataclass
class PlatformResult:
    """Result of publishing to a single platform."""

    platform: str
    success: bool

    # Step A (accepted)
    platform_video_id: Optional[str] = None
    platform_url: Optional[str] = None
    publish_id: Optional[str] = None

    # Audit/debug
    attempt_id: Optional[str] = None
    http_status: Optional[int] = None
    response_payload: Optional[Dict[str, Any]] = None

    # Errors
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    # Optional engagement
    views: int = 0
    likes: int = 0

    # Step B (confirmed)
    verify_status: str = "pending"  # pending/confirmed/rejected/unknown


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
    reframe_mode: str = "auto"  # auto | pad | crop | none

    entitlements: Optional[Entitlements] = None
    user_settings: Dict[str, Any] = field(default_factory=dict)

    temp_dir: Optional[Path] = None
    local_video_path: Optional[Path] = None
    local_telemetry_path: Optional[Path] = None
    processed_video_path: Optional[Path] = None
    thumbnail_path: Optional[Path] = None

    platform_videos: Dict[str, Path] = field(default_factory=dict)
    video_info: Dict[str, Any] = field(default_factory=dict)

    ai_title: Optional[str] = None
    ai_caption: Optional[str] = None
    ai_hashtags: List[str] = field(default_factory=list)

    telemetry_data: Optional[TelemetryData] = None
    trill_score: Optional[TrillScore] = None

    # Back-compat aliases used by older stages
    telemetry: Optional[TelemetryData] = None
    trill: Optional[TrillScore] = None
    hud_applied: bool = False

    platform_results: List[PlatformResult] = field(default_factory=list)
    output_artifacts: Dict[str, str] = field(default_factory=dict)

    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    cancel_requested: bool = False

    # Explicit error tracking
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    put_cost: int = 0
    aic_cost: int = 0
    compute_seconds: float = 0.0

    def __post_init__(self):
        # Keep legacy and new fields in sync
        if self.telemetry is None and self.telemetry_data is not None:
            self.telemetry = self.telemetry_data
        if self.telemetry_data is None and self.telemetry is not None:
            self.telemetry_data = self.telemetry

        if self.trill is None and self.trill_score is not None:
            self.trill = self.trill_score
        if self.trill_score is None and self.trill is not None:
            self.trill_score = self.trill

    def mark_stage(self, stage: str):
        self.stage = stage

    def mark_error(self, code: str, message: str, retryable: bool = False):
        self.error_code = code
        self.error_message = message
        self.errors.append(
            {
                "code": code,
                "message": message,
                "stage": self.stage,
                "retryable": retryable,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def get_failed_platforms(self) -> List[str]:
        return [r.platform for r in self.platform_results if not r.success]

    def get_final_video_path(self) -> Optional[Path]:
        return self.processed_video_path or self.local_video_path

    def get_video_for_platform(self, platform: str) -> Optional[Path]:
        if platform in self.platform_videos:
            return self.platform_videos[platform]
        if self.processed_video_path and self.processed_video_path.exists():
            return self.processed_video_path
        return self.local_video_path

    def get_effective_title(self) -> str:
        return self.ai_title or self.title or self.filename

    def get_effective_caption(self) -> str:
        return self.ai_caption or self.caption or ""

    def get_effective_hashtags(self) -> List[str]:
        return self.ai_hashtags if self.ai_hashtags else self.hashtags

    def is_success(self) -> bool:
        return any(r.success for r in self.platform_results)

    def is_partial_success(self) -> bool:
        return any(r.success for r in self.platform_results) and any(
            (not r.success) for r in self.platform_results
        )

    def get_success_platforms(self) -> List[str]:
        return [r.platform for r in self.platform_results if r.success]


def create_context(job_data: dict, upload_record: dict, user_settings: dict, entitlements: Entitlements) -> JobContext:
    # Resolve reframe_mode: job payload > upload record > user setting > "auto"
    reframe_mode = (
        job_data.get("reframe_mode")
        or upload_record.get("reframe_mode")
        or (user_settings or {}).get("reframe_mode")
        or "auto"
    )

    return JobContext(
        job_id=job_data.get("job_id", ""),
        upload_id=str(upload_record.get("id", "")),
        user_id=str(upload_record.get("user_id", "")),
        idempotency_key=job_data.get("idempotency_key", ""),
        source_r2_key=upload_record.get("r2_key", ""),
        telemetry_r2_key=upload_record.get("telemetry_r2_key"),
        filename=upload_record.get("filename", ""),
        file_size=upload_record.get("file_size", 0),
        platforms=upload_record.get("platforms", []) or [],
        title=upload_record.get("title", ""),
        caption=upload_record.get("caption", ""),
        hashtags=upload_record.get("hashtags", []) or [],
        privacy=upload_record.get("privacy", "public") or "public",
        reframe_mode=reframe_mode,
        user_settings=user_settings or {},
        entitlements=entitlements,
    )
