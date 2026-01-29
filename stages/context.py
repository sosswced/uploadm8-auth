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
    
    platform_results: List[PlatformResult] = field(default_factory=list)
    output_artifacts: Dict[str, str] = field(default_factory=dict)
    
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    cancel_requested: bool = False
    
    put_cost: int = 0
    aic_cost: int = 0
    compute_seconds: float = 0.0
    
    def mark_stage(self, stage: str):
        self.stage = stage
    
    def mark_error(self, code: str, message: str, retryable: bool = False):
        self.errors.append({"code": code, "message": message, "stage": self.stage, "retryable": retryable, "timestamp": datetime.now(timezone.utc).isoformat()})
    
    def get_final_video_path(self) -> Optional[Path]:
        return self.processed_video_path or self.local_video_path
    
    def get_video_for_platform(self, platform: str) -> Optional[Path]:
        """Get the best video file for a specific platform"""
        # First check platform-specific transcoded versions
        if platform in self.platform_videos:
            return self.platform_videos[platform]
        # Fall back to processed video
        if self.processed_video_path and self.processed_video_path.exists():
            return self.processed_video_path
        # Fall back to original
        return self.local_video_path
    
    def get_effective_title(self) -> str:
        return self.ai_title or self.title or self.filename
    
    def get_effective_caption(self) -> str:
        return self.ai_caption or self.caption or ""
    
    def get_effective_hashtags(self) -> List[str]:
        return self.ai_hashtags if self.ai_hashtags else self.hashtags
    
    def is_success(self) -> bool:
        return any(r.success for r in self.platform_results)
    
    def get_success_platforms(self) -> List[str]:
        return [r.platform for r in self.platform_results if r.success]


def create_context(job_data: dict, upload_record: dict, user_settings: dict, entitlements: Entitlements) -> JobContext:
    return JobContext(
        job_id=job_data.get("job_id", ""),
        upload_id=str(upload_record.get("id", "")),
        user_id=str(upload_record.get("user_id", "")),
        idempotency_key=job_data.get("idempotency_key", ""),
        source_r2_key=upload_record.get("r2_key", ""),
        telemetry_r2_key=upload_record.get("telemetry_r2_key"),
        filename=upload_record.get("filename", ""),
        file_size=upload_record.get("file_size", 0),
        platforms=upload_record.get("platforms", []),
        title=upload_record.get("title", ""),
        caption=upload_record.get("caption", ""),
        hashtags=upload_record.get("hashtags", []),
        privacy=upload_record.get("privacy", "public"),
        entitlements=entitlements,
        user_settings=user_settings or {},
        put_cost=upload_record.get("put_reserved", 0),
        aic_cost=upload_record.get("aic_reserved", 0),
    )
