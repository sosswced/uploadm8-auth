"""
UploadM8 Job Context
====================
Single context object passed through all stages.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class TelemetryData:
    """Parsed telemetry data from .map file."""
    data_points: List[Dict[str, float]] = field(default_factory=list)
    max_speed: float = 0.0
    avg_speed: float = 0.0
    total_duration: float = 0.0
    distance_miles: float = 0.0


@dataclass
class TrillScore:
    """Trill score calculation results."""
    score: int = 0
    bucket: str = "chill"  # chill, spirited, sendIt, euphoric, gloryBoy
    speed_score: float = 0.0
    speeding_score: float = 0.0
    euphoria_score: float = 0.0
    consistency_score: float = 0.0
    excessive_speed: bool = False
    title_modifier: str = ""
    hashtags: List[str] = field(default_factory=list)


@dataclass
class CaptionResult:
    """Generated caption and title."""
    title: str = ""
    caption: str = ""
    hashtags: List[str] = field(default_factory=list)
    generated_by: str = "manual"  # manual, trill, ai


@dataclass
class PlatformResult:
    """Result of publishing to a single platform."""
    platform: str = ""
    success: bool = False
    publish_id: Optional[str] = None
    video_id: Optional[str] = None
    url: Optional[str] = None
    error: Optional[str] = None


@dataclass
class Entitlements:
    """User's tier entitlements."""
    tier: str = "starter"
    can_generate_captions: bool = False
    can_burn_hud: bool = False
    can_use_ai_captions: bool = False
    max_uploads_per_month: int = 10
    max_accounts: int = 1
    priority_processing: bool = False
    can_schedule: bool = False
    can_export: bool = False


@dataclass
class JobContext:
    """
    Context object passed through all processing stages.
    Each stage reads and augments this context.
    """
    # Identity
    job_id: str = ""
    upload_id: str = ""
    user_id: str = ""
    
    # Source files
    source_r2_key: str = ""
    telemetry_r2_key: Optional[str] = None
    filename: str = ""
    file_size: int = 0
    
    # Local temp paths (populated by download stage)
    temp_dir: Optional[Path] = None
    local_video_path: Optional[Path] = None
    local_telemetry_path: Optional[Path] = None
    processed_video_path: Optional[Path] = None
    
    # Upload metadata
    platforms: List[str] = field(default_factory=list)
    original_title: str = ""
    original_caption: str = ""
    privacy: str = "public"
    scheduled_time: Optional[datetime] = None
    schedule_mode: str = "immediate"
    
    # User preferences
    user_settings: Dict[str, Any] = field(default_factory=dict)
    discord_webhook: Optional[str] = None
    
    # Entitlements
    entitlements: Entitlements = field(default_factory=Entitlements)
    
    # Stage results
    telemetry: Optional[TelemetryData] = None
    trill: Optional[TrillScore] = None
    caption: Optional[CaptionResult] = None
    
    # Processing state
    processed_r2_key: Optional[str] = None
    hud_applied: bool = False
    transcoded: bool = False
    
    # Publish results
    platform_results: List[PlatformResult] = field(default_factory=list)
    
    # Status
    status: str = "pending"
    error_code: Optional[str] = None
    error_detail: Optional[str] = None
    
    # Timing
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    
    @property
    def has_telemetry(self) -> bool:
        return bool(self.telemetry_r2_key)
    
    @property
    def final_title(self) -> str:
        """Get the final title (generated or original)."""
        if self.caption and self.caption.title:
            return self.caption.title
        return self.original_title or self.filename
    
    @property
    def final_caption(self) -> str:
        """Get the final caption with hashtags."""
        base = ""
        if self.caption and self.caption.caption:
            base = self.caption.caption
        else:
            base = self.original_caption
        
        hashtags = []
        if self.caption and self.caption.hashtags:
            hashtags.extend(self.caption.hashtags)
        if self.trill and self.trill.hashtags:
            hashtags.extend(self.trill.hashtags)
        
        if hashtags:
            unique_tags = list(dict.fromkeys(hashtags))
            base = f"{base}\n\n{' '.join(unique_tags)}".strip()
        
        return base
    
    @property
    def all_succeeded(self) -> bool:
        if not self.platform_results:
            return False
        return all(r.success for r in self.platform_results)
    
    @property
    def any_succeeded(self) -> bool:
        return any(r.success for r in self.platform_results)
    
    def get_failed_platforms(self) -> List[str]:
        return [r.platform for r in self.platform_results if not r.success]
    
    def get_succeeded_platforms(self) -> List[str]:
        return [r.platform for r in self.platform_results if r.success]


def create_context(
    job_data: dict,
    upload_record: dict,
    user_settings: dict,
    entitlements: Entitlements
) -> JobContext:
    """Create a JobContext from job payload and database records."""
    return JobContext(
        job_id=job_data.get("job_id", ""),
        upload_id=job_data.get("upload_id", ""),
        user_id=job_data.get("user_id", ""),
        source_r2_key=upload_record.get("r2_key", ""),
        telemetry_r2_key=upload_record.get("telemetry_r2_key"),
        filename=upload_record.get("filename", ""),
        file_size=upload_record.get("file_size", 0),
        platforms=upload_record.get("platforms", []) or [],
        original_title=upload_record.get("title", ""),
        original_caption=upload_record.get("caption", ""),
        privacy=upload_record.get("privacy", "public"),
        scheduled_time=upload_record.get("scheduled_time"),
        schedule_mode=upload_record.get("schedule_mode", "immediate"),
        user_settings=user_settings,
        discord_webhook=user_settings.get("discord_webhook"),
        entitlements=entitlements,
    )
