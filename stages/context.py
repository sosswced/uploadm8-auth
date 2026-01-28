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
    """
    Processing context that flows through all pipeline stages.
    
    Contains:
    - Job identification
    - User info and entitlements
    - File paths
    - Processing outputs
    - Platform results
    - Timestamps
    """
    
    # Job identification
    job_id: str
    upload_id: str
    user_id: str
    idempotency_key: str = ""
    
    # Job state tracking (server-authoritative)
    state: str = "queued"  # queued | processing | cancel_requested | cancelled | failed | succeeded
    stage: str = "init"    # init | download | telemetry | captions | hud | watermark | thumbnail | uploading | notify | done
    attempt_count: int = 1
    max_attempts: int = 3
    
    # Source info
    source_r2_key: str = ""
    telemetry_r2_key: Optional[str] = None
    filename: str = ""
    file_size: int = 0
    content_type: str = "video/mp4"
    
    # Upload metadata
    platforms: List[str] = field(default_factory=list)
    title: str = ""
    caption: str = ""
    hashtags: List[str] = field(default_factory=list)
    privacy: str = "public"
    scheduled_time: Optional[datetime] = None
    schedule_mode: str = "immediate"  # immediate | smart | manual
    
    # User settings (from user_settings table)
    user_settings: Dict[str, Any] = field(default_factory=dict)
    
    # Entitlements (loaded from user tier)
    entitlements: Optional[Entitlements] = None
    
    # Temp directory and local paths
    temp_dir: Optional[Path] = None
    local_video_path: Optional[Path] = None
    local_telemetry_path: Optional[Path] = None
    
    # Processing outputs
    processed_video_path: Optional[Path] = None
    processed_r2_key: Optional[str] = None
    thumbnail_path: Optional[Path] = None
    thumbnail_r2_key: Optional[str] = None
    
    # AI generated content
    ai_title: Optional[str] = None
    ai_caption: Optional[str] = None
    ai_hashtags: List[str] = field(default_factory=list)
    
    # Telemetry data (parsed from .map file)
    telemetry_data: Optional[Dict[str, Any]] = None
    max_speed_mph: Optional[float] = None
    
    # Output artifacts (R2 keys)
    output_artifacts: Dict[str, str] = field(default_factory=dict)
    
    # Platform results
    platform_results: List[PlatformResult] = field(default_factory=list)
    
    # Timing
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    
    # Error tracking
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    # Cancel flag
    cancel_requested: bool = False
    
    def mark_stage(self, stage: str):
        """Update current stage."""
        self.stage = stage
    
    def mark_error(self, code: str, message: str, details: dict = None):
        """Record an error."""
        self.error_code = code
        self.error_message = message
        self.errors.append({
            "code": code,
            "message": message,
            "details": details or {},
            "stage": self.stage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    
    def add_platform_result(self, result: PlatformResult):
        """Add a platform upload result."""
        self.platform_results.append(result)
    
    def get_final_video_path(self) -> Optional[Path]:
        """Get the final video path (processed or original)."""
        return self.processed_video_path or self.local_video_path
    
    def get_final_r2_key(self) -> str:
        """Get the final R2 key (processed or original)."""
        return self.processed_r2_key or self.source_r2_key
    
    def get_effective_title(self) -> str:
        """Get title, falling back to AI-generated or filename."""
        return self.title or self.ai_title or Path(self.filename).stem
    
    def get_effective_caption(self) -> str:
        """Get caption, falling back to AI-generated."""
        return self.caption or self.ai_caption or ""
    
    def get_effective_hashtags(self) -> List[str]:
        """Get hashtags, combining user-provided and AI-generated."""
        combined = list(self.hashtags) + list(self.ai_hashtags)
        # Dedupe while preserving order
        seen = set()
        result = []
        for h in combined:
            h_lower = h.lower().strip().lstrip('#')
            if h_lower and h_lower not in seen:
                seen.add(h_lower)
                result.append(h_lower)
        
        # Apply limit from entitlements
        max_tags = 9999
        if self.entitlements:
            max_tags = self.entitlements.max_hashtags if not self.entitlements.unlimited_hashtags else 9999
        
        return result[:max_tags]
    
    def is_success(self) -> bool:
        """Check if job completed successfully."""
        if not self.platform_results:
            return False
        return any(r.success for r in self.platform_results)
    
    def is_partial_success(self) -> bool:
        """Check if some platforms succeeded but not all."""
        if not self.platform_results:
            return False
        successes = sum(1 for r in self.platform_results if r.success)
        return 0 < successes < len(self.platform_results)
    
    def get_success_platforms(self) -> List[str]:
        """Get list of platforms that succeeded."""
        return [r.platform for r in self.platform_results if r.success]
    
    def get_failed_platforms(self) -> List[str]:
        """Get list of platforms that failed."""
        return [r.platform for r in self.platform_results if not r.success]
    
    def to_summary_dict(self) -> dict:
        """Convert to summary for database storage."""
        return {
            "job_id": self.job_id,
            "upload_id": self.upload_id,
            "state": self.state,
            "stage": self.stage,
            "attempt_count": self.attempt_count,
            "platforms": self.platforms,
            "platform_results": [
                {
                    "platform": r.platform,
                    "success": r.success,
                    "video_id": r.platform_video_id,
                    "url": r.platform_url,
                    "error": r.error_message,
                }
                for r in self.platform_results
            ],
            "output_artifacts": self.output_artifacts,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


def create_context(
    job_data: dict,
    upload_record: dict,
    user_settings: dict,
    entitlements: Entitlements
) -> JobContext:
    """Create a JobContext from job queue data and database records."""
    
    ctx = JobContext(
        job_id=job_data.get("job_id", ""),
        upload_id=job_data.get("upload_id", ""),
        user_id=job_data.get("user_id", ""),
        idempotency_key=job_data.get("idempotency_key", ""),
        attempt_count=job_data.get("attempt_count", 1),
    )
    
    # From upload record
    if upload_record:
        ctx.source_r2_key = upload_record.get("r2_key", "")
        ctx.telemetry_r2_key = upload_record.get("telemetry_r2_key")
        ctx.filename = upload_record.get("filename", "")
        ctx.file_size = upload_record.get("file_size", 0)
        ctx.platforms = upload_record.get("platforms", [])
        ctx.title = upload_record.get("title", "")
        ctx.caption = upload_record.get("caption", "")
        ctx.privacy = upload_record.get("privacy", "public")
        ctx.scheduled_time = upload_record.get("scheduled_time")
        ctx.schedule_mode = upload_record.get("schedule_mode", "immediate")
        
        # Parse hashtags from caption if present
        caption = ctx.caption or ""
        if '#' in caption:
            import re
            tags = re.findall(r'#(\w+)', caption)
            ctx.hashtags = tags
    
    # User settings
    ctx.user_settings = user_settings or {}
    
    # Entitlements
    ctx.entitlements = entitlements
    
    return ctx
