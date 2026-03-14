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

    # GPS / location — populated by telemetry_stage after reverse geocoding
    start_lat: Optional[float] = None
    start_lon: Optional[float] = None
    mid_lat: Optional[float] = None       # midpoint of the route (best representative point)
    mid_lon: Optional[float] = None
    location_city: Optional[str] = None
    location_state: Optional[str] = None
    location_country: Optional[str] = None
    location_display: Optional[str] = None  # e.g. "Kansas City, MO" or "Los Angeles, CA"
    location_road: Optional[str] = None     # road/highway name if available

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

    # Single best thumbnail (auto-selected by quality score)
    thumbnail_path: Optional[Path] = None
    # All candidate thumbnails generated (user can pick preferred later)
    thumbnail_paths: List[Path] = field(default_factory=list)
    # Sharpness scores keyed by str(path) — higher = sharper
    thumbnail_scores: Dict[str, float] = field(default_factory=dict)

    platform_videos: Dict[str, Path] = field(default_factory=dict)
    video_info: Dict[str, Any] = field(default_factory=dict)

    ai_title: Optional[str] = None
    ai_caption: Optional[str] = None
    ai_hashtags: List[str] = field(default_factory=list)
    # Few-shot examples from upload_caption_memory (set by caption_stage when db_pool provided)
    caption_memory_examples: List[Dict[str, Any]] = field(default_factory=list)

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

    def get_effective_hashtags(self, platform: str = "") -> List[str]:
        """
        Build the final hashtag list for publishing.

        Merge order: always_hashtags (FIRST) → platform_hashtags for this platform →
        base upload hashtags → AI additions.
        blocked_hashtags are filtered out at every level.

        Uses the same pipeline for always_hashtags and platform_hashtags so both
        are applied consistently. platform_hashtags keys: tiktok, youtube, instagram, facebook.
        """
        us = self.user_settings or {}

        # ── Always-on hashtags from settings page ────────────────────────
        raw_always = us.get("alwaysHashtags") or us.get("always_hashtags") or []
        if isinstance(raw_always, str):
            raw_always = [
                t.strip() for t in raw_always.replace(",", " ").split() if t.strip()
            ]
        always_tags: List[str] = list(raw_always) if isinstance(raw_always, list) else []

        # ── Platform-specific hashtags (same logic as always_hashtags) ────
        platform_tags: List[str] = []
        if platform:
            ph = us.get("platformHashtags") or us.get("platform_hashtags") or {}
            if isinstance(ph, str):
                try:
                    import json as _j
                    ph = _j.loads(ph)
                except Exception:
                    ph = {}
            if isinstance(ph, dict):
                raw = ph.get(platform) or ph.get(platform.lower()) or []
                if isinstance(raw, str):
                    raw = [t.strip() for t in raw.replace(",", " ").split() if t.strip()]
                platform_tags = list(raw) if isinstance(raw, list) else []

        # ── Blocked hashtags from settings page ──────────────────────────
        raw_blocked = us.get("blockedHashtags") or us.get("blocked_hashtags") or []
        if isinstance(raw_blocked, str):
            raw_blocked = [
                t.strip() for t in raw_blocked.replace(",", " ").split() if t.strip()
            ]
        blocked_set = {
            str(t).strip().lstrip("#").lower()
            for t in (raw_blocked if isinstance(raw_blocked, list) else [])
        }

        # ── Merge: always → platform → base → AI (same pipeline for all) ─
        base = list(self.hashtags or [])
        ai   = list(self.ai_hashtags or [])

        seen: set = set()
        merged: List[str] = []

        for tag in always_tags + platform_tags + base + ai:
            t = str(tag).strip().lstrip("#").lower()
            if not t or t in seen or t in blocked_set:
                continue
            seen.add(t)
            merged.append(tag if str(tag).startswith("#") else f"#{tag}")

        return merged

    def is_success(self) -> bool:
        return any(r.success for r in self.platform_results)

    def is_partial_success(self) -> bool:
        return any(r.success for r in self.platform_results) and any(
            (not r.success) for r in self.platform_results
        )

    def get_success_platforms(self) -> List[str]:
        return [r.platform for r in self.platform_results if r.success]

    # ── Convenience location properties ───────────────────────────────────────
    # The worker and some pipeline stages access ctx.location_name directly.
    # These proxy the TelemetryData fields so callers don't need to drill into
    # ctx.telemetry — and they're always safe (return None when no telemetry).

    @property
    def location_name(self) -> Optional[str]:
        """Reverse-geocoded display string, e.g. 'Kansas City, MO'.
        Proxies TelemetryData.location_display."""
        t = self.telemetry or self.telemetry_data
        return t.location_display if t else None

    @property
    def location_city(self) -> Optional[str]:
        """City from reverse geocoding."""
        t = self.telemetry or self.telemetry_data
        return t.location_city if t else None

    @property
    def location_state(self) -> Optional[str]:
        """State/region from reverse geocoding."""
        t = self.telemetry or self.telemetry_data
        return t.location_state if t else None

    @property
    def location_road(self) -> Optional[str]:
        """Road/highway name from reverse geocoding."""
        t = self.telemetry or self.telemetry_data
        return t.location_road if t else None


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
        target_accounts=upload_record.get("target_accounts", []) or [],
        title=upload_record.get("title", ""),
        caption=upload_record.get("caption", ""),
        hashtags=upload_record.get("hashtags", []) or [],
        privacy=upload_record.get("privacy", "public") or "public",
        reframe_mode=reframe_mode,
        user_settings=user_settings or {},
        entitlements=entitlements,
    )
