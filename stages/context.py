"""
UploadM8 Job Context
====================
Carries all state through the processing pipeline.
"""

from __future__ import annotations

import math
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.helpers import coerce_hashtag_list, coerce_jsonb_dict, sanitize_hashtag_body, strip_stray_hashtag_json_blob

from .entitlements import Entitlements


_PLACEHOLDER_TITLE_VALUES = {
    "video",
    "my video",
    "new video",
    "untitled",
    "untitled video",
    "upload",
    "uploadm8 video",
    "open road",
    "open road adventure",
    "road trip",
    "road trip vibes",
}

_PLACEHOLDER_CAPTION_VALUES = _PLACEHOLDER_TITLE_VALUES | {
    "check this out",
    "watch this",
    "new upload",
    "new clip",
    "road trip vibes",
    "travel vibes",
    "scenic drive",
    "open road vibes",
}


def is_placeholder_upload_title(title: str, filename: str = "") -> bool:
    """True for client defaults / stock titles that should not mask AI hydration."""
    t = strip_stray_hashtag_json_blob(str(title or "")).strip()
    if not t:
        return True
    low = re.sub(r"\s+", " ", t.lower()).strip()
    if low in _PLACEHOLDER_TITLE_VALUES:
        return True
    if re.search(
        r"\b(open road|endless horizons?|vast skies|scenic drive|adventure awaits|"
        r"journey unfolds|watch serenity meet motion|road ahead)\b",
        low,
    ):
        return True
    fname = str(filename or "").strip()
    if fname:
        stem = re.sub(r"\.[A-Za-z0-9]{2,5}$", "", Path(fname).name).strip()
        norm_title = re.sub(r"[^a-z0-9]+", "", low)
        norm_stem = re.sub(r"[^a-z0-9]+", "", stem.lower())
        if norm_stem and norm_title == norm_stem:
            return True
    return False


def is_placeholder_upload_caption(caption: str) -> bool:
    """True for stock/manual defaults that should not suppress hydrated captions."""
    c = strip_stray_hashtag_json_blob(str(caption or "")).strip()
    if not c:
        return True
    low = re.sub(r"\s+", " ", c.lower()).strip()
    if low in _PLACEHOLDER_CAPTION_VALUES:
        return True
    return bool(
        re.search(
            r"\b(open road|endless horizons?|vast skies|scenic drive|adventure awaits|"
            r"journey unfolds|watch serenity meet motion|good vibes|travel vibes|"
            r"road trip vibes)\b",
            low,
        )
    )


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
    # When route start is far from midpoint, second Nominatim lookup at start GPS
    location_start_display: Optional[str] = None

    # US Census gazetteer + PAD-US (worker: telemetry_stage; gazetteer file + PostGIS)
    gazetteer_place_name: Optional[str] = None
    gazetteer_state_usps: Optional[str] = None
    near_padus: bool = False
    padus_unit_name: Optional[str] = None

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
    # Speed-only subtotal before scenic/route significance boost (vision + geo + VI).
    base_score: int = 0
    scenic_boost: float = 0.0
    scenic_breakdown: Dict[str, float] = field(default_factory=dict)
    scenic_factors: List[str] = field(default_factory=list)

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
    """Result of publishing to a single platform/account."""

    platform: str
    success: bool

    # Step A (accepted)
    platform_video_id: Optional[str] = None
    platform_url: Optional[str] = None
    publish_id: Optional[str] = None

    # Account identity — set by publish_stage so we always know which platform_tokens row was used
    token_row_id: Optional[str] = None  # platform_tokens.id (UUID)
    account_id: Optional[str] = None  # platform's own account ID
    account_username: Optional[str] = None  # e.g. cedybandz5254
    account_name: Optional[str] = None  # e.g. Cedy Bandz
    account_avatar: Optional[str] = None  # avatar URL

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
    # TikTok Content Posting API export choices (per-account map under by_account)
    tiktok_post_settings: Dict[str, Any] = field(default_factory=dict)
    reframe_mode: str = "auto"  # auto | pad | crop | none
    # Scheduled publish (uploads.scheduled_time) — used for deferred/staged user comms.
    scheduled_time: Optional[datetime] = None

    entitlements: Optional[Entitlements] = None
    user_settings: Dict[str, Any] = field(default_factory=dict)
    # Free-tier burn-in settings (from admin_settings + worker; watermark_stage reads these).
    watermark_text: Optional[str] = None
    watermark_settings: Dict[str, Any] = field(default_factory=dict)

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
    # M8 caption engine: per-platform AI hashtag variants (tiktok, instagram, …)
    m8_platform_hashtags: Dict[str, List[str]] = field(default_factory=dict)
    m8_platform_titles: Dict[str, str] = field(default_factory=dict)
    m8_platform_captions: Dict[str, str] = field(default_factory=dict)
    # Optional: full style x tone x voice sweep from one M8 call (see m8_engine caption_evidence_matrix)
    m8_caption_evidence_matrix: Dict[str, Any] = field(default_factory=dict)
    # Per-platform claim↔evidence bindings from M8 grounding pass 2
    m8_platform_claims: Dict[str, Any] = field(default_factory=dict)
    # Populated by audio_stage (Whisper) when enabled — injected into caption prompts.
    ai_transcript: Optional[str] = None
    audio_path: Optional[Path] = None
    audio_context: Dict[str, Any] = field(default_factory=dict)
    vision_context: Dict[str, Any] = field(default_factory=dict)
    visual_recognition: Dict[str, Any] = field(default_factory=dict)
    video_understanding: Dict[str, Any] = field(default_factory=dict)
    video_intelligence_context: Dict[str, Any] = field(default_factory=dict)
    # Compact recognition view (object_tracks / on_screen_text /
    # person_segments / logos) extracted from video_intelligence_context for
    # fast access by the recognition aggregator + hydration enforcer +
    # thumbnail studio. Populated by stages.video_intelligence_stage.
    video_intelligence: Dict[str, Any] = field(default_factory=dict)
    # Per-upload recognition summary (top objects/text/logos, hydration_score)
    # persisted to upload_recognition_summary by services.recognition_engine.
    recognition_summary: Dict[str, Any] = field(default_factory=dict)
    # Burned-in dashcam HUD timeline (date/time/lat/lon/speed/heading parsed
    # from on-screen overlay across the full clip). Populated by
    # stages.dashcam_osd_stage. May also backfill ctx.telemetry_data when the
    # user did not supply a .map file.
    dashcam_osd_context: Dict[str, Any] = field(default_factory=dict)
    # Few-shot examples from upload_caption_memory (set by caption_stage when db_pool provided)
    caption_memory_examples: List[Dict[str, Any]] = field(default_factory=list)

    # SerpAPI / YouTube Data title trend sample (caption_stage → M8)
    trend_intel_context: Optional[Dict[str, Any]] = None

    telemetry_data: Optional[TelemetryData] = None
    trill_score: Optional[TrillScore] = None
    vehicle_make_id: Optional[int] = None
    vehicle_model_id: Optional[int] = None
    vehicle_make_name: Optional[str] = None
    vehicle_model_name: Optional[str] = None

    # Back-compat aliases used by older stages
    telemetry: Optional[TelemetryData] = None
    trill: Optional[TrillScore] = None
    platform_results: List[PlatformResult] = field(default_factory=list)
    output_artifacts: Dict[str, str] = field(default_factory=dict)
    # When AI_TRACE_ENABLED: ordered events for one `[ai-pipeline-summary]` log per upload.
    pipeline_ai_trace: List[Dict[str, Any]] = field(default_factory=list)
    # Canonical multimodal snapshot for thumbnails, Pikzels, M8 prompts, and UI (`output_artifacts`).
    hydration_payload: Optional[Dict[str, Any]] = None

    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    cancel_requested: bool = False

    # Smart deferred publish: when set, publish_stage only attempts these platforms
    # (lowercase names) and skips targets already present in platform_results.
    deferred_publish_platform_filter: Optional[set] = None

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

    def get_effective_title(self, platform: str = "") -> str:
        # Explicit user copy wins, but client defaults / stock titles must not mask hydration.
        t = (self.title or "").strip()
        ai = (self.ai_title or "").strip()
        if t and not (ai and is_placeholder_upload_title(t, self.filename)):
            return t
        pl = (platform or "").strip().lower()
        if pl:
            m8_titles = getattr(self, "m8_platform_titles", None) or {}
            if isinstance(m8_titles, dict) and m8_titles:
                tt = m8_titles.get(pl) or m8_titles.get(platform or "")
                if tt is None:
                    for mk, mv in m8_titles.items():
                        if str(mk).strip().lower() == pl:
                            tt = mv
                            break
                if tt and str(tt).strip():
                    return str(tt).strip()
        if ai:
            return ai
        return (self.filename or "").strip() or ""

    def get_effective_caption(self, platform: str = "") -> str:
        """Best caption for display or publish.

        User ``caption`` wins. Otherwise, when ``platform`` is set and M8 wrote
        ``m8_platform_captions[platform]``, that prose is used so each network can
        ship its own variant (aligned with ``get_effective_hashtags``). Falls back
        to ``ai_caption``. Strips AI JSON-hashtag glitches from all sources.
        """
        c = (self.caption or "").strip()
        ac = (self.ai_caption or "").strip()
        if c and not (ac and is_placeholder_upload_caption(c)):
            return strip_stray_hashtag_json_blob(c)
        pl = (platform or "").strip().lower()
        if pl:
            m8_caps = getattr(self, "m8_platform_captions", None) or {}
            if isinstance(m8_caps, dict) and m8_caps:
                cap = m8_caps.get(pl) or m8_caps.get(platform or "")
                if cap is None:
                    for mk, mv in m8_caps.items():
                        if str(mk).strip().lower() == pl:
                            cap = mv
                            break
                if cap and str(cap).strip():
                    return strip_stray_hashtag_json_blob(str(cap).strip())
        return strip_stray_hashtag_json_blob(ac)

    def get_effective_hashtags(self, platform: str = "") -> List[str]:
        """
        Build the final hashtag list for publishing.

        Merge order: always_hashtags (FIRST) → user platform_hashtags for this platform →
        base upload hashtags (explicit per-upload intent) → M8 per-platform AI hashtags →
        generic AI list. Under ``maxHashtags``, earlier buckets win so manual upload tags
        are not starved by model output.
        blocked_hashtags are filtered out at every level.

        platform_hashtags keys are usually tiktok / youtube / instagram / facebook (any casing).
        """
        us = self.user_settings or {}

        # ── Always-on hashtags from settings page ────────────────────────
        raw_always = us.get("alwaysHashtags") or us.get("always_hashtags") or []
        always_tags: List[str] = coerce_hashtag_list(raw_always)

        # ── User platform-specific hashtags (settings / preferences) ───────
        platform_tags: List[str] = []
        pl = (platform or "").strip().lower()
        if pl:
            ph = us.get("platformHashtags") or us.get("platform_hashtags") or {}
            if isinstance(ph, str):
                try:
                    import json as _j

                    ph = _j.loads(ph)
                except Exception:
                    ph = {}
            if isinstance(ph, dict):
                raw = None
                for key in (
                    platform,
                    pl,
                    (platform or "").strip().title(),
                    pl.title() if pl else "",
                ):
                    if key and key in ph and ph[key]:
                        raw = ph[key]
                        break
                if raw is None:
                    alias = {"youtube": ("google",), "instagram": ("ig",), "facebook": ("fb",)}.get(pl, ())
                    for a in alias:
                        if a in ph and ph[a]:
                            raw = ph[a]
                            break
                if raw is None:
                    for k, v in ph.items():
                        if str(k).strip().lower() == pl and v:
                            raw = v
                            break
                if raw is None:
                    raw = []
                platform_tags = coerce_hashtag_list(raw)

        # ── M8 caption engine: per-platform AI hashtag variants ────────────
        m8_tags: List[str] = []
        if pl:
            m8_map = getattr(self, "m8_platform_hashtags", None) or {}
            if isinstance(m8_map, dict) and m8_map:
                m8_raw = m8_map.get(pl) or m8_map.get(platform or "")
                if m8_raw is None:
                    for mk, mv in m8_map.items():
                        if str(mk).strip().lower() == pl:
                            m8_raw = mv
                            break
                m8_tags = coerce_hashtag_list(m8_raw or [])

        # ── Blocked hashtags from settings page ──────────────────────────
        raw_blocked = us.get("blockedHashtags") or us.get("blocked_hashtags") or []
        blocked_set: set = set()
        for t in coerce_hashtag_list(raw_blocked):
            b = sanitize_hashtag_body(str(t))
            if b:
                blocked_set.add(b)

        # ── Merge: always → user platform → base (upload) → M8 → generic AI ──
        base = coerce_hashtag_list(self.hashtags)
        ai = coerce_hashtag_list(self.ai_hashtags)

        seen: set = set()
        merged: List[str] = []

        for tag in always_tags + platform_tags + base + m8_tags + ai:
            body = sanitize_hashtag_body(tag)
            if not body or body in seen or body in blocked_set:
                continue
            seen.add(body)
            merged.append(f"#{body}")

        # "Max total hashtags" — cap final merged list (merge order preserved: always → platform → …).
        try:
            raw_cap = us.get("maxHashtags")
            if raw_cap is None:
                raw_cap = us.get("max_hashtags")
            cap = int(raw_cap) if raw_cap is not None and str(raw_cap).strip() != "" else 50
        except (TypeError, ValueError):
            cap = 50
        cap = max(1, min(cap, 50))
        if len(merged) > cap:
            merged = merged[:cap]

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

    def get_thumbnail_brief_vars(self, category: Optional[str] = None) -> Dict[str, str]:
        """
        Build variables for THUMBNAIL_BRIEF_PROMPT. Used by thumbnail_stage to
        generate platform-aware thumbnail briefs (headline, badge, props, etc.).
        Pass category from thumbnail_stage (from _detect_category) when available.
        """
        trill = self.trill or self.trill_score
        telemetry = self.telemetry or self.telemetry_data
        max_mph = 0.0
        if telemetry:
            max_mph = getattr(telemetry, "max_speed_mph", 0) or 0
        platforms = [str(p).lower() for p in (self.platforms or [])]
        platforms_csv = ",".join(platforms) if platforms else "youtube,instagram,facebook,tiktok"
        fusion_summary = build_fusion_summary_text(self)
        hydration_story = build_hydration_story_text(self, max_chars=700)

        trill_bits: List[str] = []
        if trill:
            if getattr(trill, "score", None) is not None:
                trill_bits.append(f"score: {getattr(trill, 'score')}")
            if getattr(trill, "bucket", None):
                trill_bits.append(f"bucket: {getattr(trill, 'bucket')}")
            if getattr(trill, "title_modifier", None):
                trill_bits.append(f"title modifier: {getattr(trill, 'title_modifier')}")
            th = getattr(trill, "hashtags", None)
            if isinstance(th, list) and th:
                trill_bits.append("tag ideas: " + ", ".join(str(x).lstrip("#") for x in th[:8]))

        geo_bits: List[str] = []
        osd_bits: List[str] = []
        tag_bits: List[str] = []
        if telemetry:
            for attr, label in (
                ("location_display", "mid-route"),
                ("location_start_display", "route start"),
                ("location_road", "road"),
                ("location_city", "city"),
                ("location_state", "region"),
                ("gazetteer_place_name", "nearest Census place"),
                ("padus_unit_name", "protected area"),
            ):
                v = getattr(telemetry, attr, None)
                if v:
                    geo_bits.append(f"{label}: {v}")
                    body = sanitize_hashtag_body(str(v))
                    if body:
                        tag_bits.append(body)
            if getattr(telemetry, "near_padus", False) and not getattr(telemetry, "padus_unit_name", None):
                geo_bits.append("protected/public lands near route")
                tag_bits.append("publiclands")
            if getattr(telemetry, "mid_lat", None) is not None and getattr(telemetry, "mid_lon", None) is not None:
                try:
                    geo_bits.append(f"GPS midpoint: {float(telemetry.mid_lat):.5f}, {float(telemetry.mid_lon):.5f}")
                except (TypeError, ValueError):
                    pass

        osd = self.dashcam_osd_context or {}
        if isinstance(osd, dict) and osd and not osd.get("skipped"):
            fs = osd.get("first_seen") or {}
            ls = osd.get("last_seen") or {}
            if fs.get("date") or fs.get("time"):
                osd_bits.append(f"HUD start: {fs.get('date') or '?'} {fs.get('time') or ''}".strip())
            if ls.get("date") or ls.get("time"):
                osd_bits.append(f"HUD end: {ls.get('date') or '?'} {ls.get('time') or ''}".strip())
            if osd.get("max_speed_mph"):
                geo_bits.append(f"HUD peak speed: {osd.get('max_speed_mph')} mph")
                osd_bits.append(f"peak speed: {osd.get('max_speed_mph')} mph")
            if osd.get("avg_speed_mph"):
                osd_bits.append(f"avg speed: {osd.get('avg_speed_mph')} mph")
            if osd.get("driver_name"):
                osd_bits.append(f"driver/HUD name: {osd.get('driver_name')}")
            if osd.get("speed_unit_detected"):
                osd_bits.append(f"speed unit: {osd.get('speed_unit_detected')}")
            if osd.get("telemetry_backfilled"):
                osd_bits.append("telemetry backfilled from HUD")
            path = osd.get("gps_path") or []
            if isinstance(path, list) and path:
                geo_bits.append(f"HUD GPS fixes: {len(path)}")
                osd_bits.append(f"GPS fixes: {len(path)}")

        ac = self.audio_context or {}
        music_bits: List[str] = []
        speech_bits: List[str] = []
        if isinstance(ac, dict):
            transcript = (self.ai_transcript or str(ac.get("transcript") or "")).strip()
            if transcript:
                speech_bits.append("transcript excerpt: " + _trim_text(transcript, 420))
            if ac.get("gpt_audio_summary"):
                speech_bits.append("speech/audio summary: " + _trim_text(str(ac.get("gpt_audio_summary")), 260))
            if ac.get("fusion_narrative"):
                speech_bits.append("audio narrative: " + _trim_text(str(ac.get("fusion_narrative")), 260))
            role = str(ac.get("transcript_role") or "").strip()
            if role:
                speech_bits.append(f"transcript role: {role}")
            if ac.get("music_artist"):
                music_bits.append(f"artist: {ac.get('music_artist')}")
                body = sanitize_hashtag_body(str(ac.get("music_artist")))
                if body:
                    tag_bits.append(body)
            if ac.get("music_title"):
                music_bits.append(f"track: {ac.get('music_title')}")
                body = sanitize_hashtag_body(str(ac.get("music_title")))
                if body:
                    tag_bits.append(body)
            if ac.get("music_genre"):
                music_bits.append(f"genre: {ac.get('music_genre')}")
                body = sanitize_hashtag_body(str(ac.get("music_genre")))
                if body:
                    tag_bits.append(body)
            yev = ac.get("yamnet_events")
            if isinstance(yev, list) and yev:
                music_bits.append("ambient audio: " + ", ".join(str(x) for x in yev[:8]))

        # Dedup, preserve order, and expose as #tag hints for the brief generator.
        seen_tags: set = set()
        clean_tags: List[str] = []
        for t in tag_bits:
            body = sanitize_hashtag_body(str(t))
            if body and body not in seen_tags:
                seen_tags.add(body)
                clean_tags.append(f"#{body}")

        out: Dict[str, str] = {
            "effective_title": self.get_effective_title() or self.filename or "Video",
            "effective_caption": self.get_effective_caption() or "",
            "category": category or getattr(self, "thumbnail_category", None) or "general",
            "location_name": self.location_name or "",
            "trill_bucket": trill.bucket if trill else "",
            "max_speed_mph": str(max_mph),
            "platforms_csv": platforms_csv,
            "hydration_story": hydration_story or "",
            "fusion_summary": fusion_summary or "",
            "geo_context": "; ".join(geo_bits[:10]),
            "osd_context": "; ".join(osd_bits[:10]),
            "trill_context": "; ".join(trill_bits[:10]),
            "music_context": "; ".join(music_bits[:8]),
            "speech_context": "; ".join(speech_bits[:6]),
            "signal_hashtags": ", ".join(clean_tags[:12]),
        }

        hp = getattr(self, "hydration_payload", None)
        if isinstance(hp, dict):
            from services.hydration_payload import hydration_brief_strings

            for k, v in hydration_brief_strings(hp).items():
                out[k] = v
            c2 = str(hp.get("category") or "").strip().lower()
            if c2:
                out["category"] = c2

        return out


# ============================================================
# Thumbnail Brief Generator — base prompt for thumbnail_stage
# ============================================================
THUMBNAIL_BRIEF_PROMPT = """You are a thumbnail/cover strategist for short-form video publishing.
Return ONLY valid JSON. No markdown.

CONTEXT
- Effective title: {effective_title}
- Caption hint: {effective_caption}
- Category: {category}
- Location: {location_name}
- Trill bucket: {trill_bucket}
- Max speed mph: {max_speed_mph}
- Target platforms: {platforms_csv}
- Hydration story: {hydration_story}
- Fused video evidence: {fusion_summary}
- Geo/route evidence: {geo_context}
- OSD/HUD evidence: {osd_context}
- Trill evidence: {trill_context}
- Music/audio evidence: {music_context}
- Speech/Whisper evidence: {speech_context}
- Signal hashtags to preserve when relevant: {signal_hashtags}

HARD RULES
- No profanity, no hate, no nudity, no weapons emphasis, no illegal claims.
- No copyrighted logos/brand marks (YouTube logo, TikTok logo, etc).
- ACCURACY: Headlines and badges must reflect what is actually in the video. No misleading claims (e.g. "TOP 5" when it's not a list, "NEW" when it's not new). Describe visible content truthfully.
- Do NOT use generic filler text like "EXCITING MOMENTS", "UNBELIEVABLE MOMENTS", "AMAZING MOMENT", "MUST WATCH", "WATCH THIS", "EPIC CLIP", or "CRAZY MOMENT".
- The selected headline must include a concrete noun, place, object, action, OCR phrase, or route/speed detail from the context.
- If geo/route evidence is present, use at least one location/road/protected-area detail in visual props, notes, or headline options when truthful.
- If OSD/HUD or Trill evidence is present, preserve speed, HUD timeline, or Trill energy in the thumbnail strategy when it matches the visible clip.
- If music/audio evidence is present, preserve artist/track/genre as context for vibe and hashtag strategy, but do not imply the creator owns the song.
- If Speech/Whisper evidence is present, use a short concrete spoken phrase only when it is clear, useful, and not private/sensitive.
- Text: 2–6 words total, ALL CAPS, 1–2 lines, mobile readable.
- Provide 3 headline options; select 1.
- Include 1 badge only when it accurately fits (e.g. FAST only if speed/telemetry present; HOW TO only if it's a tutorial).
- Always include 1 directional element (arrow/circle/glow box).
- Provide 2 prop ideas that match the category.
- Never mention AI/automation/API/internal product words.

PLATFORM RULES
- youtube: design a 16:9 custom thumbnail, strong text ok.
- instagram: design a 9:16 cover image safe for center-crop to 1:1 grid; keep key elements centered.
- facebook: design a 9:16 cover safe for 1:1 crop; keep key elements centered.
- tiktok: design a 9:16 generated cover preview for display, but also output a recommended thumb_offset_seconds because publish APIs may use frame selection instead of custom cover upload.

OUTPUT SCHEMA
{{
  "selected_headline": "string",
  "headline_options": ["string","string","string"],
  "badge_text": "string",
  "badge_style": "red"|"yellow"|"white"|"black",
  "directional_element": "arrow_up"|"arrow_right"|"circle"|"glow_box",
  "props": ["string","string"],
  "emotion_cue": "shocked"|"excited"|"serious"|"laughing",
  "color_mood": "red_black"|"blue_black"|"gold_black"|"neon",
  "platform_plan": {{
    "youtube": {{"enabled": true, "canvas": "16:9"}},
    "instagram": {{"enabled": true, "canvas": "9:16", "safe_center_pct": 60}},
    "facebook": {{"enabled": true, "canvas": "9:16", "safe_center_pct": 60}},
    "tiktok": {{"enabled": true, "canvas": "9:16", "thumb_offset_seconds": 1.5}}
  }},
  "notes": "1 sentence max"
}}
"""


def _normalize_upload_hashtags_field(raw: Any) -> List[str]:
    """Coerce uploads.hashtags (jsonb list, JSON string, or broken string) to flat tokens.

    If the raw value is a string carrying a stray JSON-array hashtag blob (the same
    kind we strip from caption text), recover the inner tokens via
    ``coerce_hashtag_list`` so per-upload hashtags never reach the publish stage as
    a single mashed token.
    """
    if isinstance(raw, str):
        # Pull out individual tags from any embedded #"[\"a\" #\"b\"]" blob, then
        # also keep any clean tags around it.
        cleaned = strip_stray_hashtag_json_blob(raw)
        recovered = coerce_hashtag_list(raw)
        # Prefer recovered tokens (handles blob-only strings); fall back to cleaned text.
        tokens = recovered if recovered else coerce_hashtag_list(cleaned)
        return [str(t) for t in tokens if str(t).strip()]
    return [str(t) for t in coerce_hashtag_list(raw) if str(t).strip()]


def create_context(job_data: dict, upload_record: dict, user_settings: dict, entitlements: Entitlements) -> JobContext:
    # Resolve reframe_mode: job payload > upload record > user setting > "auto"
    reframe_mode = (
        job_data.get("reframe_mode")
        or upload_record.get("reframe_mode")
        or (user_settings or {}).get("reframe_mode")
        or "auto"
    )

    ctx = JobContext(
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
        title=strip_stray_hashtag_json_blob(str(upload_record.get("title") or "")),
        caption=strip_stray_hashtag_json_blob(str(upload_record.get("caption") or "")),
        hashtags=_normalize_upload_hashtags_field(upload_record.get("hashtags")),
        privacy=upload_record.get("privacy", "public") or "public",
        reframe_mode=reframe_mode,
        scheduled_time=upload_record.get("scheduled_time"),
        user_settings=user_settings or {},
        entitlements=entitlements,
    )
    ctx.ai_title = strip_stray_hashtag_json_blob(
        str(upload_record.get("ai_generated_title") or upload_record.get("ai_title") or "")
    ) or None
    ctx.ai_caption = strip_stray_hashtag_json_blob(
        str(upload_record.get("ai_generated_caption") or upload_record.get("ai_caption") or "")
    ) or None
    ctx.ai_hashtags = coerce_hashtag_list(upload_record.get("ai_generated_hashtags") or upload_record.get("ai_hashtags") or [])

    artifacts = coerce_jsonb_dict(upload_record.get("output_artifacts"), default={})
    if artifacts:
        ctx.output_artifacts = {str(k): str(v) for k, v in artifacts.items() if v is not None}

    def _artifact_json_dict(key: str) -> Dict[str, Any]:
        raw = artifacts.get(key)
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                val = json.loads(raw)
                return val if isinstance(val, dict) else {}
            except (TypeError, ValueError, json.JSONDecodeError):
                return {}
        return {}

    ctx.m8_platform_titles = {
        str(k).lower(): str(v)
        for k, v in _artifact_json_dict("m8_platform_titles").items()
        if str(v).strip()
    }
    ctx.m8_platform_captions = {
        str(k).lower(): strip_stray_hashtag_json_blob(str(v).strip())
        for k, v in _artifact_json_dict("m8_platform_captions").items()
        if str(v).strip()
    }
    ctx.m8_platform_hashtags = {
        str(k).lower(): coerce_hashtag_list(v)
        for k, v in _artifact_json_dict("m8_platform_hashtags").items()
        if coerce_hashtag_list(v)
    }
    if not ctx.ai_title and ctx.m8_platform_titles:
        ctx.ai_title = (
            ctx.m8_platform_titles.get("youtube")
            or next(iter(ctx.m8_platform_titles.values()))
        )
    if not ctx.ai_caption and ctx.m8_platform_captions:
        for pref in ("tiktok", "instagram", "facebook", "youtube"):
            if pref in ctx.m8_platform_captions:
                ctx.ai_caption = ctx.m8_platform_captions[pref]
                break
        if not ctx.ai_caption:
            ctx.ai_caption = next(iter(ctx.m8_platform_captions.values()))
    if not ctx.ai_hashtags and ctx.m8_platform_hashtags:
        if "tiktok" in ctx.m8_platform_hashtags:
            ctx.ai_hashtags = ctx.m8_platform_hashtags["tiktok"]
        else:
            ctx.ai_hashtags = next(iter(ctx.m8_platform_hashtags.values()))
    try:
        ctx.vehicle_make_id = upload_record.get("vehicle_make_id")
        if ctx.vehicle_make_id is not None:
            ctx.vehicle_make_id = int(ctx.vehicle_make_id)
    except (TypeError, ValueError):
        ctx.vehicle_make_id = None
    try:
        ctx.vehicle_model_id = upload_record.get("vehicle_model_id")
        if ctx.vehicle_model_id is not None:
            ctx.vehicle_model_id = int(ctx.vehicle_model_id)
    except (TypeError, ValueError):
        ctx.vehicle_model_id = None

    uprefs = coerce_jsonb_dict(upload_record.get("user_preferences"), default={})
    if uprefs:
        raw_tt = uprefs.get("tiktok_post_settings") or uprefs.get("tiktokPostSettings")
        if isinstance(raw_tt, dict):
            ctx.tiktok_post_settings = raw_tt
        # Overlay full thumbnail snapshot from this upload so Studio/Upload toggles
        # (engine, persona, strength, default strategy) beat account defaults.
        for key in (
            "thumbnail_pikzels_enabled",
            "thumbnailPikzelsEnabled",
            "thumbnail_studio_engine_enabled",
            "thumbnailStudioEngineEnabled",
            "thumbnail_studio_enabled",
            "thumbnailStudioEnabled",
            "thumbnail_persona_enabled",
            "thumbnailPersonaEnabled",
            "thumbnail_default_persona_id",
            "thumbnailDefaultPersonaId",
            "thumbnail_persona_strength",
            "thumbnailPersonaStrength",
            "thumbnail_studio_default_strategy",
            "thumbnailStudioDefaultStrategy",
            "thumbnail_render_pipeline",
            "thumbnailRenderPipeline",
            "auto_thumbnails",
            "autoThumbnails",
            "styled_thumbnails",
            "styledThumbnails",
        ):
            if key in uprefs and uprefs[key] is not None:
                ctx.user_settings[key] = uprefs[key]

    return ctx


# ============================================================
# Multimodal fusion — M8 caption engine + scene graph
# ============================================================
# These helpers must stay importable from this module (see stages/m8_engine.py).
# They turn late-stage ctx fields into one prioritized digest so captions track
# the *current* upload instead of filename/category templates alone.

_LANDMARK_SUBSTRINGS: tuple[str, ...] = (
    "stadium",
    "arena",
    "downtown",
    "skyline",
    "bridge",
    "monument",
    "museum",
    "airport",
    "station",
    "plaza",
    "tower",
    "beach",
    "raceway",
    "racetrack",
    "speedway",
    "circuit",
    "highway",
    "interstate",
    "mile marker",
    "national park",
    "state park",
    "campus",
    "historic",
    "landmark",
)


def extract_landmark_hints(
    labels: Sequence[Any],
    ocr_text: str,
    gcv_landmark_names: Optional[Sequence[Any]] = None,
) -> List[str]:
    """
    Pull short venue / place cues from vision labels + OCR for caption grounding.
    Conservative: only labels whose text suggests a place, plus a few OCR lines.
    ``gcv_landmark_names`` comes from Google Cloud Vision LANDMARK_DETECTION when present.
    """
    hints: List[str] = []
    seen: set[str] = set()

    for raw in gcv_landmark_names or []:
        s = str(raw).strip()
        if len(s) < 2:
            continue
        low = s.lower()
        if low in seen:
            continue
        seen.add(low)
        hints.append(s[:160])
        if len(hints) >= 8:
            break

    for raw in labels or []:
        s = str(raw).strip()
        if len(s) < 3:
            continue
        low = s.lower()
        if not any(k in low for k in _LANDMARK_SUBSTRINGS):
            continue
        key = low[:120]
        if key in seen:
            continue
        seen.add(key)
        hints.append(s[:160])
        if len(hints) >= 10:
            break

    for line in (ocr_text or "").splitlines():
        t = re.sub(r"\s+", " ", line.strip())
        if len(t) < 4 or len(t) > 100:
            continue
        low = t.lower()
        if low in seen:
            continue
        # Likely signage / venue names (not full paragraphs)
        if len(t.split()) > 14:
            continue
        seen.add(low)
        hints.append(t)
        if len(hints) >= 14:
            break

    return hints[:14]


def _trim_text(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if max_chars <= 0 or not t:
        return ""
    if len(t) <= max_chars:
        return t
    return t[: max(1, max_chars - 1)].rstrip() + "…"


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles (same convention as telemetry_stage)."""
    r_mi = 3958.8
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(
        d_lon / 2
    ) ** 2
    return r_mi * 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def _uniform_route_sample_indices(n: int, k: int) -> List[int]:
    """Evenly spaced indices including first and last (for polyline caps)."""
    if n <= 0:
        return []
    k = max(1, min(k, n))
    if k == 1:
        return [0]
    if k >= n:
        return list(range(n))
    idx: set[int] = {0, n - 1}
    denom = max(k - 1, 1)
    for i in range(k):
        idx.add(int(round(i * (n - 1) / denom)))
    return sorted(idx)


def compute_route_spatial_summary(
    points: Sequence[Dict[str, Any]],
    *,
    max_polyline_points: int = 32,
) -> Tuple[Optional[Dict[str, float]], List[List[float]]]:
    """
    Bounding box + speed-tagged polyline sample from .map ``points`` (lat, lon, speed_mph).

    Used by ``build_multimodal_scene_digest`` and ``m8_engine.build_scene_graph`` (``geo``).
    """
    if not points:
        return None, []

    clean: List[Tuple[float, float, float]] = []
    for p in points:
        if not isinstance(p, dict):
            continue
        try:
            la = float(p.get("lat"))
            lo = float(p.get("lon"))
            sp = float(p.get("speed_mph") or 0.0)
        except (TypeError, ValueError):
            continue
        if not (-90.0 <= la <= 90.0 and -180.0 <= lo <= 180.0):
            continue
        if abs(la) < 1e-9 and abs(lo) < 1e-9:
            continue
        clean.append((la, lo, sp))

    if not clean:
        return None, []

    lats = [c[0] for c in clean]
    lons = [c[1] for c in clean]
    bbox = {
        "min_lat": min(lats),
        "max_lat": max(lats),
        "min_lon": min(lons),
        "max_lon": max(lons),
    }

    cap = max(2, min(int(max_polyline_points or 32), 48))
    idxs = _uniform_route_sample_indices(len(clean), cap)
    poly: List[List[float]] = []
    for i in idxs:
        la, lo, sp = clean[i]
        poly.append([round(la, 5), round(lo, 5), round(sp, 1)])

    return bbox, poly


def format_route_spatial_digest(
    points: Sequence[Dict[str, Any]],
    *,
    max_polyline_points: int = 32,
    max_chars: int = 4000,
) -> str:
    """Human-readable bbox + sampled polyline for multimodal digest."""
    bbox, poly = compute_route_spatial_summary(points, max_polyline_points=max_polyline_points)
    if not bbox:
        return ""
    lines: List[str] = []
    diag = _haversine_miles(bbox["min_lat"], bbox["min_lon"], bbox["max_lat"], bbox["max_lon"])
    lines.append(
        "Route bbox (WGS84): "
        f"SW {bbox['min_lat']:.4f},{bbox['min_lon']:.4f} "
        f"NE {bbox['max_lat']:.4f},{bbox['max_lon']:.4f} "
        f"(diagonal ~{diag:.1f} mi)"
    )
    n_full = len([p for p in points if isinstance(p, dict) and p.get("lat") is not None and p.get("lon") is not None])
    if poly:
        seg = "; ".join(f"{p[0]:.5f},{p[1]:.5f}@{p[2]:.0f}mph" for p in poly)
        lines.append(f"Sampled polyline ({len(poly)} of {n_full} GPS fixes, ~even spacing): {seg}")
    return _trim_text("\n".join(lines), max_chars)


def format_route_trill_hint(points: Sequence[Dict[str, Any]]) -> str:
    """One short line for legacy Trill beat (digest holds full sample)."""
    if not points or len(points) < 2:
        return ""
    bbox, _ = compute_route_spatial_summary(points, max_polyline_points=2)
    if not bbox:
        return ""
    n = len([p for p in points if isinstance(p, dict) and p.get("lat") is not None and p.get("lon") is not None])
    diag = _haversine_miles(bbox["min_lat"], bbox["min_lon"], bbox["max_lat"], bbox["max_lon"])
    return f"GPS path: {n} fixes, route span ~{diag:.1f} mi (bbox + sampled polyline in model digest)."


def _video_probe_lines(ctx: JobContext) -> str:
    """Short ffprobe-style facts for grounding (duration, geometry, codecs)."""
    vi = ctx.video_info or {}
    if not isinstance(vi, dict) or not vi:
        return ""
    bits: List[str] = []
    try:
        d = float(vi.get("duration") or 0)
        if d > 0:
            bits.append(f"duration ~{d:.1f}s")
    except (TypeError, ValueError):
        pass
    try:
        w = int(vi.get("width") or 0)
        h = int(vi.get("height") or 0)
        if w > 0 and h > 0:
            bits.append(f"{w}x{h}")
    except (TypeError, ValueError):
        pass
    fps = vi.get("fps")
    if fps is not None:
        try:
            fv = float(fps)
            if fv > 0:
                bits.append(f"~{fv:.2f} fps")
        except (TypeError, ValueError):
            pass
    vc = vi.get("video_codec")
    ac = vi.get("audio_codec")
    if vc:
        bits.append(f"video codec: {vc}")
    if ac:
        bits.append(f"audio codec: {ac}")
    return "; ".join(bits)


def _video_intelligence_digest_body(vi: Dict[str, Any]) -> str:
    """
    Compact full-clip intelligence: top labels plus time-coded segment rows and shot stats.
    """
    lines: List[str] = []
    tl = vi.get("top_labels") or []
    if isinstance(tl, list) and tl:
        lines.append("Top labels: " + "; ".join(str(x) for x in tl[:24]))

    segs = vi.get("segment_labels") or []
    if isinstance(segs, list) and segs:
        ranked = sorted(
            (s for s in segs if isinstance(s, dict) and (str(s.get("description") or "").strip())),
            key=lambda s: float(s.get("confidence") or 0.0),
            reverse=True,
        )[:18]
        if ranked:
            lines.append("Segment timeline (by confidence):")
            for s in ranked:
                desc = str(s.get("description") or "").strip()[:90]
                try:
                    c = float(s.get("confidence") or 0.0)
                except (TypeError, ValueError):
                    c = 0.0
                try:
                    t0 = float(s.get("start_s") or 0.0)
                    t1 = float(s.get("end_s") or 0.0)
                except (TypeError, ValueError):
                    t0, t1 = 0.0, 0.0
                lines.append(f"  {t0:.1f}-{t1:.1f}s: {desc} ({c:.2f})")

    shots = vi.get("shots") or []
    if isinstance(shots, list) and shots:
        lens: List[float] = []
        for sh in shots[:120]:
            if not isinstance(sh, dict):
                continue
            try:
                a = float(sh.get("start_s") or 0.0)
                b = float(sh.get("end_s") or 0.0)
            except (TypeError, ValueError):
                continue
            if b > a:
                lens.append(b - a)
        if lens:
            lens.sort()
            med = lens[len(lens) // 2]
            lines.append(f"Shot boundaries: {len(shots)} shots (median visible span ~{med:.1f}s)")

    st = str(vi.get("summary_text") or "").strip()
    if st and not any(st in ln for ln in lines):
        lines.insert(0, st[:800])

    return "\n".join(lines).strip()


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return re.sub(r"\s+", " ", text)
    return ""


def _unique_phrases(values: Sequence[Any], *, limit: int = 6) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, dict):
            raw = value.get("description") or value.get("text") or value.get("label") or ""
        else:
            raw = value
        text = re.sub(r"\s*\([0-9.]+\)\s*$", "", str(raw or "").strip())
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text[:80])
        if len(out) >= limit:
            break
    return out


def build_hydration_story_text(ctx: JobContext, *, max_chars: int = 700) -> str:
    """Small factual paragraph that paints the scene from all hydration signals.

    This is intentionally shorter than ``build_multimodal_scene_digest``. It is
    the reusable "what is in this video?" paragraph for M8, Pikzels, diagnostics,
    and hydration reports.
    """
    max_chars = max(180, min(int(max_chars or 700), 1600))

    from core.vision_entities import (
        build_scene_hook_line,
        collect_visual_entities,
        visual_entity_story_clauses,
    )

    tel = ctx.telemetry or ctx.telemetry_data
    osd = ctx.dashcam_osd_context or {}
    ac = ctx.audio_context or {}
    vc = ctx.vision_context or {}
    vic = ctx.video_intelligence_context or {}
    vi = ctx.video_intelligence or {}

    clauses: List[str] = []

    # Recording/time facts.
    fs = osd.get("first_seen") if isinstance(osd, dict) else {}
    ls = osd.get("last_seen") if isinstance(osd, dict) else {}
    time_bits: List[str] = []
    if isinstance(fs, dict) and (fs.get("date") or fs.get("time")):
        time_bits.append(f"HUD start {str(fs.get('date') or '').strip()} {str(fs.get('time') or '').strip()}".strip())
    if isinstance(ls, dict) and (ls.get("date") or ls.get("time")) and (
        ls.get("date") != (fs or {}).get("date") or ls.get("time") != (fs or {}).get("time")
    ):
        time_bits.append(f"HUD end {str(ls.get('date') or '').strip()} {str(ls.get('time') or '').strip()}".strip())
    if time_bits:
        clauses.append("Recording facts: " + "; ".join(time_bits) + ".")

    # Place and coordinates.
    place_bits: List[str] = []
    if tel:
        place = _first_nonempty(
            getattr(tel, "location_display", None),
            getattr(tel, "gazetteer_place_name", None),
            getattr(tel, "location_city", None),
            getattr(tel, "location_road", None),
            getattr(tel, "padus_unit_name", None),
        )
        if place:
            place_bits.append(place)
        state = _first_nonempty(getattr(tel, "gazetteer_state_usps", None), getattr(tel, "location_state", None))
        if state and place and state.lower() not in place.lower():
            place_bits.append(state)
        lat = getattr(tel, "mid_lat", None) or getattr(tel, "start_lat", None)
        lon = getattr(tel, "mid_lon", None) or getattr(tel, "start_lon", None)
        if lat is not None and lon is not None:
            try:
                place_bits.append(f"{float(lat):.5f}, {float(lon):.5f}")
            except (TypeError, ValueError):
                pass
    if not place_bits and isinstance(osd, dict):
        gps_path = osd.get("gps_path") or []
        if isinstance(gps_path, list) and gps_path:
            first = gps_path[0]
            if isinstance(first, (list, tuple)) and len(first) >= 2:
                try:
                    place_bits.append(f"{float(first[0]):.5f}, {float(first[1]):.5f}")
                except (TypeError, ValueError):
                    pass
    if place_bits:
        clauses.append("Location context: " + " near ".join(place_bits[:2]) + (f" ({place_bits[-1]})" if len(place_bits) > 2 else "") + ".")

    # Motion/vehicle/person facts.
    motion_bits: List[str] = []
    max_speed = 0.0
    if tel:
        try:
            max_speed = max(max_speed, float(getattr(tel, "max_speed_mph", 0) or 0))
        except (TypeError, ValueError):
            pass
    if isinstance(osd, dict):
        try:
            max_speed = max(max_speed, float(osd.get("max_speed_mph") or 0))
        except (TypeError, ValueError):
            pass
    if max_speed >= 5:
        motion_bits.append(f"peak speed about {int(round(max_speed))} MPH")
    if isinstance(osd, dict) and osd.get("driver_name"):
        motion_bits.append(f"driver label {str(osd.get('driver_name')).strip()[:40]}")

    entity_bundle = collect_visual_entities(
        vision_context=vc if isinstance(vc, dict) else None,
        video_intelligence=vi if isinstance(vi, dict) else None,
        video_intelligence_context=vic if isinstance(vic, dict) else None,
        category=str(
            getattr(ctx, "thumbnail_category", None)
            or (getattr(ctx, "hydration_payload", None) or {}).get("category")
            or "general"
        ),
        filename=str(getattr(ctx, "filename", "") or ""),
    )
    if entity_bundle.vehicles:
        motion_bits.append("vehicles/models: " + ", ".join(entity_bundle.vehicles[:4]))
    elif entity_bundle.brands:
        motion_bits.append("brand cues: " + ", ".join(entity_bundle.brands[:3]))
    person_count = 0
    if isinstance(vi, dict):
        person_count += len(vi.get("person_segments") or [])
    if isinstance(vic, dict):
        person_count += len(vic.get("person_segments") or [])
    if not person_count and isinstance(vc, dict):
        try:
            person_count = int(vc.get("face_count") or 0)
        except (TypeError, ValueError):
            person_count = 0
    if person_count:
        motion_bits.append(f"person/faces signal in {person_count} segment(s)")
    if motion_bits:
        clauses.append("Subject and motion: " + "; ".join(motion_bits) + ".")

    # Visual scene facts — prefer specific entities over generic Vision labels.
    rec_summary = ""
    if isinstance(vc, dict):
        rec_summary = str(vc.get("recognition_summary") or "").strip()
    if not rec_summary:
        vr = getattr(ctx, "visual_recognition", None) or {}
        if isinstance(vr, dict):
            from services.google_visual_recognition import build_recognition_narrative

            rec_summary = build_recognition_narrative(vr)
    if rec_summary:
        clauses.append(rec_summary if rec_summary.endswith(".") else rec_summary + ".")
    else:
        entity_clauses = visual_entity_story_clauses(entity_bundle)
        clauses.extend(entity_clauses)

    ocr = ""
    if isinstance(vic, dict):
        ocr_rows = vic.get("on_screen_text") or vic.get("text_detections") or []
        ocr_names = _unique_phrases(ocr_rows, limit=3)
        ocr = "; ".join(ocr_names)
    if not ocr and isinstance(vc, dict):
        ocr = str(vc.get("ocr_text") or "").strip().replace("\n", " | ")
    if ocr and not entity_bundle.signage:
        clauses.append("On-screen text: " + ocr[:140] + ".")

    # Audio/music/speech.
    audio_bits: List[str] = []
    if isinstance(ac, dict):
        artist = str(ac.get("music_artist") or "").strip()
        track = str(ac.get("music_title") or "").strip()
        genre = str(ac.get("music_genre") or "").strip()
        if artist and track:
            audio_bits.append(f"music detected: {artist} - {track}")
        elif artist or track:
            audio_bits.append("music detected: " + (artist or track))
        if genre:
            audio_bits.append(f"genre {genre}")
        transcript = _first_nonempty(ctx.ai_transcript, ac.get("transcript"))
        structured = ac.get("transcript_structured") or {}
        if isinstance(structured, dict) and structured.get("key_phrase"):
            transcript = str(structured.get("key_phrase") or "").strip()
        if transcript:
            audio_bits.append('speech/transcript cue: "' + transcript[:120].rstrip(".!?") + '"')
        top_sound = str(ac.get("top_sound_class") or "").strip()
        if top_sound:
            audio_bits.append(f"ambient sound {top_sound}")
    if audio_bits:
        clauses.append("Audio context: " + "; ".join(audio_bits) + ".")

    place_for_hook = ""
    if tel:
        place_for_hook = _first_nonempty(
            getattr(tel, "location_city", None),
            getattr(tel, "gazetteer_place_name", None),
            getattr(tel, "location_road", None),
        )
    music_artist = music_title = ""
    if isinstance(ac, dict):
        music_artist = str(ac.get("music_artist") or "").strip()
        music_title = str(ac.get("music_title") or "").strip()
    hook = build_scene_hook_line(
        place=place_for_hook,
        max_speed_mph=max_speed,
        music_artist=music_artist,
        music_title=music_title,
        bundle=entity_bundle,
    )
    if hook:
        clauses.append("Scene hook: " + hook + ".")

    if not clauses:
        fname = (ctx.filename or "uploaded video").strip()
        return f"Hydration story: {fname} has no strong analysis signals yet; use the actual frame and filename only."[:max_chars]

    story = "Hydration story: " + " ".join(clauses)
    if len(story) > max_chars:
        story = story[: max_chars - 1].rstrip() + "."
    return story


def build_video_story_timeline(ctx: JobContext, *, max_events: int = 80) -> List[Dict[str, Any]]:
    """Build an ordered ``[{t_seconds, kind, text}, ...]`` timeline fused from VI,
    Vision, Dashcam OSD, telemetry, and audio analysis on ``ctx``.

    This is the "what happens in the video over time" view that feeds the M8
    prompt, the user/admin Discord embeds, the upload email, and the queue.html
    detail panel. Each event is JSON-safe and bounded so the artifact stays
    well under the JSONB column size limits.
    """
    max_events = max(8, min(int(max_events or 80), 200))
    events: List[Dict[str, Any]] = []

    vi = ctx.video_intelligence or {}
    vic = ctx.video_intelligence_context or {}
    osd = ctx.dashcam_osd_context or {}
    ac = ctx.audio_context or {}
    vc = ctx.vision_context or {}
    tel = ctx.telemetry or ctx.telemetry_data

    def _t(v: Any) -> Optional[float]:
        try:
            return round(float(v), 2)
        except (TypeError, ValueError):
            return None

    def _add(t: Any, kind: str, text: str) -> None:
        text_clean = re.sub(r"\s+", " ", str(text or "")).strip()
        if not text_clean:
            return
        ts = _t(t)
        if ts is None or ts < 0:
            ts = 0.0
        events.append({"t_seconds": ts, "kind": str(kind)[:24], "text": text_clean[:240]})

    # ── Video Intelligence: shot changes (cap to 10 boundaries) ──────────
    shots = []
    if isinstance(vic, dict):
        shots = list(vic.get("shots") or [])
    if not shots and isinstance(vi, dict):
        shots = list(vi.get("shots") or [])
    for i, shot in enumerate(shots[:10]):
        if not isinstance(shot, dict):
            continue
        st = shot.get("start_s")
        if st is None:
            continue
        _add(st, "shot", f"Shot {i + 1} begins")

    # ── VI object tracks (top by confidence; one event per object start) ─
    obj_tracks = []
    if isinstance(vi, dict):
        obj_tracks.extend(vi.get("object_tracks") or [])
    if isinstance(vic, dict):
        obj_tracks.extend(vic.get("object_tracks") or [])
    seen_obj: set[str] = set()
    for ot in sorted(obj_tracks, key=lambda x: -float(x.get("confidence") or 0))[:12]:
        if not isinstance(ot, dict):
            continue
        desc = str(ot.get("description") or "").strip()
        if not desc or desc.lower() in seen_obj:
            continue
        seen_obj.add(desc.lower())
        _add(ot.get("start_s"), "object", desc)

    # ── VI on-screen text (highest-confidence first, cap 8) ──────────────
    ost = []
    if isinstance(vi, dict):
        ost.extend(vi.get("on_screen_text") or [])
    if isinstance(vic, dict):
        ost.extend(vic.get("on_screen_text") or [])
    for ts_row in sorted(ost, key=lambda x: -float(x.get("confidence") or 0))[:8]:
        if not isinstance(ts_row, dict):
            continue
        txt = str(ts_row.get("text") or "").strip()
        if not txt:
            continue
        _add(ts_row.get("start_s"), "on_screen_text", txt)

    # ── VI logos (Tesla, In-N-Out, etc.) ────────────────────────────────
    logos = []
    if isinstance(vi, dict):
        logos.extend(vi.get("logos") or [])
    if isinstance(vic, dict):
        logos.extend(vic.get("logos") or [])
    seen_logo: set[str] = set()
    for lg in logos[:6]:
        if not isinstance(lg, dict):
            continue
        name = str(lg.get("description") or lg.get("name") or "").strip()
        if not name or name.lower() in seen_logo:
            continue
        seen_logo.add(name.lower())
        _add(lg.get("start_s") or 0.0, "logo", f"Logo: {name}")

    # ── Vision (Google Cloud Vision) sampled-frame labels ───────────────
    if isinstance(vc, dict):
        fracs = vc.get("vision_sample_fractions") or []
        labels = vc.get("label_names") or []
        landmarks = vc.get("landmark_names") or []
        ocr = str(vc.get("ocr_text") or "").strip()
        try:
            duration_for_fracs = float(vc.get("video_duration_s") or 0.0) or 0.0
        except (TypeError, ValueError):
            duration_for_fracs = 0.0
        # Use fraction-based timing when we know the duration; otherwise
        # treat the whole clip as a single t=0 frame.
        if isinstance(fracs, list) and fracs and duration_for_fracs > 0:
            for idx, frac in enumerate(fracs[:6]):
                try:
                    t = float(frac) * duration_for_fracs
                except (TypeError, ValueError):
                    continue
                _add(t, "vision_frame", f"Vision sample {idx + 1}/{len(fracs)} @ {t:.1f}s")
        elif labels or landmarks or ocr:
            _add(0.0, "vision_frame", "Vision sampled frame")
        for lbl in (labels or [])[:6]:
            txt = str(lbl).strip()
            if txt:
                _add(0.0, "vision_label", txt)
        for lm in (landmarks or [])[:3]:
            txt = str(lm).strip()
            if txt:
                _add(0.0, "landmark", txt)
        if ocr:
            _add(0.0, "vision_ocr", ocr.replace("\n", " ")[:200])

    # ── Dashcam OSD (start/end stamps, per-frame speed, GPS path beats) ──
    if isinstance(osd, dict):
        fs = osd.get("first_seen") or {}
        ls = osd.get("last_seen") or {}
        if isinstance(fs, dict) and (fs.get("date") or fs.get("time")):
            _add(
                0.0,
                "osd_start",
                f"HUD start {str(fs.get('date') or '').strip()} {str(fs.get('time') or '').strip()}".strip(),
            )
        if isinstance(ls, dict) and (ls.get("date") or ls.get("time")):
            _add(
                ls.get("t_s") or 1e9,
                "osd_end",
                f"HUD end {str(ls.get('date') or '').strip()} {str(ls.get('time') or '').strip()}".strip(),
            )
        try:
            max_mph_osd = float(osd.get("max_speed_mph") or 0)
        except (TypeError, ValueError):
            max_mph_osd = 0.0
        if max_mph_osd >= 5:
            _add(osd.get("max_speed_t_s") or 0.0, "osd_speed", f"OSD peak {int(round(max_mph_osd))} MPH")
        speed_series = osd.get("speed_series") if isinstance(osd, dict) else None
        if isinstance(speed_series, list) and speed_series:
            step = max(1, len(speed_series) // 5)
            for entry in speed_series[::step][:5]:
                if not isinstance(entry, dict):
                    continue
                try:
                    s_mph = float(entry.get("mph") or entry.get("speed_mph") or 0)
                except (TypeError, ValueError):
                    s_mph = 0.0
                if s_mph <= 0:
                    continue
                _add(entry.get("t_s") or entry.get("ts_s") or 0.0, "osd_speed_beat", f"{int(round(s_mph))} MPH")
        gps_path = osd.get("gps_path") if isinstance(osd, dict) else None
        if isinstance(gps_path, list) and gps_path:
            step = max(1, len(gps_path) // 4)
            for i, pt in enumerate(gps_path[::step][:4]):
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    try:
                        lat = float(pt[0]); lon = float(pt[1])
                        _add(0.0, "osd_gps", f"GPS waypoint {i + 1}: {lat:.4f}, {lon:.4f}")
                    except (TypeError, ValueError):
                        continue

    # ── Telemetry highlights ─────────────────────────────────────────────
    if tel is not None:
        place = _first_nonempty(
            getattr(tel, "location_display", None),
            getattr(tel, "gazetteer_place_name", None),
            getattr(tel, "location_city", None),
            getattr(tel, "padus_unit_name", None),
        )
        if place:
            _add(0.0, "geo_place", f"Location: {place}")
        road = _first_nonempty(getattr(tel, "location_road", None))
        if road:
            _add(0.0, "geo_road", f"Road: {road}")
        try:
            max_mph_tel = float(getattr(tel, "max_speed_mph", 0) or 0)
        except (TypeError, ValueError):
            max_mph_tel = 0.0
        if max_mph_tel >= 5:
            _add(0.0, "telemetry_speed", f"Telemetry peak {int(round(max_mph_tel))} MPH")

    # ── Audio: transcript segments (cap 10, ordered by time) ─────────────
    if isinstance(ac, dict):
        ts_segs = ac.get("transcript_segments") or []
        if isinstance(ts_segs, list):
            for s in ts_segs[:10]:
                if not isinstance(s, dict):
                    continue
                _add(s.get("start"), "transcript", str(s.get("text") or "").strip())
        artist = str(ac.get("music_artist") or "").strip()
        track = str(ac.get("music_title") or "").strip()
        if artist or track:
            label = f"{artist} — {track}" if (artist and track) else (artist or track)
            _add(0.0, "music", f"Music detected: {label}")
        yn = ac.get("yamnet_events") or []
        if isinstance(yn, list):
            seen_y: set[str] = set()
            for ev in yn[:8]:
                txt = str(ev).strip() if not isinstance(ev, dict) else str(ev.get("label") or ev.get("name") or "")
                if not txt or txt.lower() in seen_y:
                    continue
                seen_y.add(txt.lower())
                _add(0.0 if not isinstance(ev, dict) else (ev.get("t_s") or 0.0), "yamnet", txt)
        top_sound = str(ac.get("top_sound_class") or "").strip()
        if top_sound:
            _add(0.0, "ambient_sound", f"Ambient: {top_sound}")

    # Sort by time, deduplicate exact same (kind, text) within 0.5s windows,
    # cap to ``max_events``.
    events.sort(key=lambda e: (float(e.get("t_seconds") or 0.0), str(e.get("kind") or "")))
    deduped: List[Dict[str, Any]] = []
    last_keys: List[Tuple[str, str, float]] = []
    for ev in events:
        k = (str(ev.get("kind") or ""), str(ev.get("text") or "").lower())
        t = float(ev.get("t_seconds") or 0.0)
        if any(prev_k == k[0] and prev_text == k[1] and abs(t - prev_t) < 0.5 for prev_k, prev_text, prev_t in last_keys):
            continue
        last_keys.append((k[0], k[1], t))
        deduped.append(ev)
        if len(deduped) >= max_events:
            break

    return deduped


def _vision_digest_header(vc: Dict[str, Any]) -> str:
    try:
        n_mf = int(vc.get("vision_multi_frame") or 1)
    except (TypeError, ValueError):
        n_mf = 1
    n_mf = max(1, n_mf)
    fracs = vc.get("vision_sample_fractions") or []
    pct_hint = ""
    if n_mf > 1 and isinstance(fracs, list) and fracs:
        try:
            pcts = [f"{float(x) * 100:.0f}%" for x in fracs[:6]]
            pct_hint = " (~" + ", ".join(pcts) + " along timeline)"
        except (TypeError, ValueError):
            pct_hint = ""
    if n_mf > 1:
        return f"=== GOOGLE VISION ({n_mf} frames sampled & merged){pct_hint} ==="
    return "=== GOOGLE VISION (single key frame) ==="


def build_multimodal_scene_digest(ctx: JobContext, *, max_chars: int = 10000) -> str:
    """
    Single narrative digest for M8: **full-clip signals first**, then multi-sample
    Vision (merged JPEGs along the timeline), full-clip Video Intelligence timeline,
    audio/transcript/music signals, probe metadata, then drive telemetry + Trill.
    """
    max_chars = max(500, min(int(max_chars or 10000), 50000))
    parts: List[str] = []
    budget = max_chars

    def push(header: str, body: str, share: float) -> None:
        nonlocal budget
        if budget < 80:
            return
        slice_n = max(120, int(max_chars * share))
        slice_n = min(slice_n, budget - len(header) - 4)
        chunk = _trim_text(body, slice_n)
        if not chunk:
            return
        block = f"{header}\n{chunk}\n"
        parts.append(block)
        budget -= len(block)

    vu = ctx.video_understanding or {}
    hydration_story = build_hydration_story_text(ctx, max_chars=900)
    if hydration_story:
        push("=== HYDRATION STORY (short factual scene picture) ===", hydration_story, 0.10)

    if isinstance(vu, dict):
        scene = str(vu.get("scene_description") or vu.get("description") or "").strip()
        tsug = str(vu.get("title_suggestion") or "").strip()
        body = scene
        if tsug:
            body = (body + "\nSuggested title angle: " + tsug).strip()
        push("=== FULL VIDEO (scene understanding) ===", body, 0.30)

    probe = _video_probe_lines(ctx)
    if probe:
        push("=== VIDEO FILE (probe) ===", probe, 0.05)

    vi = ctx.video_intelligence_context or {}
    if isinstance(vi, dict) and not vi.get("error"):
        vi_body = _video_intelligence_digest_body(vi)
        if vi_body:
            push("=== VIDEO INTELLIGENCE (Google - full clip + timeline) ===", vi_body, 0.18)

    arts = getattr(ctx, "output_artifacts", None) or {}
    if isinstance(arts, dict):
        shot_list = arts.get("shot_list_v1")
        if isinstance(shot_list, list) and shot_list:
            shot_lines: List[str] = []
            for row in shot_list[:24]:
                if not isinstance(row, dict):
                    continue
                t = row.get("t_seconds")
                kind = str(row.get("kind") or "beat")
                text = str(row.get("text") or "").strip()
                if not text:
                    continue
                try:
                    t_s = f"{float(t):.1f}s" if t is not None else "?"
                except (TypeError, ValueError):
                    t_s = "?"
                shot_lines.append(f"[{t_s} {kind}] {text[:160]}")
            if shot_lines:
                push(
                    "=== SHOT LIST (narrate in order; do not invent beats) ===",
                    "\n".join(shot_lines),
                    0.16,
                )

    pe = getattr(ctx, "place_evidence", None)
    if not isinstance(pe, dict) and isinstance(arts, dict):
        pe = arts.get("place_evidence_v1")
    if isinstance(pe, dict) and pe:
        pe_lines: List[str] = []
        if pe.get("places"):
            pe_lines.append("Places: " + ", ".join(str(x) for x in pe["places"][:10]))
        if pe.get("beaches"):
            pe_lines.append("Beaches: " + ", ".join(str(x) for x in pe["beaches"][:6]))
        if pe.get("monuments"):
            pe_lines.append("Monuments: " + ", ".join(str(x) for x in pe["monuments"][:6]))
        if pe.get("stadiums"):
            pe_lines.append("Stadiums: " + ", ".join(str(x) for x in pe["stadiums"][:6]))
        if pe.get("sports_teams"):
            pe_lines.append("Teams: " + ", ".join(str(x) for x in pe["sports_teams"][:6]))
        if pe.get("license_plates"):
            pe_lines.append("License plates (OCR): " + ", ".join(str(x) for x in pe["license_plates"][:4]))
        geo = pe.get("geocode_from_landmark") or {}
        if isinstance(geo, dict) and geo.get("location_display"):
            pe_lines.append(f"Geocoded from landmark: {geo.get('location_display')}")
        if pe_lines:
            push("=== PLACE EVIDENCE (no .map fallback lanes) ===", "\n".join(pe_lines), 0.14)

    vc = ctx.vision_context or {}
    if isinstance(vc, dict) and vc and not vc.get("skipped"):
        lines: List[str] = []
        labels = vc.get("label_names") or []
        if isinstance(labels, list) and labels:
            lines.append("Merged labels (union across samples): " + ", ".join(str(x) for x in labels[:32]))
        ocr = (vc.get("ocr_text") or "").strip()
        if ocr:
            lines.append(
                "OCR / on-screen text (blocks separated by --- per sample): " + ocr[:5200]
            )
        fc = vc.get("face_count")
        if fc is not None:
            lines.append(f"Faces (richest sampled frame): {fc}")
        lm = vc.get("landmark_names") or []
        if isinstance(lm, list) and lm:
            lines.append("Detected landmarks: " + ", ".join(str(x) for x in lm[:12]))
        lg = vc.get("logo_names") or []
        if isinstance(lg, list) and lg:
            lines.append("Detected logos / brands: " + ", ".join(str(x) for x in lg[:12]))
        rec = str(vc.get("recognition_summary") or "").strip()
        if rec:
            lines.insert(0, rec)
        web = vc.get("web_entities") or []
        if isinstance(web, list) and web:
            web_names = [
                (w.get("description") if isinstance(w, dict) else str(w))
                for w in web[:12]
            ]
            lines.append("Web entity matches: " + ", ".join(str(x) for x in web_names if x))
        loc = vc.get("localized_objects") or []
        if isinstance(loc, list) and loc:
            loc_names = [(o.get("name") if isinstance(o, dict) else str(o)) for o in loc[:12]]
            lines.append("Localized objects: " + ", ".join(str(x) for x in loc_names if x))
        dc = vc.get("dominant_colors") or []
        if isinstance(dc, list) and dc:
            lines.append(
                "Dominant colors: "
                + ", ".join(
                    str((c.get("name") if isinstance(c, dict) else c)) for c in dc[:6]
                )
            )
        if lines:
            push(_vision_digest_header(vc), "\n".join(lines), 0.22)

    ac = ctx.audio_context or {}
    if isinstance(ac, dict):
        lines_a: List[str] = []
        tx = (ctx.ai_transcript or ac.get("transcript") or "").strip()
        if tx:
            role = str(ac.get("transcript_role") or "").strip()
            lang = str(ac.get("language") or "").strip()
            head = "Transcript"
            if role or lang:
                head += f" (role={role or 'unknown'}, lang={lang or 'unknown'})"
            lines_a.append(head + ":\n" + tx[:7500])
        gas = (ac.get("gpt_audio_summary") or "").strip()
        if gas:
            lines_a.append("Audio summary: " + gas[:1600])
        spf = str(ac.get("sound_profile") or "").strip()
        if spf:
            lines_a.append("Sound profile: " + spf[:500])
        tsc = str(ac.get("top_sound_class") or "").strip()
        if tsc:
            lines_a.append("Top sound class: " + tsc[:160])
        cts = ac.get("content_signals") or []
        if isinstance(cts, list) and cts:
            lines_a.append("Content signals: " + ", ".join(str(x) for x in cts[:28]))
        fn = str(ac.get("fusion_narrative") or "").strip()
        if fn:
            lines_a.append("Fusion narrative (audio+context): " + fn[:1800])
        mus = []
        if ac.get("music_title"):
            mus.append(f"track={ac.get('music_title')}")
        if ac.get("music_artist"):
            mus.append(f"artist={ac.get('music_artist')}")
        if ac.get("music_genre"):
            mus.append(f"genre={ac.get('music_genre')}")
        if mus:
            lines_a.append("Music ID: " + ", ".join(mus))
        if ac.get("copyright_risk"):
            lines_a.append("Audio: third-party / copyright risk flagged - do not claim ownership of the track.")
        if ac.get("music_detected") and not mus:
            lines_a.append("Music detected (no reliable title/artist metadata).")
        yev = ac.get("yamnet_events")
        if isinstance(yev, list) and yev:
            lines_a.append("Ambient sound events: " + ", ".join(str(x) for x in yev[:24]))
        he = ac.get("hume_emotions")
        if isinstance(he, dict) and he.get("dominant_emotion"):
            lines_a.append(
                f"Dominant speech emotion: {he.get('dominant_emotion')} "
                f"(intensity {he.get('emotional_intensity', '')})"
            )
        if lines_a:
            push("=== AUDIO & SPEECH ===", "\n\n".join(lines_a), 0.22)

    osd = ctx.dashcam_osd_context or {}
    if isinstance(osd, dict) and osd and not osd.get("skipped"):
        osd_lines: List[str] = []
        fs = osd.get("first_seen") or {}
        ls = osd.get("last_seen") or {}
        if fs.get("date") or fs.get("time"):
            osd_lines.append(
                f"Clip start (HUD): {fs.get('date') or '?'} {fs.get('time') or ''}".strip()
            )
        if (ls.get("date") or ls.get("time")) and (ls.get("date") != fs.get("date") or ls.get("time") != fs.get("time")):
            osd_lines.append(
                f"Clip end (HUD): {ls.get('date') or '?'} {ls.get('time') or ''}".strip()
            )
        if fs.get("lat") is not None and fs.get("lon") is not None:
            try:
                osd_lines.append(f"GPS at clip start (HUD): {float(fs['lat']):.5f}, {float(fs['lon']):.5f}")
            except (TypeError, ValueError):
                pass
        if ls.get("lat") is not None and ls.get("lon") is not None and (
            ls.get("lat") != fs.get("lat") or ls.get("lon") != fs.get("lon")
        ):
            try:
                osd_lines.append(f"GPS at clip end (HUD): {float(ls['lat']):.5f}, {float(ls['lon']):.5f}")
            except (TypeError, ValueError):
                pass
        peak = osd.get("max_speed_mph")
        if peak:
            at_s = osd.get("max_speed_at_s")
            tail = f" at ~{float(at_s):.0f}s into clip" if at_s is not None else ""
            osd_lines.append(f"Peak speed (HUD): {peak} mph{tail}")
        avg = osd.get("avg_speed_mph")
        if avg:
            osd_lines.append(f"Average speed (HUD): {avg} mph")
        unit = osd.get("speed_unit_detected")
        if unit:
            osd_lines.append(f"On-screen speed unit: {unit}")
        driver = osd.get("driver_name")
        if driver:
            osd_lines.append(f"Driver name on HUD: {driver}")
        path = osd.get("gps_path") or []
        if isinstance(path, list) and len(path) >= 2:
            osd_lines.append(f"OSD GPS fixes recovered: {len(path)} (across HUD timeline)")
        cov = osd.get("coverage_pct")
        if cov is not None:
            try:
                osd_lines.append(f"HUD recognition coverage: {float(cov) * 100:.0f}% of sampled frames")
            except (TypeError, ValueError):
                pass
        if osd.get("telemetry_backfilled"):
            osd_lines.append("Telemetry was backfilled from HUD (no .map file uploaded).")
        if osd_lines:
            push("=== DASHCAM HUD (burned-in overlay) ===", "\n".join(osd_lines), 0.10)

    tel = ctx.telemetry or ctx.telemetry_data
    tr = ctx.trill or ctx.trill_score
    if tel or tr:
        loc_bits: List[str] = []
        if tel:
            for attr, label in (
                ("location_display", "Location (mid-route)"),
                ("location_start_display", "Route start"),
                ("location_road", "Road"),
                ("location_city", "City"),
                ("location_state", "Region"),
                ("location_country", "Country"),
            ):
                v = getattr(tel, attr, None)
                if v:
                    loc_bits.append(f"{label}: {v}")
            gp = getattr(tel, "gazetteer_place_name", None)
            if gp:
                loc_bits.append(f"Census nearest place: {gp}")
            gus = getattr(tel, "gazetteer_state_usps", None)
            if gus:
                loc_bits.append(f"Census place state (USPS): {gus}")
            pun = getattr(tel, "padus_unit_name", None)
            if pun:
                loc_bits.append(f"Protected area (PADUS): {pun}")
            elif getattr(tel, "near_padus", False):
                loc_bits.append("Route intersects protected lands (PADUS).")
            # WGS84 decimal degrees (same as scene_graph geo.*); disambiguates cities, cross-border routes
            mla, mlo = getattr(tel, "mid_lat", None), getattr(tel, "mid_lon", None)
            if mla is not None and mlo is not None:
                try:
                    loc_bits.append(f"GPS mid-route (WGS84): {float(mla):.5f}, {float(mlo):.5f}")
                except (TypeError, ValueError):
                    pass
            sla, slo = getattr(tel, "start_lat", None), getattr(tel, "start_lon", None)
            if sla is not None and slo is not None:
                try:
                    loc_bits.append(f"GPS route start (WGS84): {float(sla):.5f}, {float(slo):.5f}")
                except (TypeError, ValueError):
                    pass
            pts = getattr(tel, "points", None) or []
            if isinstance(pts, list) and len(pts) >= 2:
                rtxt = format_route_spatial_digest(pts, max_polyline_points=36, max_chars=4200)
                if rtxt:
                    loc_bits.append(rtxt)
        if tr:
            loc_bits.append(f"Trill: score={getattr(tr, 'score', '')} bucket={getattr(tr, 'bucket', '')}")
            tm = str(getattr(tr, "title_modifier", "") or "").strip()
            if tm:
                loc_bits.append(f"Trill title modifier: {tm}")
            th = getattr(tr, "hashtags", None) or []
            if isinstance(th, list) and th:
                loc_bits.append("Trill hashtag ideas: " + ", ".join(str(x).lstrip("#") for x in th[:14]))
        if tel:
            mph = getattr(tel, "max_speed_mph", None)
            if mph:
                loc_bits.append(f"Peak speed: {mph} mph")
            dist = getattr(tel, "total_distance_miles", None)
            if dist:
                loc_bits.append(f"Distance: {dist} mi")
            dur = getattr(tel, "duration_seconds", None)
            if dur is not None and float(dur or 0) > 0:
                loc_bits.append(f"Route duration (from .map): {float(dur):.0f}s")
            npts = len(getattr(tel, "points", None) or [])
            if npts:
                loc_bits.append(f"GPS samples in route: {npts}")
            alt = getattr(tel, "max_altitude_ft", None)
            if alt is not None and float(alt or 0) > 0:
                loc_bits.append(f"Max altitude (from .map): {float(alt):.0f} ft")
            sp = getattr(tel, "speeding_seconds", None)
            if sp is not None and float(sp or 0) > 1:
                loc_bits.append(f"Time above speeding threshold: {float(sp):.0f}s")
            eu = getattr(tel, "euphoria_seconds", None)
            if eu is not None and float(eu or 0) > 1:
                loc_bits.append(f"High-energy / euphoria seconds (telemetry): {float(eu):.0f}s")
        if loc_bits:
            push("=== DRIVE / TELEMETRY + TRILL ===", "\n".join(loc_bits), 0.14)

    out = "\n".join(parts).strip()
    return _trim_text(out, max_chars)


def build_fusion_summary_text(ctx: JobContext) -> str:
    """2–4 sentence executive summary for the scene graph JSON."""
    digest = build_multimodal_scene_digest(ctx, max_chars=4500)
    if not digest:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", digest.replace("\n", " "))
    picked: List[str] = []
    for s in sentences:
        t = s.strip()
        if len(t) < 40:
            continue
        picked.append(t)
        if len(picked) >= 4:
            break
    if not picked:
        picked = [_trim_text(digest, 900)]

    out = " ".join(picked).strip()
    return _trim_text(out, 4000)


def build_fusion_caption_rules(ctx: JobContext) -> str:
    """Hard prompt constraints derived from fused evidence (anti-stale / anti-wrong)."""
    lines: List[str] = [
        "FUSION RULES (obey all that apply):",
        "- Anchor the hook in the **newest, most specific** facts below (scene understanding > full-clip video intelligence + timeline > merged multi-frame Vision > transcript).",
        "- If transcript and visuals disagree, trust **visuals + scene understanding** for on-screen facts; use transcript for quoted speech only.",
        "- Do not recycle generic automotive / lifestyle templates unless telemetry or labels explicitly support them.",
        "- When **landmarks**, **logos**, or **.map / GPS** fields are present, cite at least one of them explicitly (place name, brand, or route statistic) - do not ignore them.",
    ]
    ac = ctx.audio_context or {}
    if isinstance(ac, dict):
        role = str(ac.get("transcript_role") or "").strip().lower()
        if role in ("third_party_lyrics", "third_party_music", "song"):
            lines.append(
                "- Audio role is third-party lyrics/music: never claim the creator wrote, produced, or performed the track."
            )
        if str(ac.get("fusion_narrative") or "").strip():
            lines.append(
                "- Audio fusion narrative is present: use it as a thematic anchor when it does not contradict on-screen evidence."
            )
        cs = ac.get("content_signals") or []
        if isinstance(cs, list) and cs:
            lines.append(
                "- Structured audio content_signals are present: prefer named concrete signals over generic hype."
            )

    vu = ctx.video_understanding or {}
    if isinstance(vu, dict) and (vu.get("scene_description") or vu.get("description")):
        lines.append("- Scene understanding is present: the caption must reflect at least one concrete detail from it.")

    vc = ctx.vision_context or {}
    if isinstance(vc, dict) and (vc.get("ocr_text") or "").strip():
        lines.append("- OCR text is present: mention a specific on-screen word/phrase when it helps accuracy (scores, brands, place names).")
    if isinstance(vc, dict) and (vc.get("landmark_names") or vc.get("logo_names")):
        lines.append(
            "- Google Vision reported landmark(s) and/or logo(s): treat those as high-confidence on-screen entities."
        )
    if isinstance(vc, dict):
        try:
            mf = int(vc.get("vision_multi_frame") or 1)
        except (TypeError, ValueError):
            mf = 1
        if mf > 1:
            lines.append(
                "- Google Vision merged multiple frames along the clip: OCR blocks may come from different moments (--- separators); integrate the strongest truthful specifics."
            )

    vi_ctx = ctx.video_intelligence_context or {}
    if isinstance(vi_ctx, dict) and not vi_ctx.get("error"):
        segs = vi_ctx.get("segment_labels") or []
        if isinstance(segs, list) and segs:
            lines.append(
                "- Video Intelligence segment timeline is present: prefer at least one time-grounded label or action when it matches the story."
            )

    tr = ctx.trill or ctx.trill_score
    if tr and str(getattr(tr, "title_modifier", "") or "").strip():
        lines.append(
            "- Trill title modifier is present: you may echo that energy in the hook when it matches visible/telemetry context."
        )

    osd_ctx = ctx.dashcam_osd_context or {}
    if isinstance(osd_ctx, dict) and osd_ctx and not osd_ctx.get("skipped"):
        if osd_ctx.get("max_speed_mph") or (osd_ctx.get("first_seen") or {}).get("lat") is not None:
            lines.append(
                "- Dashcam HUD (burned-in overlay) was decoded: prefer its date, time, GPS, "
                "peak/avg speed and driver name as ground truth for time-and-place beats; "
                "they were read directly off the video and beat any guess."
            )
        if osd_ctx.get("telemetry_backfilled"):
            lines.append(
                "- HUD GPS was used to backfill telemetry (no .map file uploaded): treat "
                "the route extent and peak speed as real, not estimated."
            )

    tel = ctx.telemetry or ctx.telemetry_data
    if tel and getattr(tel, "mid_lat", None) is not None and getattr(tel, "mid_lon", None) is not None:
        lines.append(
            "- GPS (WGS84) appears in digest/geo: use with reverse-geocoded place names for accuracy; "
            "prefer human place names in the public caption unless the user clearly wants coordinates."
        )
    if tel and isinstance(getattr(tel, "points", None), list) and len(getattr(tel, "points") or []) >= 4:
        lines.append(
            "- Route bbox + sampled polyline may appear: use them for extent, direction, and shape; "
            "do not invent a different path or region than these GPS samples support."
        )
    if tel and (getattr(tel, "max_speed_mph", 0) or 0) > 1:
        lines.append("- Telemetry shows meaningful speed: reference mph, road, or place when writing automotive beats.")
    if tel and getattr(tel, "location_start_display", None) and getattr(tel, "location_display", None):
        if str(tel.location_start_display).strip() != str(tel.location_display).strip():
            lines.append(
                "- Route start and mid-route geocodes differ: you may contrast where the drive began vs where this clip sits on the route."
            )

    return "\n".join(lines) + "\n"
