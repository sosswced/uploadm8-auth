"""
UploadM8 Job Context
====================
Carries all state through the processing pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core.helpers import coerce_hashtag_list, sanitize_hashtag_body

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
    # When route start is far from midpoint, second Nominatim lookup at start GPS
    location_start_display: Optional[str] = None

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
    # M8 caption engine: per-platform AI hashtag variants (tiktok, instagram, …)
    m8_platform_hashtags: Dict[str, List[str]] = field(default_factory=dict)
    # Populated by audio_stage (Whisper) when enabled — injected into caption prompts.
    ai_transcript: Optional[str] = None
    audio_path: Optional[Path] = None
    audio_context: Dict[str, Any] = field(default_factory=dict)
    vision_context: Dict[str, Any] = field(default_factory=dict)
    video_understanding: Dict[str, Any] = field(default_factory=dict)
    video_intelligence_context: Dict[str, Any] = field(default_factory=dict)
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
        # Upload form / DB title wins over AI so explicit user copy is never replaced.
        t = (self.title or "").strip()
        if t:
            return t
        if (self.ai_title or "").strip():
            return (self.ai_title or "").strip()
        return (self.filename or "").strip() or ""

    def get_effective_caption(self) -> str:
        c = (self.caption or "").strip()
        if c:
            return c
        return (self.ai_caption or "").strip()

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
        return {
            "effective_title": self.get_effective_title() or self.filename or "Video",
            "effective_caption": self.get_effective_caption() or "",
            "category": category or getattr(self, "thumbnail_category", None) or "general",
            "location_name": self.location_name or "",
            "trill_bucket": trill.bucket if trill else "",
            "max_speed_mph": str(max_mph),
            "platforms_csv": platforms_csv,
        }


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

HARD RULES
- No profanity, no hate, no nudity, no weapons emphasis, no illegal claims.
- No copyrighted logos/brand marks (YouTube logo, TikTok logo, etc).
- ACCURACY: Headlines and badges must reflect what is actually in the video. No misleading claims (e.g. "TOP 5" when it's not a list, "NEW" when it's not new). Describe visible content truthfully.
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
- tiktok: assume no custom thumbnail via API; instead output a recommended thumb_offset_seconds to pick a frame and a text strategy that does NOT rely on overlays.

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
    """Coerce uploads.hashtags (jsonb list, JSON string, or broken string) to flat tokens."""
    return [str(t) for t in coerce_hashtag_list(raw) if str(t).strip()]


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
        hashtags=_normalize_upload_hashtags_field(upload_record.get("hashtags")),
        privacy=upload_record.get("privacy", "public") or "public",
        reframe_mode=reframe_mode,
        user_settings=user_settings or {},
        entitlements=entitlements,
    )


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


def build_multimodal_scene_digest(ctx: JobContext, *, max_chars: int = 10000) -> str:
    """
    Single narrative digest for M8: **full-clip signals first**, then sampled frame
    vision, then audio/transcript, then drive telemetry. Ordering reduces
    “stale template” captions that overweight filename or sparse early cues.
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
    if isinstance(vu, dict):
        scene = str(vu.get("scene_description") or vu.get("description") or "").strip()
        tsug = str(vu.get("title_suggestion") or "").strip()
        body = scene
        if tsug:
            body = (body + "\nSuggested title angle: " + tsug).strip()
        push("=== FULL VIDEO (scene understanding) ===", body, 0.34)

    vi = ctx.video_intelligence_context or {}
    if isinstance(vi, dict) and not vi.get("error"):
        tl = vi.get("top_labels") or []
        if isinstance(tl, list) and tl:
            push(
                "=== VIDEO INTELLIGENCE (Google — full clip labels) ===",
                "\n".join(str(x) for x in tl[:22]),
                0.14,
            )
        elif (vi.get("summary_text") or "").strip():
            push(
                "=== VIDEO INTELLIGENCE (Google — full clip labels) ===",
                str(vi.get("summary_text") or ""),
                0.12,
            )

    vc = ctx.vision_context or {}
    if isinstance(vc, dict) and vc and not vc.get("skipped"):
        lines: List[str] = []
        labels = vc.get("label_names") or []
        if isinstance(labels, list) and labels:
            lines.append("Frame labels: " + ", ".join(str(x) for x in labels[:28]))
        ocr = (vc.get("ocr_text") or "").strip()
        if ocr:
            lines.append("OCR / on-screen text: " + ocr[:2400])
        fc = vc.get("face_count")
        if fc is not None:
            lines.append(f"Faces (sampled frame): {fc}")
        lm = vc.get("landmark_names") or []
        if isinstance(lm, list) and lm:
            lines.append("Detected landmarks: " + ", ".join(str(x) for x in lm[:10]))
        lg = vc.get("logo_names") or []
        if isinstance(lg, list) and lg:
            lines.append("Detected logos / brands: " + ", ".join(str(x) for x in lg[:10]))
        if lines:
            push("=== SAMPLE FRAME (Google Vision — one key frame) ===", "\n".join(lines), 0.16)

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
            lines_a.append(head + ":\n" + tx[:7000])
        gas = (ac.get("gpt_audio_summary") or "").strip()
        if gas:
            lines_a.append("Audio summary: " + gas[:900])
        mus = []
        if ac.get("music_title"):
            mus.append(f"track={ac.get('music_title')}")
        if ac.get("music_artist"):
            mus.append(f"artist={ac.get('music_artist')}")
        if ac.get("music_genre"):
            mus.append(f"genre={ac.get('music_genre')}")
        if mus:
            lines_a.append("Music ID: " + ", ".join(mus))
        yev = ac.get("yamnet_events")
        if isinstance(yev, list) and yev:
            lines_a.append("Ambient sound events: " + ", ".join(str(x) for x in yev[:20]))
        he = ac.get("hume_emotions")
        if isinstance(he, dict) and he.get("dominant_emotion"):
            lines_a.append(
                f"Dominant speech emotion: {he.get('dominant_emotion')} "
                f"(intensity {he.get('emotional_intensity', '')})"
            )
        if lines_a:
            push("=== AUDIO & SPEECH ===", "\n\n".join(lines_a), 0.22)

    tel = ctx.telemetry or ctx.telemetry_data
    if tel:
        loc_bits: List[str] = []
        for attr, label in (
            ("location_display", "Location (mid-route)"),
            ("location_start_display", "Route start"),
            ("location_road", "Road"),
            ("location_city", "City"),
            ("location_state", "Region"),
        ):
            v = getattr(tel, attr, None)
            if v:
                loc_bits.append(f"{label}: {v}")
        tr = ctx.trill or ctx.trill_score
        if tr:
            loc_bits.append(f"Trill: score={getattr(tr, 'score', '')} bucket={getattr(tr, 'bucket', '')}")
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
            loc_bits.append(f"GPS samples in .map: {npts}")
        alt = getattr(tel, "max_altitude_ft", None)
        if alt is not None and float(alt or 0) > 0:
            loc_bits.append(f"Max altitude (from .map): {float(alt):.0f} ft")
        sp = getattr(tel, "speeding_seconds", None)
        if sp is not None and float(sp or 0) > 1:
            loc_bits.append(f"Time above speeding threshold: {float(sp):.0f}s")
        if loc_bits:
            push("=== DRIVE / TELEMETRY (.map + reverse geocode) ===", "\n".join(loc_bits), 0.12)

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
        "- Anchor the hook in the **newest, most specific** facts below (scene understanding > video labels > frame OCR > transcript).",
        "- If transcript and visuals disagree, trust **visuals + scene understanding** for on-screen facts; use transcript for quoted speech only.",
        "- Do not recycle generic automotive / lifestyle templates unless telemetry or labels explicitly support them.",
        "- When **landmarks**, **logos**, or **.map / GPS** fields are present, cite at least one of them explicitly (place name, brand, or route statistic) — do not ignore them.",
    ]
    ac = ctx.audio_context or {}
    if isinstance(ac, dict):
        role = str(ac.get("transcript_role") or "").strip().lower()
        if role in ("third_party_lyrics", "third_party_music", "song"):
            lines.append(
                "- Audio role is third-party lyrics/music: never claim the creator wrote, produced, or performed the track."
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

    tel = ctx.telemetry or ctx.telemetry_data
    if tel and (getattr(tel, "max_speed_mph", 0) or 0) > 1:
        lines.append("- Telemetry shows meaningful speed: reference mph, road, or place when writing automotive beats.")
    if tel and getattr(tel, "location_start_display", None) and getattr(tel, "location_display", None):
        if str(tel.location_start_display).strip() != str(tel.location_display).strip():
            lines.append(
                "- Route start and mid-route geocodes differ: you may contrast where the drive began vs where this clip sits on the route."
            )

    return "\n".join(lines) + "\n"
