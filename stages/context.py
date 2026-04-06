"""
UploadM8 Job Context
====================
Carries all state through the processing pipeline.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from .entitlements import Entitlements
from .safe_parse import json_dict

logger = logging.getLogger("uploadm8-worker.context")


def expand_hashtag_items(items: Any) -> List[str]:
    """
    Flatten hashtag inputs from DB/UI. Handles nested lists and JSON-looking strings
    (e.g. '["tag1","tag2"]' stored as a single element) so publish never emits '#"[\"a\"]"' artifacts.
    """
    def _clean_token(raw: Any) -> str:
        token = re.sub(r"[^a-z0-9_]", "", str(raw or "").strip().lower().lstrip("#"))
        return token[:50]

    out: List[str] = []
    if items is None:
        return out
    if isinstance(items, str):
        s = items.strip()
        if not s:
            return out
        if s.startswith("[") and "]" in s:
            try:
                parsed = json.loads(s)
                return expand_hashtag_items(parsed)
            except json.JSONDecodeError as e:
                logger.debug("expand_hashtag_items: JSON parse failed for bracket string: %s", e)
        cleaned = []
        for t in s.replace(",", " ").split():
            token = _clean_token(t)
            if token:
                cleaned.append(token)
        return cleaned
    if not isinstance(items, (list, tuple)):
        return out
    for item in items:
        if item is None:
            continue
        s = str(item).strip()
        if not s:
            continue
        if (s.startswith("[") and s.endswith("]")) or (s.startswith('"[') and "]" in s):
            try:
                cleaned = s.strip('"').strip("'")
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    out.extend(expand_hashtag_items(parsed))
                    continue
            except json.JSONDecodeError as e:
                logger.debug("expand_hashtag_items: nested list JSON parse failed: %s", e)
        if s.startswith("#") and "[" in s:
            try:
                parsed = json.loads(s[1:])
                if isinstance(parsed, list):
                    out.extend(expand_hashtag_items(parsed))
                    continue
            except json.JSONDecodeError as e:
                logger.debug("expand_hashtag_items: #prefix JSON parse failed: %s", e)
        token = _clean_token(s)
        if token:
            out.append(token)
    deduped: List[str] = []
    seen: set = set()
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        deduped.append(t)
    return deduped


_LANDMARK_HINT_WORDS = frozenset({
    "building", "monument", "architecture", "tower", "landmark", "church", "cathedral",
    "castle", "bridge", "downtown", "skyline", "statue", "museum", "historic", "palace",
    "plaza", "square", "arena", "stadium", "temple", "mosque", "synagogue", "lighthouse",
    "ruins", "national park", "memorial",
})


def extract_landmark_hints(label_names: List[str], ocr_text: str) -> List[str]:
    """Heuristic hints for architecture / POI language from Vision labels + OCR (no extra API)."""
    out: List[str] = []
    seen: set = set()
    for lbl in label_names or []:
        s = str(lbl).strip()
        low = s.lower()
        if not s:
            continue
        if any(w in low for w in _LANDMARK_HINT_WORDS):
            if low not in seen:
                seen.add(low)
                out.append(s)
    ob = (ocr_text or "").lower()
    for w in _LANDMARK_HINT_WORDS:
        if w in ob and w not in seen:
            seen.add(w)
            out.append(w)
    return out[:40]


def build_multimodal_scene_digest(ctx: "JobContext", max_chars: int = 12000) -> str:
    """
    Dense, high-recall digest for M8 + caption prompts: background audio, music, speech role,
    full vision labels, OCR, faces, geo (lat/lon + place), telemetry, video understanding.
    Intended to be copied wholesale into prompts — not a substitute for Scene Graph JSON.
    """
    sections: List[str] = []
    ac = ctx.audio_context or {}
    vc = ctx.vision_context or {}
    vu = ctx.video_understanding or {}
    tel = ctx.telemetry or ctx.telemetry_data
    artifacts = getattr(ctx, "output_artifacts", None) or {}

    # ── Audio environment (noise, events, music) ───────────────────────────
    audio_lines: List[str] = []
    if ac.get("sound_profile"):
        audio_lines.append(f"Sound profile: {ac['sound_profile']}")
    if ac.get("top_sound_class"):
        audio_lines.append(f"Top sound class: {ac.get('top_sound_class')}")
    yev = ac.get("yamnet_events") or []
    if yev:
        audio_lines.append(
            "YAMNet / background & environment sounds: " + ", ".join(str(x) for x in yev)
        )
    ol = artifacts.get("audio_labels")
    if ol and ol not in (yev, str(yev)):
        audio_lines.append(f"Extra audio labels: {ol}")
    if ac.get("music_detected"):
        mg = ac.get("music_genre") or ""
        mt = ac.get("music_title") or ""
        ma = ac.get("music_artist") or ""
        line = f"Identified music: {mt} — {ma}".strip(" —")
        if mg:
            line += f" (genre: {mg})"
        audio_lines.append(line)
    if ac.get("transcript_role"):
        audio_lines.append(f"Transcript role: {ac['transcript_role']}")
    sig = ac.get("content_signals") or []
    if isinstance(sig, list) and sig:
        audio_lines.append("Content signals: " + ", ".join(str(x) for x in sig[:24]))
    if audio_lines:
        sections.append("AUDIO (environment + music + role)\n" + "\n".join(audio_lines))

    # ── Geo / route (lat-lon unlimited when present) ─────────────────────────
    geo_lines: List[str] = []
    if tel:
        if getattr(tel, "location_display", None):
            geo_lines.append(f"Place (reverse geocode): {tel.location_display}")
        if getattr(tel, "location_road", None):
            geo_lines.append(f"Road / route: {tel.location_road}")
        for lat, lon, lab in (
            (getattr(tel, "mid_lat", None), getattr(tel, "mid_lon", None), "mid route"),
            (getattr(tel, "start_lat", None), getattr(tel, "start_lon", None), "start"),
        ):
            if lat is not None and lon is not None:
                geo_lines.append(f"Coordinates ({lab}): {lat:.6f}, {lon:.6f}")
        alt = getattr(tel, "max_altitude_ft", None) or 0
        if alt and alt > 10:
            geo_lines.append(f"Max altitude (ft): {alt:.0f}")
        dist = getattr(tel, "total_distance_miles", None) or 0
        if dist and dist > 0.05:
            geo_lines.append(f"Route distance (mi): {dist:.2f}")
        mph = getattr(tel, "max_speed_mph", 0) or 0
        if mph and mph > 3:
            geo_lines.append(f"Peak speed (mph): {mph:.0f}")
    if ctx.location_name and not any("Place" in x for x in geo_lines):
        geo_lines.append(f"Location hint: {ctx.location_name}")
    if geo_lines:
        sections.append("GEO / TELEMETRY\n" + "\n".join(geo_lines))

    # ── Vision: people, objects, vehicles, text (full lists — no arbitrary tiny caps) ──
    vis_lines: List[str] = []
    labels = vc.get("label_names") or []
    if labels:
        vis_lines.append("Scene & object labels (Google Vision): " + ", ".join(str(x) for x in labels))
    fc = vc.get("face_count")
    if fc is not None:
        vis_lines.append(
            f"People: face_count={fc}, has_faces={vc.get('has_faces')}, "
            f"expressive_faces={vc.get('expressive')}"
        )
    ocr = (vc.get("ocr_text") or "").strip()
    if ocr:
        vis_lines.append("On-screen text (OCR, full): " + ocr[:8000])
    lh = extract_landmark_hints(labels, ocr)
    if lh:
        vis_lines.append("Architecture / landmark language: " + ", ".join(lh))
    if vis_lines:
        sections.append("VISION\n" + "\n".join(vis_lines))

    sd = (vu.get("scene_description") or vu.get("description") or "").strip()
    if sd:
        sections.append("VIDEO UNDERSTANDING (Twelve Labs / similar)\n" + sd[:8000])

    if ac.get("fusion_narrative"):
        sections.append(
            "FUSION NARRATIVE (audio GPT)\n" + str(ac.get("fusion_narrative"))[:2000]
        )

    vi = getattr(ctx, "video_intelligence_context", None) or {}
    if vi and not vi.get("error"):
        vi_lines: List[str] = []
        if vi.get("summary_text"):
            vi_lines.append(str(vi["summary_text"])[:2000])
        tops = vi.get("top_labels") or []
        if tops:
            vi_lines.append("Top VI labels: " + ", ".join(str(x) for x in tops[:20]))
        n_shots = len(vi.get("shots") or [])
        if n_shots:
            vi_lines.append(f"Detected shot boundaries: {n_shots} segments")
        if vi_lines:
            sections.append("VIDEO INTELLIGENCE (full clip)\n" + "\n".join(vi_lines))

    text = "\n\n".join(sections)
    return text[:max_chars]


def build_lyrics_creative_playbook(ctx: "JobContext") -> str:
    """
    When lyrics are present, give concrete creative angles (still attribution-safe).
    """
    ac = ctx.audio_context or {}
    role = (ac.get("transcript_role") or "").strip()
    lyrics_like = role in ("third_party_lyrics", "third_party_music", "mixed_speech_and_music")
    if not lyrics_like:
        if not (ac.get("music_detected") and (ac.get("transcript") or "").strip()):
            return ""
    mt = ac.get("music_title") or ""
    ma = ac.get("music_artist") or ""
    track_hint = f' ({mt} — {ma})'.strip() if (mt or ma) else ""

    lines = [
        "Atmosphere, not identity: let ONE short lyric-flavored line echo the scene (e.g. night drive + road imagery) — you are describing the clip, not performing as the artist.",
        "Soundtrack framing: 'this song + this view' or 'the energy when this comes on in the car' — vibe without claiming you wrote or released the track.",
        "Contrast hook: if lyrics feel heavy but the visuals are calm (or the opposite), name that tension in one clause — it reads human, not bot.",
        "Quote discipline: avoid long verbatim verses; at most a 2–6 word fragment as flavor, or paraphrase the mood.",
        "Hashtag angles: song title fragment, artist + genre, setting + era, car/scene niche — not #newmusic / #original unless the user explicitly said so.",
        f"Track context for tone only{track_hint}: use genre/mood to pick adjectives; still anchor nouns in what the camera actually shows.",
    ]
    return (
        "━━ LYRICS + SCENE (creative integration — obey FUSION / ATTRIBUTION rules first) ━━\n"
        + "\n".join(f"• {ln}" for ln in lines)
    )


def build_fusion_summary_text(ctx: "JobContext") -> str:
    """
    One-paragraph fusion of audio + vision + geo for thumbnail/caption prompts.
    Does not call external APIs — derives from fields already on the context.
    """
    parts: List[str] = []
    ac = ctx.audio_context or {}
    vc = ctx.vision_context or {}
    vu = ctx.video_understanding or {}
    tel = ctx.telemetry or ctx.telemetry_data

    role = (ac.get("transcript_role") or "").strip()
    if role:
        parts.append(f"Transcript role: {role.replace('_', ' ')}.")
    if ac.get("music_detected"):
        mt = ac.get("music_title") or ""
        ma = ac.get("music_artist") or ""
        if mt or ma:
            parts.append(f"Background/identified audio: {mt} — {ma}".strip(" —"))
        if ac.get("music_genre"):
            parts.append(f"Genre: {ac['music_genre']}.")
    if ac.get("sound_profile") or ac.get("top_sound_class"):
        parts.append(
            f"Audio environment: {ac.get('sound_profile', '')} / {ac.get('top_sound_class', '')}".strip(" /")
        )
    labels = vc.get("label_names") or []
    if labels:
        parts.append("Visual focus: " + ", ".join(str(x) for x in labels) + ".")
    ocr = (vc.get("ocr_text") or "").strip()
    if ocr:
        parts.append("On-screen text (OCR): " + ocr[:400] + ("…" if len(ocr) > 400 else ""))
    if vu.get("scene_description") or vu.get("description"):
        sd = (vu.get("scene_description") or vu.get("description") or "")[:600]
        parts.append("Video understanding: " + sd + ("…" if len(sd) >= 600 else ""))
    vi_ctx = getattr(ctx, "video_intelligence_context", None) or {}
    if vi_ctx and not vi_ctx.get("error"):
        vsum = (vi_ctx.get("summary_text") or "").strip()
        if vsum:
            parts.append(vsum[:500] + ("…" if len(vsum) > 500 else ""))
    if tel:
        mph = getattr(tel, "max_speed_mph", 0) or 0
        if mph and mph > 5:
            parts.append(f"Telemetry: peak about {mph:.0f} mph.")
        if getattr(tel, "location_display", None):
            parts.append(f"Location: {tel.location_display}.")
        mla, mlo = getattr(tel, "mid_lat", None), getattr(tel, "mid_lon", None)
        if mla is not None and mlo is not None:
            parts.append(f"Route coordinates (mid): {mla:.5f}, {mlo:.5f}.")
    lh = extract_landmark_hints(labels, ocr)
    if lh:
        parts.append("Landmark/architecture hints: " + ", ".join(lh[:12]) + ".")
    return " ".join(parts)[:4000]


def build_fusion_caption_rules(ctx: "JobContext") -> str:
    """Hard rules for caption JSON prompt when audio is ambiguous."""
    ac = ctx.audio_context or {}
    lines: List[str] = []
    role = (ac.get("transcript_role") or "").strip()
    if role in ("third_party_lyrics", "third_party_music"):
        lines.append(
            "The transcript is primarily SONG LYRICS from a recorded track — NOT the creator's "
            "original speech or songwriting. Do NOT say 'I wrote', 'my track', 'my song', "
            "'original music', or 'new music' unless the user's own caption/title explicitly claims that."
        )
        lines.append(
            "Describe the real-world scene (drive, place, vibe, energy) instead of narrating as the vocalist."
        )
        lines.append(
            "Do NOT write in first person as if you are singing, performing, or channeling the artist's "
            "voice. No 'I channel', 'my performance', 'my vocals', or lip-sync POV unless the visuals "
            "clearly show a performance to camera."
        )
    elif role == "mixed_speech_and_music":
        lines.append(
            "The transcript may mix spoken speech and music — attribute only clear speech to the creator; "
            "do not treat lyrics as autobiographical truth."
        )
    elif ac.get("music_detected") and (ac.get("music_title") or ac.get("copyright_risk")):
        lines.append(
            f"If you reference the soundtrack, you may name the vibe or setting; avoid claiming "
            f"you created the song \"{ac.get('music_title', '')}\" unless the user explicitly says so."
        )
    if not lines:
        base = ""
    else:
        base = "━━ FUSION / ATTRIBUTION (audio vs video — follow strictly) ━━\n" + "\n".join(
            f"• {ln}" for ln in lines
        )
    playbook = build_lyrics_creative_playbook(ctx)
    if playbook:
        return (base + "\n\n" + playbook) if base else playbook
    return base


def resolve_fused_thumbnail_category(ctx: "JobContext") -> Optional[str]:
    """
    When audio classifies as music but the video is clearly automotive FOOTAGE
    (dashcam, highway, speed), prefer the visual category for thumbnails.

    Returns a thumbnail slug (e.g. automotive) or None to let normal detection run.
    """
    ac = ctx.audio_context or {}
    canonical = ctx.get_canonical_category()
    if canonical != "music":
        return None
    if not ac.get("music_detected") and not (ac.get("transcript") or "").strip():
        return None

    vc = ctx.vision_context or {}
    labels = " ".join(str(x).lower() for x in (vc.get("label_names") or [])[:25])
    road_keys = (
        "road", "highway", "car", "vehicle", "windshield", "automotive", "driving",
        "motor vehicle", "lane", "traffic", "tire", "wheel",
    )
    if any(k in labels for k in road_keys):
        return "automotive"

    tel = ctx.telemetry or ctx.telemetry_data
    if tel:
        mph = getattr(tel, "max_speed_mph", 0) or 0
        if mph >= 35:
            return "automotive"
        if (getattr(tel, "total_distance_miles", 0) or 0) > 1.0:
            return "automotive"

    fname = (ctx.filename or "").lower()
    if any(x in fname for x in ("dash", "cam", "drive", "road", "trip", "hwy", "highway")):
        return "automotive"

    return None


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
    # Same numeric value as `score`; kept so callers can use either keyword (`total=` is legacy / JSON).
    total: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self.total != 0 and self.score == 0:
            object.__setattr__(self, "score", int(self.total))
        object.__setattr__(self, "total", int(self.score))

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
    comments: int = 0
    shares: int = 0

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

    # Audio context — populated by audio_stage when Whisper transcription succeeds
    ai_transcript: Optional[str] = None
    audio_path: Optional[Path] = None   # temp WAV path; cleaned up with temp_dir

    telemetry_data: Optional[TelemetryData] = None
    trill_score: Optional[TrillScore] = None

    # Back-compat aliases used by older stages
    telemetry: Optional[TelemetryData] = None
    trill: Optional[TrillScore] = None
    hud_applied: bool = False

    platform_results: List[PlatformResult] = field(default_factory=list)
    output_artifacts: Dict[str, str] = field(default_factory=dict)
    # Built during pipeline; finalized to uploads.pipeline_manifest for diagnostics UI
    pipeline_diag: Optional[Dict[str, Any]] = None

    # Set by worker for distributed publish throttling / circuit breaker (optional)
    redis_client: Any = None

    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    cancel_requested: bool = False

    # Explicit error tracking
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    # JSON-safe dict persisted on uploads.failure_diag for support / user diagnostics
    failure_diag: Optional[Dict[str, Any]] = None

    put_cost: int = 0
    aic_cost: int = 0
    compute_seconds: float = 0.0

    # ── Intelligence context fields (populated by new stages) ────────────────
    # audio_stage: Whisper + ACRCloud + YAMNet + Hume + GPT classification
    audio_context: Optional[Dict[str, Any]] = None

    # vision_stage: Google Cloud Vision face detection + OCR + labels
    vision_context: Optional[Dict[str, Any]] = None

    # twelvelabs_stage: Deep video understanding (scene description + title)
    video_understanding: Optional[Dict[str, Any]] = None

    # video_intelligence_stage: Google Video Intelligence labels + shot boundaries
    video_intelligence_context: Optional[Dict[str, Any]] = None

    # thumbnail_stage: dominant colors from best frame (before brief)
    frame_color_palette: Optional[Dict[str, Any]] = None

    # thumbnail_stage: SerpAPI / YouTube search trend hints (optional)
    trend_intel_context: Optional[Dict[str, Any]] = None

    # thumbnail_stage: AI creative brief for rendering
    thumbnail_brief: Optional[Dict[str, Any]] = None

    # ── M8 Engine (multimodal caption brain — stages/m8_engine.py) ───────────
    m8_scene_graph: Optional[Dict[str, Any]] = None
    m8_engine_output: Optional[Dict[str, Any]] = None
    m8_engine_meta: Optional[Dict[str, Any]] = None
    content_strategy: Optional[Dict[str, Any]] = None
    m8_platform_titles: Dict[str, str] = field(default_factory=dict)
    m8_platform_captions: Dict[str, str] = field(default_factory=dict)
    m8_platform_hashtags: Dict[str, List[str]] = field(default_factory=dict)

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
        """Best title for publishing.

        Single-video manual metadata (self.title) must win over saved settings and AI.
        For uploads without manual title, fall back to platform settings, then AI/model output.
        """
        us = self.user_settings or {}
        # Single-video manual override always wins.
        if isinstance(self.title, str) and self.title.strip():
            return self.title.strip()
        if platform:
            pt = us.get("platformTitles") or us.get("platform_titles") or {}
            if isinstance(pt, dict) and pt:
                pv = pt.get(platform) or pt.get((platform or "").lower())
                if isinstance(pv, str) and pv.strip():
                    return pv.strip()
            m8t = (self.m8_platform_titles or {}).get(platform) or (
                (self.m8_platform_titles or {}).get((platform or "").lower())
            )
            if isinstance(m8t, str) and m8t.strip():
                return m8t.strip()
        # No manual override present: AI / understanding / filename fallbacks.
        tl_title = (self.video_understanding or {}).get("title_suggestion", "")
        return self.ai_title or tl_title or self.filename

    def get_effective_caption(self, platform: str = "") -> str:
        """Best caption.

        Single-video manual metadata (self.caption) must win over saved settings and AI.
        For uploads without manual caption, fall back to platform settings, then AI.
        """
        us = self.user_settings or {}
        # Single-video manual override always wins.
        if isinstance(self.caption, str) and self.caption.strip():
            return self.caption.strip()
        if platform:
            pc = us.get("platformCaptions") or us.get("platform_captions") or {}
            if isinstance(pc, dict) and pc:
                cv = pc.get(platform) or pc.get((platform or "").lower())
                if isinstance(cv, str) and cv.strip():
                    return cv.strip()
            m8c = (self.m8_platform_captions or {}).get(platform) or (
                (self.m8_platform_captions or {}).get((platform or "").lower())
            )
            if isinstance(m8c, str) and m8c.strip():
                return m8c.strip()
        # No manual override present: AI fallback.
        return self.ai_caption or ""

    def get_effective_hashtags(self, platform: str = "") -> List[str]:
        """
        Build the final hashtag list for publishing.

        Merge order: base upload hashtags (manual; kept even if blocked) → always_hashtags →
        platform_hashtags for this platform → AI additions.
        blocked_hashtags apply to saved settings and AI tags, not to manual upload tags.

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
        always_tags: List[str] = expand_hashtag_items(
            raw_always if isinstance(raw_always, list) else []
        )

        # ── Platform-specific hashtags (same logic as always_hashtags) ────
        # Case-insensitive key lookup: frontend may send "TikTok", "Instagram", etc.
        platform_tags: List[str] = []
        if platform:
            ph = us.get("platformHashtags") or us.get("platform_hashtags") or {}
            if isinstance(ph, str):
                ph = json_dict(ph, default={}, context="user_settings.platform_hashtags")
            if isinstance(ph, dict):
                raw = ph.get(platform) or ph.get((platform or "").lower()) or ph.get((platform or "").title()) or []
                if not raw:
                    key_lower = (platform or "").lower()
                    for k, v in ph.items():
                        if str(k).lower() == key_lower:
                            raw = v
                            break
                if isinstance(raw, str):
                    raw = [t.strip() for t in raw.replace(",", " ").split() if t.strip()]
                platform_tags = expand_hashtag_items(raw) if isinstance(raw, list) else []

        # ── Blocked hashtags from settings page ──────────────────────────
        raw_blocked = us.get("blockedHashtags") or us.get("blocked_hashtags") or []
        if isinstance(raw_blocked, str):
            raw_blocked = [
                t.strip() for t in raw_blocked.replace(",", " ").split() if t.strip()
            ]
        blocked_set = set(expand_hashtag_items(raw_blocked if isinstance(raw_blocked, list) else []))

        # ── Merge: base(manual) → always → platform → AI (hashtags merge only; no overwrite) ─
        # Manual upload hashtags must never be removed by saved blocked tags.
        base = expand_hashtag_items(self.hashtags or [])
        ai = expand_hashtag_items(self.ai_hashtags or [])
        if platform:
            m8h = (self.m8_platform_hashtags or {}).get(platform) or (
                (self.m8_platform_hashtags or {}).get((platform or "").lower())
            )
            if m8h:
                ai = expand_hashtag_items(m8h)

        seen: set = set()
        merged: List[str] = []

        # 1) Manual/base tags: keep as entered (except empty/dupe).
        for tag in base:
            t = str(tag).strip().lstrip("#").lower()
            if not t or t in seen:
                continue
            seen.add(t)
            merged.append(f"#{t}")

        # 2) Saved settings + AI tags: apply blocked list.
        for tag in always_tags + platform_tags + ai:
            t = str(tag).strip().lstrip("#").lower()
            if not t or t in seen or t in blocked_set:
                continue
            seen.add(t)
            merged.append(f"#{t}")

        return merged

    def is_success(self) -> bool:
        return any(r.success for r in self.platform_results)

    def is_partial_success(self) -> bool:
        return any(r.success for r in self.platform_results) and any(
            (not r.success) for r in self.platform_results
        )

    def get_success_platforms(self) -> List[str]:
        return [r.platform for r in self.platform_results if r.success]

    # ── Audio context helpers ─────────────────────────────────────────────────
    def get_audio_category(self) -> str:
        return (self.audio_context or {}).get("category", "other")

    def get_audio_emotion(self) -> str:
        return (self.audio_context or {}).get("emotional_tone", "hype_energetic")

    def get_thumbnail_mood(self) -> str:
        return (self.audio_context or {}).get("thumbnail_mood", "bold_dramatic")

    def get_transcript(self) -> str:
        return (self.audio_context or {}).get("transcript", "")

    def has_copyright_risk(self) -> bool:
        return bool((self.audio_context or {}).get("copyright_risk", False))

    def get_caption_style(self) -> str:
        return (self.audio_context or {}).get("caption_style", "hype_street")

    def get_audio_keywords(self) -> List[str]:
        return (self.audio_context or {}).get("suggested_keywords", [])

    def get_hume_dominant_emotion(self) -> str:
        return (self.audio_context or {}).get("hume_emotions", {}).get("dominant_emotion", "")

    def get_yamnet_profile(self) -> str:
        return (self.audio_context or {}).get("sound_profile", "unknown")

    def get_canonical_category(self) -> Optional[str]:
        """
        Unified category from supercharged signals. Used by caption and thumbnail
        to stay consistent. Priority: audio_context > video_understanding (if we
        infer from scene) > None (caller falls back to _detect_category).

        Maps audio categories (e.g. music_performance, food_cooking) to
        caption/thumbnail slugs (music, food).
        """
        ac = self.audio_context or {}
        raw = ac.get("category", "other")
        if not raw or raw == "other":
            return None
        # Map audio categories to caption/thumbnail category slugs
        _AUDIO_TO_CANONICAL = {
            "automotive": "automotive",
            "sports_extreme": "sports",
            "gaming": "gaming",
            "music_performance": "music",
            "food_cooking": "food",
            "travel_vlog": "travel",
            "fitness_workout": "fitness",
            "comedy_entertainment": "comedy",
            "educational": "education",
            "lifestyle_fashion": "lifestyle",
            "pets_animals": "pets",
            "nature_outdoors": "travel",
            "business_finance": "real_estate",
            "technology": "tech",
            "art_creative": "lifestyle",
            "family_kids": "lifestyle",
            "news_commentary": "general",
        }
        return _AUDIO_TO_CANONICAL.get(raw, raw)


    # ── Vision context helpers ────────────────────────────────────────────────
    def has_expressive_faces(self) -> bool:
        return bool((self.vision_context or {}).get("expressive", False))

    def get_ocr_text(self) -> str:
        return (self.vision_context or {}).get("ocr_text", "")

    def get_scene_labels(self) -> List[str]:
        return (self.vision_context or {}).get("label_names", [])

    # ── Twelve Labs helpers ───────────────────────────────────────────────────
    def get_scene_description(self) -> str:
        return (self.video_understanding or {}).get("scene_description", "")

    def get_tl_title_suggestion(self) -> str:
        return (self.video_understanding or {}).get("title_suggestion", "")

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
        Pass category from thumbnail_stage when available; otherwise uses
        canonical_category (audio) or general.
        """
        trill = self.trill or self.trill_score
        telemetry = self.telemetry or self.telemetry_data
        max_mph = 0.0
        if telemetry:
            max_mph = getattr(telemetry, "max_speed_mph", 0) or 0
        platforms = [str(p).lower() for p in (self.platforms or [])]
        platforms_csv = ",".join(platforms) if platforms else "youtube,instagram,facebook,tiktok"
        ac = self.audio_context or {}
        vc = self.vision_context or {}
        vu = self.video_understanding or {}
        # Category: caller arg > canonical (audio) > thumbnail_category > general
        cat = category or self.get_canonical_category() or getattr(self, "thumbnail_category", None) or "general"
        # Supercharged: emotional tone, suggested keywords for caption/thumbnail alignment
        emotional_tone = ac.get("emotional_tone", "")
        suggested_kw = ac.get("suggested_keywords", [])
        keywords_str = ", ".join(str(x) for x in suggested_kw) if suggested_kw else ""
        labels = vc.get("label_names") or []
        labels_str = ", ".join(str(x) for x in labels) if labels else ""
        ocr = (vc.get("ocr_text") or "").strip()
        fusion_summary = build_fusion_summary_text(self)
        scene = (vu.get("scene_description") or vu.get("description") or "")[:1200]
        geo_bits = []
        lat_lon = ""
        if telemetry:
            if getattr(telemetry, "location_display", None):
                geo_bits.append(f"route_area={telemetry.location_display}")
            if getattr(telemetry, "location_road", None):
                geo_bits.append(f"road={telemetry.location_road}")
            mla = getattr(telemetry, "mid_lat", None)
            mlo = getattr(telemetry, "mid_lon", None)
            if mla is not None and mlo is not None:
                lat_lon = f"{mla:.6f}, {mlo:.6f}"
                geo_bits.append(f"lat_lon_mid={lat_lon}")
        geo_context = "; ".join(geo_bits) if geo_bits else ""
        music_line = ""
        if ac.get("music_detected"):
            music_line = f"Recognized track: {ac.get('music_title', '')} — {ac.get('music_artist', '')}".strip()
            if ac.get("music_genre"):
                music_line += f" (genre: {ac['music_genre']})"
        yamnet_parts = ac.get("yamnet_events") or []
        yamnet_line = ", ".join(str(x) for x in yamnet_parts) if yamnet_parts else (ac.get("sound_profile") or "")
        landmark_hints = ", ".join(extract_landmark_hints(labels, ocr))
        fc = vc.get("face_count")
        people_vehicles = ""
        if fc is not None:
            people_vehicles = f"people_faces={fc}, expressive={vc.get('expressive')}"
        multimodal_digest = build_multimodal_scene_digest(self, max_chars=3500)
        vi = getattr(self, "video_intelligence_context", None) or {}
        vi_summary = (vi.get("summary_text") or "").strip()
        if not vi_summary and vi.get("top_labels"):
            vi_summary = "Video Intelligence: " + ", ".join(str(x) for x in (vi.get("top_labels") or [])[:12])
        pal = getattr(self, "frame_color_palette", None) or {}
        palette_hint = ""
        if isinstance(pal, dict) and pal:
            palette_hint = (
                f"primary={pal.get('primary_hex', '')} accent={pal.get('accent_hex', '')} "
                f"mood={pal.get('mood_hint', '')} all={','.join(pal.get('all_hex') or [])}"
            )
        ti = getattr(self, "trend_intel_context", None) or {}
        trend_intel = ""
        if isinstance(ti, dict) and ti.get("summary"):
            trend_intel = str(ti.get("summary") or "")[:1400]
        return {
            "effective_title": self.get_effective_title() or self.filename or "Video",
            "effective_caption": self.get_effective_caption() or "",
            "category": cat,
            "location_name": self.location_name or "",
            "trill_bucket": trill.bucket if trill else "",
            "max_speed_mph": str(max_mph),
            "platforms_csv": platforms_csv,
            "emotional_tone": emotional_tone,
            "suggested_keywords": keywords_str,
            "vision_labels": labels_str,
            "vision_ocr": ocr[:600] if ocr else "",
            "video_scene": scene,
            "geo_context": geo_context,
            "fusion_summary": fusion_summary,
            "music_line": music_line,
            "yamnet_line": yamnet_line or "",
            "lat_lon": lat_lon,
            "landmark_hints": landmark_hints,
            "people_vehicles": people_vehicles,
            "multimodal_digest": multimodal_digest,
            "video_intelligence": vi_summary,
            "frame_color_palette": palette_hint,
            "trend_intel": trend_intel,
        }

    def get_fusion_caption_rules(self) -> str:
        """
        Short rules for caption_stage so Whisper lyrics + ACR music do not get
        misread as the creator's own words or original song.
        """
        return build_fusion_caption_rules(self)


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
- Emotional tone (from audio): {emotional_tone}
- Suggested keywords (align with caption): {suggested_keywords}
- Google Vision / scene labels (if any): {vision_labels}
- On-screen text (OCR, if any): {vision_ocr}
- Video understanding (if any): {video_scene}
- Geo / route (telemetry): {geo_context}
- Lat/lon (mid route, if any): {lat_lon}
- Background / environment audio (YAMNet / sound profile): {yamnet_line}
- Landmark & architecture hints: {landmark_hints}
- People / faces summary: {people_vehicles}
- Fused narrative (sound + picture + place): {fusion_summary}
- Identified music (if any): {music_line}
- Full multimodal digest (use for accuracy — objects, noise, geo, OCR): {multimodal_digest}
- Google Video Intelligence — full-clip labels / shots (if any): {video_intelligence}
- Dominant colors from best frame (hex + warm/cool hint, if any): {frame_color_palette}
- Category trend / SERP title hints (if any — phrasing patterns only, do not invent facts): {trend_intel}

MANDATE
- Treat labels, OCR, geo coordinates, telemetry, music ID, AND background-sound profile as first-class evidence.
- If lyrics/soundtrack are identified but visuals show a drive, place, or activity, the thumbnail story is the SCENE + VIBE, not a fake music release.

HARD RULES
- No profanity, no hate, no nudity, no weapons emphasis, no illegal claims.
- No copyrighted logos/brand marks (YouTube logo, TikTok logo, etc).
- ACCURACY: Headlines and badges must reflect what is actually in the video. No misleading claims (e.g. "TOP 5" when it's not a list, "NEW" when it's not new). Describe visible content truthfully.
- If identified music is present but the video is driving/road/dashcam footage, treat the STORY as the drive/journey — not a "new song drop" or "I wrote this".
- BADGES: Never use generic hype stickers ("INSANE", "CRAZY", "EPIC", "MUST SEE") unless they directly match a visible peak moment. Prefer specific badges tied to telemetry or scene (e.g. "74 MPH", "DESERT RUN", "NIGHT DRIVE") when data supports them. Use an empty string for badge_text when no honest badge fits.
- Text: 2–4 words total, ALL CAPS, 1–2 lines, mobile readable.
- Provide 3 headline options; select 1.
- Include 1 badge only when it accurately fits (e.g. FAST only if speed/telemetry present; HOW TO only if it's a tutorial).
- Directional element is optional; use "none" when the frame is already strong and clean.
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
  "directional_element": "arrow_up"|"arrow_right"|"circle"|"glow_box"|"none",
  "props": ["string","string"],
  "emotion_cue": "shocked"|"excited"|"serious"|"laughing",
  "color_mood": "red_black"|"blue_black"|"gold_black"|"neon",
  "text_effect": "glitch"|"chrome"|"fire"|"neon"|"clean"|null,
  "platform_plan": {{
    "youtube": {{"enabled": true, "canvas": "16:9"}},
    "instagram": {{"enabled": true, "canvas": "9:16", "safe_center_pct": 60}},
    "facebook": {{"enabled": true, "canvas": "9:16", "safe_center_pct": 60}},
    "tiktok": {{"enabled": true, "canvas": "9:16", "thumb_offset_seconds": 1.5}}
  }},
  "notes": "1 sentence max"
}}
"""

# Backward-compat alias for older worker images or stale imports with a typo.
THUMNAIL_BRIEF_PROMPT = THUMBNAIL_BRIEF_PROMPT


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
        target_accounts=list(dict.fromkeys(str(t) for t in (upload_record.get("target_accounts") or []) if t)),
        title=upload_record.get("title", ""),
        caption=upload_record.get("caption", ""),
        hashtags=expand_hashtag_items(upload_record.get("hashtags") or []),
        privacy=upload_record.get("privacy", "public") or "public",
        reframe_mode=reframe_mode,
        user_settings=user_settings or {},
        entitlements=entitlements,
        put_cost=upload_record.get("put_reserved", 0),
        aic_cost=upload_record.get("aic_reserved", 0),
    )
