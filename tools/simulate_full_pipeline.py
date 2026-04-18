"""
Local end-to-end simulation of the worker processing pipeline (no Redis, no R2, no DB).

Mirrors worker.py order after download:
  transcode (or fast path) -> audio (Whisper + stack) -> vision -> Twelve Labs ->
  video intelligence (optional) -> thumbnail -> caption (title/caption/hashtags)

Caption/title/hashtags go through run_caption_stage: by default the M8 engine (stages/m8_engine.py)
builds a scene graph from audio, vision, Twelve Labs, telemetry/GPS, and video intelligence when
those stages ran; set UPLOADM8_M8_CAPTION_ENGINE=false to force the legacy single-prompt path only.

Usage (from repo root):
  python tools/simulate_full_pipeline.py path/to/video.mp4

  # Exercise caption style / tone / voice and hashtag rules (comma lists):
  python tools/simulate_full_pipeline.py clip.mp4 \\
    --caption-style punchy --caption-tone cinematic --caption-voice hypebeast \\
    --always-hashtags "tester,qwe" --blocked-hashtags "no,bad" \\
    --platform-tiktok "1,2" --platform-youtube "3,4" \\
    --ai-hashtag-count 5 --hashtag-style mixed

  # --- Local iteration: real .env, skip expensive vision APIs, still get audio + thumb + M8 ---
  python tools/simulate_full_pipeline.py your.mp4 \\
    --skip-vision --skip-12labs --skip-video-intelligence --no-billing

  # --- Telemetry .map + video (Trill must be on or the .map is not ingested) ---
  python tools/simulate_full_pipeline.py your.mp4 \\
    --telemetry-map path/to/session.map --trill-enabled \\
    --skip-vision --skip-12labs --skip-video-intelligence --no-billing

  # Optional: JSON snapshots per stage for comparing different test clips
  python tools/simulate_full_pipeline.py clip_a.mp4 --debug-dump-dir ./_dbg_clip_a

  # Windows helper (same defaults): tools/local_context_test.ps1

Requires:
  - .env with OPENAI_API_KEY (and optional keys for Vision, Twelve Labs, ACR, etc.)
  - FFmpeg/ffprobe on PATH
  - Google Vision / Video Intelligence: set GOOGLE_APPLICATION_CREDENTIALS to your service-account JSON,
    or place a single social-media-up-*.json in the repo root (see stages/vision_stage.py).

Options:
  --full-transcode   Run worker-style deduplicated transcode (slow; imports worker)
  --skip-vision      Skip Google Cloud Vision
  --skip-twelvelabs Skip Twelve Labs indexing
  --skip-video-intelligence  Skip Google Video Intelligence (when enabled in .env)
  --no-styled-thumb  Styled thumbnail overlays off (faster)
  See argparse below for Upload Preferences / billing flags.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
import tempfile
import uuid
import random
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [simulate] %(message)s",
)
log = logging.getLogger("simulate")


def _configure_console_encoding() -> None:
    """
    Avoid Windows cp1252 stdout/stderr crashes when AI text contains emoji/symbols.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _ctx_snapshot(ctx) -> Dict[str, Any]:
    """Small, stable debug snapshot for stage-by-stage diffs."""
    t = getattr(ctx, "telemetry_data", None) or getattr(ctx, "telemetry", None)
    trill = getattr(ctx, "trill_score", None) or getattr(ctx, "trill", None)
    return {
        "stage_trace": list(getattr(ctx, "stage_trace", []) or []),
        "video": {
            "local_video_path": str(getattr(ctx, "local_video_path", "") or ""),
            "processed_video_path": str(getattr(ctx, "processed_video_path", "") or ""),
            "thumbnail_path": str(getattr(ctx, "thumbnail_path", "") or ""),
            "thumbnail_paths_count": len(getattr(ctx, "thumbnail_paths", []) or []),
        },
        "ai": {
            "title": getattr(ctx, "ai_title", None),
            "caption_len": len(getattr(ctx, "ai_caption", "") or ""),
            "hashtags": list(getattr(ctx, "ai_hashtags", []) or []),
            "transcript_len": len(getattr(ctx, "ai_transcript", "") or ""),
        },
        "video_intelligence": {
            "has_context": bool(getattr(ctx, "video_intelligence_context", None)),
            "top_labels": list((getattr(ctx, "video_intelligence_context", None) or {}).get("top_labels", [])[:5])
            if isinstance(getattr(ctx, "video_intelligence_context", None), dict)
            else [],
        },
        "telemetry": {
            "has_telemetry": bool(t),
            "max_speed_mph": getattr(t, "max_speed_mph", None) if t else None,
            "avg_speed_mph": getattr(t, "avg_speed_mph", None) if t else None,
            "location_display": getattr(t, "location_display", None) if t else None,
            "has_trill_score": bool(trill),
            "trill_total": getattr(trill, "total", None) if trill else None,
            "trill_bucket": getattr(trill, "bucket", None) if trill else None,
        },
        "artifacts": dict(getattr(ctx, "output_artifacts", {}) or {}),
        "errors_count": len(getattr(ctx, "errors", []) or []),
    }


def _dump_ctx(ctx, dump_dir: Path, stage_name: str) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    p = dump_dir / f"{stage_name}.json"
    p.write_text(json.dumps(_ctx_snapshot(ctx), indent=2, default=str), encoding="utf-8")
    log.info("Debug dump: %s", p)


def _emit_decision_trace(args: argparse.Namespace, ent: Any, user_settings: Dict[str, Any]) -> None:
    """Print explicit decision-flow inputs before stage execution."""
    trace = {
        "tier": getattr(args, "tier", ""),
        "entitlements": {
            "can_ai": getattr(ent, "can_ai", None),
            "can_watermark": getattr(ent, "can_watermark", None),
            "can_burn_hud": getattr(ent, "can_burn_hud", None),
            "max_caption_frames": getattr(ent, "max_caption_frames", None),
            "ai_depth": getattr(ent, "ai_depth", None),
            "max_thumbnails": getattr(ent, "max_thumbnails", None),
            "can_custom_thumbnails": getattr(ent, "can_custom_thumbnails", None),
            "can_ai_thumbnail_styling": getattr(ent, "can_ai_thumbnail_styling", None),
            "priority_class": getattr(ent, "priority_class", None),
            "queue_depth": getattr(ent, "queue_depth", None),
        },
        "preferences": {
            "auto_captions": user_settings.get("auto_captions"),
            "auto_thumbnails": user_settings.get("auto_thumbnails"),
            "styled_thumbnails": user_settings.get("styled_thumbnails"),
            "ai_hashtags_enabled": user_settings.get("ai_hashtags_enabled"),
            "caption_style": user_settings.get("caption_style"),
            "caption_tone": user_settings.get("caption_tone"),
            "caption_voice": user_settings.get("caption_voice"),
            "caption_frame_count": user_settings.get("caption_frame_count"),
            "trill_enabled": user_settings.get("trill_enabled"),
            "trill_min_score": user_settings.get("trill_min_score"),
            "trill_ai_enhance": user_settings.get("trill_ai_enhance"),
            "hud_enabled": user_settings.get("hud_enabled"),
            "speeding_mph": user_settings.get("speeding_mph"),
            "euphoria_mph": user_settings.get("euphoria_mph"),
            "always_hashtags": user_settings.get("always_hashtags"),
            "blocked_hashtags": user_settings.get("blocked_hashtags"),
            "platform_hashtags": user_settings.get("platform_hashtags"),
            "ai_hashtag_style": user_settings.get("ai_hashtag_style"),
            "ai_hashtag_count": user_settings.get("ai_hashtag_count"),
            "max_hashtags": user_settings.get("max_hashtags"),
        },
        "runtime_flags": {
            "skip_vision": bool(getattr(args, "skip_vision", False)),
            "skip_12labs": bool(getattr(args, "skip_12labs", False)),
            "skip_video_intelligence": bool(getattr(args, "skip_video_intelligence", False)),
            "skip_caption": bool(getattr(args, "skip_caption", False)),
            "full_transcode": bool(getattr(args, "full_transcode", False)),
            "telemetry_map": bool(getattr(args, "telemetry_map", "")),
        },
    }
    print("\n" + "-" * 60)
    print("DECISION TRACE (inputs controlling stage decisions)")
    print("-" * 60)
    print(json.dumps(trace, indent=2, default=str))
    print("-" * 60)


def _parse_tag_csv(s: str) -> List[str]:
    """Split comma/space-separated tags; keep # prefix if user included it."""
    if not (s or "").strip():
        return []
    out: List[str] = []
    for part in s.replace(",", " ").split():
        t = part.strip()
        if t:
            out.append(t)
    return out


def _effective_caption_frames(requested: int, ent: Any) -> int:
    """Match stages/caption_stage.py clamp: min(user, tier max, 12)."""
    max_caption_frames = getattr(ent, "max_caption_frames", None) or 6
    return max(2, min(min(int(requested), int(max_caption_frames)), 12))


def build_user_settings(args: argparse.Namespace, ent: Any) -> Dict[str, Any]:
    """Map CLI -> ctx.user_settings (camelCase + snake_case where worker expects both)."""
    cf_user = min(int(args.caption_frames), int(getattr(ent, "max_caption_frames", 20) or 20))
    platform_ht: Dict[str, List[str]] = {}
    if getattr(args, "platform_hashtags_json", None):
        try:
            raw = json.loads(args.platform_hashtags_json)
            if isinstance(raw, dict):
                for k, v in raw.items():
                    key = str(k).lower()
                    if isinstance(v, list):
                        platform_ht[key] = [str(x).strip() for x in v if str(x).strip()]
                    elif isinstance(v, str):
                        platform_ht[key] = _parse_tag_csv(v)
        except json.JSONDecodeError as e:
            raise SystemExit(f"Invalid --platform-hashtags-json: {e}") from e
    else:
        for plat, attr in (
            ("tiktok", "platform_tiktok"),
            ("youtube", "platform_youtube"),
            ("instagram", "platform_instagram"),
            ("facebook", "platform_facebook"),
        ):
            val = getattr(args, attr, "") or ""
            if str(val).strip():
                platform_ht[plat] = _parse_tag_csv(str(val))

    always = _parse_tag_csv(getattr(args, "always_hashtags", "") or "")
    blocked = _parse_tag_csv(getattr(args, "blocked_hashtags", "") or "")

    auto_cap = not getattr(args, "no_auto_captions", False)
    auto_thumb = not getattr(args, "no_auto_thumbnails", False)
    styled = not getattr(args, "no_styled_thumb", False)
    ai_tags = not getattr(args, "no_ai_hashtags", False)
    use_audio = not getattr(args, "no_audio_context", False)

    us: Dict[str, Any] = {
        "autoCaptions": auto_cap,
        "auto_captions": auto_cap,
        "autoThumbnails": auto_thumb,
        "auto_thumbnails": auto_thumb,
        "styledThumbnails": styled,
        "styled_thumbnails": styled,
        "aiHashtagsEnabled": ai_tags,
        "ai_hashtags_enabled": ai_tags,
        "use_audio_context": use_audio,
        "useAudioContext": use_audio,
        "audio_transcription": use_audio,
        "audioTranscription": use_audio,
        "captionStyle": args.caption_style,
        "caption_style": args.caption_style,
        "captionTone": args.caption_tone,
        "caption_tone": args.caption_tone,
        "captionVoice": args.caption_voice,
        "caption_voice": args.caption_voice,
        "captionFrameCount": cf_user,
        "caption_frame_count": cf_user,
        "aiHashtagStyle": args.hashtag_style,
        "ai_hashtag_style": args.hashtag_style,
        "aiHashtagCount": int(args.ai_hashtag_count),
        "ai_hashtag_count": int(args.ai_hashtag_count),
        "maxHashtags": int(args.max_hashtags),
        "max_hashtags": int(args.max_hashtags),
        "hashtagPosition": args.hashtag_position,
        "hashtag_position": args.hashtag_position,
        "alwaysHashtags": always,
        "always_hashtags": list(always),
        "blockedHashtags": blocked,
        "blocked_hashtags": list(blocked),
        "platformHashtags": platform_ht,
        "platform_hashtags": platform_ht,
        "trillOpenaiModel": args.openai_model,
        "_openai_model_override": args.openai_model,
        "privacy": getattr(args, "privacy", "public"),
        "defaultPrivacy": getattr(args, "privacy", "public"),
        "thumbnailInterval": int(getattr(args, "thumbnail_interval", 10) or 10),
        "thumbnail_interval": int(getattr(args, "thumbnail_interval", 10) or 10),
        # Trill / telemetry preferences
        "trillEnabled": bool(getattr(args, "trill_enabled", False)),
        "trill_enabled": bool(getattr(args, "trill_enabled", False)),
        "trillMinScore": int(getattr(args, "trill_min_score", 60) or 60),
        "trill_min_score": int(getattr(args, "trill_min_score", 60) or 60),
        "trillAiEnhance": bool(getattr(args, "trill_ai_enhance", True)),
        "trill_ai_enhance": bool(getattr(args, "trill_ai_enhance", True)),
        "trillHudEnabled": bool(getattr(args, "trill_hud_enabled", False)),
        "trill_hud_enabled": bool(getattr(args, "trill_hud_enabled", False)),
        "hud_enabled": bool(getattr(args, "hud_enabled", False)),
        "speeding_mph": int(getattr(args, "speeding_mph", 50) or 50),
        "euphoria_mph": int(getattr(args, "euphoria_mph", 101) or 101),
    }
    return us


def _apply_random_caption_ai_overrides(args: argparse.Namespace, ent: Any, rng: random.Random) -> None:
    """
    Randomize Caption & Hashtag-related settings for weak-point hunting.

    This mutates `args` so downstream `build_user_settings()` uses randomized values.
    """
    # Caption Style / Tone
    args.caption_style = rng.choice(["story", "punchy", "factual"])
    args.caption_tone = rng.choice(["authentic", "hype", "cinematic", "calm"])

    # Caption Voice / Persona
    from stages.caption_stage import VOICE_PROFILES

    args.caption_voice = rng.choice(sorted(VOICE_PROFILES.keys()))

    # AI Caption Scan Depth
    # Must align with caption_stage behavior: clamp to 2..12 after ent.max_caption_frames and user setting.
    tier_max = int(getattr(ent, "max_caption_frames", 20) or 20)
    upper = max(2, min(tier_max, 12))
    args.caption_frames = rng.randint(2, upper)

    # Hashtag settings
    args.hashtag_style = rng.choice(["trending", "niche", "mixed"])
    args.hashtag_position = rng.choice(["start", "end", "caption", "comment"])

    # Prefer values that are within caption_stage clamp (1..50)
    args.ai_hashtag_count = rng.randint(3, 12)
    args.max_hashtags = max(args.ai_hashtag_count, rng.randint(8, 20))

    # Always / blocked tags and platform-specific tags come from settings page.
    # These are only base/filters; AI adds its own hashtags later in caption_stage.
    # Realistic niche-style tags for merge/block tests - avoid meta words that look like AI slop
    # (e.g. "caption", "cinematic") so runs reflect production hashtag quality.
    tag_pool = [
        "tester",
        "qwe",
        "no",
        "bad",
        "euphoria",
        "thrill",
        "drift",
        "creator",
        "studio",
        "nightdrive",
        "dashcam",
        "audio",
        "topup",
        "mentor",
        "teacher",
        "hype",
        "carsoftiktok",
        "bestfriend",
    ]

    # Always include 0..3 tags
    always_k = rng.randint(0, 3)
    always_tags = rng.sample(tag_pool, k=always_k) if always_k else []

    # Block 0..2 tags, sometimes overlapping with always to test filtering
    blocked_k = rng.randint(0, 2)
    blocked_tags = rng.sample(tag_pool, k=blocked_k) if blocked_k else []
    if always_tags and rng.random() < 0.35:
        blocked_tags.append(rng.choice(always_tags))
    # Dedup while preserving order
    seen = set()
    blocked_tags = [t for t in blocked_tags if not (t in seen or seen.add(t))]

    args.always_hashtags = ",".join(always_tags)
    args.blocked_hashtags = ",".join(blocked_tags)

    # Platform-specific hashtags
    # Override platform_hashtags_json so the platform-* CLI inputs win.
    args.platform_hashtags_json = ""

    platforms = [p.strip().lower() for p in (args.platforms or "").split(",") if p.strip()]
    platform_attrs = {
        "tiktok": "platform_tiktok",
        "youtube": "platform_youtube",
        "instagram": "platform_instagram",
        "facebook": "platform_facebook",
    }
    # Clear all supported platform attrs first
    for attr in platform_attrs.values():
        setattr(args, attr, "")

    # Assign 0..3 tags per platform
    for plat in platforms:
        if plat not in platform_attrs:
            continue
        k = rng.randint(0, 3)
        tags = rng.sample(tag_pool, k=k) if k else []
        setattr(args, platform_attrs[plat], ",".join(tags))


def _print_billing_block(
    ent: Any,
    num_platforms: int,
    args: argparse.Namespace,
    effective_pipeline_frames: int,
) -> None:
    from stages.entitlements import compute_aic_cost, compute_upload_cost

    use_ai = bool(getattr(args, "billing_use_ai", True))
    use_hud = bool(getattr(args, "billing_hud", False))
    put, aic_presign = compute_upload_cost(
        entitlements=ent,
        num_platforms=num_platforms,
        use_ai=use_ai,
        use_hud=use_hud,
        num_thumbnails=getattr(ent, "max_thumbnails", None),
    )
    # Presign uses tier max_caption_frames for AIC (see app.py). Show alternate line for pipeline frames.
    aic_if_pipeline_frames = 0
    if use_ai and getattr(ent, "can_ai", False):
        depth = getattr(ent, "ai_depth", None) or "basic"
        aic_if_pipeline_frames = compute_aic_cost(str(depth), int(effective_pipeline_frames))
    print("\n" + "-" * 60)
    print("BILLING ESTIMATE (per-service AIC + duration scale — stages/ai_service_costs.py)")
    print("-" * 60)
    print(f"  Platforms (count):     {num_platforms}")
    print(f"  Presign-style PUT/AIC: {put} PUT, {aic_presign} AIC")
    print(
        "    (AIC = sum of enabled pipeline services × duration multiplier + frame surcharge; "
        "see compute_upload_cost in entitlements.py)"
    )
    if use_ai and getattr(ent, "can_ai", False):
        print(
            f"  AIC if priced by actual pipeline frame count ({effective_pipeline_frames}): "
            f"{aic_if_pipeline_frames} AIC (compute_aic_cost(ai_depth, frames))"
        )
    print(
        f"  Flags for this estimate: billing_use_ai={use_ai}, billing_hud={use_hud}, "
        f"tier={getattr(args, 'tier', '')}"
    )
    print("-" * 60)


async def _fast_transcode_path(ctx) -> None:
    """Set video_info + platform_videos from source file (no FFmpeg re-encode)."""
    from stages.transcode_stage import get_video_info

    src = ctx.local_video_path
    if not src or not Path(src).exists():
        raise RuntimeError("No local_video_path")
    info = await get_video_info(Path(src))
    ctx.video_info = {
        "width": info.width,
        "height": info.height,
        "duration": info.duration,
        "fps": info.fps,
        "video_codec": info.video_codec,
        "audio_codec": info.audio_codec,
    }
    ctx.platform_videos = {}
    for p in ctx.platforms or []:
        ctx.platform_videos[p] = Path(src)
    ctx.processed_video_path = Path(src)


async def run_pipeline(args: argparse.Namespace) -> int:
    video = Path(args.video).resolve()
    if not video.exists():
        log.error("File not found: %s", video)
        return 1

    from stages.context import create_context
    from stages.entitlements import get_entitlements_for_tier
    from stages.errors import SkipStage, log_stage_skip
    from stages.audio_stage import run_audio_context_stage
    from stages.vision_stage import run_vision_stage
    from stages.twelvelabs_stage import run_twelvelabs_stage
    from stages.video_intelligence_stage import run_video_intelligence_stage
    from stages.thumbnail_stage import run_thumbnail_stage
    from stages.caption_stage import run_caption_stage
    from stages.hud_stage import run_hud_stage
    from stages.watermark_stage import run_watermark_stage
    from stages.telemetry_stage import run_telemetry_stage

    upload_id = str(uuid.uuid4())
    user_id = "simulate-user"
    job_id = str(uuid.uuid4())

    ent = get_entitlements_for_tier(args.tier)
    if getattr(args, "randomize_caption_ai", False):
        seed = getattr(args, "random_seed", None)
        rng = random.Random(seed)
        _apply_random_caption_ai_overrides(args, ent, rng)
    user_settings = build_user_settings(args, ent)
    if getattr(args, "trace_decisions", False):
        _emit_decision_trace(args, ent, user_settings)

    upload_record = {
        "id": upload_id,
        "user_id": user_id,
        "r2_key": f"simulate/{user_id}/{upload_id}/{video.name}",
        "filename": video.name,
        "file_size": video.stat().st_size,
        "platforms": [p.strip().lower() for p in args.platforms.split(",") if p.strip()],
        "title": args.title or "",
        "caption": args.caption or "",
        "hashtags": [],
        "privacy": getattr(args, "privacy", "public") or "public",
        "user_preferences": json.dumps(user_settings),
    }
    job_data = {"job_id": job_id, "idempotency_key": job_id}
    ctx = create_context(job_data, upload_record, user_settings, ent)
    ctx.user_settings["_openai_model_override"] = (
        ctx.user_settings.get("trillOpenaiModel")
        or ctx.user_settings.get("trill_openai_model")
        or "gpt-4o-mini"
    )

    out_thumb = Path(__file__).resolve().parent / "_simulate_full_thumb.jpg"
    dump_dir = Path(args.debug_dump_dir).resolve() if getattr(args, "debug_dump_dir", "") else None
    stop_after = str(getattr(args, "stop_after_stage", "") or "").strip().lower()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ctx.temp_dir = tmp_path
        dest = tmp_path / video.name
        shutil.copy2(video, dest)
        ctx.local_video_path = dest
        if getattr(args, "telemetry_map", None):
            map_src = Path(args.telemetry_map).resolve()
            if not map_src.exists():
                log.error("Telemetry .map file not found: %s", map_src)
                return 1
            map_dest = tmp_path / map_src.name
            shutil.copy2(map_src, map_dest)
            ctx.local_telemetry_path = map_dest
        log.info("Simulated download -> %s", dest)
        if dump_dir:
            _dump_ctx(ctx, dump_dir, "00_start")

        # -- Telemetry / HUD / watermark --
        ctx.telemetry_data = None
        ctx.telemetry = None
        ctx.trill_score = None
        us0 = ctx.user_settings or {}
        want_telemetry = bool(
            us0.get("trill_enabled") or us0.get("hud_enabled"),
        )
        if ctx.local_telemetry_path and want_telemetry:
            try:
                ctx = await run_telemetry_stage(ctx)
                if ctx.trill_score:
                    log.info(
                        "Telemetry ok trill_score=%s bucket=%s",
                        getattr(ctx.trill_score, "total", None),
                        getattr(ctx.trill_score, "bucket", None),
                    )
            except SkipStage as e:
                log_stage_skip(log, "Telemetry", e.reason)
            except Exception as e:
                log.warning("Telemetry error: %s", e)
        if dump_dir:
            _dump_ctx(ctx, dump_dir, "10_telemetry")
        if stop_after == "telemetry":
            log.info("Stopping after telemetry (--stop-after-stage=telemetry)")
            return 0
        try:
            await run_hud_stage(ctx)
        except SkipStage as e:
            log_stage_skip(log, "HUD", e.reason)
        except Exception as e:
            log.warning("HUD: %s", e)
        if dump_dir:
            _dump_ctx(ctx, dump_dir, "11_hud")
        if stop_after == "hud":
            log.info("Stopping after hud (--stop-after-stage=hud)")
            return 0
        try:
            await run_watermark_stage(ctx)
        except SkipStage as e:
            log_stage_skip(log, "Watermark", e.reason)
        except Exception as e:
            log.warning("Watermark: %s", e)
        if dump_dir:
            _dump_ctx(ctx, dump_dir, "12_watermark")
        if stop_after == "watermark":
            log.info("Stopping after watermark (--stop-after-stage=watermark)")
            return 0

        # -- Transcode --
        if args.full_transcode:
            try:
                from worker import _run_deduplicated_transcode

                ctx = await _run_deduplicated_transcode(ctx)
            except SkipStage as e:
                log_stage_skip(log, "Transcode", f"{e.reason} — using fast path")
                await _fast_transcode_path(ctx)
            except Exception as e:
                log.warning("Transcode error: %s - using fast path", e)
                await _fast_transcode_path(ctx)
        else:
            await _fast_transcode_path(ctx)
        if dump_dir:
            _dump_ctx(ctx, dump_dir, "20_transcode")
        if stop_after == "transcode":
            log.info("Stopping after transcode (--stop-after-stage=transcode)")
            return 0

        # -- Audio (Whisper + ACR + YAMNet + GPT classification) --
        try:
            ctx = await run_audio_context_stage(ctx)
            log.info(
                "Audio ok category=%s mood=%s",
                (ctx.audio_context or {}).get("category"),
                (ctx.audio_context or {}).get("thumbnail_mood"),
            )
        except SkipStage as e:
            log_stage_skip(log, "Audio", e.reason)
            ctx.audio_context = ctx.audio_context or {}
        except Exception as e:
            log.warning("Audio error: %s", e)
            ctx.audio_context = {}
        if dump_dir:
            _dump_ctx(ctx, dump_dir, "30_audio")
        if stop_after == "audio":
            log.info("Stopping after audio (--stop-after-stage=audio)")
            return 0

        # -- Vision --
        if not args.skip_vision:
            try:
                ctx = await run_vision_stage(ctx)
            except SkipStage as e:
                log_stage_skip(log, "Vision", e.reason)
                ctx.vision_context = {}
            except Exception as e:
                log.warning("Vision error: %s", e)
                ctx.vision_context = {}
        else:
            ctx.vision_context = {}
        if dump_dir:
            _dump_ctx(ctx, dump_dir, "40_vision")
        if stop_after == "vision":
            log.info("Stopping after vision (--stop-after-stage=vision)")
            return 0

        # -- Twelve Labs --
        if not args.skip_12labs:
            try:
                ctx = await run_twelvelabs_stage(ctx)
            except SkipStage as e:
                log_stage_skip(log, "Twelve Labs", e.reason)
                ctx.video_understanding = {}
            except Exception as e:
                log.warning("Twelve Labs error: %s", e)
                ctx.video_understanding = {}
        else:
            ctx.video_understanding = {}
        if dump_dir:
            _dump_ctx(ctx, dump_dir, "50_twelvelabs")
        if stop_after == "twelvelabs":
            log.info("Stopping after twelvelabs (--stop-after-stage=twelvelabs)")
            return 0

        # -- Google Video Intelligence (full-clip labels / shots; same order as worker) --
        if not getattr(args, "skip_video_intelligence", False):
            try:
                ctx = await run_video_intelligence_stage(ctx)
            except SkipStage as e:
                log_stage_skip(log, "Video Intelligence", e.reason)
                ctx.video_intelligence_context = getattr(ctx, "video_intelligence_context", None) or {}
            except Exception as e:
                log.warning("Video Intelligence error: %s", e)
                ctx.video_intelligence_context = getattr(ctx, "video_intelligence_context", None) or {}
        else:
            log.info("Video Intelligence skipped (--skip-video-intelligence)")
            ctx.video_intelligence_context = getattr(ctx, "video_intelligence_context", None) or {}
        if dump_dir:
            _dump_ctx(ctx, dump_dir, "55_video_intelligence")
        if stop_after == "video_intelligence":
            log.info("Stopping after video_intelligence (--stop-after-stage=video_intelligence)")
            return 0

        # -- Thumbnail --
        try:
            ctx = await run_thumbnail_stage(ctx)
        except SkipStage as e:
            log_stage_skip(log, "Thumbnail", e.reason)
        except Exception as e:
            log.warning("Thumbnail error: %s", e)

        if ctx.thumbnail_path and Path(ctx.thumbnail_path).exists():
            shutil.copy2(ctx.thumbnail_path, out_thumb)
            log.info("Wrote %s", out_thumb)
        if dump_dir:
            _dump_ctx(ctx, dump_dir, "60_thumbnail")
        if stop_after == "thumbnail":
            log.info("Stopping after thumbnail (--stop-after-stage=thumbnail)")
            return 0

        # -- Caption (title / caption / hashtags) --
        if not getattr(args, "skip_caption", False):
            try:
                ctx = await run_caption_stage(ctx, db_pool=None)
            except SkipStage as e:
                log_stage_skip(log, "Caption", e.reason)
            except Exception as e:
                log.warning("Caption error: %s", e)
        else:
            log.info("Caption stage skipped (--skip-caption)")
        if dump_dir:
            _dump_ctx(ctx, dump_dir, "70_caption")

    try:
        from stages.playwright_stage import close_browser

        await close_browser()
    except Exception:
        pass

    # -- Summary (outside temp dir) --
    eff_frames = _effective_caption_frames(int(args.caption_frames), ent)
    plats = [p.strip().lower() for p in args.platforms.split(",") if p.strip()]

    print("\n" + "=" * 60)
    print("SIMULATION RESULT")
    print("=" * 60)
    print(
        f"  Preferences: style={args.caption_style} tone={args.caption_tone} "
        f"voice={args.caption_voice} | caption frames requested={args.caption_frames} "
        f"-> pipeline effective={eff_frames} (capped by tier + max 12 in caption_stage)"
    )
    print(
        f"  Trill: enabled={bool(ctx.user_settings.get('trill_enabled'))} "
        f"| telemetry_map={'yes' if getattr(args, 'telemetry_map', None) else 'no'} "
        f"| hud_enabled={bool(ctx.user_settings.get('hud_enabled'))}"
    )
    if ctx.trill_score:
        print(
            f"  Trill score: {getattr(ctx.trill_score, 'total', '-')} "
            f"(bucket={getattr(ctx.trill_score, 'bucket', '-')})"
        )
    tr = (ctx.ai_transcript or "")[:500]
    if tr:
        print(f"Transcript (excerpt): {tr}{'...' if len(ctx.ai_transcript or '') > 500 else ''}")
    print(f"AI title:     {ctx.ai_title or '-'}")
    print(f"AI caption:   {(ctx.ai_caption or '-')[:800]}{'...' if ctx.ai_caption and len(ctx.ai_caption) > 800 else ''}")
    print(f"AI hashtags (raw from caption stage): {ctx.ai_hashtags or []}")
    print("Effective hashtags per platform (always + platform + base + AI, blocked removed):")
    for plat in plats:
        try:
            eff = ctx.get_effective_hashtags(plat)
            print(f"  {plat}: {eff}")
        except Exception as ex:
            print(f"  {plat}: (error: {ex})")
    print(f"Thumbnail:    {out_thumb if out_thumb.exists() else '-'}")
    try:
        _art = ctx.output_artifacts or {}
        _sig_raw = _art.get("thumbnail_style_signatures")
        _rej_raw = _art.get("thumbnail_qa_rejections")
        _sig = json.loads(_sig_raw) if isinstance(_sig_raw, str) and _sig_raw.strip() else (_sig_raw or {})
        _rej = json.loads(_rej_raw) if isinstance(_rej_raw, str) and _rej_raw.strip() else (_rej_raw or {})
        if isinstance(_sig, dict) and _sig:
            print("Thumbnail style QA (winner):")
            for plat, meta in _sig.items():
                if not isinstance(meta, dict):
                    continue
                qa = meta.get("qa") if isinstance(meta.get("qa"), dict) else {}
                print(
                    f"  {plat}: score={meta.get('score')} pack={meta.get('style_pack')} "
                    f"text_area={qa.get('text_area_ratio')} luma_std={qa.get('global_luma_std')} "
                    f"text_std={qa.get('text_region_std')} focal={qa.get('focal_strength')}"
                )
        if isinstance(_rej, dict) and _rej:
            print("Thumbnail QA rejects (sample):")
            for plat, rows in _rej.items():
                if not isinstance(rows, list) or not rows:
                    continue
                preview = rows[:3]
                for row in preview:
                    if not isinstance(row, dict):
                        continue
                    print(
                        f"  {plat}: nonce={row.get('nonce')} text_area={row.get('text_area_ratio')} "
                        f"luma_std={row.get('global_luma_std')} text_std={row.get('text_region_std')} "
                        f"focal={row.get('focal_strength')}"
                    )
    except Exception:
        pass
    if (ctx.output_artifacts or {}).get("watermarked_video"):
        print(f"Watermark:    applied ({ctx.output_artifacts.get('watermarked_video')})")
    else:
        print("Watermark:    not applied")
    print("=" * 60)
    if not getattr(args, "no_billing", False):
        _print_billing_block(ent, len(plats), args, eff_frames)
    return 0


def main() -> None:
    _configure_console_encoding()
    p = argparse.ArgumentParser(description="Simulate full upload processing pipeline locally")
    p.add_argument(
        "video",
        nargs="?",
        type=Path,
        default=None,
        help="Path to a video file (omit only with --list-voices)",
    )
    p.add_argument(
        "--list-voices",
        action="store_true",
        help="Print valid --caption-voice keys (VOICE_PROFILES) and exit",
    )
    p.add_argument("--tier", default="creator_pro", help="Entitlements tier slug (see get_entitlements_for_tier)")
    p.add_argument("--platforms", default="youtube,tiktok", help="Comma-separated platforms")
    p.add_argument("--title", default="", help="Optional upload title")
    p.add_argument("--caption", default="", help="Optional upload caption (skips AI caption if set)")
    p.add_argument("--privacy", default="public", help="Upload default privacy (stored on upload record)")
    p.add_argument("--caption-frames", type=int, default=6, dest="caption_frames")
    p.add_argument("--caption-style", default="story", choices=("story", "punchy", "factual"))
    p.add_argument(
        "--caption-tone",
        default="authentic",
        choices=("authentic", "hype", "cinematic", "calm"),
    )
    p.add_argument(
        "--caption-voice",
        default="default",
        choices=(
            "default",
            "mentor",
            "hypebeast",
            "best_friend",
            "teacher",
            "cinematic_narrator",
        ),
    )
    p.add_argument("--hashtag-style", default="mixed", choices=("trending", "niche", "mixed"))
    p.add_argument("--ai-hashtag-count", type=int, default=15, dest="ai_hashtag_count")
    p.add_argument("--max-hashtags", type=int, default=15, dest="max_hashtags")
    p.add_argument(
        "--hashtag-position",
        default="start",
        choices=("start", "end", "caption", "comment"),
        dest="hashtag_position",
    )
    p.add_argument(
        "--always-hashtags",
        default="",
        help="Comma-separated; always merged first (see get_effective_hashtags)",
    )
    p.add_argument("--blocked-hashtags", default="", help="Comma-separated; never used")
    p.add_argument("--platform-tiktok", default="", dest="platform_tiktok")
    p.add_argument("--platform-youtube", default="", dest="platform_youtube")
    p.add_argument("--platform-instagram", default="", dest="platform_instagram")
    p.add_argument("--platform-facebook", default="", dest="platform_facebook")
    p.add_argument(
        "--platform-hashtags-json",
        default="",
        help='JSON object, e.g. {"tiktok":["a"],"youtube":["b"]} (overrides --platform-*)',
    )
    p.add_argument("--thumbnail-interval", type=int, default=10, dest="thumbnail_interval")
    p.add_argument("--openai-model", default="gpt-4o-mini", dest="openai_model")
    p.add_argument("--telemetry-map", default="", dest="telemetry_map",
                   help="Path to .map telemetry file for Trill/HUD testing")
    p.add_argument("--no-auto-captions", action="store_true", dest="no_auto_captions")
    p.add_argument("--no-auto-thumbnails", action="store_true", dest="no_auto_thumbnails")
    p.add_argument("--no-styled-thumb", action="store_true", dest="no_styled_thumb")
    p.add_argument("--no-ai-hashtags", action="store_true", dest="no_ai_hashtags")
    p.add_argument("--no-audio-context", action="store_true", dest="no_audio_context")
    p.add_argument("--skip-caption", action="store_true", dest="skip_caption")
    p.add_argument("--billing-use-ai", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--billing-hud", action="store_true", help="Include HUD burn in PUT estimate")
    p.add_argument("--no-billing", action="store_true", dest="no_billing")
    p.add_argument("--full-transcode", action="store_true", help="Run worker deduplicated transcode (slow)")
    p.add_argument("--skip-vision", action="store_true")
    p.add_argument("--skip-12labs", action="store_true", dest="skip_12labs")
    p.add_argument(
        "--skip-video-intelligence",
        action="store_true",
        dest="skip_video_intelligence",
        help="Skip Google Video Intelligence stage (worker uses upload prefs + GCP credentials)",
    )
    p.add_argument("--trill-enabled", action=argparse.BooleanOptionalAction, default=False, dest="trill_enabled")
    p.add_argument("--trill-ai-enhance", action=argparse.BooleanOptionalAction, default=True, dest="trill_ai_enhance")
    p.add_argument("--trill-hud-enabled", action=argparse.BooleanOptionalAction, default=False, dest="trill_hud_enabled")
    p.add_argument("--trill-min-score", type=int, default=60, dest="trill_min_score")
    p.add_argument("--hud-enabled", action=argparse.BooleanOptionalAction, default=False, dest="hud_enabled")
    p.add_argument("--speeding-mph", type=int, default=50, dest="speeding_mph")
    p.add_argument("--euphoria-mph", type=int, default=101, dest="euphoria_mph")
    p.add_argument("--randomize-caption-ai", action="store_true", dest="randomize_caption_ai",
                   help="Randomize Caption & AI Settings (style/tone/voice + scan depth + hashtag rules)")
    p.add_argument("--random-seed", type=int, default=None, dest="random_seed",
                   help="Seed for randomization (deterministic). Default: non-deterministic.")
    p.add_argument("--random-runs", type=int, default=1, dest="random_runs",
                   help="How many randomized runs to execute sequentially (use with care; calls external APIs)")
    p.add_argument("--debug-dump-dir", default="", dest="debug_dump_dir",
                   help="Write stage-by-stage context snapshots as JSON into this folder")
    p.add_argument("--stop-after-stage", default="", dest="stop_after_stage",
                   choices=("", "telemetry", "hud", "watermark", "transcode", "audio", "vision", "twelvelabs", "video_intelligence", "thumbnail"),
                   help="Stop pipeline right after this stage for focused debugging")
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"), dest="log_level",
                   help="Global logging verbosity for simulate + stage loggers")
    p.add_argument("--trace-decisions", action="store_true", dest="trace_decisions",
                   help="Print decision-flow inputs (entitlements + prefs + runtime flags)")
    args = p.parse_args()
    logging.getLogger().setLevel(getattr(logging, str(args.log_level).upper(), logging.INFO))
    if args.list_voices:
        from stages.caption_stage import VOICE_PROFILES

        print("Valid --caption-voice values:", ", ".join(sorted(VOICE_PROFILES.keys())))
        raise SystemExit(0)
    if args.video is None:
        p.error("video: path to a video file is required")

    random_runs = max(1, int(getattr(args, "random_runs", 1) or 1))
    if not getattr(args, "randomize_caption_ai", False) or random_runs == 1:
        raise SystemExit(asyncio.run(run_pipeline(args)))

    # Run randomized sweeps sequentially.
    base_seed = args.random_seed
    if base_seed is None:
        base_seed = int(time.time() * 1000) % (2**31 - 1)
    exit_code = 0
    for i in range(random_runs):
        args.random_seed = int(base_seed) + i
        code = asyncio.run(run_pipeline(args))
        exit_code = code if code != 0 else exit_code
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
