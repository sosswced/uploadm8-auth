"""
Worker-pipeline thumbnail renderer — Pikzels public API v2.

Posts the AI-selected source frame + a creative brief to
``POST https://api.pikzels.com/v2/thumbnail/image`` and writes the rendered
image to ``output_path``. Auth is X-Api-Key (env: ``PIKZELS_API_KEY`` preferred,
``THUMB_RENDER_API_KEY`` fallback) — same key the Thumbnail Studio uses.

The legacy ``app.pikzels.com/platform/api/thumbnail`` integration was removed:
it returned non-renderable payloads (no image bytes / 404), which is why
``thumbnail_r2_key`` was never populated and the dashboard/queue showed no
thumbnails for AI-styled jobs.

Public function names are preserved so the existing call sites in
``stages/thumbnail_stage.py`` keep working (``render_thumbnail_with_studio_renderer``,
``studio_renderer_enabled``, plus the legacy aliases at the bottom).
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from core.thumbnail_text import (
    is_evidence_empty_fallback_headline,
    is_generic_thumbnail_headline,
)
from services.pikzels_v2 import (
    V2_THUMBNAIL_EDIT,
    V2_THUMBNAIL_IMAGE,
    V2_THUMBNAIL_TEXT,
    resolve_public_api_key,
)
from services.thumbnail_studio import closeness_to_pikzels_image_weight
from services.pikzels_v2_client import (
    coerce_pikzels_v2_image_base64_fields,
    pikzels_timeout_seconds,
    pikzels_v2_post,
)
from services.provider_error_trace import append_provider_error

# Strong guard when persona/style refs are sent — model sometimes copies reference typography.
_PERSONA_STYLE_TEXT_GUARD = (
    "Persona or style reference applies to likeness, lighting, and composition only — "
    "do NOT reproduce any text, typography, logos, watermarks, or captions that appear "
    "only on the reference image. "
)

_DASHCAM_POV_FIDELITY_GUARD = (
    "DASHCAM POV FIDELITY: edit the supplied in-car forward-facing frame only. "
    "Preserve the real road, vehicle hood or windshield, trees, sky, and any HUD timestamp "
    "overlay exactly. Do NOT add faces, people, celebrities, reaction shots, stock subjects, "
    "neon compass circles, or graphic overlays not in the source frame."
)

logger = logging.getLogger("uploadm8-worker.thumb_renderer")

MIN_THUMB_SIZE = 2048  # bytes — smaller responses are treated as render failures.
DEFAULT_PIKZELS_MODEL = (os.environ.get("PIKZELS_THUMBNAIL_MODEL") or "pkz_4").strip() or "pkz_4"
# Pikzels accepts long prompts; we cap to control cost/latency. Prioritize fused evidence first
# (see ``_build_pikzels_v2_prompt``) so truncation cuts generic composition hints, not facts.
# Pikzels public API rejects prompts over 1000 chars (VALIDATION_ERROR). Default below limit.
# Abbreviations (_PROMPT_LABEL_ABBREVS) pack more hydration evidence into that budget.
PIKZELS_API_PROMPT_HARD_MAX = 1000
_PIKZELS_IMAGE_PROMPT_MAX = max(
    400,
    min(PIKZELS_API_PROMPT_HARD_MAX, int(os.environ.get("PIKZELS_THUMBNAIL_PROMPT_MAX", "950") or 950)),
)

# Long phrase → short token (applied before length trim).
_PROMPT_LABEL_ABBREVS: List[tuple[str, str]] = [
    ("Canonical geo (hydration_payload): ", "Geo: "),
    ("Canonical dashcam/OSD (hydration_payload): ", "OSD: "),
    ("Canonical music (hydration_payload): ", "Music: "),
    ("Canonical speech/transcript (hydration_payload): ", "Speech: "),
    ("Canonical vision labels (hydration_payload): ", "Vis: "),
    ("Canonical vision OCR (hydration_payload): ", "OCR: "),
    ("Canonical Trill (hydration_payload): ", "Trill: "),
    ("UploadM8 fused evidence (canonical hydration_payload): ", "Fusion: "),
    ("UploadM8 hydration story (canonical): ", "Story: "),
    ("Canonical anchor phrase (hydration_payload): ", "Anchor: "),
    ("Canonical signal hashtags (hydration_payload): ", "Tags: "),
    ("UploadM8 fused evidence (ground truth): ", "Fusion: "),
    ("UploadM8 hydration story (scene paragraph): ", "Story: "),
    ("Pikzels creative brief: ", "Brief: "),
    ("User-selected default thumbnail strategy: ", "Layout: "),
    ("Geo/route context to reflect truthfully: ", "Geo: "),
    ("Dashcam HUD/OSD context to preserve: ", "OSD: "),
    ("Trill driving-energy context: ", "Trill: "),
    ("Music/audio vibe context, do not imply ownership: ", "Music: "),
    ("Speech/Whisper context for truthful hooks: ", "Speech: "),
    ("Relevant signal tags for strategy: ", "Tags: "),
    ("Layout and style notes: ", "Layout: "),
    ("Hydration focus: ", "Hydr: "),
    ("Ground in canonical hydration: ", "Hydr: "),
    ("peak speed ", "spd "),
    ("driver/OSD ", "drv "),
    ("recording start ", "rec "),
    ("road/highway ", "rd "),
    ("coordinates ", "coords "),
    ("bucket ", "bkt "),
    (" score ", " sc "),
]

_PROMPT_TAIL_DROP_MARKERS: List[str] = [
    "MrBeast-style YouTube thumbnail composition",
    "YT thumb style:",
    "Natural cinematic grade on existing scene only",
    "Use the supplied source frame as factual grounding",
    "Ground on supplied frame",
    "16:9 widescreen YouTube canvas",
    "9:16 vertical canvas",
]


def abbreviate_pikzels_prompt(prompt: str) -> str:
    """Shorten verbose hydration labels so more evidence fits under the API cap."""
    s = re.sub(r"\s+", " ", str(prompt or "").strip())
    for old, new in _PROMPT_LABEL_ABBREVS:
        s = s.replace(old, new)
    # Compact telemetry speed tokens only — do not rewrite headline text like "64 MPH".
    s = re.sub(r"\bspd\s+(\d+(?:\.\d+)?)\s*mph\b", r"spd \1mph", s, flags=re.I)
    s = re.sub(r"\bpeak speed[:\s]+(\d+(?:\.\d+)?)\s*mph\b", r"spd \1mph", s, flags=re.I)
    s = re.sub(r"\btranscript\b", "txcript", s, flags=re.I)
    s = re.sub(r"\brecording\b", "rec", s, flags=re.I)
    s = re.sub(r"\bhighway\b", "hwy", s, flags=re.I)
    return re.sub(r"\s{2,}", " ", s).strip()


def fit_pikzels_prompt_to_budget(prompt: str, cap: Optional[int] = None) -> str:
    """Abbreviate, then drop low-priority tail hints, then word-safe trim."""
    limit = int(cap) if cap is not None else min(_PIKZELS_IMAGE_PROMPT_MAX, PIKZELS_API_PROMPT_HARD_MAX)
    limit = max(400, min(PIKZELS_API_PROMPT_HARD_MAX, limit))
    s = abbreviate_pikzels_prompt(prompt)
    if len(s) <= limit:
        return s
    for marker in _PROMPT_TAIL_DROP_MARKERS:
        if len(s) <= limit:
            break
        idx = s.rfind(marker)
        if idx > int(limit * 0.45):
            s = s[:idx].rstrip(" .;,")
    if len(s) > limit:
        s = s[:limit].rsplit(" ", 1)[0].rstrip(" .;,")
    return s


def clamp_pikzels_image_prompt(prompt: str) -> str:
    """Final guard before POST — abbreviate and never exceed Pikzels API hard limit."""
    return fit_pikzels_prompt_to_budget(str(prompt or "").strip())

# Internal-only brief notes that must never block hydration or consume prompt budget.
_THUMB_NOTES_PLACEHOLDERS = frozenset({"No AI — evidence-based brief", "Fallback brief"})

# Per-platform aspect ratio for ``/v2/thumbnail/image``.
_PLATFORM_FORMAT: Dict[str, str] = {
    "youtube": "16:9",
    "instagram": "9:16",
    "facebook": "9:16",
    "tiktok": "9:16",
}


def studio_renderer_enabled() -> bool:
    """True iff a Pikzels v2 API key is configured."""
    return bool(resolve_public_api_key())


# ── Prompt assembly ─────────────────────────────────────────────────────────


_COLOR_MOOD_HINTS: Dict[str, str] = {
    "red_black": "high-contrast red and black palette, neon red accents",
    "yellow_black": "bold yellow and black palette, comic-book contrast",
    "blue_white": "clean cinematic blue and white palette",
    "warm_sunset": "warm sunset oranges and magenta highlights",
    "cool_studio": "cool studio lighting, neutral background",
    "neon_cyber": "neon cyberpunk magenta and cyan glow",
}

_EMOTION_HINTS: Dict[str, str] = {
    "excited": "wide eyes, mouth open in surprise, high-energy expression",
    "shocked": "eyebrows raised, jaw dropped, dramatic shock",
    "angry": "intense furrowed brow, fierce expression",
    "curious": "tilted head, intrigued look",
    "happy": "genuine smile, bright eyes",
    "serious": "focused gaze, confident expression",
    "playful": "smirk, winking or playful gesture",
}


def _truthy(v: Any) -> bool:
    return bool(v) and str(v).strip().lower() not in ("none", "null", "false", "0")


def _build_pikzels_v2_prompt(
    brief: Dict[str, Any],
    *,
    category: str,
    platform: str,
    trace_sink: Optional[Any] = None,
    hydration_payload: Optional[Dict[str, Any]] = None,
    platform_color: Optional[str] = None,
    accent_color: Optional[str] = None,
) -> str:
    """Compose a single-string render prompt from the GPT-generated brief."""
    if not isinstance(brief, dict):
        brief = {}
    dashcam_pov = bool(brief.get("_uploadm8_dashcam_pov")) or (category or "").strip().lower() == "dashcam"
    parts: List[str] = []

    headline = str(brief.get("selected_headline") or "").strip()
    if not headline:
        opts = brief.get("headline_options") or []
        if isinstance(opts, list):
            for o in opts:
                if isinstance(o, str) and o.strip():
                    headline = o.strip()
                    break
                if isinstance(o, dict):
                    t = str(o.get("text") or o.get("headline") or "").strip()
                    if t:
                        headline = t
                        break

    # Pikzels' image model has a strong tendency to stamp its own clickbait
    # text on the canvas ("UNBELIEVABLE MOMENTS", "EVENT MOMENTS", "MUST WATCH",
    # "WATCH THIS") when our prompt is anything less than a HARD instruction.
    # The previous "Do not add generic text such as …" wording was ignored in
    # production. We now place the no-text directive FIRST, repeat it twice,
    # and only allow text rendering when the headline is both non-generic AND
    # contains at least one specific signal (a digit, an uppercase proper-noun
    # word, or a token over 5 chars). Even then, we cap to ~30 chars and
    # forbid any other on-image text.
    def _headline_is_concrete(h: str) -> bool:
        if not h or is_generic_thumbnail_headline(h) or is_evidence_empty_fallback_headline(h):
            return False
        if any(ch.isdigit() for ch in h):
            return True
        # at least one capitalised proper-noun-ish token of 4+ chars (not all-caps stop word)
        for tok in h.split():
            t = tok.strip(" ,.;:!?-_")
            if len(t) >= 4 and t[:1].isupper() and not t.isupper():
                return True
            if len(t) >= 6:
                return True
        return False

    if _headline_is_concrete(headline):
        parts.append(
            f'Render exactly one short headline reading "{headline[:30]}" in large bold '
            "display typography in the lower third of the image. Render NO OTHER text, "
            "letters, words, numbers, captions, banners, watermarks, signatures, "
            "labels, or written content anywhere else on the image."
        )
    else:
        parts.append(
            "STRICT NO-TEXT MODE: render absolutely NO text, letters, words, "
            "numbers, captions, headlines, banners, watermarks, signatures, "
            "labels, or any written content anywhere on the image. Do not add "
            "phrases like \"UNBELIEVABLE MOMENTS\", \"EVENT MOMENTS\", \"MUST "
            "WATCH\", \"WATCH THIS\", \"EPIC MOMENT\", or any similar clickbait "
            "wording. Pure photographic composition only — no typography of any kind."
        )

    if dashcam_pov:
        parts.append(_DASHCAM_POV_FIDELITY_GUARD)

    prioritized: List[str] = []

    hp = hydration_payload if isinstance(hydration_payload, dict) else None
    if hp:
        ev = hp.get("evidence") if isinstance(hp.get("evidence"), dict) else {}
        geo = ev.get("geo") if isinstance(ev.get("geo"), dict) else {}
        road = str(geo.get("road") or "").strip()
        city = str(geo.get("city") or "").strip()
        st = str(geo.get("state") or "").strip()
        geo_parts: List[str] = []
        disp = str(geo.get("display") or "").strip()
        if disp:
            geo_parts.append(disp[:120])
        if road:
            geo_parts.append(f"rd {road}")
        if city and st:
            geo_parts.append(f"{city}, {st}")
        elif city:
            geo_parts.append(city)
        lat, lon = geo.get("lat"), geo.get("lon")
        if lat is not None and lon is not None:
            try:
                geo_parts.append(f"gps {float(lat):.4f},{float(lon):.4f}")
            except (TypeError, ValueError):
                pass
        if geo_parts:
            prioritized.append("Geo: " + "; ".join(geo_parts)[:200])

        osd = ev.get("osd") if isinstance(ev.get("osd"), dict) else {}
        osd_parts: List[str] = []
        msm = osd.get("max_speed_mph")
        if msm is not None:
            try:
                osd_parts.append(f"spd {float(msm):.0f}mph")
            except (TypeError, ValueError):
                osd_parts.append(f"spd {msm}mph")
        dn = str(osd.get("driver_name") or "").strip()
        if dn:
            osd_parts.append(f"drv {dn[:40]}")
        fss = str(osd.get("first_seen") or "").strip()
        if fss:
            osd_parts.append(f"rec {fss[:32]}")
        if osd_parts:
            prioritized.append("OSD: " + "; ".join(osd_parts)[:160])

        mus = ev.get("music") if isinstance(ev.get("music"), dict) else {}
        ma, mt = str(mus.get("artist") or "").strip(), str(mus.get("title") or "").strip()
        if ma or mt:
            prioritized.append("Music: " + " — ".join(p for p in (ma, mt) if p)[:140])

        spch = ev.get("speech") if isinstance(ev.get("speech"), dict) else {}
        phrase = str(spch.get("phrase") or "").strip()
        if phrase:
            prioritized.append(f"Speech: {phrase[:200]}")

        vis = ev.get("vision") if isinstance(ev.get("vision"), dict) else {}
        vlabels = vis.get("labels") if isinstance(vis.get("labels"), list) else []
        if vlabels:
            prioritized.append(
                "Vis: " + ", ".join(str(x) for x in vlabels[:8])[:160]
            )
        voc = str(vis.get("ocr") or "").strip()[:100]
        if voc:
            prioritized.append(f"OCR: {voc}")

        tri = ev.get("trill") if isinstance(ev.get("trill"), dict) else {}
        tbuck = str(tri.get("bucket") or "").strip()
        tsco = tri.get("score")
        if tbuck or tsco is not None:
            if tsco is not None:
                try:
                    prioritized.append(
                        f"Trill: bkt {tbuck} sc {float(tsco):.0f}"[:80] if tbuck else f"Trill: sc {float(tsco):.0f}"[:40]
                    )
                except (TypeError, ValueError):
                    if tbuck:
                        prioritized.append(f"Trill: {tbuck}"[:60])
            elif tbuck:
                prioritized.append(f"Trill: {tbuck}"[:60])

        cfs = str(hp.get("fusion_summary") or "").strip()
        if cfs:
            prioritized.append("Fusion: " + cfs[:280])

        hstory = str(hp.get("hydration_story") or "").strip()
        if hstory:
            prioritized.append("Story: " + hstory[:220])

        anch = str(hp.get("anchor_phrase") or "").strip()
        if anch:
            prioritized.append(f"Anchor: {anch[:120]}")

        sigs = hp.get("signal_hashtags")
        if isinstance(sigs, list) and sigs:
            prioritized.append(
                "Tags: " + ", ".join(str(x) for x in sigs[:8])[:120]
            )

    fusion = str(brief.get("fusion_summary") or "").strip()
    if fusion and not any(p.startswith("Fusion:") for p in prioritized):
        prioritized.append(f"Fusion: {fusion[:280]}")

    hydration_story_slice = str(brief.get("hydration_story") or "").strip()
    if hydration_story_slice and len(fusion) < 120 and not any(p.startswith("Story:") for p in prioritized):
        prioritized.append(f"Story: {hydration_story_slice[:220]}")

    text_brief = str(brief.get("pikzels_text_brief") or brief.get("engine_text_brief") or "").strip()
    if text_brief:
        prioritized.append(f"Brief: {text_brief[:240]}")

    default_strategy = brief.get("default_strategy")
    if isinstance(default_strategy, dict) and default_strategy:
        ds_bits: List[str] = []
        if default_strategy.get("layout_name") or default_strategy.get("layout_pattern"):
            ds_bits.append(
                f"{str(default_strategy.get('layout_name') or '').strip()} {str(default_strategy.get('layout_pattern') or '').strip()}".strip()
            )
        if default_strategy.get("audience_niche"):
            ds_bits.append(f"audience {str(default_strategy.get('audience_niche')).replace('_', ' ')}")
        if default_strategy.get("competitor_gap_mode"):
            ds_bits.append("differentiated competitor-gap variant")
        if ds_bits:
            prioritized.append(
                "User-selected default thumbnail strategy: "
                + "; ".join(ds_bits)[:360]
                + ". Keep this layout family consistent while adapting content to this upload's evidence."
            )

    notes = str(brief.get("notes") or "").strip()
    if notes and notes not in _THUMB_NOTES_PLACEHOLDERS and len(notes) <= 220:
        prioritized.append(notes)

    geo_context = str(brief.get("geo_context") or "").strip()
    if geo_context and not any(p.startswith("Geo:") for p in prioritized):
        prioritized.append(f"Geo: {geo_context[:200]}")

    osd_context = str(brief.get("osd_context") or "").strip()
    if osd_context and not any(p.startswith("OSD:") for p in prioritized):
        prioritized.append(f"OSD: {osd_context[:160]}")

    trill_context = str(brief.get("trill_context") or "").strip()
    if trill_context and not any(p.startswith("Trill:") for p in prioritized):
        prioritized.append(f"Trill: {trill_context[:120]}")

    music_context = str(brief.get("music_context") or "").strip()
    if music_context and not any(p.startswith("Music:") for p in prioritized):
        prioritized.append(f"Music: {music_context[:140]}")

    speech_context = str(brief.get("speech_context") or "").strip()
    if speech_context and not any(p.startswith("Speech:") for p in prioritized):
        prioritized.append(f"Speech: {speech_context[:200]}")

    signal_hashtags = str(brief.get("signal_hashtags") or "").strip()
    if signal_hashtags and not any(p.startswith("Tags:") for p in prioritized):
        prioritized.append(f"Tags: {signal_hashtags[:120]}")

    styling: List[str] = []
    if not dashcam_pov:
        badge_text = str(brief.get("badge_text") or "").strip()
        badge_style = str(brief.get("badge_style") or "").strip().lower()
        if badge_text:
            if badge_style:
                styling.append(f'a {badge_style} circular badge with the word "{badge_text[:14]}"')
            else:
                styling.append(f'a circular badge with the word "{badge_text[:14]}"')

        direction = str(brief.get("directional_element") or "").strip().lower()
        if direction and direction not in ("none", "null"):
            styling.append(f"a bold {direction} directional element pointing at the subject")

        props = brief.get("props") or []
        if isinstance(props, list):
            clean_props = [str(p).strip() for p in props if isinstance(p, (str, int, float)) and str(p).strip()]
            if clean_props:
                styling.append(f"props: {', '.join(clean_props[:5])}")

        emotion = str(brief.get("emotion_cue") or "").strip().lower()
        if emotion:
            styling.append(_EMOTION_HINTS.get(emotion, f"{emotion} facial expression"))

    color_mood = str(brief.get("color_mood") or "").strip().lower()
    if color_mood:
        styling.append(_COLOR_MOOD_HINTS.get(color_mood, color_mood.replace("_", " ") + " color palette"))

    plat_color = str(platform_color or "").strip()
    if plat_color:
        styling.append(
            f"platform badge and corner indicator use solid color {plat_color}"
        )
    accent = str(accent_color or "").strip()
    if accent:
        styling.append(
            f"accent highlight strokes, arrows, circles, and focal glow use {accent}"
        )

    tail: List[str] = []
    if category and category != "general" and not dashcam_pov:
        tail.append(f"{category} content category visual cues")

    canvas_hint = "16:9 YT" if (platform or "").lower() == "youtube" else "9:16 vert, safe crop"
    tail.append(canvas_hint)

    if dashcam_pov:
        tail.append("Natural cinematic grade on existing scene only; no new subjects")
    else:
        tail.append("Ground on supplied frame; match visible content")
        tail.append("YT thumb style: hi contrast, sharp subj, dramatic light")

    ordered = parts + prioritized + styling + tail
    prompt = ". ".join(p for p in ordered if p)
    # Public API: prompt cannot contain URLs (YouTube links in notes/fusion would 4xx).
    prompt = re.sub(r"https?://[^\s]+", "", prompt, flags=re.IGNORECASE)
    prompt = re.sub(r"\s{2,}", " ", prompt).strip()
    prompt = fit_pikzels_prompt_to_budget(prompt)
    if callable(trace_sink):
        try:
            trace_sink(
                "pikzels_prompt_built",
                {
                    "platform": (platform or "").strip().lower(),
                    "category": (category or "").strip().lower(),
                    "headline_preview": str(headline or "")[:80],
                    "headline_concrete": bool(_headline_is_concrete(headline)),
                    "fusion_chars": len(fusion),
                    "hydration_story_chars": len(hydration_story_slice),
                    "prompt_len": len(prompt),
                    "prompt_head": prompt[:140],
                    "prompt_tail": prompt[-140:] if len(prompt) > 280 else prompt,
                },
            )
        except Exception:
            pass
    return prompt or "MrBeast-style high-contrast YouTube thumbnail with bold text and dramatic subject"


# ── Response decoding ───────────────────────────────────────────────────────


def _decode_response_b64(data: Dict[str, Any]) -> Optional[bytes]:
    """Inline base64 image (covers OpenAPI ``image_base64`` and the ``data:`` form)."""
    if not isinstance(data, dict):
        return None
    candidates: List[Any] = [
        data.get("image_base64"),
        data.get("b64_json"),
        data.get("image_b64"),
    ]
    nested = data.get("data")
    if isinstance(nested, dict):
        candidates.extend([nested.get("image_base64"), nested.get("b64_json"), nested.get("image_b64")])
    if isinstance(nested, list) and nested and isinstance(nested[0], dict):
        first = nested[0]
        candidates.extend([first.get("image_base64"), first.get("b64_json"), first.get("image_b64")])
    for raw in candidates:
        if not isinstance(raw, str):
            continue
        s = raw.strip()
        if not s:
            continue
        if s.lower().startswith("data:image") and ";base64," in s.lower():
            s = s.split(",", 1)[1]
        try:
            return base64.b64decode(s)
        except (binascii.Error, TypeError, ValueError):
            continue
    return None


def _walk_find_cdn_url(obj: Any, depth: int = 0) -> str:
    """Pikzels v2 sometimes nests the CDN URL inside arbitrary JSON shapes."""
    if depth > 8 or obj is None:
        return ""
    if isinstance(obj, str) and obj.startswith("http") and "pikzels.com" in obj:
        return obj.strip()
    if isinstance(obj, dict):
        for k in ("output", "image_url", "url", "thumbnail_url", "output_url", "cdn_url", "pikzels_cdn_url"):
            v = obj.get(k)
            if isinstance(v, str) and v.startswith("http"):
                return v.strip()
        for v in obj.values():
            u = _walk_find_cdn_url(v, depth + 1)
            if u:
                return u
    if isinstance(obj, list):
        for it in obj[:40]:
            u = _walk_find_cdn_url(it, depth + 1)
            if u:
                return u
    return ""


async def _download_with_pikzels_auth(url: str, timeout: float) -> Optional[bytes]:
    headers: Dict[str, str] = {}
    if "pikzels.com" in (url or "").lower():
        key = resolve_public_api_key()
        if key:
            headers["X-Api-Key"] = key
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            r = await client.get(url, headers=headers or None)
            if r.status_code == 200 and r.content:
                return r.content
            logger.debug("[thumb-renderer] download HTTP %s len=%s", r.status_code, len(r.content or b""))
    except (httpx.HTTPError, OSError) as e:
        logger.debug("[thumb-renderer] download failed: %s", e)
    return None


async def _pikzels_v2_response_to_bytes(data: Dict[str, Any], timeout: float) -> Optional[bytes]:
    raw = _decode_response_b64(data)
    if raw:
        return raw
    url = _walk_find_cdn_url(data)
    if url:
        return await _download_with_pikzels_auth(url, timeout=timeout)
    return None


# ── Optional creative brief (text endpoint) ─────────────────────────────────


async def generate_pikzels_text_brief(*, source_title: str, niche: str, context_summary: str = "") -> str:
    """One ``/v2/thumbnail/text`` call; merged into the prompt if it succeeds."""
    if not resolve_public_api_key():
        return ""
    ctx = f" Context: {context_summary[:500]}." if str(context_summary or "").strip() else ""
    prompt = (
        f"YouTube thumbnail brief. Title: {source_title or 'untitled'}. Niche: {niche or 'general'}. "
        f"{ctx} Reply with two short sentences: (1) emotional hook angle, (2) visual layout emphasis."
    )[:1000]
    status, data = await pikzels_v2_post(
        V2_THUMBNAIL_TEXT,
        {"prompt": prompt, "model": DEFAULT_PIKZELS_MODEL, "format": "16:9"},
    )
    if status >= 400 or not isinstance(data, dict):
        return ""
    for k in ("text", "output", "prompt", "description", "thumbnail_text"):
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()[:600]
    nested = data.get("data")
    if isinstance(nested, list) and nested and isinstance(nested[0], dict):
        for k in ("text", "output", "prompt"):
            v = nested[0].get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()[:600]
    return ""


_pikzels_v2_text_brief = generate_pikzels_text_brief


# ── Main entrypoint (call-site signature preserved) ─────────────────────────


async def render_thumbnail_with_studio_renderer(
    base_frame_path: Path,
    brief: Dict[str, Any],
    platform: str,
    output_path: Path,
    *,
    upload_id: str = "",
    category: str = "",
    persona: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
    job_context: Any = None,
) -> bool:
    """
    Render a styled platform thumbnail via Pikzels v2 ``/v2/thumbnail/image``.

    When ``brief`` contains ``_uploadm8_pikzels_support_image_url`` (HTTPS still, e.g.
    YouTube ``i.ytimg.com``), it is sent as ``support_image_url`` for pkz_4 / pkz_4_5
    (public API forbids URLs inside ``prompt``). Prompt assembly strips any remaining
    URL tokens to avoid validation errors.

    Returns True only when a non-empty output image is written to ``output_path``.
    All failures are non-fatal and logged at WARNING.
    """
    if not studio_renderer_enabled():
        logger.debug("[thumb-renderer] PIKZELS_API_KEY not set — skipping Pikzels v2 render")
        return False

    try:
        frame_bytes = base_frame_path.read_bytes()
    except (OSError, PermissionError) as e:
        logger.warning("[thumb-renderer] failed reading source frame: %s", e)
        return False
    if not frame_bytes:
        logger.warning("[thumb-renderer] source frame is empty: %s", base_frame_path)
        return False
    frame_b64 = base64.b64encode(frame_bytes).decode("ascii")

    plat = (platform or "").strip().lower()
    fmt = _PLATFORM_FORMAT.get(plat, "16:9")

    trace_sink = None
    if job_context is not None:
        from services.thumbnail_trace import trace_sink_factory

        trace_sink = trace_sink_factory(job_context)

    hp_raw = getattr(job_context, "hydration_payload", None) if job_context is not None else None
    hydration_payload = hp_raw if isinstance(hp_raw, dict) else None

    from services.platform_colors import platform_color_for, resolve_platform_colors

    us = getattr(job_context, "user_settings", None) if job_context is not None else None
    color_map = resolve_platform_colors(us if isinstance(us, dict) else None)

    prompt = _build_pikzels_v2_prompt(
        brief or {},
        category=category or "",
        platform=plat,
        trace_sink=trace_sink,
        hydration_payload=hydration_payload,
        platform_color=platform_color_for(color_map, plat),
        accent_color=color_map.get("accent"),
    )

    iw = "medium"
    explicit_weight = False
    if isinstance(options, dict):
        ow = str(options.get("image_weight") or "").strip().lower()
        if ow in ("low", "medium", "high"):
            iw = ow
            explicit_weight = True
        else:
            rs = options.get("reference_strength")
            if isinstance(rs, (int, float)):
                try:
                    iw = closeness_to_pikzels_image_weight(int(rs))
                    explicit_weight = True
                except (TypeError, ValueError):
                    pass
    has_persona = bool(
        isinstance(persona, dict)
        and str(persona.get("id") or persona.get("pikzonality_id") or "").strip()
    )
    if isinstance(brief, dict) and brief.get("_uploadm8_dashcam_pov"):
        iw = "high"
    elif not explicit_weight and not has_persona:
        iw = "high"

    payload: Dict[str, Any] = {
        "prompt": prompt,
        "image_base64": f"data:image/jpeg;base64,{frame_b64}",
        "image_weight": iw,
        "model": DEFAULT_PIKZELS_MODEL,
        "format": fmt,
    }

    _model_lc = str(payload.get("model") or DEFAULT_PIKZELS_MODEL).strip().lower()
    _sup_ref = ""
    if isinstance(brief, dict):
        _sup_ref = str(brief.get("_uploadm8_pikzels_support_image_url") or "").strip()
    if _sup_ref.startswith("https://") and _model_lc in ("pkz_4", "pkz_4_5"):
        payload["support_image_url"] = _sup_ref[:2000]
        logger.info(
            "[thumb-renderer] Pikzels payload includes support_image_url (YouTube still) upload=%s platform=%s",
            upload_id,
            plat,
        )
    elif _sup_ref.startswith("https://") and _model_lc:
        logger.debug(
            "[thumb-renderer] support_image_url skipped (model %s; pkz_4+ only) upload=%s",
            _model_lc,
            upload_id,
        )

    persona_uuid_set = False
    style_uuid_set = False
    if isinstance(persona, dict) and persona:
        pid = str(persona.get("id") or persona.get("pikzonality_id") or "").strip()
        if pid:
            try:
                uuid.UUID(pid)
                kind = str(persona.get("kind") or persona.get("type") or "persona").strip().lower()
                if kind == "style":
                    payload["style"] = pid[:200]
                    style_uuid_set = True
                else:
                    payload["persona"] = pid[:200]
                    persona_uuid_set = True
                logger.info(
                    "[thumb-renderer] Pikzels payload includes %s=%s upload=%s platform=%s",
                    kind, pid[:8] + "…", upload_id, plat,
                )
            except (ValueError, TypeError):
                logger.warning(
                    "[thumb-renderer] persona/style id is NOT a valid UUID — Pikzels render will run "
                    "WITHOUT persona for upload=%s platform=%s id=%s",
                    upload_id, plat, pid[:32],
                )
        else:
            logger.warning(
                "[thumb-renderer] persona dict supplied but had no id/pikzonality_id — "
                "Pikzels render will run WITHOUT persona for upload=%s platform=%s",
                upload_id, plat,
            )
    elif persona is not None:
        logger.warning(
            "[thumb-renderer] persona arg provided but not a dict — Pikzels render will run "
            "WITHOUT persona for upload=%s platform=%s persona_type=%s",
            upload_id, plat, type(persona).__name__,
        )

    # Public OpenAPI for /v2/thumbnail/image does not include persona_strength and sets
    # additionalProperties: false — sending it caused 4xx rejects whenever persona mode was on.
    if isinstance(options, dict) and options:
        style_hint = str(options.get("style_hint") or "").strip()
        if style_hint:
            payload["prompt"] = f"{payload['prompt']}. Visual style: {style_hint[:180]}".strip()[:_PIKZELS_IMAGE_PROMPT_MAX]

    if persona_uuid_set and isinstance(options, dict) and options:
        ps = options.get("persona_strength")
        if isinstance(ps, (int, float)):
            try:
                psv = max(0, min(100, int(ps)))
            except (TypeError, ValueError):
                psv = 70
            if psv >= 67:
                hint = "Strong match to the creator persona reference face and style."
            elif psv <= 33:
                hint = "Light persona influence; keep composition bold but subtle on the face."
            else:
                hint = "Balanced use of the creator persona reference."
            merged = f"{payload['prompt']}. {hint}".strip()
            payload["prompt"] = merged[:_PIKZELS_IMAGE_PROMPT_MAX]
    elif style_uuid_set:
        payload["prompt"] = (
            f"{payload['prompt']}. Follow the selected Pikzels style reference while preserving the source frame subject.".strip()[
                :_PIKZELS_IMAGE_PROMPT_MAX
            ]
        )

    if persona_uuid_set or style_uuid_set:
        payload["prompt"] = clamp_pikzels_image_prompt(
            _PERSONA_STYLE_TEXT_GUARD + str(payload.get("prompt") or "")
        )

    pre_post_len = len(str(payload.get("prompt") or ""))
    payload["prompt"] = clamp_pikzels_image_prompt(str(payload.get("prompt") or ""))
    if pre_post_len > len(payload["prompt"]):
        logger.warning(
            "[thumb-renderer] Pikzels prompt capped upload=%s platform=%s from=%d to=%d",
            upload_id,
            plat,
            pre_post_len,
            len(payload["prompt"]),
        )

    if job_context is not None:
        from services.thumbnail_trace import persist_pikzels_prompt_for_platform, trace_append

        persist_pikzels_prompt_for_platform(job_context, plat, str(payload.get("prompt") or ""))
        trace_append(
            job_context,
            "pikzels_final_prompt",
            {
                "platform": plat,
                "prompt_len": len(str(payload.get("prompt") or "")),
                "persona_uuid_set": bool(persona_uuid_set),
                "style_uuid_set": bool(style_uuid_set),
            },
        )

    coerce_pikzels_v2_image_base64_fields(payload)

    timeout = pikzels_timeout_seconds()
    try:
        status, data = await pikzels_v2_post(V2_THUMBNAIL_IMAGE, payload)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning("[thumb-renderer] Pikzels v2 POST raised: %s", e)
        if job_context is not None:
            append_provider_error(
                job_context,
                provider="pikzels",
                stage="thumbnail_stage",
                operation="thumbnail_image_post",
                message=str(e),
                exception_type=type(e).__name__,
            )
        return False

    if status >= 400 or not isinstance(data, dict):
        logger.warning(
            "[thumb-renderer] Pikzels v2 image HTTP %s upload=%s platform=%s body=%s",
            status, upload_id, plat, str(data)[:240],
        )
        if job_context is not None:
            body_blob = ""
            try:
                body_blob = json.dumps(data, default=str)[:2000] if isinstance(data, dict) else str(data)[:2000]
            except Exception:
                body_blob = str(data)[:2000]
            low = body_blob.lower()
            prompt_long = status == 400 and (
                "1000" in low
                or "prompt must" in low
                or "validation_error" in low
                or "too long" in low
            )
            append_provider_error(
                job_context,
                provider="pikzels",
                stage="thumbnail_stage",
                operation="thumbnail_image_post",
                message=(
                    "Pikzels rejected prompt length (API max 1000 chars); shorten fused evidence "
                    "or lower PIKZELS_THUMBNAIL_PROMPT_MAX"
                    if prompt_long
                    else "non-2xx or non-json response"
                ),
                http_status=status,
                provider_code="prompt_too_long" if prompt_long else None,
                response_body_snippet=body_blob[:1200],
            )
        return False

    image_bytes = await _pikzels_v2_response_to_bytes(data, timeout=timeout)
    if not image_bytes:
        logger.warning(
            "[thumb-renderer] Pikzels v2 returned no image data upload=%s platform=%s keys=%s",
            upload_id, plat, list(data.keys())[:8] if isinstance(data, dict) else None,
        )
        if job_context is not None:
            append_provider_error(
                job_context,
                provider="pikzels",
                stage="thumbnail_stage",
                operation="thumbnail_image_decode",
                message="response contained no image bytes",
                response_body_snippet=str(list(data.keys())[:20] if isinstance(data, dict) else data)[:1200],
            )
        return False

    try:
        output_path.write_bytes(image_bytes)
    except OSError as e:
        logger.warning("[thumb-renderer] failed writing output %s: %s", output_path, e)
        return False

    if output_path.exists() and output_path.stat().st_size >= MIN_THUMB_SIZE:
        logger.info(
            "[thumb-renderer] Pikzels v2 render ok upload=%s platform=%s bytes=%d",
            upload_id, plat, output_path.stat().st_size,
        )
        return True
    logger.warning(
        "[thumb-renderer] output file too small (%s bytes) upload=%s platform=%s",
        output_path.stat().st_size if output_path.exists() else 0, upload_id, plat,
    )
    return False


async def refine_thumbnail_with_pikzels_edit(
    image_path: Path,
    edit_prompt: str,
    *,
    platform: str,
    upload_id: str = "",
) -> bool:
    """
    Light pass over an existing rendered thumbnail via ``POST /v2/thumbnail/edit``.

    On success, overwrites ``image_path`` in place. Failures are non-fatal.
    """
    if not studio_renderer_enabled():
        return False
    ep = str(edit_prompt or "").strip()
    if not ep:
        return False
    try:
        raw = image_path.read_bytes()
    except OSError as e:
        logger.warning("[thumb-renderer] hydration edit: cannot read %s: %s", image_path, e)
        return False
    if not raw or len(raw) < MIN_THUMB_SIZE:
        return False
    b64 = base64.b64encode(raw).decode("ascii")
    plat = (platform or "").strip().lower()
    fmt = _PLATFORM_FORMAT.get(plat, "16:9")
    payload: Dict[str, Any] = {
        "prompt": clamp_pikzels_image_prompt(ep),
        "image_base64": f"data:image/jpeg;base64,{b64}",
        "format": fmt,
        "model": DEFAULT_PIKZELS_MODEL,
    }
    coerce_pikzels_v2_image_base64_fields(payload)
    timeout = pikzels_timeout_seconds()
    try:
        status, data = await pikzels_v2_post(V2_THUMBNAIL_EDIT, payload)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning("[thumb-renderer] hydration edit POST raised: %s", e)
        return False
    if status >= 400 or not isinstance(data, dict):
        logger.warning(
            "[thumb-renderer] hydration edit HTTP %s upload=%s platform=%s body=%s",
            status, upload_id, plat, str(data)[:240],
        )
        return False
    image_bytes = await _pikzels_v2_response_to_bytes(data, timeout=timeout)
    if not image_bytes or len(image_bytes) < MIN_THUMB_SIZE:
        return False
    try:
        image_path.write_bytes(image_bytes)
    except OSError as e:
        logger.warning("[thumb-renderer] hydration edit write failed %s: %s", image_path, e)
        return False
    logger.info(
        "[thumb-renderer] Pikzels hydration edit ok upload=%s platform=%s bytes=%d",
        upload_id, plat, len(image_bytes),
    )
    return True


# Legacy aliases — older imports across the worker still reference these.
pikzels_enabled = studio_renderer_enabled
render_thumbnail_with_pikzels = render_thumbnail_with_studio_renderer
