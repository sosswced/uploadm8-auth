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
import logging
import os
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

logger = logging.getLogger("uploadm8-worker.thumb_renderer")

MIN_THUMB_SIZE = 2048  # bytes — smaller responses are treated as render failures.
DEFAULT_PIKZELS_MODEL = (os.environ.get("PIKZELS_THUMBNAIL_MODEL") or "pkz_4").strip() or "pkz_4"
# Pikzels accepts long prompts; we cap to control cost/latency. Prioritize fused evidence first
# (see ``_build_pikzels_v2_prompt``) so truncation cuts generic composition hints, not facts.
_PIKZELS_IMAGE_PROMPT_MAX = max(400, int(os.environ.get("PIKZELS_THUMBNAIL_PROMPT_MAX", "1200") or 1200))

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
) -> str:
    """Compose a single-string render prompt from the GPT-generated brief."""
    if not isinstance(brief, dict):
        brief = {}
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

    prioritized: List[str] = []

    hp = hydration_payload if isinstance(hydration_payload, dict) else None
    if hp:
        ev = hp.get("evidence") if isinstance(hp.get("evidence"), dict) else {}
        geo = ev.get("geo") if isinstance(ev.get("geo"), dict) else {}
        road = str(geo.get("road") or "").strip()
        city = str(geo.get("city") or "").strip()
        st = str(geo.get("state") or "").strip()
        geo_parts: List[str] = []
        if road:
            geo_parts.append(f"road/highway {road}")
        if city and st:
            geo_parts.append(f"{city}, {st}")
        elif city:
            geo_parts.append(city)
        lat, lon = geo.get("lat"), geo.get("lon")
        if lat is not None and lon is not None:
            try:
                geo_parts.append(f"coords {float(lat):.4f},{float(lon):.4f}")
            except (TypeError, ValueError):
                pass
        if geo_parts:
            prioritized.append(
                "Canonical geo (hydration_payload): " + "; ".join(geo_parts)[:280]
            )

        osd = ev.get("osd") if isinstance(ev.get("osd"), dict) else {}
        osd_parts: List[str] = []
        msm = osd.get("max_speed_mph")
        if msm is not None:
            try:
                osd_parts.append(f"peak speed {float(msm):.1f} mph")
            except (TypeError, ValueError):
                osd_parts.append(f"peak speed {msm}")
        dn = str(osd.get("driver_name") or "").strip()
        if dn:
            osd_parts.append(f"driver/OSD {dn}")
        fss = str(osd.get("first_seen") or "").strip()
        if fss:
            osd_parts.append(f"recording start {fss}")
        if osd_parts:
            prioritized.append(
                "Canonical dashcam/OSD (hydration_payload): " + "; ".join(osd_parts)[:280]
            )

        mus = ev.get("music") if isinstance(ev.get("music"), dict) else {}
        ma, mt = str(mus.get("artist") or "").strip(), str(mus.get("title") or "").strip()
        if ma or mt:
            prioritized.append(
                "Canonical music (hydration_payload): "
                + " — ".join(p for p in (ma, mt) if p)[:220]
            )

        spch = ev.get("speech") if isinstance(ev.get("speech"), dict) else {}
        phrase = str(spch.get("phrase") or "").strip()
        if phrase:
            prioritized.append(
                f"Canonical speech/transcript (hydration_payload): {phrase[:260]}"
            )

        vis = ev.get("vision") if isinstance(ev.get("vision"), dict) else {}
        vlabels = vis.get("labels") if isinstance(vis.get("labels"), list) else []
        if vlabels:
            prioritized.append(
                "Canonical vision labels (hydration_payload): "
                + ", ".join(str(x) for x in vlabels[:10])[:240]
            )
        voc = str(vis.get("ocr") or "").strip()[:140]
        if voc:
            prioritized.append(
                f"Canonical vision OCR (hydration_payload): {voc}"
            )

        tri = ev.get("trill") if isinstance(ev.get("trill"), dict) else {}
        tbuck = str(tri.get("bucket") or "").strip()
        tsco = tri.get("score")
        if tbuck or tsco is not None:
            if tsco is not None:
                try:
                    prioritized.append(
                        f"Canonical Trill (hydration_payload): bucket {tbuck} score {float(tsco):.0f}"[:240]
                    )
                except (TypeError, ValueError):
                    if tbuck:
                        prioritized.append(f"Canonical Trill (hydration_payload): {tbuck}"[:220])
            elif tbuck:
                prioritized.append(f"Canonical Trill (hydration_payload): {tbuck}"[:220])

        cfs = str(hp.get("fusion_summary") or "").strip()
        if cfs:
            prioritized.append(
                "UploadM8 fused evidence (canonical hydration_payload): " + cfs[:400]
            )

        hstory = str(hp.get("hydration_story") or "").strip()
        if hstory:
            prioritized.append(
                "UploadM8 hydration story (canonical): " + hstory[:320]
            )

        anch = str(hp.get("anchor_phrase") or "").strip()
        if anch:
            prioritized.append(f"Canonical anchor phrase (hydration_payload): {anch[:220]}")

        sigs = hp.get("signal_hashtags")
        if isinstance(sigs, list) and sigs:
            prioritized.append(
                "Canonical signal hashtags (hydration_payload): "
                + ", ".join(str(x) for x in sigs[:10])[:180]
            )

    fusion = str(brief.get("fusion_summary") or "").strip()
    if fusion:
        prioritized.append(f"UploadM8 fused evidence (ground truth): {fusion[:400]}")

    hydration_story_slice = str(brief.get("hydration_story") or "").strip()
    if hydration_story_slice and len(fusion) < 120:
        prioritized.append(
            "UploadM8 hydration story (scene paragraph): " + hydration_story_slice[:320]
        )

    text_brief = str(brief.get("pikzels_text_brief") or brief.get("engine_text_brief") or "").strip()
    if text_brief:
        prioritized.append(f"Pikzels creative brief: {text_brief[:380]}")

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
    if geo_context:
        prioritized.append(f"Geo/route context to reflect truthfully: {geo_context[:260]}")

    osd_context = str(brief.get("osd_context") or "").strip()
    if osd_context:
        prioritized.append(f"Dashcam HUD/OSD context to preserve: {osd_context[:220]}")

    trill_context = str(brief.get("trill_context") or "").strip()
    if trill_context:
        prioritized.append(f"Trill driving-energy context: {trill_context[:220]}")

    music_context = str(brief.get("music_context") or "").strip()
    if music_context:
        prioritized.append(f"Music/audio vibe context, do not imply ownership: {music_context[:220]}")

    speech_context = str(brief.get("speech_context") or "").strip()
    if speech_context:
        prioritized.append(f"Speech/Whisper context for truthful hooks: {speech_context[:260]}")

    signal_hashtags = str(brief.get("signal_hashtags") or "").strip()
    if signal_hashtags:
        prioritized.append(f"Relevant signal tags for strategy: {signal_hashtags[:180]}")

    styling: List[str] = []
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

    tail: List[str] = []
    if category and category != "general":
        tail.append(f"{category} content category visual cues")

    canvas_hint = "16:9 widescreen YouTube canvas" if (platform or "").lower() == "youtube" else "9:16 vertical canvas, subject centered for safe-area cropping"
    tail.append(canvas_hint)

    tail.append("Use the supplied source frame as factual grounding; keep text and visuals specific to visible content")
    tail.append("MrBeast-style YouTube thumbnail composition: high contrast, sharp subject, dramatic lighting, eye magnet")

    ordered = parts + prioritized + styling + tail
    prompt = ". ".join(p for p in ordered if p)
    cap = _PIKZELS_IMAGE_PROMPT_MAX
    if len(prompt) > cap:
        prompt = prompt[:cap]
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

    prompt = _build_pikzels_v2_prompt(
        brief or {},
        category=category or "",
        platform=plat,
        trace_sink=trace_sink,
        hydration_payload=hydration_payload,
    )

    iw = "medium"
    if isinstance(options, dict):
        ow = str(options.get("image_weight") or "").strip().lower()
        if ow in ("low", "medium", "high"):
            iw = ow
        else:
            rs = options.get("reference_strength")
            if isinstance(rs, (int, float)):
                try:
                    iw = closeness_to_pikzels_image_weight(int(rs))
                except (TypeError, ValueError):
                    pass

    payload: Dict[str, Any] = {
        "prompt": prompt,
        "image_base64": f"data:image/jpeg;base64,{frame_b64}",
        "image_weight": iw,
        "model": DEFAULT_PIKZELS_MODEL,
        "format": fmt,
    }

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
        payload["prompt"] = (
            _PERSONA_STYLE_TEXT_GUARD + str(payload.get("prompt") or "")
        ).strip()[:_PIKZELS_IMAGE_PROMPT_MAX]

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
            append_provider_error(
                job_context,
                provider="pikzels",
                stage="thumbnail_stage",
                operation="thumbnail_image_post",
                message="non-2xx or non-json response",
                http_status=status,
                response_body_snippet=str(data)[:1200],
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
        "prompt": ep[:980],
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
