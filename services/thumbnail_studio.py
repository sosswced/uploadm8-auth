from __future__ import annotations

import asyncio
import base64
import binascii
import time
import csv
import hashlib
import io
import uuid
import json
import logging
import os
import re
import zipfile
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import httpx

from core.config import R2_BUCKET_NAME
from core.r2 import generate_presigned_download_url, get_s3_client

from services.pikzels_v2 import (
    V2_PIKZONALITY_BY_ID,
    V2_PIKZONALITY_PERSONA,
    V2_THUMBNAIL_IMAGE,
    V2_THUMBNAIL_SCORE,
    V2_THUMBNAIL_TEXT,
    resolve_public_api_key,
)
from services.pikzels_v2_client import (
    normalize_url_or_base64,
    pikzels_v2_get,
    pikzels_v2_post,
    trim_pikzonality_images,
)

_log = logging.getLogger("uploadm8.thumbnail_studio")


def sanitize_pikzels_persona_name(name: str) -> str:
    """Pikzels POST /v2/pikzonality/persona: name max 25 chars; letters, numbers, spaces, hyphens, underscores."""
    raw = (name or "").strip() or "Persona"
    safe = re.sub(r"[^\w\-\s]", "", raw, flags=re.UNICODE)
    safe = re.sub(r"\s+", " ", safe).strip()[:25] or "Persona"
    return safe


def split_face_refs_for_pikzels(refs: List[str]) -> Tuple[List[str], List[str]]:
    """Split UI refs into HTTP(S) URLs vs data:image payloads (Pikzels accepts exactly 3 of one kind)."""
    urls: List[str] = []
    b64s: List[str] = []
    for r in refs or []:
        s = str(r).strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith("https://") or low.startswith("http://"):
            urls.append(s[:8000])
        elif low.startswith("data:image"):
            b64s.append(s[:15_000_000])
    return urls, b64s


def _pikzels_api_error_message(data: Any) -> str:
    if not isinstance(data, dict):
        return "upstream_error"
    err = data.get("error")
    if isinstance(err, dict):
        return str(err.get("message") or err.get("code") or "error")[:800]
    detail = data.get("detail")
    if isinstance(detail, list):
        parts: List[str] = []
        for it in detail[:10]:
            if isinstance(it, dict):
                parts.append(str(it.get("message") or it.get("code") or it)[:220])
            else:
                parts.append(str(it)[:220])
        if parts:
            return "; ".join(parts)[:800]
    for key in ("issues", "errors"):
        arr = data.get(key)
        if isinstance(arr, list) and len(arr) > 0:
            return _pikzels_api_error_message({"detail": arr})
    if detail is not None and not isinstance(detail, (dict, list)):
        return str(detail)[:800]
    return str(data.get("message") or data.get("detail") or "upstream_error")[:800]


async def wait_pikzonality_persona_ready(
    pikzonality_id: str,
    *,
    max_wait_s: Optional[float] = None,
    interval_s: float = 2.0,
) -> Optional[str]:
    """
    Poll GET /v2/pikzonality/{id} until status completed / failed / timeout.
    Returns None on success, or an error string.
    """
    pid = (pikzonality_id or "").strip()
    if not pid:
        return "missing_pikzonality_id"
    cap = max_wait_s
    if cap is None:
        try:
            cap = float(os.environ.get("PIKZELS_PIKZONALITY_POLL_TIMEOUT_SECONDS", "300") or 300)
        except (TypeError, ValueError):
            cap = 300.0
    cap = max(30.0, min(cap, 900.0))
    deadline = time.monotonic() + cap
    path_tpl = V2_PIKZONALITY_BY_ID
    while time.monotonic() < deadline:
        code, data = await pikzels_v2_get(path_tpl.format(id=pid))
        if code >= 400:
            return _pikzels_api_error_message(data)
        if not isinstance(data, dict):
            await asyncio.sleep(interval_s)
            continue
        st = str(data.get("status") or "").strip().lower()
        if st == "completed":
            return None
        if st == "failed":
            return _pikzels_api_error_message(data) or "pikzonality_failed"
        await asyncio.sleep(interval_s)
    return "pikzels_persona_processing_timeout"


async def register_creator_persona_with_pikzels(
    *,
    name: str,
    image_refs: List[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Create a persona pikzonality at Pikzels (3 face refs) and wait until ready.
    Returns (pikzels_pikzonality_id, error_message). ID is what /v2/thumbnail/* expects in ``persona``.
    """
    if not resolve_public_api_key():
        return None, None
    urls, b64s = split_face_refs_for_pikzels(image_refs)
    body: Dict[str, Any] = {"name": sanitize_pikzels_persona_name(name)}
    if len(urls) >= 3:
        body["image_urls"] = urls[:3]
    elif len(b64s) >= 3:
        # Browser FileReader sends ``data:image/...;base64,...`` — Pikzels requires that shape.
        body["image_base64s"] = b64s[:3]
    else:
        return None, "pikzels_persona_requires_three_https_urls_or_three_data_image_urls"
    # XOR + min length: never send both keys (even empty arrays) — Pikzels rejects that.
    trim_pikzonality_images(body)
    code, data = await pikzels_v2_post(V2_PIKZONALITY_PERSONA, body)
    if code >= 400 or not isinstance(data, dict):
        return None, _pikzels_api_error_message(data)
    new_id = str(data.get("id") or data.get("request_id") or "").strip()
    if not new_id:
        return None, "pikzels_persona_create_missing_id"
    wait_err = await wait_pikzonality_persona_ready(new_id)
    if wait_err:
        return None, wait_err
    return new_id, None


_PROTECTED_MARKS = {
    "nfl",
    "nba",
    "disney",
    "marvel",
    "pixar",
    "coca cola",
    "coca-cola",
    "nike",
    "adidas",
    "apple",
    "netflix",
}


def extract_youtube_video_id(url: str) -> str:
    text = (url or "").strip()
    if not text:
        return ""
    try:
        parsed = urlparse(text)
    except ValueError as e:
        _log.debug("extract_youtube_video_id urlparse: %s", e)
        return ""

    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").strip("/")
    if "youtu.be" in host and path:
        return path.split("/")[0]

    if "youtube.com" in host:
        if path == "watch":
            return (parse_qs(parsed.query).get("v") or [""])[0]
        if path.startswith("shorts/"):
            return path.split("/", 1)[1].split("/")[0]
        if path.startswith("embed/"):
            return path.split("/", 1)[1].split("/")[0]
    return ""


async def fetch_youtube_title(url: str) -> str:
    if not (url or "").strip():
        return ""
    endpoint = "https://www.youtube.com/oembed"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(endpoint, params={"url": url, "format": "json"})
            if r.status_code != 200:
                return ""
            data = r.json() if r.content else {}
            return str(data.get("title") or "").strip()[:220]
    except Exception as e:
        _log.debug("fetch_youtube_title failed url=%s: %s", url[:80], e)
        return ""


def estimate_studio_cost(
    *,
    variant_count: int,
    has_persona: bool,
    competitor_gap_mode: bool,
    has_channel_memory: bool,
) -> Tuple[int, int, Dict[str, Any]]:
    n = max(4, min(8, int(variant_count or 4)))
    put = 4 + n
    aic = 10 + (n * 2)
    if has_persona:
        aic += 5
    if competitor_gap_mode:
        aic += 4
    if has_channel_memory:
        aic += 2
    breakdown = {
        "variant_count": n,
        "components": {
            "base_put": 4,
            "variant_put": n,
            "base_aic": 10,
            "variant_aic": n * 2,
            "persona_aic": 5 if has_persona else 0,
            "competitor_gap_aic": 4 if competitor_gap_mode else 0,
            "channel_memory_aic": 2 if has_channel_memory else 0,
        },
    }
    return int(put), int(aic), breakdown


def estimate_pikzels_v2_call_cost(op: str) -> Tuple[int, int, Dict[str, Any]]:
    """
    Per-call token debit for proxied Pikzels v2 operations (PUT + AIC).
    Tuned to be lighter than full Thumbnail Studio recreate jobs.
    """
    o = (op or "").strip().lower()
    table: Dict[str, Tuple[int, int]] = {
        "prompt": (1, 4),
        "recreate": (1, 5),
        "edit": (1, 5),
        "one_click_fix": (1, 5),
        "faceswap": (1, 6),
        "score": (0, 2),
        "titles": (0, 3),
        "persona": (1, 8),
        "style": (1, 8),
    }
    put, aic = table.get(o, (1, 4))
    return int(put), int(aic), {"pikzels_v2_op": o, "put": put, "aic": aic}


def detect_safety_flags(text: str) -> List[str]:
    t = (text or "").lower()
    flags: List[str] = []
    for mark in _PROTECTED_MARKS:
        if mark in t:
            flags.append(f"possible_protected_mark:{mark}")
    celeb_hits = re.findall(r"\b(mrbeast|kim kardashian|elon musk|drake|taylor swift)\b", t, re.I)
    for h in celeb_hits:
        flags.append(f"celebrity_reference:{str(h).lower()}")
    return flags


def pattern_profile(seed_text: str) -> Dict[str, Any]:
    h = hashlib.sha1((seed_text or "thumbnail").encode("utf-8")).hexdigest()
    x = int(h[:8], 16)
    emotions = ["shock", "curiosity", "confidence", "urgency", "achievement"]
    text_pos = ["top", "center", "bottom"]
    contrast = ["high", "medium", "very_high"]
    return {
        "face_scale": round(0.32 + (x % 33) / 100.0, 2),
        "text_position": text_pos[(x // 7) % len(text_pos)],
        "contrast_profile": contrast[(x // 13) % len(contrast)],
        "emotion_bias": emotions[(x // 19) % len(emotions)],
    }


def format_row_by_key(format_key: Optional[str]) -> Optional[Dict[str, Any]]:
    k = (format_key or "").strip()
    if not k:
        return None
    for row in format_library_rows():
        if str(row.get("key") or "") == k:
            return row
    return None


def format_library_rows() -> List[Dict[str, Any]]:
    return [
        {"key": "gaming_shock_face", "niche": "gaming", "name": "Shock Face + Big Win", "pattern": "big face, 2-4 word hook, neon edge", "social_proof": "High CTR pattern"},
        {"key": "finance_split", "niche": "finance", "name": "Before/After Split", "pattern": "split chart, money callout, clean text", "social_proof": "Used in 12k thumbnails"},
        {"key": "education_arrow", "niche": "education", "name": "Arrow To Insight", "pattern": "diagram + red arrow + 3-word promise", "social_proof": "High CTR pattern"},
        {"key": "automotive_speed", "niche": "automotive", "name": "Motion + Speed Tag", "pattern": "moving car, speed badge, punch headline", "social_proof": "Used in 8k thumbnails"},
        {"key": "lifestyle_glow", "niche": "lifestyle", "name": "Glow Portrait", "pattern": "subject closeup, glow edge, concise text", "social_proof": "High CTR pattern"},
    ]


def build_variant_suggestions(variant: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    face_scale = float(variant.get("face_scale") or 0.0)
    words = int(variant.get("headline_words") or 0)
    contrast = str(variant.get("contrast_profile") or "")
    if face_scale < 0.35:
        out.append("Face too small for mobile; crop tighter.")
    if words > 5:
        out.append("Text too long; keep hook to 3-4 words.")
    if contrast not in ("high", "very_high"):
        out.append("Low subject/background separation; increase contrast.")
    if not out:
        out.append("Strong baseline; run A/B against a stronger emotion cue.")
    return out


def generate_recreate_variants(
    *,
    youtube_title: str,
    topic: str,
    niche: str,
    closeness: int,
    variant_count: int,
    persona_name: str = "",
    competitor_gap_mode: bool = False,
    channel_memory_hint: str = "",
    format_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    n = max(4, min(8, int(variant_count or 4)))
    closeness = max(0, min(100, int(closeness or 50)))
    base = (topic or youtube_title or "Untitled Concept").strip()
    niche_clean = (niche or "general").strip().lower()
    voice = (persona_name or "default").strip()
    fmt = format_row_by_key(format_key)
    layout_name = str(fmt.get("name") or "") if fmt else ""
    layout_pattern = str(fmt.get("pattern") or "") if fmt else ""
    fmt_key = str(fmt.get("key") or "") if fmt else ""

    rows: List[Dict[str, Any]] = []
    for i in range(n):
        seed = f"{base}|{niche_clean}|{closeness}|{i}|{voice}|{channel_memory_hint}|{fmt_key}|{layout_pattern}"
        profile = pattern_profile(seed)
        # Consumer-facing subhead (avoid "safe clone" jargon in UI).
        vibe = (
            "Stays close to familiar winning looks in this niche"
            if closeness >= 70
            else "Pushes a fresher look than typical thumbnails here"
        )
        headline_words = 3 + (i % 3)
        headline = f"{base[:42]}".upper().split()
        headline = " ".join(headline[:headline_words]) or "WATCH THIS"
        ctr_score = 62.0 + (i * 4.1) + (0.08 * closeness)
        if competitor_gap_mode:
            ctr_score += 3.4
        if profile["contrast_profile"] == "very_high":
            ctr_score += 2.1
        if fmt:
            ctr_score += 1.5
        if fmt:
            thumb_layout_suffix = f' following layout "{layout_name}": {layout_pattern}.'
        else:
            thumb_layout_suffix = "."
        variant = {
            "index": i + 1,
            "name": (
                f"{layout_name} · {i + 1}"
                if layout_name
                else f"{niche_clean.title()} Variant {i + 1}"
            ),
            "headline": headline,
            "subhead": f"{vibe} · {niche_clean.replace('_', ' ').title()}",
            "persona": voice or None,
            "format_key": fmt_key or None,
            "layout_pattern": layout_pattern or None,
            "render_prompt": (
                f"Create a {niche_clean} thumbnail{thumb_layout_suffix} "
                f"Use {profile['emotion_bias']} emotion, "
                f"{profile['text_position']} text placement, {profile['contrast_profile']} contrast, "
                f"and approximately {profile['face_scale']:.2f} face scale."
            ),
            "ctr_score": round(min(98.0, ctr_score), 2),
            "face_scale": profile["face_scale"],
            "text_position": profile["text_position"],
            "contrast_profile": profile["contrast_profile"],
            "emotion": profile["emotion_bias"],
            "headline_words": headline_words,
            "watermark_preview": True,
            "safety_flags": detect_safety_flags(f"{headline} {base}"),
        }
        variant["suggestions"] = build_variant_suggestions(variant)
        rows.append(variant)
    rows.sort(key=lambda r: float(r.get("ctr_score") or 0), reverse=True)
    return rows


def youtube_reference_thumbnail_url(video_id: str) -> str:
    """Public CDN URL Pikzels can fetch (hqdefault is more reliable than maxres)."""
    vid = (video_id or "").strip()
    if not vid:
        return ""
    return f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"


def _pikzels_extract_image_url(data: Dict[str, Any]) -> str:
    if not isinstance(data, dict):
        return ""
    for k in (
        "image_url",
        "url",
        "thumbnail_url",
        "output_url",
        "cdn_url",
        "pikzels_cdn_url",
        "image",
    ):
        u = data.get(k)
        if isinstance(u, str) and u.startswith("http"):
            return u.strip()
    nested = data.get("data")
    if isinstance(nested, dict):
        for k in (
            "image_url",
            "url",
            "thumbnail_url",
            "output_url",
            "cdn_url",
            "pikzels_cdn_url",
            "image",
        ):
            u = nested.get(k)
            if isinstance(u, str) and u.startswith("http"):
                return u.strip()
        inner = nested.get("output") or nested.get("result") or nested.get("image")
        if isinstance(inner, dict):
            for k in ("image_url", "url", "thumbnail_url", "output_url", "pikzels_cdn_url"):
                u = inner.get(k)
                if isinstance(u, str) and u.startswith("http"):
                    return u.strip()
    if isinstance(nested, list) and nested:
        first = nested[0] if isinstance(nested[0], dict) else {}
        if isinstance(first, dict):
            for k in ("url", "image_url", "thumbnail_url"):
                u = first.get(k)
                if isinstance(u, str) and u.startswith("http"):
                    return u.strip()
    found = _pikzels_walk_find_cdn_url(data)
    if found:
        return found
    return ""


def _pikzels_walk_find_cdn_url(obj: Any, depth: int = 0) -> str:
    """OpenAPI variants sometimes nest the CDN URL deep in JSON — shallow scan."""
    if depth > 8 or obj is None:
        return ""
    if isinstance(obj, str) and "cdn.pikzels.com" in obj:
        for token in obj.replace(",", " ").split():
            t = token.strip("'\"").rstrip(").,;")
            if t.startswith("http") and "cdn.pikzels.com" in t:
                return t
    if isinstance(obj, dict):
        for v in obj.values():
            u = _pikzels_walk_find_cdn_url(v, depth + 1)
            if u:
                return u
    if isinstance(obj, list):
        for it in obj[:40]:
            u = _pikzels_walk_find_cdn_url(it, depth + 1)
            if u:
                return u
    return ""


def _pikzels_decode_image_bytes(data: Dict[str, Any]) -> Optional[bytes]:
    if not isinstance(data, dict):
        return None
    b64_value = (
        data.get("image_base64")
        or data.get("b64_json")
        or data.get("image_b64")
    )
    if not b64_value and isinstance(data.get("data"), list) and data["data"]:
        first = data["data"][0] if isinstance(data["data"][0], dict) else {}
        if isinstance(first, dict):
            b64_value = first.get("b64_json") or first.get("image_base64")
    if not b64_value:
        return None
    try:
        return base64.b64decode(str(b64_value))
    except (binascii.Error, TypeError, ValueError):
        return None


async def _download_bytes(url: str, timeout: float = 45.0) -> Optional[bytes]:
    headers: Dict[str, str] = {}
    try:
        host = (urlparse(url).hostname or "").lower()
        if "pikzels.com" in host:
            key = resolve_public_api_key()
            if key:
                headers["X-Api-Key"] = key
    except Exception:
        pass
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            r = await client.get(url, headers=headers or None)
            if r.status_code == 200 and r.content:
                return r.content
            _log.debug(
                "thumbnail preview download HTTP %s len=%s host=%s",
                r.status_code,
                len(r.content or b""),
                (urlparse(url).hostname or ""),
            )
    except Exception as e:
        _log.debug("thumbnail preview download failed: %s", e)
    return None


async def _pikzels_response_to_image_bytes(data: Dict[str, Any]) -> Optional[bytes]:
    raw = _pikzels_decode_image_bytes(data)
    if raw:
        return raw
    u = _pikzels_extract_image_url(data)
    if u:
        return await _download_bytes(u)
    return None


async def _r2_put_bytes(r2_key: str, body: bytes, content_type: str) -> None:
    bucket = (R2_BUCKET_NAME or "").strip()
    if not bucket:
        raise RuntimeError("R2_BUCKET_NAME not configured")

    def _put() -> None:
        get_s3_client().put_object(
            Bucket=bucket,
            Key=r2_key,
            Body=body,
            ContentType=content_type,
        )

    await asyncio.to_thread(_put)


def presign_variant_preview_url(r2_key: str, ttl: int = 3600) -> str:
    k = (r2_key or "").strip()
    if not k or not R2_BUCKET_NAME:
        return ""
    try:
        return generate_presigned_download_url(k, ttl=int(ttl))
    except Exception as e:
        _log.debug("presign_variant_preview_url: %s", e)
        return ""


def attach_preview_urls_to_variants(variants: List[Dict[str, Any]], ttl: int = 3600) -> None:
    for v in variants:
        if not isinstance(v, dict):
            continue
        key = str(v.get("preview_r2_key") or "").strip()
        if key:
            v["preview_url"] = presign_variant_preview_url(key, ttl=ttl)
            continue
        ext = str(v.get("pikzels_cdn_url") or "").strip()
        if ext.startswith("http"):
            v["preview_url"] = ext


async def _pikzels_engine_text_brief(
    *,
    source_title: str,
    niche: str,
    topic: str,
) -> str:
    """Single v2 text call — shared creative direction for all variants (no extra wallet debit)."""
    if not resolve_public_api_key():
        return ""
    prompt = (
        f"YouTube thumbnail brief. Video title: {source_title or topic or 'untitled'}. "
        f"Niche: {niche or 'general'}. "
        "Reply with exactly two short sentences: (1) emotional hook angle (2) visual layout emphasis."
    )[:1000]
    status, data = await pikzels_v2_post(
        V2_THUMBNAIL_TEXT,
        {"prompt": prompt, "model": "pkz_4", "format": "16:9"},
    )
    if status >= 400 or not isinstance(data, dict):
        return ""
    for k in ("text", "output", "prompt", "description", "thumbnail_text"):
        t = data.get(k)
        if isinstance(t, str) and t.strip():
            return t.strip()[:800]
    # Some payloads nest strings
    d0 = data.get("data")
    if isinstance(d0, list) and d0 and isinstance(d0[0], dict):
        for k in ("text", "output", "prompt"):
            t = d0[0].get(k)
            if isinstance(t, str) and t.strip():
                return t.strip()[:800]
    return ""


async def _pikzels_score_from_url(image_url: str, title: str) -> Optional[float]:
    if not image_url.strip():
        return None
    payload: Dict[str, Any] = {
        "image_url": image_url.strip(),
        "image_base64": "",
        "title": (title or "")[:200],
    }
    normalize_url_or_base64(payload, "image_url", "image_base64")
    status, data = await pikzels_v2_post(V2_THUMBNAIL_SCORE, payload)
    if status >= 400 or not isinstance(data, dict):
        return None
    for k in ("main_score", "score", "ctr_score", "total_score"):
        try:
            v = data.get(k)
            if v is None and isinstance(data.get("data"), dict):
                v = data["data"].get(k)
            if v is not None:
                f = float(v)
                if 0 <= f <= 100:
                    return round(f, 2)
                if 0 <= f <= 1:
                    return round(f * 100.0, 2)
        except (TypeError, ValueError):
            continue
    return None


async def enrich_variants_with_uploadm8_engine(
    variants: List[Dict[str, Any]],
    *,
    youtube_video_id: str,
    source_title: str,
    niche: str,
    topic: str,
    persona_name: str,
    user_id: str,
    job_id: str,
    pikzels_persona_pikzonality_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    When PIKZELS_API_KEY (or THUMB_RENDER_API_KEY) is set, run Pikzels v2 text + image + score
    and store previews on R2. Wallet is not debited again — studio recreate already charged the user.
    """
    if not variants or not resolve_public_api_key():
        return variants
    ref = youtube_reference_thumbnail_url(youtube_video_id)
    if not ref:
        for v in variants:
            if isinstance(v, dict):
                v.setdefault("engine_status", "skipped_no_video_id")
        return variants

    brief = await _pikzels_engine_text_brief(
        source_title=source_title,
        niche=niche,
        topic=topic,
    )
    sem = asyncio.Semaphore(2)

    async def one(idx: int, v: Dict[str, Any]) -> None:
        async with sem:
            pv = dict(v)
            if brief:
                pv["engine_text_brief"] = brief
                # Strip Pikzels URLs from the snippet (regex must allow https://cdn… with no gap).
                brief_visible = re.sub(
                    r"https?://[^\s]*pikzels\.com[^\s]*", "", brief, flags=re.I
                )
                brief_visible = re.sub(r"\s{2,}", " ", brief_visible).strip()
                pv["subhead"] = (
                    f"{pv.get('subhead') or ''} · {(brief_visible or brief)[:120]}".strip(" ·")
                )
            prompt = str(pv.get("render_prompt") or "")[:980]
            payload: Dict[str, Any] = {
                "prompt": prompt,
                "image_url": ref,
                "image_base64": "",
                "support_image_url": "",
                "support_image_base64": "",
                "image_weight": "medium",
                "model": "pkz_4",
                "format": "16:9",
            }
            pkz_pid = (pikzels_persona_pikzonality_id or "").strip()
            if pkz_pid:
                # Pikzels v2: ``persona`` must be the Pikzonality UUID from POST /v2/pikzonality/persona — not a display name.
                payload["persona"] = pkz_pid[:200]
            else:
                # Legacy rows may have stored a UUID in persona_name; never send arbitrary strings (API rejects).
                pn = (persona_name or "").strip()
                if pn:
                    try:
                        uuid.UUID(pn)
                        payload["persona"] = pn[:200]
                    except (ValueError, TypeError):
                        pass
            normalize_url_or_base64(payload, "image_url", "image_base64")
            normalize_url_or_base64(payload, "support_image_url", "support_image_base64")
            status, pdata = await pikzels_v2_post(V2_THUMBNAIL_IMAGE, payload)
            pv["pikzels_recreate_http_status"] = int(status)
            if status >= 400:
                pv["engine_status"] = "pikzels_recreate_error"
                pv["engine_error"] = _pikzels_api_error_message(pdata or {})[:800]
                variants[idx] = pv
                return
            salvage = ""
            if isinstance(pdata, dict):
                salvage = _pikzels_extract_image_url(pdata)
                if salvage.startswith("http"):
                    pv["pikzels_cdn_url"] = salvage
            img_bytes = await _pikzels_response_to_image_bytes(pdata or {})
            if not img_bytes or len(img_bytes) < 2048:
                if salvage.startswith("http"):
                    pv["engine_status"] = "ok"
                    scored = await _pikzels_score_from_url(
                        salvage, str(pv.get("headline") or "")
                    )
                    if scored is not None:
                        pv["ctr_score"] = scored
                        pv["pikzels_main_score"] = scored
                else:
                    pv["engine_status"] = "pikzels_no_image_bytes"
                variants[idx] = pv
                return
            r2_key = f"thumbnail-studio/previews/{user_id}/{job_id}/variant_{int(pv.get('index') or idx + 1)}.jpg"
            try:
                await _r2_put_bytes(r2_key, img_bytes, "image/jpeg")
            except Exception as e:
                _log.warning("R2 preview upload failed: %s", e)
                if salvage.startswith("http"):
                    pv["pikzels_cdn_url"] = salvage
                    pv["engine_status"] = "ok"
                    scored = await _pikzels_score_from_url(
                        salvage, str(pv.get("headline") or "")
                    )
                    if scored is not None:
                        pv["ctr_score"] = scored
                        pv["pikzels_main_score"] = scored
                    variants[idx] = pv
                    return
                pv["engine_status"] = "r2_upload_failed"
                variants[idx] = pv
                return
            pv["preview_r2_key"] = r2_key
            pv["engine_status"] = "ok"
            signed = presign_variant_preview_url(r2_key, ttl=3600)
            scored = await _pikzels_score_from_url(signed, str(pv.get("headline") or ""))
            if scored is not None:
                pv["ctr_score"] = scored
                pv["pikzels_main_score"] = scored
            variants[idx] = pv

    tasks = []
    for i in range(len(variants)):
        row = variants[i]
        if isinstance(row, dict):
            tasks.append(one(i, row))
    if tasks:
        await asyncio.gather(*tasks)

    variants.sort(key=lambda r: float(r.get("ctr_score") or 0), reverse=True)
    return variants


def build_thumbnail_ab_export_zip(
    *,
    job_id: str,
    job_row: Dict[str, Any],
    variants: List[Dict[str, Any]],
) -> bytes:
    """ZIP: full JSON, CSV summary, per-variant JSON, optional README."""
    buf = io.BytesIO()
    meta = {
        "job_id": job_id,
        "youtube_url": job_row.get("youtube_url"),
        "youtube_video_id": job_row.get("youtube_video_id"),
        "source_title": job_row.get("source_title"),
        "topic": job_row.get("topic"),
        "niche": job_row.get("niche"),
        "closeness": job_row.get("closeness"),
        "variant_count": job_row.get("variant_count"),
        "put_cost": job_row.get("put_cost"),
        "aic_cost": job_row.get("aic_cost"),
    }
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("job.json", json.dumps(meta, indent=2, default=str))
        zf.writestr("variants.json", json.dumps(variants, indent=2, default=str))
        sio = io.StringIO()
        w = csv.writer(sio)
        w.writerow(
            [
                "variant_index",
                "headline",
                "subhead",
                "ctr_score",
                "engine_status",
                "preview_r2_key",
                "render_prompt",
            ]
        )
        for v in variants:
            if not isinstance(v, dict):
                continue
            w.writerow(
                [
                    v.get("index"),
                    v.get("headline"),
                    v.get("subhead"),
                    v.get("ctr_score"),
                    v.get("engine_status"),
                    v.get("preview_r2_key"),
                    v.get("render_prompt"),
                ]
            )
        zf.writestr("summary.csv", sio.getvalue())
        for v in variants:
            if not isinstance(v, dict):
                continue
            idx = int(v.get("index") or 0) or 0
            zf.writestr(f"variant_{idx:02d}.json", json.dumps(v, indent=2, default=str))
        zf.writestr(
            "README.txt",
            "UploadM8 Thumbnail Studio — comparison pack (what each file is for)\n"
            "===============================================================\n\n"
            "job.json\n"
            "  One summary of this run: YouTube link, title, niche, how many variants, token cost.\n"
            "  You do not need to edit this file — it is for your records or support.\n\n"
            "variants.json\n"
            "  Every variant in one list: headlines, prompts, scores, and engine status fields.\n"
            "  Handy if you use a spreadsheet or another tool; the app reads the same data live.\n\n"
            "variant_01.json, variant_02.json, …\n"
            "  One file per variant (same info as inside variants.json, split out for convenience).\n\n"
            "summary.csv\n"
            "  A simple table you can open in Excel / Google Sheets: headline, score, status, etc.\n\n"
            "About scores and previews\n"
            "  Numbers like \"CTR\" in the product are model estimates from our scoring step —\n"
            "  they are not your real YouTube click‑through rate until you publish and measure.\n"
            "  Preview images are stored privately on UploadM8; use the in‑app URLs or re‑export.\n",
        )
    return buf.getvalue()


async def upload_ab_export_zip_to_r2(zip_bytes: bytes, user_id: str, job_id: str) -> str:
    key = f"thumbnail-studio/ab-packs/{user_id}/{job_id}.zip"
    await _r2_put_bytes(key, zip_bytes, "application/zip")
    return key
