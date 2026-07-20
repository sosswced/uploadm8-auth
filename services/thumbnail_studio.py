from __future__ import annotations

import asyncio
import base64
import binascii
import time
import csv
import hashlib
import html
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
from core.thumbnail_text import clean_thumbnail_headline, is_generic_thumbnail_headline

from services.pikzels_v2 import (
    V2_PIKZONALITY_BY_ID,
    V2_PIKZONALITY_PERSONA,
    V2_THUMBNAIL_FACESWAP,
    V2_THUMBNAIL_IMAGE,
    V2_THUMBNAIL_SCORE,
    V2_THUMBNAIL_TEXT,
    resolve_public_api_key,
)
from services.pikzels_errors import format_pikzels_error_message

# Durable studio previews live on R2 under thumbnail-studio/previews/…
# Objects are kept ~10 months by default (set Cloudflare R2 lifecycle to match).
# Presigned GET max is 7d (SigV4); clients re-presign via job load / r2-preview proxy.
STUDIO_PREVIEW_RETENTION_DAYS = max(
    30, min(int(os.environ.get("STUDIO_PREVIEW_RETENTION_DAYS", "300")), 3660)
)
STUDIO_PREVIEW_PRESIGN_TTL_SEC = max(
    3600, min(int(os.environ.get("STUDIO_PREVIEW_PRESIGN_TTL_SEC", str(7 * 86400))), 7 * 86400)
)

# Guided recreate: each variant = one Pikzels /v2/thumbnail/image (+ optional faceswap).
STUDIO_VARIANT_COUNT_MIN = max(1, int(os.environ.get("THUMBNAIL_STUDIO_VARIANT_MIN", "1")))
STUDIO_VARIANT_COUNT_MAX = max(
    STUDIO_VARIANT_COUNT_MIN,
    int(os.environ.get("THUMBNAIL_STUDIO_VARIANT_MAX", "8")),
)
STUDIO_VARIANT_COUNT_DEFAULT = max(
    STUDIO_VARIANT_COUNT_MIN,
    min(
        STUDIO_VARIANT_COUNT_MAX,
        int(os.environ.get("THUMBNAIL_STUDIO_VARIANT_DEFAULT", "2")),
    ),
)


def clamp_studio_variant_count(raw: Any) -> int:
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = STUDIO_VARIANT_COUNT_DEFAULT
    return max(STUDIO_VARIANT_COUNT_MIN, min(STUDIO_VARIANT_COUNT_MAX, n))
from services.pikzels_v2_client import (
    normalize_url_or_base64,
    pikzels_v2_get,
    pikzels_v2_post,
    resolve_pikzels_persona_style_xor,
    trim_pikzonality_images,
)

_log = logging.getLogger("uploadm8.thumbnail_studio")

_PIKZELS_CDN_REST_RE = re.compile(
    r"https://cdn\.pikzels\.com/rest-api/[^\s\"'<>)\]]+",
    re.I,
)


def strip_pikzels_urls_from_text(s: str) -> str:
    """Remove Pikzels CDN/API URLs from user-facing copy (briefs, subheads)."""
    if not s:
        return ""
    t = re.sub(
        r"https?://[^\s\"'<>)\]]*pikzels\.com[^\s\"'<>)\]]*",
        "",
        str(s),
        flags=re.I,
    )
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t.strip(" ·-|")


def extract_cdn_pikzels_rest_url(text: str) -> str:
    """First ``https://cdn.pikzels.com/rest-api/...`` URL in *text*, or empty."""
    if not text:
        return ""
    m = _PIKZELS_CDN_REST_RE.search(str(text))
    if not m:
        return ""
    u = m.group(0).rstrip(").,;\"'")
    return u if u.lower().startswith("https://cdn.pikzels.com/") else ""


def promote_cdn_url_from_variant_text(v: Dict[str, Any]) -> None:
    """If ``pikzels_cdn_url`` is missing, copy a CDN thumbnail URL out of stored text fields."""
    if str(v.get("pikzels_cdn_url") or "").strip().lower().startswith("http"):
        return
    for k in ("subhead", "engine_text_brief", "engine_error"):
        u = extract_cdn_pikzels_rest_url(str(v.get(k) or ""))
        if u:
            v["pikzels_cdn_url"] = u
            return


def polish_studio_variant_for_client(v: Dict[str, Any]) -> None:
    """Lift CDN URLs into ``pikzels_cdn_url`` and strip them from ``subhead`` for API clients."""
    if not isinstance(v, dict):
        return
    promote_cdn_url_from_variant_text(v)
    sh = str(v.get("subhead") or "")
    if sh:
        v["subhead"] = strip_pikzels_urls_from_text(sh)


def sanitize_pikzels_persona_name(name: str) -> str:
    """Pikzels POST /v2/pikzonality/persona: name max 25 chars; letters, numbers, spaces, hyphens, underscores."""
    raw = (name or "").strip() or "Persona"
    safe = re.sub(r"[^\w\-\s]", "", raw, flags=re.UNICODE)
    safe = re.sub(r"\s+", " ", safe).strip()[:25] or "Persona"
    return safe


# Persona rows store full data URLs in Postgres TEXT; never truncate base64 payloads
# (truncation caused Pikzels "Image could not be decoded" on link-pikzels).
_PERSONA_IMAGE_DB_MAX_CHARS = int(
    os.environ.get("PERSONA_IMAGE_DB_MAX_CHARS", "12000000") or 12_000_000
)


def truncate_persona_image_url_for_storage(url: str) -> str:
    """Cap stored length for absurd inputs; data URLs must stay intact for Pikzels re-link."""
    s = str(url or "").strip()
    if not s:
        return ""
    cap = max(32_000, min(_PERSONA_IMAGE_DB_MAX_CHARS, 20_000_000))
    if len(s) <= cap:
        return s
    return s[:cap]


def normalize_persona_face_ref_for_pikzels(ref: str) -> Optional[str]:
    """
    Re-encode browser ``data:image/*`` payloads to JPEG data URLs so Pikzels receives
    decodable bytes (WebP/PNG/alpha, odd ICC profiles, etc.).

    ``http(s)`` URLs are returned unchanged (Pikzels fetches them).
    """
    s = str(ref or "").strip()
    if not s:
        return None
    low = s.lower()
    if low.startswith("https://") or low.startswith("http://"):
        return s[:8000]
    if not low.startswith("data:image"):
        return None
    comma = s.find(",")
    header = s[:comma].lower() if comma >= 0 else low
    if comma < 0 or ";base64" not in header:
        return None
    raw_b64 = s[comma + 1 :].strip()
    try:
        raw = base64.b64decode(raw_b64, validate=False)
    except (binascii.Error, ValueError):
        return None
    if not raw:
        return None
    try:
        from PIL import Image

        im = Image.open(io.BytesIO(raw))
        im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88, optimize=True)
        out = buf.getvalue()
    except Exception as e:
        _log.debug("normalize_persona_face_ref_for_pikzels PIL failed: %s", e)
        return None
    if not out:
        return None
    b64o = base64.b64encode(out).decode("ascii")
    return f"data:image/jpeg;base64,{b64o}"


def prepare_persona_image_refs_for_pikzels(refs: List[str]) -> Tuple[List[str], Optional[str]]:
    """
    Drop unreadable entries; require ≥3 usable refs for Pikzels persona registration.

    Returns (cleaned_refs, error_message).
    """
    cleaned: List[str] = []
    for r in refs or []:
        n = normalize_persona_face_ref_for_pikzels(str(r).strip())
        if n:
            cleaned.append(n)
    if len(cleaned) < 3:
        return [], (
            "Need at least three readable face photos (JPEG/PNG/WebP). "
            "If “Link to Pikzels” used to fail with “image could not be decoded”, "
            "re-save the persona with 3–20 new selfies — a previous release truncated stored photos."
        )
    return cleaned, None


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
    """See https://docs.pikzels.com/errors for upstream ``error.code`` / ``error.message``."""
    return format_pikzels_error_message(data, max_len=800)


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


def _pikzels_duplicate_persona_name_error(message: str) -> bool:
    """True when Pikzels rejects create because this display name is already registered."""
    low = (message or "").lower()
    if "already exist" in low and "persona" in low:
        return True
    if "duplicate" in low and "name" in low:
        return True
    return False


async def register_creator_persona_with_pikzels(
    *,
    name: str,
    image_refs: List[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Create a persona pikzonality at Pikzels (3 face refs) and wait until ready.
    Returns (pikzels_pikzonality_id, error_message). ID is what /v2/thumbnail/* expects in ``persona``.

    Retries with a short random suffix when Pikzels says the name already exists (common after a
    partial link or re-save with the same label).
    """
    if not resolve_public_api_key():
        return (
            None,
            "Pikzels is not configured on this server (set PIKZELS_API_KEY or "
            "THUMB_RENDER_API_KEY). Save/link cannot complete until the key is set.",
        )
    prepared, prep_err = prepare_persona_image_refs_for_pikzels(list(image_refs or []))
    if prep_err:
        return None, prep_err
    urls, b64s = split_face_refs_for_pikzels(prepared)
    base_name = sanitize_pikzels_persona_name(name)
    last_err: Optional[str] = None

    for attempt in range(6):
        if attempt == 0:
            label = base_name
        else:
            tail = uuid.uuid4().hex[:4]
            label = sanitize_pikzels_persona_name(f"{base_name[:18]}-{tail}")
        body: Dict[str, Any] = {"name": label}
        if len(urls) >= 3:
            body["image_urls"] = urls[:3]
        elif len(b64s) >= 3:
            body["image_base64s"] = b64s[:3]
        else:
            return None, "pikzels_persona_requires_three_https_urls_or_three_data_image_urls"
        trim_pikzonality_images(body)
        code, data = await pikzels_v2_post(V2_PIKZONALITY_PERSONA, body)
        if code < 400 and isinstance(data, dict):
            new_id = str(data.get("id") or data.get("request_id") or "").strip()
            if not new_id:
                _log.warning("Pikzels persona create succeeded without id (name=%r)", label)
                return None, "pikzels_persona_create_missing_id"
            wait_err = await wait_pikzonality_persona_ready(new_id)
            if wait_err:
                _log.warning(
                    "Pikzels persona did not become ready (name=%r id=%s err=%s)",
                    label,
                    new_id[:12],
                    wait_err,
                )
                return None, wait_err
            if attempt > 0:
                _log.info(
                    "registered Pikzels persona after name collision (base=%r registered_as=%r id=%s)",
                    base_name,
                    label,
                    new_id[:12],
                )
            else:
                _log.info(
                    "registered Pikzels persona (name=%r id=%s)",
                    label,
                    new_id[:12],
                )
            return new_id, None

        err = _pikzels_api_error_message(data) if isinstance(data, dict) else "pikzels_persona_create_failed"
        last_err = err
        if attempt < 5 and _pikzels_duplicate_persona_name_error(err):
            _log.info(
                "pikzels persona name collision; retrying with new label (attempt=%s err=%s)",
                attempt + 1,
                (err or "")[:160],
            )
            continue
        _log.warning("Pikzels persona registration failed (name=%r err=%s)", label, (err or "")[:240])
        return None, err

    _log.warning("Pikzels persona registration exhausted retries (base=%r err=%s)", base_name, (last_err or "")[:240])
    return None, last_err or "pikzels_persona_create_failed"


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
    """Estimate PUT/AIC for Thumbnail Studio recreate jobs.

    Debits must still go through ``core.wallet.atomic_debit_tokens``. AIC component
    sizes for known pipeline services are anchored to ``stages.ai_service_costs.SERVICE_WEIGHTS``
    where a 1:1 mapping exists (competitor gap); studio-specific base/variant math
    stays local so wallet UX does not silently change.
    """
    from stages.ai_service_costs import SERVICE_WEIGHTS

    # Jul 2026: Pikzels image→thumb ≈ $0.36; target wallet ≥ 2.5× ≈ 90 AIC @ $0.01/AIC.
    n = clamp_studio_variant_count(variant_count)
    per_variant_aic = int(SERVICE_WEIGHTS.get("thumbnail_recreate_ai", 90) or 90)
    put = 4 + n
    aic = n * per_variant_aic
    persona_base_aic = int(SERVICE_WEIGHTS.get("persona_consistency", 95) or 95) if has_persona else 0
    # Faceswap ≈ $0.37 × 2.5 ≈ 93 AIC per variant when persona is applied.
    persona_faceswap_aic = (n * 90) if has_persona else 0
    if has_persona:
        aic += persona_base_aic + persona_faceswap_aic
    competitor_gap_aic = int(SERVICE_WEIGHTS.get("thumbnail_competitor_gap", 8)) if competitor_gap_mode else 0
    if competitor_gap_aic:
        aic += competitor_gap_aic
    channel_memory_aic = 5 if has_channel_memory else 0
    if channel_memory_aic:
        aic += channel_memory_aic
    breakdown = {
        "variant_count": n,
        "components": {
            "base_put": 4,
            "variant_put": n,
            "base_aic": 0,
            "variant_aic": n * per_variant_aic,
            "persona_aic": persona_base_aic,
            "persona_faceswap_aic": persona_faceswap_aic,
            "competitor_gap_aic": competitor_gap_aic,
            "channel_memory_aic": channel_memory_aic,
        },
        "service_weight_refs": {
            "competitor_gap_aic": "thumbnail_competitor_gap",
            "persona_aic": "persona_consistency",
            "variant_aic": "thumbnail_recreate_ai",
        },
        "pricing_note": "2.5x Pikzels vendor avg (Jul 2026); 1 AIC ≈ $0.01 retail",
        "debit_via": "atomic_debit_tokens",
    }
    return int(put), int(aic), breakdown


def estimate_pikzels_v2_call_cost(op: str) -> Tuple[int, int, Dict[str, Any]]:
    """
    Per-call token debit for proxied Pikzels v2 operations (PUT + AIC).

    Anchored to Jul 2026 Pikzels usage (~$0.36–$0.38 for PKZ-4 gen, $0.12 edit,
    $0.03 score) at ≥2.5× vendor with 1 AIC ≈ $0.01 intended retail.
    """
    o = (op or "").strip().lower()
    # (put, aic) — AIC ≈ ceil(2.5 * vendor_usd / 0.01)
    table: Dict[str, Tuple[int, int]] = {
        "prompt": (1, 20),          # lighter than full gen
        "recreate": (1, 90),        # image/text→thumb ~$0.36–0.37
        "edit": (1, 30),            # PKZ-4.5 edit ~$0.12
        "one_click_fix": (1, 30),  # treat like edit
        "faceswap": (1, 93),        # ~$0.37
        "score": (0, 8),            # ~$0.03
        "titles": (0, 10),          # text-only; modest buffer
        "persona": (1, 95),         # create persona ~$0.38
        "style": (1, 90),           # style gen ≈ recreate
    }
    put, aic = table.get(o, (1, 20))
    return int(put), int(aic), {
        "pikzels_v2_op": o,
        "put": put,
        "aic": aic,
        "pricing_note": "2.5x Pikzels vendor avg (Jul 2026); 1 AIC ≈ $0.01 retail",
    }


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


def cap_pikzels_studio_render_prompt(prompt: str, max_len: Optional[int] = None) -> str:
    """
    Keep studio recreate prompts within Pikzels' 1000-char API limit (abbreviated when possible).

    Prefer preserving the headline hook and YouTube-reference guardrails; trim tail
    (creative/hydration lanes) when necessary.
    """
    try:
        cap = int(max_len) if max_len is not None else int(
            os.environ.get("PIKZELS_THUMBNAIL_PROMPT_MAX", "950") or 950
        )
    except (TypeError, ValueError):
        cap = 950
    cap = max(400, min(1000, cap))
    p = re.sub(r"\s+", " ", (prompt or "").strip())
    if len(p) <= cap:
        return p
    if " Hydration focus:" in p:
        p = p.split(" Hydration focus:")[0].strip()
    if len(p) <= cap:
        return p
    if " Creative direction:" in p and len(p) > cap:
        head, _, tail = p.partition(" Creative direction:")
        if len(head) + 80 < cap:
            tail_budget = max(40, cap - len(head) - len(" Creative direction:"))
            p = f"{head} Creative direction:{tail[:tail_budget].rsplit(' ', 1)[0]}".strip()
    if len(p) <= cap:
        return p
    from stages.pikzels_api import fit_pikzels_prompt_to_budget

    return fit_pikzels_prompt_to_budget(p, cap)


def closeness_to_pikzels_image_weight(closeness: int) -> str:
    """
    Map Studio closeness (0–100) to Pikzels v2 recreate ``image_weight``.

    Lower closeness → freer interpretation of the YouTube still; higher → tighter
    adherence. Must be one of the weights Pikzels accepts for ``/v2/thumbnail/image``.
    """
    try:
        c = int(closeness)
    except (TypeError, ValueError):
        c = 55
    c = max(0, min(100, c))
    if c <= 34:
        return "low"
    if c <= 66:
        return "medium"
    return "high"


def audience_label(niche: str) -> str:
    """Human-readable audience label used in prompts and cards."""
    key = (niche or "general").strip().lower()
    labels = {
        "general": "broad YouTube",
        "gaming": "gaming",
        "finance": "finance / money",
        "education": "education / tutorial",
        "automotive": "automotive",
        "lifestyle": "lifestyle",
        "comedy": "comedy / meme",
        "podcast": "podcast / interview",
        "music": "music / artist",
        "sports": "sports",
        "tech": "tech / product",
        "beauty": "beauty / fashion",
        "food": "food / cooking",
        "travel": "travel / outdoors",
        "fitness": "fitness / health",
        "true_crime": "true crime / documentary",
        "real_estate": "real estate",
        "business": "business / creator economy",
        "news": "news / commentary",
    }
    return labels.get(key, key.replace("_", " ") or "broad YouTube")


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


def parse_dynamic_layout_key(key: str) -> Optional[Tuple[str, int]]:
    """
    Parse ``dyn-{niche-slug}-{index}`` keys (index is zero-padded width 4).

    The niche segment may contain hyphens (e.g. ``true-crime``); the numeric
    suffix is always the last hyphen-separated segment.
    """
    k = (key or "").strip().lower()
    if not k.startswith("dyn-"):
        return None
    rest = k[4:]
    if not rest:
        return None
    niche_slug, idx_s = rest.rsplit("-", 1)
    niche_slug = niche_slug.strip("-") or "general"
    if not idx_s.isdigit():
        return None
    return niche_slug, int(idx_s)


# Procedural layout frames — indexed by ``idx % len`` then perturbed by niche + seed.
_LAYOUT_FRAME_POOL: Tuple[Tuple[str, str], ...] = (
    ("Split reveal", "before/after split with a credible proof badge and tight headline rail"),
    ("Arrow insight", "single focal arrow path, diagram-adjacent clarity, compact typography"),
    ("Single hero", "one dominant subject, generous negative space, bold lower-third hook"),
    ("Dual tension", "two subjects in opposition, high drama crop, rivalry framing"),
    ("Macro proof", "extreme close-up of the proof object with small reaction inset"),
    ("Wide story", "environmental master with readable depth layers and one hero face"),
    ("Speed streak", "motion streaks, kinetic energy, speed or progress badge"),
    ("Clean studio", "soft key light, premium editorial grade, minimal clutter"),
    ("Neon punch", "saturated rim light, cyber edge, chunky display type zone"),
    ("Document stack", "paperwork / evidence stack vibe with one circled detail"),
    ("Map corridor", "route or map overlay hint without inventing false geography"),
    ("Timer urgency", "countdown or clock motif paired with decisive facial cue"),
    ("Podcast depth", "mic + shallow depth-of-field guest stack, warm grade"),
    ("Product hero", "device or SKU hero with spec callout zone and face inset"),
    ("Outdoor scale", "tiny human vs big landscape, horizon-led composition"),
    ("Meme stack", "stacked reaction layers, meme-native contrast, punchy hook zone"),
)

# Stable wireframe ids for Thumbnail Studio layout preview UI (maps 1:1 with _LAYOUT_FRAME_POOL order).
_LAYOUT_PREVIEW_IDS: Tuple[str, ...] = (
    "split_reveal",
    "arrow_insight",
    "single_hero",
    "dual_tension",
    "macro_proof",
    "wide_story",
    "speed_streak",
    "clean_studio",
    "neon_punch",
    "document_stack",
    "map_corridor",
    "timer_urgency",
    "podcast_depth",
    "product_hero",
    "outdoor_scale",
    "meme_stack",
)

_CURATED_LAYOUT_PREVIEW: Dict[str, str] = {
    "gaming_shock_face": "shock_face",
    "reaction_meme": "meme_stack",
    "finance_split": "split_reveal",
    "education_arrow": "arrow_insight",
    "automotive_speed": "speed_streak",
    "lifestyle_glow": "single_hero",
    "podcast_guest": "podcast_depth",
    "music_release": "neon_punch",
    "sports_showdown": "dual_tension",
    "tech_product": "product_hero",
    "beauty_transformation": "split_reveal",
    "food_crave": "macro_proof",
    "travel_wonder": "outdoor_scale",
    "fitness_result": "timer_urgency",
    "true_crime_case": "document_stack",
    "real_estate_listing": "wide_story",
}


def layout_preview_id(format_key: str) -> str:
    """Wireframe slug for guided-recreate layout chips (frontend SVG previews)."""
    k = (format_key or "").strip()
    if not k:
        return "mixed_variants"
    if k in _CURATED_LAYOUT_PREVIEW:
        return _CURATED_LAYOUT_PREVIEW[k]
    parsed = parse_dynamic_layout_key(k)
    if parsed:
        _, idx = parsed
        return _LAYOUT_PREVIEW_IDS[idx % len(_LAYOUT_PREVIEW_IDS)]
    return "single_hero"


def attach_layout_preview_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    pid = layout_preview_id(str(row.get("key") or ""))
    out["preview_id"] = pid
    return out


def dynamic_layout_row(format_key: str) -> Optional[Dict[str, Any]]:
    """Resolve a procedural ``dyn-…`` row; stable for a given key and niche."""
    parsed = parse_dynamic_layout_key(format_key)
    if not parsed:
        return None
    niche_slug, idx = parsed
    niche_key = niche_slug.replace("-", "_")
    name_t, pat_t = _LAYOUT_FRAME_POOL[idx % len(_LAYOUT_FRAME_POOL)]
    aud = audience_label(niche_key)
    twist = pattern_profile(f"{format_key}|{niche_slug}|{idx}")
    pattern = (
        f"{pat_t} Tuned for {aud} discovery feeds; {twist['emotion_bias']} energy, "
        f"{twist['text_position']} text bias, {twist['contrast_profile']} contrast, "
        f"~{twist['face_scale']:.2f} face scale vs frame."
    )
    return {
        "key": format_key.strip(),
        "niche": niche_key,
        "name": f"{name_t} · mix {idx + 1}",
        "pattern": pattern[:420],
        "social_proof": "Procedural layout",
    }


def static_format_library_rows() -> List[Dict[str, Any]]:
    """Curated layout presets (``format_key``)."""
    return [
        {"key": "gaming_shock_face", "niche": "gaming", "name": "Shock Face + Big Win", "pattern": "big face, 2-4 word hook, neon edge", "social_proof": "High CTR pattern"},
        {"key": "reaction_meme", "niche": "comedy", "name": "Reaction Meme", "pattern": "two expressive faces, bold stacked text, internet-meme contrast", "social_proof": "Creator pattern"},
        {"key": "finance_split", "niche": "finance", "name": "Before/After Split", "pattern": "split screen, proof element, clean money/text callout", "social_proof": "Used in 12k thumbnails"},
        {"key": "education_arrow", "niche": "education", "name": "Arrow To Insight", "pattern": "diagram + red arrow + 3-word promise", "social_proof": "High CTR pattern"},
        {"key": "automotive_speed", "niche": "automotive", "name": "Motion + Speed Tag", "pattern": "moving car, speed badge, punch headline", "social_proof": "Used in 8k thumbnails"},
        {"key": "lifestyle_glow", "niche": "lifestyle", "name": "Glow Portrait", "pattern": "subject closeup, glow edge, concise text", "social_proof": "High CTR pattern"},
        {"key": "podcast_guest", "niche": "podcast", "name": "Guest Quote", "pattern": "host/guest faces, quote hook, studio depth, clean name tag", "social_proof": "Podcast pattern"},
        {"key": "music_release", "niche": "music", "name": "Artist Drop", "pattern": "artist closeup, moody grade, track-title typography, motion glow", "social_proof": "Music pattern"},
        {"key": "sports_showdown", "niche": "sports", "name": "Showdown Matchup", "pattern": "two-side rivalry, scoreboard badge, intense action crop", "social_proof": "Sports pattern"},
        {"key": "tech_product", "niche": "tech", "name": "Product Hero", "pattern": "device close-up, face reaction inset, clean spec badge, blue rim light", "social_proof": "Tech pattern"},
        {"key": "beauty_transformation", "niche": "beauty", "name": "Transformation", "pattern": "before/after beauty crop, soft glow, simple result claim", "social_proof": "Beauty pattern"},
        {"key": "food_crave", "niche": "food", "name": "Crave Close-Up", "pattern": "macro food hero, hand/face reaction, warm contrast, short taste hook", "social_proof": "Food pattern"},
        {"key": "travel_wonder", "niche": "travel", "name": "Destination Wonder", "pattern": "wide place reveal, small human scale, bright location label", "social_proof": "Travel pattern"},
        {"key": "fitness_result", "niche": "fitness", "name": "Result Proof", "pattern": "body/action pose, timer/proof badge, high-energy contrast", "social_proof": "Fitness pattern"},
        {"key": "true_crime_case", "niche": "true_crime", "name": "Case File", "pattern": "dark evidence board, red circle/arrow, tense 3-word hook", "social_proof": "Story pattern"},
        {"key": "real_estate_listing", "niche": "real_estate", "name": "Property Reveal", "pattern": "wide home/interior hero, price/location badge, bright clean text", "social_proof": "Real estate pattern"},
    ]


def format_row_by_key(format_key: Optional[str]) -> Optional[Dict[str, Any]]:
    k = (format_key or "").strip()
    if not k:
        return None
    dyn = dynamic_layout_row(k)
    if dyn:
        return dyn
    for row in static_format_library_rows():
        if str(row.get("key") or "") == k:
            return row
    return None


def format_library_rows(niche: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Curated presets plus procedural ``dyn-{niche}-{nnnn}`` rows for the given niche.

    Procedural keys stay stable: the same ``format_key`` always resolves to the same
    pattern via ``format_row_by_key`` (used by Studio + upload pipeline).
    """
    rows = list(static_format_library_rows())
    slug = (niche or "general").strip().lower().replace(" ", "_") or "general"
    for i in range(32):
        row = dynamic_layout_row(f"dyn-{slug}-{i:04d}")
        if row:
            rows.append(row)
    return [attach_layout_preview_fields(r) for r in rows]


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


def _variant_creative_directive(i: int, closeness: int, competitor_gap_mode: bool) -> str:
    """
    Deterministic style directive per variant index so Pikzels gets truly different
    creative instructions instead of near-duplicates.
    """
    lanes = [
        "tight face crop, asymmetric framing, bold 2-3 word hook in upper-left, warm highlights",
        "split composition (before/after), strong red arrow cue, cool blue shadows, compact badge text",
        "clean negative space on one side, oversized headline, high local contrast on subject, minimal props",
        "dynamic diagonal composition, action blur accents, urgency color pops (orange/red), punchy typography",
        "centered hero subject with glow edge, simplified background, premium cinematic grade, short confidence hook",
        "reaction-first portrait, exaggerated expression, chunky outlined text, contrasty teal/orange palette",
        "object close-up plus small face inset, numeric proof callout, crisp white text, editorial layout",
        "story-frame composition with depth layers, directional lighting, restrained text, dramatic mood",
    ]
    base = lanes[max(0, i) % len(lanes)]
    if closeness >= 75:
        return f"{base}. Keep resemblance to the source layout while changing color and text hierarchy."
    if competitor_gap_mode:
        return f"{base}. Make it distinctly different from common niche thumbnails; prioritize novelty."
    return f"{base}. Ensure this looks materially different from other variants in this batch."


def normalize_hydration_context(raw: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Clean optional user-provided evidence that should steer thumbnail variants."""
    if not isinstance(raw, dict):
        return {}

    def _clean(key: str, limit: int) -> str:
        value = str(raw.get(key) or "").strip()
        value = re.sub(r"\s+", " ", value)
        return value[:limit]

    out = {
        "story": _clean("story", 520),
        "caption": _clean("caption", 420),
        "geo": _clean("geo", 180),
        "latitude": _clean("latitude", 40),
        "longitude": _clean("longitude", 40),
        "artist": _clean("artist", 120),
        "track": _clean("track", 160),
        "speed_mph": _clean("speed_mph", 24),
        "osd_driver": _clean("osd_driver", 80),
        "osd_recording_start": _clean("osd_recording_start", 80),
        "speech": _clean("speech", 320),
        "trill_bucket": _clean("trill_bucket", 48),
        "trill_score": _clean("trill_score", 24),
        "vision_ocr": _clean("vision_ocr", 180),
        "vision_labels": _clean("vision_labels", 220),
    }
    out = {k: v for k, v in out.items() if v}
    for key, value in raw.items():
        if str(key).startswith("_") and value is not None:
            out[str(key)] = str(value).strip()[:240]
    return out


def hydration_signal_lanes(ctx: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Turn caption/geo/music evidence into variant lanes shown in the UI."""
    clean = normalize_hydration_context(ctx)
    lanes: List[Dict[str, str]] = []
    if clean.get("story"):
        lanes.append(
            {
                "key": "story",
                "label": "Hydration story",
                "value": clean["story"],
                "directive": "Ground the thumbnail in this factual scene paragraph",
            }
        )
    if clean.get("caption"):
        lanes.append(
            {
                "key": "caption",
                "label": "Caption hook",
                "value": clean["caption"],
                "directive": "Ground the thumbnail in this caption or transcript detail",
            }
        )
    geo_bits = [clean.get("geo") or ""]
    if clean.get("latitude") and clean.get("longitude"):
        geo_bits.append(f"{clean['latitude']}, {clean['longitude']}")
    elif clean.get("latitude") or clean.get("longitude"):
        geo_bits.append(clean.get("latitude") or clean.get("longitude") or "")
    geo_value = " · ".join([x for x in geo_bits if x])
    if geo_value:
        lanes.append(
            {
                "key": "geo",
                "label": "Geo / route",
                "value": geo_value[:220],
                "directive": "Use the place, route, or coordinates as concrete visual evidence",
            }
        )
    music_value = " - ".join([x for x in (clean.get("artist"), clean.get("track")) if x])
    if music_value:
        lanes.append(
            {
                "key": "music",
                "label": "Artist / track",
                "value": music_value[:220],
                "directive": "Let the recognized music influence mood, typography, and hook",
            }
        )
    speed_bits = []
    if clean.get("speed_mph"):
        speed_bits.append(f"{clean['speed_mph']} mph peak")
    if clean.get("osd_driver"):
        speed_bits.append(f"driver/HUD {clean['osd_driver']}")
    if clean.get("osd_recording_start"):
        speed_bits.append(f"rec {clean['osd_recording_start']}")
    speed_value = " · ".join(speed_bits)
    if speed_value:
        lanes.append(
            {
                "key": "speed",
                "label": "Dashcam speed / OSD",
                "value": speed_value[:220],
                "directive": "Show speed, HUD, or dashcam telemetry as a believable visual cue",
            }
        )
    trill_bits = []
    if clean.get("trill_bucket"):
        trill_bits.append(str(clean["trill_bucket"]))
    if clean.get("trill_score"):
        trill_bits.append(f"score {clean['trill_score']}")
    trill_value = " · ".join(trill_bits)
    if trill_value:
        lanes.append(
            {
                "key": "trill",
                "label": "Trill intensity",
                "value": trill_value[:220],
                "directive": "Match the energy level implied by the Trill score bucket",
            }
        )
    if clean.get("speech"):
        lanes.append(
            {
                "key": "speech",
                "label": "Speech / transcript",
                "value": clean["speech"][:220],
                "directive": "Reflect this spoken line or transcript beat in expression or scene",
            }
        )
    vision_bits = []
    if clean.get("vision_ocr"):
        vision_bits.append(clean["vision_ocr"][:120])
    if clean.get("vision_labels"):
        vision_bits.append(clean["vision_labels"][:120])
    vision_value = " · ".join(vision_bits)
    if vision_value:
        lanes.append(
            {
                "key": "vision",
                "label": "Vision OCR / labels",
                "value": vision_value[:220],
                "directive": "Use on-frame text or vision labels as concrete scene anchors",
            }
        )
    if clean:
        all_bits = []
        if clean.get("story"):
            all_bits.append(clean["story"])
        if clean.get("caption"):
            all_bits.append(clean["caption"])
        if geo_value:
            all_bits.append(geo_value)
        if speed_value:
            all_bits.append(speed_value)
        if music_value:
            all_bits.append(music_value)
        if trill_value:
            all_bits.append(trill_value)
        if clean.get("speech"):
            all_bits.append(clean["speech"])
        if vision_value:
            all_bits.append(vision_value)
        lanes.append(
            {
                "key": "combined",
                "label": "Combined evidence",
                "value": " | ".join(all_bits)[:260],
                "directive": "Blend the strongest available evidence into one high-context thumbnail",
            }
        )
    return lanes


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
    hydration_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    n = clamp_studio_variant_count(variant_count)
    try:
        _close = int(closeness)
    except (TypeError, ValueError):
        _close = 55
    closeness = max(0, min(100, _close))
    base = (topic or youtube_title or "Untitled Concept").strip()
    niche_clean = (niche or "general").strip().lower()
    audience = audience_label(niche_clean)
    voice = (persona_name or "default").strip()
    fmt_selected = format_row_by_key(format_key) if (format_key or "").strip() else None
    hydration_lanes = hydration_signal_lanes(hydration_context)

    rows: List[Dict[str, Any]] = []
    for i in range(n):
        if fmt_selected:
            fmt = fmt_selected
        else:
            fmt = dynamic_layout_row(f"dyn-{niche_clean}-{i:04d}")
        layout_name = str(fmt.get("name") or "") if fmt else ""
        layout_pattern = str(fmt.get("pattern") or "") if fmt else ""
        fmt_key = str(fmt.get("key") or "") if fmt else ""
        seed = f"{base}|{niche_clean}|{closeness}|{i}|{voice}|{channel_memory_hint}|{fmt_key}|{layout_pattern}"
        profile = pattern_profile(seed)
        # Consumer-facing subhead (avoid "safe clone" jargon in UI).
        vibe = (
            "Stays close to familiar winning looks in this niche"
            if closeness >= 70
            else "Pushes a fresher look than typical thumbnails here"
        )
        headline_words = 3 + (i % 3)
        headline = clean_thumbnail_headline(base, max_words=headline_words, max_chars=42)
        if is_generic_thumbnail_headline(headline):
            headline = clean_thumbnail_headline(f"{niche_clean.replace('_', ' ')} concept", max_words=3) or "VIDEO HIGHLIGHT"
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
        creative_directive = _variant_creative_directive(i, closeness, competitor_gap_mode)
        reference_prompt = (
            "Use the provided YouTube thumbnail image as the visual anchor. Preserve its primary "
            "subject type, pose/framing, text density, camera angle, and color hierarchy unless "
            "the editor topic explicitly requires a small change. Do not invent unrelated cars, "
            "objects, locations, or generic stock people that are not implied by the reference or topic."
        )
        persona_prompt = ""
        if persona_name and voice.lower() != "default":
            persona_prompt = (
                f" A linked Pikzels persona named {voice[:80]} is supplied; make that persona the main "
                "visible face/character while keeping the source thumbnail's rough pose, scale, and framing."
            )
        hydration = hydration_lanes[i % len(hydration_lanes)] if hydration_lanes else None
        hydration_prompt = ""
        hydration_summary = ""
        if hydration:
            hydration_summary = f"{hydration['label']}: {hydration['value']}"
            hydration_prompt = (
                f" Hydr: {hydration.get('key', 'ev')} — {hydration['value'][:200]}."
            )
        variant = {
            "index": i + 1,
            "name": (
                f"{layout_name} · {i + 1}"
                if layout_name
                else f"{niche_clean.title()} Variant {i + 1}"
            ),
            "headline": headline,
            "subhead": f"{vibe} · {audience.title()}",
            "persona": voice or None,
            "format_key": fmt_key or None,
            "layout_pattern": layout_pattern or None,
            "render_prompt": cap_pikzels_studio_render_prompt(
                (
                    f"Recreate the provided reference thumbnail for a {audience} audience{thumb_layout_suffix} "
                    f"New thumbnail text hook: {headline}. {reference_prompt}{persona_prompt} "
                    f"Use {profile['emotion_bias']} emotion, "
                    f"{profile['text_position']} text placement, {profile['contrast_profile']} contrast, "
                    f"and approximately {profile['face_scale']:.2f} face scale. "
                    f"Creative direction: {creative_directive}"
                    f"{hydration_prompt}"
                )
            ),
            "hydration_focus": hydration["label"] if hydration else "",
            "hydration_signal": hydration["value"] if hydration else "",
            "hydration_summary": hydration_summary,
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
        if hydration_summary:
            variant["suggestions"].insert(0, f"Hydrate this concept with {hydration_summary}.")
        rows.append(variant)
    rows.sort(key=lambda r: float(r.get("ctr_score") or 0), reverse=True)
    return rows


def youtube_reference_thumbnail_url(video_id: str) -> str:
    """Public CDN URL Pikzels can fetch (hqdefault is more reliable than maxres)."""
    vid = (video_id or "").strip()
    if not vid:
        return ""
    return f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"


def _youtube_thumbnail_url_candidates(video_id: str) -> List[str]:
    """Ordered list of YouTube still URLs to try (hosts and qualities vary by rollout)."""
    vid = (video_id or "").strip()
    if not vid:
        return []
    qualities = (
        "maxresdefault.jpg",
        "hqdefault.jpg",
        "mqdefault.jpg",
        "sddefault.jpg",
        "default.jpg",
    )
    hosts = ("https://i.ytimg.com/vi", "https://img.youtube.com/vi")
    out: List[str] = []
    seen: set[str] = set()
    for host in hosts:
        for q in qualities:
            u = f"{host}/{vid}/{q}"
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


def _youtube_thumb_response_to_jpeg_data_url(raw: bytes) -> Optional[str]:
    """
    Turn ytimg CDN bytes into a JPEG ``data:`` URL for Pikzels.

    Rejects HTML error pages, normalizes WebP/PNG via PIL when available, and falls back
    to raw JPEG bytes when PIL is unavailable.
    """
    if not raw or len(raw) < 80:
        return None
    head = raw[:512].lstrip()
    low = head.lower()
    if low.startswith((b"<!doctype html", b"<html", b"<?xml")):
        return None
    try:
        from PIL import Image

        im = Image.open(io.BytesIO(raw))
        im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90, optimize=True)
        out = buf.getvalue()
        if len(out) < 400:
            return None
        b64 = base64.standard_b64encode(out).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        if raw[:2] == b"\xff\xd8" and len(raw) > 400:
            b64 = base64.standard_b64encode(raw).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
        return None


def _looks_like_youtube_thumb_url(url: str) -> bool:
    u = str(url or "").strip()
    if not u.startswith(("http://", "https://")):
        return False
    host = (urlparse(u).hostname or "").lower()
    return host.endswith(("ytimg.com", "youtube.com", "googleusercontent.com", "ggpht.com"))


def _dedupe_urls(urls: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in urls or []:
        u = html.unescape(str(raw or "").strip()).replace("\\u0026", "&")
        if not _looks_like_youtube_thumb_url(u):
            continue
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def _extract_json_object_after(text: str, marker: str) -> Optional[Dict[str, Any]]:
    idx = (text or "").find(marker)
    if idx < 0:
        return None
    start = text.find("{", idx)
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for pos in range(start, len(text)):
        ch = text[pos]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : pos + 1])
                    return parsed if isinstance(parsed, dict) else None
                except Exception:
                    return None
    return None


def _walk_thumbnail_urls(obj: Any, out: List[str]) -> None:
    if isinstance(obj, dict):
        u = obj.get("url")
        if isinstance(u, str) and _looks_like_youtube_thumb_url(u):
            out.append(u)
        for v in obj.values():
            _walk_thumbnail_urls(v, out)
    elif isinstance(obj, list):
        for item in obj[:80]:
            _walk_thumbnail_urls(item, out)


def _youtube_thumbnail_urls_from_watch_html(text: str) -> List[str]:
    """Extract thumbnail candidates from the watch page, not just predictable CDN paths."""
    candidates: List[str] = []
    page = text or ""

    for tag in re.findall(r"<meta\s+[^>]*>", page, flags=re.I):
        ident = re.search(r"""(?:property|name)=["']([^"']+)["']""", tag, flags=re.I)
        if not ident or ident.group(1).lower() not in ("og:image", "twitter:image"):
            continue
        content = re.search(r"""content=["']([^"']+)["']""", tag, flags=re.I)
        if content:
            candidates.append(content.group(1))

    # Schema.org JSON-LD and inline JS often expose thumbnailUrl even when predictable
    # /vi/<id>/hqdefault paths return 404 placeholders.
    for m in re.finditer(r""""thumbnailUrl"\s*:\s*(?:"([^"]+)"|\[\s*"([^"]+)")""", page):
        candidates.append(m.group(1) or m.group(2) or "")

    player = _extract_json_object_after(page, "ytInitialPlayerResponse")
    if player:
        _walk_thumbnail_urls(player, candidates)

    return _dedupe_urls(candidates)


async def _download_youtube_thumb_candidate(
    client: httpx.AsyncClient,
    url: str,
    *,
    referer: str,
) -> str:
    try:
        r = await client.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                ),
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": referer,
                "Origin": "https://www.youtube.com",
            },
        )
        if r.status_code != 200 or not r.content:
            return ""
        ct = (r.headers.get("content-type") or "").lower()
        if "text/html" in ct:
            return ""
        return _youtube_thumb_response_to_jpeg_data_url(r.content) or ""
    except Exception as e:
        _log.debug("youtube thumb candidate failed url=%s err=%s", url[:140], e)
        return ""


async def _youtube_thumbnail_from_watch_page(video_id: str) -> str:
    vid = (video_id or "").strip()
    if not vid:
        return ""
    watch = f"https://www.youtube.com/watch?v={vid}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        async with httpx.AsyncClient(timeout=25.0, follow_redirects=True) as client:
            r = await client.get(watch, headers=headers)
            if r.status_code != 200 or not r.text:
                return ""
            for u in _youtube_thumbnail_urls_from_watch_html(r.text):
                data_url = await _download_youtube_thumb_candidate(client, u, referer=watch)
                if data_url:
                    return data_url
    except Exception as e:
        _log.debug("youtube watch page thumb failed vid=%s: %s", vid[:16], e)
    return ""


async def _youtube_thumbnail_from_data_api(video_id: str) -> str:
    """Optional strongest source when YOUTUBE_DATA_API_KEY is configured."""
    key = (os.environ.get("YOUTUBE_DATA_API_KEY") or "").strip()
    vid = (video_id or "").strip()
    if not key or not vid:
        return ""
    watch = f"https://www.youtube.com/watch?v={vid}"
    try:
        async with httpx.AsyncClient(timeout=18.0, follow_redirects=True) as client:
            r = await client.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params={"part": "snippet", "id": vid, "key": key},
            )
            if r.status_code != 200:
                return ""
            data = r.json()
            items = data.get("items") if isinstance(data, dict) else None
            if not isinstance(items, list) or not items:
                return ""
            thumbs = ((items[0] or {}).get("snippet") or {}).get("thumbnails") or {}
            ordered: List[str] = []
            for name in ("maxres", "standard", "high", "medium", "default"):
                row = thumbs.get(name)
                if isinstance(row, dict) and row.get("url"):
                    ordered.append(str(row["url"]))
            for u in _dedupe_urls(ordered):
                data_url = await _download_youtube_thumb_candidate(client, u, referer=watch)
                if data_url:
                    return data_url
    except Exception as e:
        _log.debug("youtube data api thumb failed vid=%s: %s", vid[:16], e)
    return ""


async def _youtube_thumbnail_from_oembed(video_id: str) -> str:
    """
    Fallback when direct ``i.ytimg.com`` / ``img.youtube.com`` fetches fail.

    YouTube's oEmbed document includes a ``thumbnail_url`` that is usually fetchable with
    the same browser-like headers as normal page embeds.
    """
    vid = (video_id or "").strip()
    if not vid:
        return ""
    watch = f"https://www.youtube.com/watch?v={vid}"
    ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    oembed_headers = {
        "User-Agent": ua,
        "Accept": "application/json",
        "Referer": "https://www.youtube.com/",
    }
    try:
        async with httpx.AsyncClient(timeout=18.0, follow_redirects=True) as client:
            r = await client.get(
                "https://www.youtube.com/oembed",
                params={"url": watch, "format": "json"},
                headers=oembed_headers,
            )
            if r.status_code != 200:
                return ""
            data = r.json()
            if not isinstance(data, dict):
                return ""
            thumb = str(data.get("thumbnail_url") or "").strip()
            if not thumb.startswith("http"):
                return ""
            r2 = await client.get(
                thumb,
                headers={
                    "User-Agent": ua,
                    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": watch,
                    "Origin": "https://www.youtube.com",
                },
            )
            if r2.status_code != 200 or not r2.content:
                return ""
            raw = r2.content
            ct = (r2.headers.get("content-type") or "").lower()
            if "text/html" in ct:
                return ""
            return _youtube_thumb_response_to_jpeg_data_url(raw) or ""
    except Exception as e:
        _log.debug("youtube oembed thumb failed vid=%s: %s", vid[:16], e)
    return ""


async def _youtube_reference_thumbnail_as_data_url(video_id: str) -> str:
    """
    Download the video's default thumbnail on UploadM8 and return ``data:image/jpeg;base64,...``.

    Pikzels frequently returns ``INVALID_IMAGE`` when asked to fetch ``i.ytimg.com`` URLs
    themselves (egress / bot checks). Sending bytes avoids that and fixes identical
    "stale" fallback thumbnails when every recreate variant hit the same failed URL path.
    """
    vid = (video_id or "").strip()
    if not vid:
        return ""
    watch = f"https://www.youtube.com/watch?v={vid}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": watch,
        "Origin": "https://www.youtube.com",
        "Sec-Fetch-Dest": "image",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "cross-site",
    }
    last_note = ""
    try:
        async with httpx.AsyncClient(timeout=35.0, follow_redirects=True) as client:
            for u in _youtube_thumbnail_url_candidates(vid):
                try:
                    r = await client.get(u, headers=headers)
                    if r.status_code != 200 or not r.content:
                        last_note = f"{u} status={r.status_code} len={len(r.content or b'')}"
                        continue
                    raw = r.content
                    ct = (r.headers.get("content-type") or "").lower()
                    if "text/html" in ct:
                        last_note = f"{u} content-type={ct!r}"
                        continue
                    data_url = _youtube_thumb_response_to_jpeg_data_url(raw)
                    if data_url:
                        return data_url
                    last_note = f"{u} not decodable as image len={len(raw)}"
                except httpx.HTTPError as e:
                    last_note = f"{u} err={e}"
                    continue
    except Exception as e:
        _log.debug("youtube thumb download failed vid=%s: %s", vid[:16], e)
    if last_note:
        _log.warning(
            "youtube thumb download exhausted vid=%s — %s",
            (video_id or "")[:16],
            last_note[:220],
        )
    watch_page = await _youtube_thumbnail_from_watch_page(vid)
    if watch_page:
        return watch_page
    data_api = await _youtube_thumbnail_from_data_api(vid)
    if data_api:
        return data_api
    oembed = await _youtube_thumbnail_from_oembed(vid)
    if oembed:
        return oembed
    return ""


def _pikzels_extract_image_url(data: Dict[str, Any]) -> str:
    if not isinstance(data, dict):
        return ""
    for k in (
        "output",
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
            "output",
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


def _jpeg_data_url(raw: bytes) -> str:
    if not raw:
        return ""
    return f"data:image/jpeg;base64,{base64.standard_b64encode(raw).decode('ascii')}"


async def _pikzels_faceswap_image_bytes(
    *,
    image_bytes: Optional[bytes],
    image_url: str,
    face_ref: str,
) -> Tuple[Optional[bytes], int, str]:
    """
    Best-effort identity pass for linked personas.

    Pikzels recreate accepts ``persona``, but real-world outputs can ignore identity when the
    reference/layout prompt is strong. FaceSwap is the follow-up API designed for putting the
    saved person onto an existing thumbnail, so guided persona runs use it after image generation.
    """
    face = str(face_ref or "").strip()
    if not face:
        return None, 0, ""
    payload: Dict[str, Any] = {"format": "16:9"}
    if image_bytes:
        payload["image_base64"] = _jpeg_data_url(image_bytes)[:14_000_000]
    elif str(image_url or "").startswith("http"):
        payload["image_url"] = str(image_url).strip()
    else:
        return None, 0, "missing_generated_image"

    if face.lower().startswith("data:image"):
        payload["face_image_base64"] = face[:14_000_000]
    else:
        payload["face_image"] = face[:8000]
    normalize_url_or_base64(payload, "image_url", "image_base64")

    status, data = await pikzels_v2_post(V2_THUMBNAIL_FACESWAP, payload)
    if status >= 400:
        return None, int(status), _pikzels_api_error_message(data or {})[:500]
    swapped = await _pikzels_response_to_image_bytes(data or {})
    if not swapped or len(swapped) < 2048:
        return None, int(status), "faceswap_no_image_bytes"
    return swapped, int(status), ""


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


def presign_variant_preview_url(r2_key: str, ttl: Optional[int] = None) -> str:
    k = (r2_key or "").strip()
    if not k or not R2_BUCKET_NAME:
        return ""
    try:
        return generate_presigned_download_url(
            k, ttl=int(ttl if ttl is not None else STUDIO_PREVIEW_PRESIGN_TTL_SEC)
        )
    except Exception as e:
        _log.debug("presign_variant_preview_url: %s", e)
        return ""


def attach_preview_urls_to_variants(
    variants: List[Dict[str, Any]], ttl: Optional[int] = None
) -> None:
    """Attach fresh preview_url. Prefer R2 (re-presigned); CDN only as fallback."""
    use_ttl = int(ttl if ttl is not None else STUDIO_PREVIEW_PRESIGN_TTL_SEC)
    for v in variants:
        if not isinstance(v, dict):
            continue
        polish_studio_variant_for_client(v)
        key = str(v.get("preview_r2_key") or "").strip()
        if key:
            v["preview_url"] = presign_variant_preview_url(key, ttl=use_ttl)
            v["preview_storage"] = "r2"
            v["preview_retention_days"] = STUDIO_PREVIEW_RETENTION_DAYS
            continue
        ext = str(v.get("pikzels_cdn_url") or "").strip()
        if ext.startswith("http"):
            v["preview_url"] = ext
            v["preview_storage"] = "pikzels_cdn"


def resolve_variant_cdn_url(variant: Dict[str, Any]) -> str:
    """Best-effort Pikzels CDN URL from stored variant JSON (no network)."""
    if not isinstance(variant, dict):
        return ""
    polish_studio_variant_for_client(variant)
    raw = str(variant.get("pikzels_cdn_url") or "").strip()
    if raw.lower().startswith("https://cdn.pikzels.com/"):
        return raw
    extracted = _pikzels_extract_image_url(variant)
    if str(extracted or "").lower().startswith("https://cdn.pikzels.com/"):
        return str(extracted).strip()
    for k in ("subhead", "engine_text_brief", "engine_error"):
        u = extract_cdn_pikzels_rest_url(str(variant.get(k) or ""))
        if u:
            return u
    return ""


async def backfill_variant_preview_to_r2(
    *,
    user_id: str,
    job_id: str,
    variant_id: str,
    variant: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Free repair: if the variant has no ``preview_r2_key`` but a live CDN URL,
    download once and mirror to R2. No Pikzels regenerate / no wallet debit.

    Returns ``{status, preview_r2_key?, reason?}`` where status is one of:
    ``already_r2``, ``mirrored``, ``cdn_gone``, ``no_cdn``, ``error``.
    """
    v = dict(variant) if isinstance(variant, dict) else {}
    existing = str(v.get("preview_r2_key") or "").strip()
    if existing.startswith("thumbnail-studio/"):
        return {"status": "already_r2", "preview_r2_key": existing, "variant_id": variant_id}

    cdn = resolve_variant_cdn_url(v)
    if not cdn:
        return {"status": "no_cdn", "variant_id": variant_id, "reason": "no_pikzels_cdn_url"}

    img = await _download_bytes(cdn)
    if not img or len(img) < 2048:
        return {
            "status": "cdn_gone",
            "variant_id": variant_id,
            "reason": "cdn_unavailable_or_expired",
        }

    idx = int(v.get("index") or 1)
    r2_key = f"thumbnail-studio/previews/{user_id}/{job_id}/variant_{idx}.jpg"
    # Avoid collisions when rank_idx differs from index — include variant id suffix.
    if variant_id:
        short = str(variant_id).replace("-", "")[:12]
        r2_key = f"thumbnail-studio/previews/{user_id}/{job_id}/variant_{idx}_{short}.jpg"
    try:
        await _r2_put_bytes(r2_key, img, "image/jpeg")
    except Exception as e:
        _log.warning("backfill R2 put failed variant=%s: %s", variant_id, e)
        return {"status": "error", "variant_id": variant_id, "reason": str(e)[:300]}

    v["preview_r2_key"] = r2_key
    v["pikzels_cdn_url"] = cdn
    v["preview_backfilled_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    v["preview_storage"] = "r2"
    return {
        "status": "mirrored",
        "variant_id": variant_id,
        "preview_r2_key": r2_key,
        "variant": v,
    }


async def backfill_job_variants_to_r2(
    *,
    user_id: str,
    job_id: str,
    variants: List[Dict[str, Any]],
    max_parallel: int = 3,
) -> Dict[str, Any]:
    """
    Free CDN→R2 backfill for a job's variants. Mutates variants that succeed
    (sets ``preview_r2_key``). Caller must persist updated ``variant_json``.
    """
    sem = asyncio.Semaphore(max(1, min(int(max_parallel or 3), 4)))
    summary: Dict[str, Any] = {
        "job_id": job_id,
        "mirrored": 0,
        "already_r2": 0,
        "cdn_gone": 0,
        "no_cdn": 0,
        "error": 0,
        "updated_variants": [],
        "free": True,
        "note": (
            "No wallet debit. Only works while Pikzels CDN still has the image; "
            "expired CDN links need a paid regenerate."
        ),
    }

    async def one(v: Dict[str, Any]) -> None:
        async with sem:
            vid = str(v.get("variant_id") or "")
            result = await backfill_variant_preview_to_r2(
                user_id=user_id,
                job_id=job_id,
                variant_id=vid,
                variant=v,
            )
            st = str(result.get("status") or "error")
            summary[st] = int(summary.get(st) or 0) + 1
            if st == "mirrored" and isinstance(result.get("variant"), dict):
                v.update(result["variant"])
                v["variant_id"] = vid
                summary["updated_variants"].append(vid)

    await asyncio.gather(*[one(v) for v in variants if isinstance(v, dict)])
    return summary


async def _pikzels_engine_text_brief(
    *,
    source_title: str,
    niche: str,
    topic: str,
) -> str:
    """Single v2 text call — shared creative direction for all variants (no extra wallet debit)."""
    if not resolve_public_api_key():
        return ""
    title_line = (source_title or topic or "untitled").strip() or "untitled"
    topic_clean = (topic or "").strip()
    steer = ""
    if topic_clean and topic_clean.lower() != title_line.lower():
        steer = f" Additional creative steer from the editor: {topic_clean[:280]}."
    prompt = (
        f"YouTube thumbnail brief. Video title: {title_line}. "
        f"Niche: {niche or 'general'}.{steer} "
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


def build_studio_pikzels_prompt(
    variant: Dict[str, Any],
    *,
    hydration_context: Optional[Dict[str, Any]] = None,
    engine_text_brief: str = "",
    niche: str = "general",
) -> str:
    """
    Studio recreate: merge layout/render notes with canonical hydration evidence
    (geo, speed/OSD, music, speech, vision, Trill) via ``_build_pikzels_v2_prompt``.
    """
    from services.hydration_from_upload_row import flat_context_to_hydration_payload
    from services.hydration_payload import hydration_brief_strings_compact
    from stages.pikzels_api import _build_pikzels_v2_prompt, clamp_pikzels_image_prompt

    layout_notes = cap_pikzels_studio_render_prompt(str(variant.get("render_prompt") or ""))
    hp = flat_context_to_hydration_payload(hydration_context)
    ev = hp.get("evidence") if isinstance(hp, dict) and isinstance(hp.get("evidence"), dict) else {}
    has_evidence = any(isinstance(v, dict) and v for v in ev.values())
    if not has_evidence:
        return clamp_pikzels_image_prompt(layout_notes)

    brief_dict: Dict[str, Any] = {
        "selected_headline": str(variant.get("headline") or "").strip(),
        "pikzels_text_brief": (engine_text_brief or "").strip(),
        "notes": layout_notes[:420] if layout_notes else "",
    }
    for key, value in hydration_brief_strings_compact(hp).items():
        brief_dict.setdefault(key, value)
    brief_dict.setdefault("fusion_summary", str(hp.get("fusion_summary") or ""))
    brief_dict.setdefault("hydration_story", str(hp.get("hydration_story") or ""))

    prompt = _build_pikzels_v2_prompt(
        brief_dict,
        category=(niche or "general").strip().lower() or "general",
        platform="youtube",
        hydration_payload=hp,
    )
    if layout_notes and "Layout:" not in prompt:
        prompt = f"{prompt} Layout: {layout_notes[:200]}".strip()
    return clamp_pikzels_image_prompt(cap_pikzels_studio_render_prompt(prompt))


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
    closeness: int = 55,
    persona_face_ref: str = "",
    reference_image_data_url: str = "",
    hydration_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    When PIKZELS_API_KEY (or THUMB_RENDER_API_KEY) is set, run Pikzels v2 text + image + score
    and store previews on R2. Wallet is not debited again — studio recreate already charged the user.

    ``closeness`` maps to each recreate call's ``image_weight`` (low / medium / high) so the
    Studio slider matches Pikzels' reference-strength control; prompts still carry UploadM8
    layout and hydration text from ``generate_recreate_variants``.

    When a linked persona is selected, ``persona`` is sent on the create-from-image call and
    ``/v2/thumbnail/faceswap`` runs as a best-effort identity pass before storing the preview.
    """
    if not variants or not resolve_public_api_key():
        return variants
    ref = youtube_reference_thumbnail_url(youtube_video_id)
    if not ref:
        for v in variants:
            if isinstance(v, dict):
                v.setdefault("engine_status", "skipped_no_video_id")
        return variants

    # One download per job — shared by all variants (Pikzels XOR: base64 preferred over URL).
    # Do not fall back to ytimg ``image_url``: Pikzels often cannot fetch it and may
    # generate unrelated stock imagery or return INVALID_IMAGE.
    ref_data_url = str(reference_image_data_url or "").strip()
    if not ref_data_url:
        ref_data_url = await _youtube_reference_thumbnail_as_data_url(youtube_video_id)
    if not ref_data_url:
        _log.warning(
            "no local YouTube reference bytes for vid=%s — skipping Pikzels recreate calls",
            (youtube_video_id or "")[:16],
        )
        for v in variants:
            if isinstance(v, dict):
                v["engine_status"] = "youtube_reference_unavailable"
                v["engine_error"] = (
                    "UploadM8 could not download a usable public thumbnail for this YouTube video."
                )
        return variants

    brief = await _pikzels_engine_text_brief(
        source_title=source_title,
        niche=niche,
        topic=topic,
    )
    image_weight = closeness_to_pikzels_image_weight(closeness)
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
                merged = (
                    f"{pv.get('subhead') or ''} · {(brief_visible or brief)[:120]}".strip(" ·")
                )
                pv["subhead"] = strip_pikzels_urls_from_text(merged)
            prompt = build_studio_pikzels_prompt(
                pv,
                hydration_context=hydration_context,
                engine_text_brief=str(brief or ""),
                niche=niche or "general",
            )
            pv["pikzels_prompt_len"] = len(prompt)
            pv["pikzels_prompt_has_hydration"] = bool(hydration_context)
            payload: Dict[str, Any] = {
                "prompt": prompt,
                "image_weight": image_weight,
                "model": "pkz_4",
                "format": "16:9",
            }
            if ref_data_url:
                payload["image_base64"] = ref_data_url[:14_000_000]
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
            # Belt-and-suspenders: never send style + persona together.
            resolve_pikzels_persona_style_xor(payload)
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
            if persona_face_ref and (img_bytes or salvage.startswith("http")):
                swapped, fs_status, fs_error = await _pikzels_faceswap_image_bytes(
                    image_bytes=img_bytes,
                    image_url=salvage,
                    face_ref=persona_face_ref,
                )
                pv["pikzels_faceswap_http_status"] = fs_status
                if swapped:
                    img_bytes = swapped
                    pv["faceswap_applied"] = True
                    pv.pop("pikzels_cdn_url", None)
                elif fs_error:
                    pv["faceswap_error"] = fs_error[:500]
            if not img_bytes or len(img_bytes) < 2048:
                if salvage.startswith("http"):
                    # Prefer durable R2 mirror so Saved runs survive CDN expiry.
                    mirrored = await _download_bytes(salvage)
                    if mirrored and len(mirrored) >= 2048:
                        r2_key = (
                            f"thumbnail-studio/previews/{user_id}/{job_id}/"
                            f"variant_{int(pv.get('index') or idx + 1)}.jpg"
                        )
                        try:
                            await _r2_put_bytes(r2_key, mirrored, "image/jpeg")
                            pv["preview_r2_key"] = r2_key
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
                        except Exception as e:
                            _log.warning("R2 CDN mirror upload failed: %s", e)
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
                if salvage.startswith("http") and not pv.get("faceswap_applied"):
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
            signed = presign_variant_preview_url(r2_key)
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
