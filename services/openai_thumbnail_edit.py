"""OpenAI image-edit thumbnail fallback (footage-anchored, never pure generation).

Runs between Pikzels and the PIL template in the styled render order: when the
studio renderer fails (402 credits, HTTP errors, missing aspect leaders), the
sharpness-best REAL frame is edited into a styled cover via ``images.edit``
instead of degrading straight to the flat text-overlay template.

Guardrails (plan Phase 3):
  * ``OPENAI_THUMBNAIL_EDIT_ENABLED`` env kill-switch — off means byte-identical
    legacy behavior (Pikzels → template).
  * Tier gate mirrors Pikzels (``can_ai_thumbnail_styling`` entitlement).
  * Aspect-leader + cache pattern: max ONE live call per aspect (16:9 / 9:16),
    so an upload never spends more than 2 edit calls.
  * The prompt carries the content-identity ``do_not_invent`` contract — the
    edit stylizes what is on the frame, it never adds people, text-unrelated
    props, or unverified numbers.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("uploadm8-worker.thumbnail.openai-edit")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_IMAGE_EDIT_MODEL = os.environ.get("OPENAI_THUMBNAIL_EDIT_MODEL", "gpt-image-1")
OPENAI_IMAGE_EDIT_TIMEOUT_SEC = float(os.environ.get("OPENAI_THUMBNAIL_EDIT_TIMEOUT_SEC", "90"))
MAX_EDIT_CALLS_PER_UPLOAD = 2

_ASPECT_TO_SIZE = {"16:9": "1536x1024", "9:16": "1024x1536"}
_PLATFORM_CANVAS = {
    "youtube": (1280, 720),
    "instagram": (720, 1280),
    "facebook": (720, 1280),
    "tiktok": (720, 1280),
}
MIN_COVER_BYTES = 2048


def openai_thumbnail_edit_enabled() -> bool:
    """Kill-switch + key check. Off ⇒ legacy Pikzels → template behavior."""
    flag = os.environ.get("OPENAI_THUMBNAIL_EDIT_ENABLED", "").strip().lower()
    return flag in ("1", "true", "yes", "on") and bool(OPENAI_API_KEY)


def openai_thumbnail_edit_eligible(us: Dict[str, Any], entitlements: Any) -> bool:
    """Enabled + same tier gate as Pikzels (AI thumbnail styling entitlement)."""
    if not openai_thumbnail_edit_enabled():
        return False
    return bool(getattr(entitlements, "can_ai_thumbnail_styling", False) if entitlements else False)


def build_openai_edit_prompt(
    brief: Dict[str, Any],
    identity: Optional[Dict[str, Any]] = None,
    *,
    platform: str = "",
) -> str:
    """Styling prompt from the sanitized brief + content-identity contract.

    Everything the model may emphasize comes from verified evidence (headline,
    hero facts); everything it must not fabricate comes from ``do_not_invent``.
    """
    ident = identity or {}
    headline = str(brief.get("selected_headline") or "").strip()
    color_mood = str(brief.get("color_mood") or "").strip()
    subject = str(ident.get("subject") or "").strip()

    facts: List[str] = []
    for f in (ident.get("hero_facts") or [])[:3]:
        if isinstance(f, dict) and f.get("text"):
            facts.append(str(f["text"]))

    lines: List[str] = [
        "Transform this real video frame into a scroll-stopping social media cover.",
        "Keep the ACTUAL scene and composition — enhance color grading, contrast,",
        "lighting drama, and clarity like a professional thumbnail designer.",
    ]
    if subject:
        lines.append(f"The footage shows: {subject}.")
    if facts:
        lines.append("Emphasize what is genuinely there: " + "; ".join(facts) + ".")
    if headline:
        lines.append(
            f'Add the headline text "{headline}" in bold, high-contrast sans-serif '
            "lettering with a subtle dark outline, positioned so it never covers "
            "the main subject."
        )
    if color_mood:
        lines.append(f"Color mood: {color_mood.replace('_', ' ')}.")
    dni = [str(d) for d in (ident.get("do_not_invent") or [])[:4]]
    dni.append("do not add people, faces, watermarks, logos, or extra text beyond the headline")
    dni.append("do not invent objects or scenery that are not in the frame")
    lines.append("STRICT RULES: " + "; ".join(dni) + ".")
    return " ".join(lines)


async def generate_openai_edited_cover(
    base_path: Path,
    prompt: str,
    aspect_fmt: str,
    out_raw_path: Path,
    *,
    upload_id: str = "",
) -> bool:
    """One live ``images.edit`` call — edits the real frame at the given aspect.

    Saves the raw edited image (PNG) to ``out_raw_path``. Fail-soft: any error
    returns False and the caller falls through to the PIL template.
    """
    if not OPENAI_API_KEY:
        return False
    size = _ASPECT_TO_SIZE.get(aspect_fmt, "1536x1024")
    try:
        from stages.outbound_rl import outbound_slot

        with open(base_path, "rb") as fh:
            frame_bytes = fh.read()
        async with outbound_slot("openai"):
            async with httpx.AsyncClient(timeout=OPENAI_IMAGE_EDIT_TIMEOUT_SEC) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/images/edits",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    data={
                        "model": OPENAI_IMAGE_EDIT_MODEL,
                        "prompt": prompt[:4000],
                        "size": size,
                        "n": "1",
                    },
                    files={"image": ("frame.jpg", frame_bytes, "image/jpeg")},
                )
        if resp.status_code != 200:
            body = (resp.text or "")[:300]
            logger.warning(
                "[thumb-openai-edit] HTTP %s upload=%s: %s", resp.status_code, upload_id, body
            )
            return False
        data = resp.json().get("data") or []
        b64 = (data[0] or {}).get("b64_json") if data else None
        if not b64:
            logger.warning("[thumb-openai-edit] empty image payload upload=%s", upload_id)
            return False
        out_raw_path.write_bytes(base64.b64decode(b64))
        return out_raw_path.exists() and out_raw_path.stat().st_size >= MIN_COVER_BYTES
    except Exception as e:  # fail-soft by contract — template remains the last resort
        logger.warning("[thumb-openai-edit] call failed (non-fatal) upload=%s: %s", upload_id, e)
        return False


def finalize_platform_cover(raw_path: Path, platform: str, out_path: Path) -> bool:
    """Cover-crop the cached edited image onto the platform canvas as JPEG."""
    try:
        from PIL import Image
    except ImportError:
        logger.warning("[thumb-openai-edit] Pillow not installed — cannot finalize cover")
        return False
    target_w, target_h = _PLATFORM_CANVAS.get(str(platform or "").lower(), (1280, 720))
    try:
        img = Image.open(raw_path).convert("RGB")
        iw, ih = img.size
        scale = max(target_w / iw, target_h / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        x0, y0 = (nw - target_w) // 2, (nh - target_h) // 2
        img = img.crop((x0, y0, x0 + target_w, y0 + target_h))
        img.save(out_path, "JPEG", quality=90)
        return out_path.exists() and out_path.stat().st_size >= MIN_COVER_BYTES
    except Exception as e:
        logger.warning("[thumb-openai-edit] finalize failed: %s", e)
        return False


__all__ = [
    "MAX_EDIT_CALLS_PER_UPLOAD",
    "openai_thumbnail_edit_enabled",
    "openai_thumbnail_edit_eligible",
    "build_openai_edit_prompt",
    "generate_openai_edited_cover",
    "finalize_platform_cover",
]
