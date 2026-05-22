"""
User platform badge / accent colors from ``user_color_preferences``.

Used by the thumbnail pipeline (template + Pikzels prompts) and available in
``ctx.user_settings`` after ``load_user_settings`` merges the color row.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

DEFAULT_PLATFORM_COLORS: Dict[str, str] = {
    "tiktok": "#000000",
    "youtube": "#FF0000",
    "instagram": "#E4405F",
    "facebook": "#1877F2",
    "accent": "#3B82F6",
}

_PLATFORM_KEYS = ("tiktok", "youtube", "instagram", "facebook")
_DB_KEY_MAP = {
    "tiktok": "tiktok_color",
    "youtube": "youtube_color",
    "instagram": "instagram_color",
    "facebook": "facebook_color",
    "accent": "accent_color",
}


def _normalize_hex(value: Any, fallback: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return fallback
    if not raw.startswith("#"):
        raw = f"#{raw}"
    body = raw[1:]
    if len(body) == 3 and all(c in "0123456789abcdefABCDEF" for c in body):
        body = "".join(c * 2 for c in body)
    if len(body) != 6 or not all(c in "0123456789abcdefABCDEF" for c in body):
        return fallback
    return f"#{body.upper()}"


def normalize_platform_colors(raw: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """Return a complete platform color map with validated hex values."""
    src = dict(raw or {})
    out = dict(DEFAULT_PLATFORM_COLORS)
    nested = src.get("platform_colors") or src.get("platformColors")
    if isinstance(nested, dict):
        src = {**src, **nested}
    for key in _PLATFORM_KEYS + ("accent",):
        db_key = _DB_KEY_MAP[key]
        val = src.get(key)
        if val is None:
            val = src.get(db_key)
        out[key] = _normalize_hex(val, DEFAULT_PLATFORM_COLORS[key])
    return out


def resolve_platform_colors(settings: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """Read platform colors from worker/user settings dict."""
    if not isinstance(settings, Mapping):
        return dict(DEFAULT_PLATFORM_COLORS)
    nested = settings.get("platform_colors") or settings.get("platformColors")
    if isinstance(nested, Mapping):
        return normalize_platform_colors(nested)
    flat: Dict[str, Any] = {}
    for key in _PLATFORM_KEYS + ("accent",):
        flat[_DB_KEY_MAP[key]] = settings.get(_DB_KEY_MAP[key])
    if any(flat.values()):
        return normalize_platform_colors(flat)
    return dict(DEFAULT_PLATFORM_COLORS)


def platform_color_for(settings: Optional[Mapping[str, Any]], platform: str) -> str:
    if isinstance(settings, Mapping) and any(k in settings for k in _PLATFORM_KEYS):
        colors = normalize_platform_colors(settings)
    else:
        colors = resolve_platform_colors(settings)
    key = str(platform or "").strip().lower()
    if key == "google":
        key = "youtube"
    if key == "meta":
        key = "facebook"
    return colors.get(key) or DEFAULT_PLATFORM_COLORS.get(key, "#6B7280")


def contrasting_text_color(bg_hex: str) -> str:
    """Pick white or near-black text for a solid badge background."""
    body = str(bg_hex or "").strip().lstrip("#")
    if len(body) == 3:
        body = "".join(c * 2 for c in body)
    try:
        r = int(body[0:2], 16)
        g = int(body[2:4], 16)
        b = int(body[4:6], 16)
    except (ValueError, IndexError):
        return "#FFFFFF"
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#FFFFFF" if luminance < 0.55 else "#1A1A1A"
