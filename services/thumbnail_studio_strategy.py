"""Shared readers for Thumbnail Studio default strategy stored on users.preferences."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def read_thumbnail_studio_default_strategy(prefs: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Resolve the Studio "make default" strategy object from preference blobs.

    Canonical keys (written by Thumbnail Studio feedback):
      thumbnailStudioDefaultStrategy / thumbnail_studio_default_strategy
    Legacy aliases (older readers / niche merge):
      thumbnailDefaultStrategy / thumbnail_default_strategy
    """
    if not isinstance(prefs, Mapping):
        return {}
    for key in (
        "thumbnailStudioDefaultStrategy",
        "thumbnail_studio_default_strategy",
        "thumbnailDefaultStrategy",
        "thumbnail_default_strategy",
    ):
        raw = prefs.get(key)
        if isinstance(raw, dict) and raw:
            return dict(raw)
    return {}


def strategy_audience_niche(prefs: Optional[Mapping[str, Any]], default: str = "general") -> str:
    nested = read_thumbnail_studio_default_strategy(prefs)
    niche = nested.get("audience_niche") or nested.get("niche")
    if niche is None and isinstance(prefs, Mapping):
        niche = prefs.get("audience_niche") or prefs.get("audienceNiche")
    s = str(niche or default).strip().lower() or default
    return s


def strategy_summary_line(prefs: Optional[Mapping[str, Any]]) -> Optional[str]:
    """One-line human summary for Upload/Settings UI."""
    strat = read_thumbnail_studio_default_strategy(prefs)
    if not strat:
        return None
    parts = []
    layout = (
        strat.get("layout_name")
        or strat.get("layout_pattern")
        or strat.get("format_key")
        or strat.get("layout")
    )
    if layout:
        parts.append(f"layout {str(layout).replace('_', ' ')}")
    niche = strat.get("audience_niche") or strat.get("niche")
    if niche:
        parts.append(f"niche {niche}")
    yt = strat.get("reference_youtube_url") or strat.get("youtube_url")
    if yt:
        parts.append("YouTube ref set")
    persona = strat.get("persona_id") or strat.get("persona_name")
    if persona:
        parts.append("persona linked")
    if not parts:
        return "Studio default strategy saved"
    return " · ".join(parts)
