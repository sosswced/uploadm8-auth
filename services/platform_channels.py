"""
Publish destinations: canonical platform slugs, display names, and analytics aliases.

Upload rows store short slugs in ``uploads.platforms[]`` (e.g. ``instagram``, ``facebook``).
Marketing and UI refer to **Instagram Reels** and **Facebook Reels** — same integration,
clearer naming. Optional analytics filters can narrow the catalog slice to reel rows only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

# Values allowed in uploads.platforms[] and platform_tokens.platform (canonical).
CANONICAL_PUBLISH_PLATFORMS: FrozenSet[str] = frozenset(
    {"tiktok", "youtube", "instagram", "facebook", "twitter", "linkedin", "threads"}
)

# alias_lower -> (canonical_platform, pci_content_kind_filter or None)
# pci_content_kind_filter limits platform_content_items rows in canonical engagement rollup.
_ANALYTICS_ALIASES: Dict[str, Tuple[str, Optional[str]]] = {
    "instagram_reels": ("instagram", "reel"),
    "ig_reels": ("instagram", "reel"),
    "instagram-reels": ("instagram", "reel"),
    "instagramreels": ("instagram", "reel"),
    "facebook_reels": ("facebook", "reel"),
    "fb_reels": ("facebook", "reel"),
    "facebook-reels": ("facebook", "reel"),
    "facebookreels": ("facebook", "reel"),
    "youtube_shorts": ("youtube", None),
    "yt_shorts": ("youtube", None),
}

# Human-facing labels for UI, exports, and API (canonical slug -> default label).
PLATFORM_DISPLAY_LABELS: Dict[str, str] = {
    "tiktok": "TikTok",
    "youtube": "YouTube",
    "instagram": "Instagram Reels",
    "facebook": "Facebook Reels",
    "twitter": "X (Twitter)",
    "linkedin": "LinkedIn",
    "threads": "Threads",
}


@dataclass(frozen=True)
class AnalyticsPlatformFilter:
    """Resolved GET /api/analytics/overview ?platform= filter."""

    platform: Optional[str]
    """Canonical slug for uploads.platforms[] / SQL, or None for all."""

    catalog_content_kind: Optional[str]
    """If ``reel``, limit ``platform_content_items`` to reel rows for that platform(s)."""

    display_name: str
    """Label for dashboards and API responses."""

    raw_query: Optional[str]
    """Normalized query token the client sent (e.g. ``instagram_reels``), or None for all."""


def platform_display_label(canonical: str) -> str:
    """Default product label for a canonical platform slug."""
    return PLATFORM_DISPLAY_LABELS.get(
        canonical.lower().strip(),
        canonical.replace("_", " ").title(),
    )


def list_analytics_platform_query_values() -> List[str]:
    """All accepted ?platform= tokens (for 400 messages and OpenAPI)."""
    base = sorted(CANONICAL_PUBLISH_PLATFORMS)
    extra = sorted(set(_ANALYTICS_ALIASES.keys()))
    return sorted(set(base + extra + ["all"]))


def resolve_analytics_platform_filter(platform: Optional[str]) -> AnalyticsPlatformFilter:
    """
    Map ``?platform=`` to canonical slug + optional PCI reel-only slice.

    * ``all`` / empty → all platforms, no PCI kind filter.
    * ``instagram_reels`` → canonical ``instagram``, catalog rows with reel kind only.
    * ``instagram`` → canonical ``instagram``, all catalog kinds for that platform.
    """
    if platform is None:
        return AnalyticsPlatformFilter(
            platform=None,
            catalog_content_kind=None,
            display_name="All platforms",
            raw_query=None,
        )
    p = str(platform).strip().lower()
    if not p or p == "all":
        return AnalyticsPlatformFilter(
            platform=None,
            catalog_content_kind=None,
            display_name="All platforms",
            raw_query=None,
        )

    if p in _ANALYTICS_ALIASES:
        canon, ck = _ANALYTICS_ALIASES[p]
        return AnalyticsPlatformFilter(
            platform=canon,
            catalog_content_kind=ck,
            display_name=platform_display_label(canon),
            raw_query=p,
        )

    if p in CANONICAL_PUBLISH_PLATFORMS:
        return AnalyticsPlatformFilter(
            platform=p,
            catalog_content_kind=None,
            display_name=platform_display_label(p),
            raw_query=p,
        )

    raise ValueError("invalid_platform")
