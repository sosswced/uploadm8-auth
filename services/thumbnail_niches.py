"""Canonical audience / niche options for Thumbnail Studio and upload thumbnails."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# (value, label) — single source for API + UI population
THUMBNAIL_NICHE_OPTIONS: Tuple[Tuple[str, str], ...] = (
    ("general", "General"),
    ("gaming", "Gaming"),
    ("finance", "Finance"),
    ("education", "Education"),
    ("automotive", "Automotive"),
    ("lifestyle", "Lifestyle"),
    ("comedy", "Comedy / Memes"),
    ("podcast", "Podcast / Interview"),
    ("music", "Music / Artist"),
    ("sports", "Sports"),
    ("tech", "Tech / Product"),
    ("beauty", "Beauty / Fashion"),
    ("food", "Food / Cooking"),
    ("travel", "Travel / Outdoors"),
    ("fitness", "Fitness / Health"),
    ("true_crime", "True Crime / Documentary"),
    ("real_estate", "Real Estate"),
    ("business", "Business / Creator Economy"),
    ("news", "News / Commentary"),
)

_VALID_NICHES = frozenset(v for v, _ in THUMBNAIL_NICHE_OPTIONS)


def normalize_niche(value: Optional[str], *, default: str = "general") -> str:
    key = (value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if key in _VALID_NICHES:
        return key
    return default if default in _VALID_NICHES else "general"


def niche_options_payload() -> List[Dict[str, str]]:
    return [{"value": v, "label": lbl} for v, lbl in THUMBNAIL_NICHE_OPTIONS]
