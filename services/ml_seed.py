"""
Synthetic seed bootstrap for the UploadM8 ML engine.

Cold-start safety valve: when real labeled data has no class variance, an
env-gated seed lets the pipeline train end-to-end and produce a non-trivial
model *locally* for validation. Seeded rows are appended only to the local
parquet used for training — they are NEVER pushed to a Hugging Face dataset
bucket, and seeded models are not promoted (see services/ml_engine.py).

Kept dependency-light (pandas + stdlib random) and governed by the feature
registry so seed columns always match the real schema.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List

from services.ml_feature_registry import active_cat, active_num, label, label_fallback

_TIERS = ["free", "starter", "creator_lite", "creator_pro", "studio", "agency"]
_PLATFORMS = ["tiktok", "youtube", "instagram", "facebook"]
_CATEGORIES = ["cars", "gaming", "music", "comedy", "fitness", "food", "travel"]
_CAPTION_STYLES = ["story", "listicle", "hook", "question"]
_TONES = ["authentic", "hype", "calm", "bold"]


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _cat_value(name: str, r: random.Random) -> str:
    if name == "subscription_tier":
        return r.choice(_TIERS)
    if name == "platform":
        return r.choice(_PLATFORMS)
    if name == "content_category":
        return r.choice(_CATEGORIES)
    if name == "caption_style":
        return r.choice(_CAPTION_STYLES)
    if name in ("caption_tone", "caption_voice"):
        return r.choice(_TONES)
    return r.choice(["a", "b", "c"])


def _seed_rows(loop: str, n: int, base_seed: int = 1337) -> List[dict]:
    r = _rng(base_seed)
    num_cols = active_num(loop)
    cat_cols = active_cat(loop)
    target = label(loop)
    fallback = label_fallback(loop)
    rows: List[dict] = []
    for i in range(int(n)):
        positive = 1 if i % 2 == 0 else 0
        row: dict = {}
        for c in num_cols:
            # Give positives a mild upward shift so the model finds signal.
            base = r.random()
            row[c] = round(base * 100.0 + (35.0 if positive else 0.0), 4)
        for c in cat_cols:
            row[c] = _cat_value(c, r)
        row[target] = positive
        if fallback:
            row[fallback] = positive if r.random() > 0.2 else 1 - positive
        row["is_seed"] = 1
        rows.append(row)
    return rows


def _append(path: str, rows: List[dict]) -> int:
    import pandas as pd

    seed_df = pd.DataFrame.from_records(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():
        try:
            existing = pd.read_parquet(out)
            combined = pd.concat([existing, seed_df], ignore_index=True)
        except Exception:
            combined = seed_df
    else:
        combined = seed_df
    combined.to_parquet(out, index=False)
    return len(seed_df)


def seed_promo_parquet(path: str, n: int = 60) -> int:
    """Append synthetic promo rows (both label classes) to the local parquet."""
    return _append(path, _seed_rows("promo", n, base_seed=2026))


def seed_content_parquet(path: str, n: int = 60) -> int:
    """Append synthetic content-success rows (both label classes) to the local parquet."""
    return _append(path, _seed_rows("content", n, base_seed=4099))
