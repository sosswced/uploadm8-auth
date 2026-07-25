"""Learned hero-fact class priors per content cluster (Phase 5).

When enough publish outcomes exist, ``rebuild_hero_fact_priors`` ranks which
hero-fact classes win engagement for each ``identity_domain_tag`` cluster.
The ranking feeds ``_hero_fact_headlines`` so gardening/food/travel/driving
each prefer the fact class that historically converts — without hardcoding.

Until a priors file exists (or row counts are too low), callers get an empty
prior map and fall through to the identity's native ranking. Safe no-op.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("uploadm8.hero_fact_priors")

DEFAULT_PRIORS_PATH = Path(__file__).resolve().parents[1] / "data" / "ml" / "hero_fact_priors_v1.json"
MIN_ROWS_PER_CLUSTER = 25

# Static bootstrap priors used only when no learned file exists — domain-agnostic
# class preference order, NOT keyword lists. Speed is intentionally mid-pack so
# it never dominates non-driving content before learned data arrives.
_BOOTSTRAP_GLOBAL: List[str] = [
    "landmark",
    "logo",
    "place",
    "count",
    "entity",
    "on_screen_text",
    "music",
    "transcript",
    "speed",
    "sound",
]


def _load_priors(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or DEFAULT_PRIORS_PATH
    try:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("version"):
                return data
    except Exception as e:
        logger.debug("hero_fact_priors load failed: %s", e)
    return {
        "version": 1,
        "global": list(_BOOTSTRAP_GLOBAL),
        "clusters": {},
        "row_counts": {},
        "source": "bootstrap",
    }


def class_rank_for_cluster(
    domain_tag: str,
    *,
    priors: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Ordered hero-fact class preference for a content cluster."""
    data = priors if isinstance(priors, dict) else _load_priors()
    tag = str(domain_tag or "").strip().lower()
    clusters = data.get("clusters") if isinstance(data.get("clusters"), dict) else {}
    if tag and isinstance(clusters.get(tag), list) and clusters[tag]:
        return [str(c) for c in clusters[tag] if c]
    global_order = data.get("global") if isinstance(data.get("global"), list) else _BOOTSTRAP_GLOBAL
    return [str(c) for c in global_order if c]


def rank_hero_facts(
    hero_facts: Sequence[Dict[str, Any]],
    *,
    domain_tag: str = "",
    priors: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Re-order hero facts using learned (or bootstrap) class priors.

    Stable within a class: original relative order is preserved (identity score
    already ranked them). Facts with unknown classes sort last.
    """
    order = class_rank_for_cluster(domain_tag, priors=priors)
    index = {c: i for i, c in enumerate(order)}

    def _key(f: Dict[str, Any]) -> Tuple[int, float]:
        cls = str(f.get("class") or "").strip().lower()
        # Unknown classes after known ones; within a class keep identity score.
        try:
            score = -float(f.get("score") or 0)
        except (TypeError, ValueError):
            score = 0.0
        return (index.get(cls, len(order) + 1), score)

    return sorted([f for f in hero_facts if isinstance(f, dict)], key=_key)


def rebuild_hero_fact_priors(
    rows: Sequence[Dict[str, Any]],
    *,
    out_path: Optional[Path] = None,
    min_rows: int = MIN_ROWS_PER_CLUSTER,
) -> Dict[str, Any]:
    """Aggregate publish outcomes → per-cluster hero-fact class ranking.

    Each row should carry:
      * ``identity_domain_tag`` (cluster key)
      * ``identity_headline_class`` (which class was used on the cover)
      * ``is_hot`` or ``views_per_day`` / ``hotness_score`` (outcome)

    Clusters below ``min_rows`` inherit the global ranking. Writes JSON to
    ``data/ml/hero_fact_priors_v1.json`` by default so the next thumbnail stage
    picks it up with no deploy flag.
    """
    from collections import defaultdict

    # cluster -> class -> (wins, weight)
    cluster_wins: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    global_wins: Dict[str, float] = defaultdict(float)
    row_counts: Dict[str, int] = defaultdict(int)

    for row in rows:
        if not isinstance(row, dict):
            continue
        tag = str(row.get("identity_domain_tag") or "").strip().lower()
        cls = str(row.get("identity_headline_class") or row.get("identity_hero_fact_class") or "").strip().lower()
        if not cls:
            continue
        try:
            weight = float(row.get("hotness_score") or row.get("views_per_day") or 0)
        except (TypeError, ValueError):
            weight = 0.0
        if row.get("is_hot"):
            weight = max(weight, 1.0)
        weight = max(weight, 0.1)
        global_wins[cls] += weight
        if tag:
            cluster_wins[tag][cls] += weight
            row_counts[tag] += 1

    def _ranked(wins: Dict[str, float]) -> List[str]:
        ranked = sorted(wins.items(), key=lambda kv: (-kv[1], kv[0]))
        ordered = [c for c, _ in ranked]
        # Append bootstrap classes not yet seen so the order stays complete.
        for c in _BOOTSTRAP_GLOBAL:
            if c not in ordered:
                ordered.append(c)
        return ordered

    clusters_out: Dict[str, List[str]] = {}
    for tag, wins in cluster_wins.items():
        if row_counts[tag] >= min_rows:
            clusters_out[tag] = _ranked(wins)

    payload = {
        "version": 1,
        "global": _ranked(global_wins) if global_wins else list(_BOOTSTRAP_GLOBAL),
        "clusters": clusters_out,
        "row_counts": dict(row_counts),
        "source": "learned" if clusters_out or global_wins else "bootstrap",
        "min_rows": int(min_rows),
    }
    dest = out_path or DEFAULT_PRIORS_PATH
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        logger.info(
            "hero_fact_priors wrote %s clusters=%s source=%s",
            dest,
            list(clusters_out.keys()),
            payload["source"],
        )
    except Exception as e:
        logger.warning("hero_fact_priors write failed: %s", e)
    return payload


__all__ = [
    "DEFAULT_PRIORS_PATH",
    "MIN_ROWS_PER_CLUSTER",
    "class_rank_for_cluster",
    "rank_hero_facts",
    "rebuild_hero_fact_priors",
]
