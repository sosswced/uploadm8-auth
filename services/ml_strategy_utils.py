"""
Shared helpers for strategy scoring / thumbnail ML bias.

Uses empirical-Bayes style shrinkage toward a global prior so sparse samples
don't dominate decisions. Pure functions — easy to unit test.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# Typical engagement-rate scale for (likes+comments+shares)/views * 100 on social.
_DEFAULT_PRIOR_MEAN = 2.0
_DEFAULT_PRIOR_STRENGTH = 6.0


def empirical_bayes_mean(raw_mean: float, n: int, *, prior_mean: float = _DEFAULT_PRIOR_MEAN, prior_strength: float = _DEFAULT_PRIOR_STRENGTH) -> float:
    """Shrink raw_mean toward prior_mean; more weight to data as n grows."""
    n = max(0, int(n or 0))
    if n <= 0:
        return float(prior_mean)
    w = float(n) * float(raw_mean) + float(prior_strength) * float(prior_mean)
    return w / (float(n) + float(prior_strength))


def prefer_ai_thumbnail_vs_sharpness(
    *,
    sharp_mean: Optional[float],
    sharp_samples: int,
    ai_mean: Optional[float],
    ai_samples: int,
    min_samples_ai: int = 4,
    min_samples_sharp: int = 2,
    margin: float = 0.12,
    prior_mean: float = _DEFAULT_PRIOR_MEAN,
    prior_strength: float = _DEFAULT_PRIOR_STRENGTH,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Return (use_ai, detail) comparing AI frame selection vs sharpness-only.

    Requires enough samples on the AI arm before trusting a win; otherwise we
    default to sharpness (cheaper / stable) unless AI is clearly ahead.
    """
    sm = float(sharp_mean) if sharp_mean is not None else None
    am = float(ai_mean) if ai_mean is not None else None
    ss = max(0, int(sharp_samples or 0))
    asn = max(0, int(ai_samples or 0))

    eb_sharp = empirical_bayes_mean(sm if sm is not None else prior_mean, ss, prior_mean=prior_mean, prior_strength=prior_strength)
    eb_ai = empirical_bayes_mean(am if am is not None else prior_mean, asn, prior_mean=prior_mean, prior_strength=prior_strength)

    # Insufficient AI evidence → do not prefer AI (avoid noisy early wins).
    if asn < min_samples_ai:
        return False, {
            "reason": "insufficient_ai_samples",
            "eb_sharp": eb_sharp,
            "eb_ai": eb_ai,
            "sharp_samples": ss,
            "ai_samples": asn,
            "margin": margin,
        }

    # If sharpness has almost no data, allow AI if AI looks better (still needs min_samples_ai).
    if ss < min_samples_sharp and eb_ai > eb_sharp:
        return True, {
            "reason": "ai_wins_vs_weak_sharp_baseline",
            "eb_sharp": eb_sharp,
            "eb_ai": eb_ai,
            "sharp_samples": ss,
            "ai_samples": asn,
            "margin": margin,
        }

    use = bool(eb_ai > eb_sharp + float(margin))
    return use, {
        "reason": "eb_compare" if use else "sharp_preferred_by_margin_or_tie",
        "eb_sharp": eb_sharp,
        "eb_ai": eb_ai,
        "sharp_samples": ss,
        "ai_samples": asn,
        "margin": margin,
    }
