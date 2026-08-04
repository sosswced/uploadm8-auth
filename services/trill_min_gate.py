"""Min-Trill publish gate — optionally skip uploads below the user's threshold.

``trill_min_score`` alone only suppresses Trill hype tags. When
``trill_skip_low_score`` / ``trillSkipLowScore`` is enabled, the worker aborts
the upload (cancelled + wallet refund) before publish so batch queues can
skip low-scoring clips.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple


def _settings_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off", ""):
            return False
    return bool(value)


def trill_min_score(user_settings: Mapping[str, Any] | None) -> int:
    us = user_settings or {}
    raw = us.get("trill_min_score")
    if raw is None:
        raw = us.get("trillMinScore")
    if raw is None:
        return 0
    try:
        return max(0, min(100, int(raw)))
    except (TypeError, ValueError):
        return 0


def trill_skip_low_score_enabled(user_settings: Mapping[str, Any] | None) -> bool:
    us = user_settings or {}
    raw = us.get("trill_skip_low_score")
    if raw is None:
        raw = us.get("trillSkipLowScore")
    return _settings_bool(raw, False)


def should_skip_low_trill(
    user_settings: Mapping[str, Any] | None,
    score: Optional[float],
    *,
    allow_scenic_headroom: bool = False,
    scenic_max_boost: float = 28.0,
) -> Tuple[bool, str]:
    """Return ``(should_skip, reason)``.

    When ``allow_scenic_headroom`` is True (early gate after telemetry), only
    skip if even the maximum scenic boost could not reach the minimum — so
    borderline clips still get Vision/OSD scenic enrichment.
    """
    if not trill_skip_low_score_enabled(user_settings):
        return False, ""
    if score is None:
        return False, ""
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        return False, ""
    min_score = trill_min_score(user_settings)
    if min_score <= 0:
        return False, ""

    effective = score_f
    if allow_scenic_headroom:
        try:
            headroom = max(0.0, float(scenic_max_boost))
        except (TypeError, ValueError):
            headroom = 28.0
        effective = score_f + headroom

    if effective >= float(min_score):
        return False, ""

    if allow_scenic_headroom:
        reason = (
            f"Skipped: Trill score {score_f:.0f} below minimum {min_score} "
            f"(cannot reach with scenic boost)"
        )
    else:
        reason = f"Skipped: Trill score {score_f:.0f} below minimum {min_score}"
    return True, reason
