"""
UploadM8 smart scheduling — optimal upload time calculation per platform.

Blends (when DB signals are supplied from ``services.smart_schedule_insights``):
  • Soft **local hot windows** per platform (time frames, not fixed clock hours)
  • Fleet-wide hourly signals from successful uploads (UTC publish hour → local)
  • Per-user hourly signals (same, scoped to the creator)
  • Optional momentum multipliers (recent window vs older baseline)
  • Trained ``m8_publish_hour_priors`` when fresh (PCI ``published_at`` model)

Day occupancy / blocked offsets hard-deconflict calendar days inside the
scheduling window; when the window is full, slots spill into the expand
horizon (``num_days * 2``) so later uploads still see those days.

Pure helpers live here; SQL aggregation lives in ``services/smart_schedule_insights``.
"""

from __future__ import annotations

import hashlib
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from core.helpers import _now_utc

logger = logging.getLogger("uploadm8-api")

_JITTER_MAX_SECONDS = 30 * 60  # ±30 minutes from anchor

# Bump when refreshing research-backed hot windows (YYYY-MM).
STATIC_PRIOR_RESEARCH_VERSION = "2026-08"
# Sources (local audience time): Buffer / Sprout / Later 2026 consensus.
# These are soft TIME FRAMES — not hardcoded post clocks. Sampling logic
# spreads posts *around* each window; M8 + fleet/user signals still override.

# Local-hour frames [start_hour, end_hour] inclusive (user timezone).
# ``weight`` is relative importance of that frame vs others on the same platform.
PLATFORM_HOT_WINDOWS: Dict[str, List[Dict[str, Any]]] = {
    "tiktok": [
        {"start_hour": 14, "end_hour": 17, "weight": 1.0, "label": "afternoon"},
        {"start_hour": 19, "end_hour": 22, "weight": 0.85, "label": "evening"},
    ],
    "youtube": [
        {"start_hour": 16, "end_hour": 21, "weight": 1.0, "label": "late_afternoon_evening"},
    ],
    "instagram": [
        {"start_hour": 12, "end_hour": 16, "weight": 0.95, "label": "midday"},
        {"start_hour": 18, "end_hour": 21, "weight": 1.0, "label": "evening"},
        {"start_hour": 9, "end_hour": 10, "weight": 0.55, "label": "morning_pulse"},
    ],
    "facebook": [
        {"start_hour": 9, "end_hour": 12, "weight": 1.0, "label": "morning"},
        {"start_hour": 18, "end_hour": 20, "weight": 0.9, "label": "early_evening"},
    ],
}

# Preferred weekdays (Mon=0 … Sun=6) for soft day scoring — never hard-blocked.
PLATFORM_OPTIMAL_DAYS = {
    "tiktok": [0, 1, 2, 3, 4],
    "youtube": [3, 4, 5],
    "instagram": [0, 1, 2, 3],
    "facebook": [0, 1, 2, 3],
}

_EPS = 1e-9
# Residual mass outside hot windows so data-driven signals can still win.
_OUTSIDE_WINDOW_FLOOR = 0.04


def _rng_from_seed(seed: Optional[str]) -> random.Random:
    if not seed:
        return random.Random()
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _resolve_tz(tz_name: str) -> ZoneInfo:
    try:
        return ZoneInfo((tz_name or "UTC").strip())
    except Exception:
        return ZoneInfo("UTC")


def _normalize_hour_window(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        start = int(raw.get("start_hour"))
        end = int(raw.get("end_hour"))
        weight = float(raw.get("weight", 1.0))
    except (TypeError, ValueError):
        return None
    start = max(0, min(23, start))
    end = max(0, min(23, end))
    if end < start:
        start, end = end, start
    if weight <= 0:
        return None
    return {
        "start_hour": start,
        "end_hour": end,
        "weight": weight,
        "label": str(raw.get("label") or ""),
    }


def platform_hot_windows(platform: str) -> List[Dict[str, Any]]:
    """Normalized local hot windows for a platform (fallback: tiktok frames)."""
    key = str(platform or "").strip().lower()
    raw = PLATFORM_HOT_WINDOWS.get(key) or PLATFORM_HOT_WINDOWS["tiktok"]
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        norm = _normalize_hour_window(item)
        if norm:
            out.append(norm)
    return out or list(PLATFORM_HOT_WINDOWS["tiktok"])


def _window_hour_mass(start: int, end: int, hour: int) -> float:
    """
    Soft bump inside an inclusive hour frame — peak near center, taper at edges.

    Posts land *around* the timeframe rather than on a single hardcoded clock hour.
    """
    if hour < start or hour > end:
        return 0.0
    span = max(1.0, (end - start) / 2.0)
    mid = (start + end) / 2.0
    dist = abs(hour - mid) / span
    # Cosine-ish taper: 1.0 at center → ~0.55 at edges
    return 0.55 + 0.45 * max(0.0, 1.0 - dist)


def hot_windows_to_prior_24(windows: List[Dict[str, Any]]) -> List[float]:
    """Expand hot windows into a normalized 24-bin local-hour prior."""
    w = [_OUTSIDE_WINDOW_FLOOR + _EPS] * 24
    for win in windows:
        norm = _normalize_hour_window(win) if "start_hour" in win else None
        if not norm:
            continue
        start, end, weight = norm["start_hour"], norm["end_hour"], norm["weight"]
        for h in range(start, end + 1):
            w[h] += weight * _window_hour_mass(start, end, h)
    s = sum(w)
    return [x / s for x in w]


def static_hour_prior_24(platform: str) -> List[float]:
    """Soft local-hour prior from platform hot windows (not discrete clock peaks)."""
    return hot_windows_to_prior_24(platform_hot_windows(platform))


def soft_bias_toward_hot_windows(
    local_weights: List[float],
    platform: str,
    *,
    strength: float = 0.35,
) -> List[float]:
    """
    Gently pull any local-hour weight vector toward the platform's hot frames.

    ``strength`` in [0, 1]: 0 = unchanged, 1 = fully replace with window prior.
    Keeps data-driven peaks while still preferring to post *around* the frames.
    """
    if not local_weights or len(local_weights) != 24:
        return static_hour_prior_24(platform)
    s = max(0.0, min(1.0, float(strength)))
    if s <= _EPS:
        total = sum(max(0.0, float(x)) for x in local_weights) + _EPS
        return [max(0.0, float(x)) / total for x in local_weights]
    prior = static_hour_prior_24(platform)
    mixed = [(1.0 - s) * max(0.0, float(a)) + s * float(b) for a, b in zip(local_weights, prior)]
    total = sum(mixed) + _EPS
    return [x / total for x in mixed]


def hour_in_hot_windows(hour: int, platform: str) -> bool:
    h = int(hour) % 24
    for win in platform_hot_windows(platform):
        if win["start_hour"] <= h <= win["end_hour"]:
            return True
    return False


def utc_weights_as_local(
    utc_weights: List[float],
    tz: ZoneInfo,
    ref: datetime,
) -> List[float]:
    """Re-index UTC hour weights into the user's local-hour buckets (DST-aware offset at ref)."""
    aware = ref.replace(tzinfo=timezone.utc) if ref.tzinfo is None else ref.astimezone(timezone.utc)
    offset = aware.astimezone(tz).utcoffset()
    offset_h = int((offset.total_seconds() if offset else 0) // 3600)
    local_w = [0.0] * 24
    for utc_h, wt in enumerate(utc_weights):
        local_h = (utc_h + offset_h) % 24
        local_w[local_h] += max(0.0, float(wt))
    s = sum(local_w)
    if s <= _EPS:
        return utc_weights
    return [x / s for x in local_w]


def _pick_weighted_hour(hour_weights: List[float], rng: random.Random) -> int:
    total = sum(hour_weights)
    if total <= _EPS:
        return rng.randint(0, 23)
    r = rng.uniform(0.0, total)
    c = 0.0
    for h, wt in enumerate(hour_weights):
        c += wt
        if r <= c:
            return h
    return 23


def _apply_subsecond_jitter(
    anchor: datetime,
    now: datetime,
    *,
    rng: random.Random,
) -> datetime:
    """Spread within ±30 minutes of the chosen anchor (stored as UTC)."""
    jitter = rng.randint(-_JITTER_MAX_SECONDS, _JITTER_MAX_SECONDS)
    out = anchor + timedelta(seconds=jitter)
    out = out.replace(microsecond=0)
    if out <= now:
        out += timedelta(days=1)
    return out


def clamp_smart_schedule_days(num_days: Any, *, default: int = 14) -> int:
    """Normalize Smart Schedule window to 1–730 days (never 0 / NaN)."""
    try:
        n = int(num_days)
    except (TypeError, ValueError):
        n = int(default)
    if n < 1:
        n = int(default) if int(default) >= 1 else 14
    return max(1, min(730, n))


def smart_schedule_expand_horizon(num_days: int) -> int:
    """
    Spill horizon when the primary window is full of occupied days.

    Same formula as dense-batch expansion: ``min(730, num_days * 2)``.
    """
    n = clamp_smart_schedule_days(num_days)
    return max(n, min(730, n * 2))


def _normalize_day_occupancy(raw: Any) -> Dict[int, int]:
    """
    Coerce occupancy input to ``{day_offset: slot_count}``.

    Accepts a dict of counts, or a legacy set/list of blocked offsets (count=1 each).
    """
    if not raw:
        return {}
    if isinstance(raw, dict):
        out: Dict[int, int] = {}
        for k, v in raw.items():
            try:
                offset = int(k)
                count = int(v)
            except (TypeError, ValueError):
                continue
            if offset >= 1 and count > 0:
                out[offset] = out.get(offset, 0) + count
        return out
    out: Dict[int, int] = {}
    try:
        iterable = list(raw)
    except TypeError:
        return {}
    for item in iterable:
        try:
            offset = int(item)
        except (TypeError, ValueError):
            continue
        if offset >= 1:
            out[offset] = out.get(offset, 0) + 1
    return out


def _blocked_day_set(raw: Any) -> set:
    """Offsets that already have a publish slot (hard deconflict)."""
    return set(_normalize_day_occupancy(raw).keys())


def _pick_day_offset(
    now: datetime,
    platform: str,
    num_days: int,
    used_days: set,
    day_occupancy: Optional[Any],
    rng: random.Random,
) -> int:
    """
    Prefer free days inside ``1..num_days``; when the window is full, spill into
    the expand horizon. Never reuses ``used_days`` or blocked occupancy offsets.
    """
    num_days = clamp_smart_schedule_days(num_days)
    expand_to = smart_schedule_expand_horizon(num_days)
    optimal_days = PLATFORM_OPTIMAL_DAYS.get(platform, [0, 1, 2, 3, 4])
    blocked = _blocked_day_set(day_occupancy)

    available_days: list = []
    for day_offset in range(1, num_days + 1):
        if day_offset in blocked or day_offset in used_days:
            continue
        target_date = now + timedelta(days=day_offset)
        weekday = target_date.weekday()
        priority = 2 if weekday in optimal_days else 1
        available_days.append((day_offset, priority, weekday))

    if available_days:
        available_days.sort(key=lambda x: (-x[1], rng.random()))
        return available_days[0][0]

    pool = [
        d
        for d in range(1, num_days + 1)
        if d not in used_days and d not in blocked
    ]
    if pool:
        return rng.choice(pool)

    # Window exhausted (dense batch / short window): expand past num_days
    # rather than colliding with blocked/used offsets.
    for day_offset in range(num_days + 1, expand_to + 1):
        if day_offset in used_days or day_offset in blocked:
            continue
        return day_offset

    # Last resort: unique offset that respects blocked + used (deterministic via rng).
    for _ in range(64):
        candidate = rng.randint(1, expand_to)
        if candidate not in used_days and candidate not in blocked:
            return candidate

    # Walk past expand_to until unique — never collide with blocked days.
    candidate = expand_to + 1
    guard = 0
    while candidate in used_days or candidate in blocked:
        candidate += 1
        guard += 1
        if guard > 1024:
            return expand_to + 1 + len(used_days) + len(blocked)
    return candidate


def calculate_smart_schedule(
    platforms: List[str],
    num_days: int = 14,
    user_timezone: str = "UTC",
    *,
    hour_weights_by_platform: Optional[Dict[str, List[float]]] = None,
    hour_weights_are_local: bool = True,
    window_bias_strength: float = 0.35,
    blocked_day_offsets: Optional[Any] = None,
    day_occupancy: Optional[Any] = None,
    random_seed: Optional[str] = None,
) -> Dict[str, datetime]:
    """
    Calculate smart upload times per platform (stored as UTC).

    ``hour_weights_by_platform`` are **local-hour** priors by default (research +
    blended insights). Pass ``hour_weights_are_local=False`` only for raw UTC
    vectors that still need remapping into ``user_timezone``.

    ``window_bias_strength`` softly pulls the final hour distribution toward
    ``PLATFORM_HOT_WINDOWS`` so posts land *around* those frames without
    hardcoding a single clock time (0 disables).

    Day offsets prefer free days inside ``num_days``. When the window is full,
    slots may spill into the expand horizon (``num_days * 2``, capped at 730)
    so they never collide with ``blocked_day_offsets`` / occupancy.

    ``random_seed``: when set (e.g. upload_id), preview and presign produce identical slots.
    """
    num_days = clamp_smart_schedule_days(num_days)
    tz = _resolve_tz(user_timezone)
    rng = _rng_from_seed(random_seed)
    now = _now_utc()
    schedule: Dict[str, datetime] = {}
    used_days: set = set()
    occupancy = _normalize_day_occupancy(day_occupancy if day_occupancy is not None else blocked_day_offsets)

    plats = sorted({str(p).strip().lower() for p in platforms if str(p).strip()})
    for platform in plats:
        hour_weights = None
        if hour_weights_by_platform:
            hour_weights = hour_weights_by_platform.get(platform)
            if hour_weights is None:
                for k, v in hour_weights_by_platform.items():
                    if str(k).strip().lower() == platform:
                        hour_weights = v
                        break
        if not hour_weights or len(hour_weights) != 24:
            local_weights = static_hour_prior_24(platform)
        else:
            s = sum(max(0.0, float(x)) for x in hour_weights) + _EPS
            normalized = [max(0.0, float(x)) / s for x in hour_weights]
            if hour_weights_are_local:
                local_weights = normalized
            else:
                local_weights = utc_weights_as_local(normalized, tz, now)

        local_weights = soft_bias_toward_hot_windows(
            local_weights, platform, strength=window_bias_strength
        )

        day_offset = _pick_day_offset(now, platform, num_days, used_days, occupancy, rng)
        used_days.add(day_offset)
        occupancy[day_offset] = occupancy.get(day_offset, 0) + 1

        chosen_local_hour = _pick_weighted_hour(local_weights, rng)
        target_date = (now + timedelta(days=day_offset)).date()
        # Uniform minute inside the chosen hour keeps slots "around" the frame.
        local_minute = rng.randint(0, 59)
        local_dt = datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            chosen_local_hour,
            local_minute,
            0,
            tzinfo=tz,
        )
        anchor_utc = local_dt.astimezone(timezone.utc)
        schedule[platform] = _apply_subsecond_jitter(anchor_utc, now, rng=rng)

    return schedule


async def get_existing_scheduled_days(
    conn,
    user_id: str,
    num_days: int = 14,
    *,
    exclude_upload_id: Optional[str] = None,
) -> Dict[int, int]:
    """
    Occupancy map of day offsets that already have a scheduled slot.

    Tracks through the **expand horizon** (not just ``num_days``) so spill
    slots past the primary window still deconflict later uploads.
    """
    now = _now_utc()
    num_days = clamp_smart_schedule_days(num_days)
    horizon = smart_schedule_expand_horizon(num_days)
    end_date = now + timedelta(days=horizon)

    exclude_clause = ""
    params: list = [user_id, now, end_date]
    if exclude_upload_id:
        exclude_clause = "AND id != $4::uuid"
        params.append(exclude_upload_id)

    existing = await conn.fetch(
        f"""
        SELECT scheduled_time, schedule_mode, schedule_metadata
        FROM uploads
        WHERE user_id = $1
        AND (
            (scheduled_time >= $2 AND scheduled_time <= $3)
            OR (schedule_mode = 'smart' AND schedule_metadata IS NOT NULL)
        )
        AND status IN ('pending', 'queued', 'scheduled', 'staged', 'ready_to_publish')
        {exclude_clause}
    """,
        *params,
    )

    from services.schedule_slots import day_occupancy_from_today

    occupancy: Dict[int, int] = {}
    today = now.date()
    for row in existing:
        for offset, count in day_occupancy_from_today(
            dict(row), today=today, num_days=num_days, horizon=horizon
        ).items():
            occupancy[offset] = occupancy.get(offset, 0) + count

    return occupancy
