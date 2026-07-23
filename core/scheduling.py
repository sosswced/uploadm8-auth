"""
UploadM8 smart scheduling — optimal upload time calculation per platform.

Blends (when DB signals are supplied from ``services.smart_schedule_insights``):
  • Static UTC engagement priors (research-shaped defaults)
  • Fleet-wide hourly signals from successful uploads (log-views by publish hour, UTC)
  • Per-user hourly signals (same, scoped to the creator)
  • Optional momentum multipliers (recent window vs older baseline)

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

# Platform-specific optimal posting times (in UTC)
# Based on general social media engagement research
PLATFORM_OPTIMAL_TIMES = {
    "tiktok": [
        {"hour": 7, "minute": 0, "weight": 0.8},
        {"hour": 12, "minute": 0, "weight": 0.9},
        {"hour": 15, "minute": 0, "weight": 0.7},
        {"hour": 19, "minute": 0, "weight": 1.0},
        {"hour": 21, "minute": 0, "weight": 0.95},
        {"hour": 23, "minute": 0, "weight": 0.6},
    ],
    "youtube": [
        {"hour": 12, "minute": 0, "weight": 0.7},
        {"hour": 14, "minute": 0, "weight": 0.8},
        {"hour": 17, "minute": 0, "weight": 0.9},
        {"hour": 19, "minute": 0, "weight": 1.0},
        {"hour": 21, "minute": 0, "weight": 0.95},
    ],
    "instagram": [
        {"hour": 6, "minute": 0, "weight": 0.7},
        {"hour": 11, "minute": 0, "weight": 0.85},
        {"hour": 13, "minute": 0, "weight": 0.9},
        {"hour": 17, "minute": 0, "weight": 0.8},
        {"hour": 19, "minute": 0, "weight": 1.0},
        {"hour": 21, "minute": 0, "weight": 0.9},
    ],
    "facebook": [
        {"hour": 9, "minute": 0, "weight": 0.8},
        {"hour": 11, "minute": 0, "weight": 0.7},
        {"hour": 13, "minute": 0, "weight": 0.9},
        {"hour": 16, "minute": 0, "weight": 0.85},
        {"hour": 19, "minute": 0, "weight": 1.0},
        {"hour": 20, "minute": 0, "weight": 0.9},
    ],
}

PLATFORM_OPTIMAL_DAYS = {
    "tiktok": [1, 2, 3, 4],
    "youtube": [3, 4, 5],
    "instagram": [0, 1, 2, 4],
    "facebook": [0, 1, 2, 3],
}

_EPS = 1e-9


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


def static_hour_prior_24(platform: str) -> List[float]:
    """Map PLATFORM_OPTIMAL_TIMES into a 24-slot UTC hour prior (normalized)."""
    optimal_times = PLATFORM_OPTIMAL_TIMES.get(platform, PLATFORM_OPTIMAL_TIMES["tiktok"])
    w = [0.02 + _EPS] * 24
    for t in optimal_times:
        h = int(t["hour"]) % 24
        w[h] += float(t["weight"])
    s = sum(w)
    return [x / s for x in w]


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


def _pick_day_offset(
    now: datetime,
    platform: str,
    num_days: int,
    used_days: set,
    blocked_days: Optional[set],
    rng: random.Random,
) -> int:
    num_days = clamp_smart_schedule_days(num_days)
    optimal_days = PLATFORM_OPTIMAL_DAYS.get(platform, [0, 1, 2, 3, 4])
    available_days: list = []
    for day_offset in range(1, num_days + 1):
        if blocked_days and day_offset in blocked_days:
            continue
        target_date = now + timedelta(days=day_offset)
        weekday = target_date.weekday()
        if day_offset not in used_days:
            priority = 2 if weekday in optimal_days else 1
            available_days.append((day_offset, priority, weekday))

    if not available_days:
        pool = [
            d
            for d in range(1, num_days + 1)
            if d not in used_days and (blocked_days is None or d not in blocked_days)
        ]
        if pool:
            return rng.choice(pool)
        # Window exhausted (dense batch / short window): expand past num_days
        # rather than colliding with blocked/used offsets.
        expand_to = max(num_days + 1, min(730, num_days * 2))
        for day_offset in range(num_days + 1, expand_to + 1):
            if day_offset in used_days:
                continue
            if blocked_days and day_offset in blocked_days:
                continue
            return day_offset
        # Last resort: unique offset beyond used set (still deterministic via rng).
        for _ in range(64):
            candidate = rng.randint(1, max(num_days, expand_to))
            if candidate not in used_days:
                return candidate
        return num_days + 1 + len(used_days)

    available_days.sort(key=lambda x: (-x[1], rng.random()))
    return available_days[0][0]


def calculate_smart_schedule(
    platforms: List[str],
    num_days: int = 14,
    user_timezone: str = "UTC",
    *,
    hour_weights_by_platform: Optional[Dict[str, List[float]]] = None,
    blocked_day_offsets: Optional[set] = None,
    random_seed: Optional[str] = None,
) -> Dict[str, datetime]:
    """
    Calculate smart upload times per platform (UTC).

    Hour priors are shifted into ``user_timezone`` for slot selection; results are UTC.

    ``random_seed``: when set (e.g. upload_id), preview and presign produce identical slots.
    """
    num_days = clamp_smart_schedule_days(num_days)
    tz = _resolve_tz(user_timezone)
    rng = _rng_from_seed(random_seed)
    now = _now_utc()
    schedule: Dict[str, datetime] = {}
    used_days: set = set()

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
            hour_weights = static_hour_prior_24(platform)
        else:
            s = sum(max(0.0, float(x)) for x in hour_weights) + _EPS
            hour_weights = [max(0.0, float(x)) / s for x in hour_weights]

        local_weights = utc_weights_as_local(hour_weights, tz, now)

        day_offset = _pick_day_offset(now, platform, num_days, used_days, blocked_day_offsets, rng)
        used_days.add(day_offset)

        chosen_local_hour = _pick_weighted_hour(local_weights, rng)
        target_date = (now + timedelta(days=day_offset)).date()
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
) -> set:
    """Calendar day offsets (from today) that already have a scheduled slot for this user."""
    now = _now_utc()
    end_date = now + timedelta(days=num_days)

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

    from services.schedule_slots import day_offsets_from_today

    used_days: set = set()
    today = now.date()
    for row in existing:
        used_days |= day_offsets_from_today(dict(row), today=today, num_days=num_days)

    return used_days
