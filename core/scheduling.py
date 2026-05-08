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

import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from core.helpers import _now_utc

logger = logging.getLogger("uploadm8-api")


# Platform-specific optimal posting times (in UTC)
# Based on general social media engagement research
PLATFORM_OPTIMAL_TIMES = {
    "tiktok": [
        {"hour": 7, "minute": 0, "weight": 0.8},  # 7 AM - morning scroll
        {"hour": 12, "minute": 0, "weight": 0.9},  # 12 PM - lunch break
        {"hour": 15, "minute": 0, "weight": 0.7},  # 3 PM - afternoon break
        {"hour": 19, "minute": 0, "weight": 1.0},  # 7 PM - evening prime time
        {"hour": 21, "minute": 0, "weight": 0.95},  # 9 PM - night engagement
        {"hour": 23, "minute": 0, "weight": 0.6},  # 11 PM - late night
    ],
    "youtube": [
        {"hour": 12, "minute": 0, "weight": 0.7},  # 12 PM - lunch views
        {"hour": 14, "minute": 0, "weight": 0.8},  # 2 PM - afternoon
        {"hour": 17, "minute": 0, "weight": 0.9},  # 5 PM - after work/school
        {"hour": 19, "minute": 0, "weight": 1.0},  # 7 PM - prime time
        {"hour": 21, "minute": 0, "weight": 0.95},  # 9 PM - evening viewing
    ],
    "instagram": [
        {"hour": 6, "minute": 0, "weight": 0.7},  # 6 AM - early morning
        {"hour": 11, "minute": 0, "weight": 0.85},  # 11 AM - mid-morning
        {"hour": 13, "minute": 0, "weight": 0.9},  # 1 PM - lunch
        {"hour": 17, "minute": 0, "weight": 0.8},  # 5 PM - commute
        {"hour": 19, "minute": 0, "weight": 1.0},  # 7 PM - prime time
        {"hour": 21, "minute": 0, "weight": 0.9},  # 9 PM - evening
    ],
    "facebook": [
        {"hour": 9, "minute": 0, "weight": 0.8},  # 9 AM - morning check
        {"hour": 11, "minute": 0, "weight": 0.7},  # 11 AM - mid-morning
        {"hour": 13, "minute": 0, "weight": 0.9},  # 1 PM - lunch break
        {"hour": 16, "minute": 0, "weight": 0.85},  # 4 PM - afternoon
        {"hour": 19, "minute": 0, "weight": 1.0},  # 7 PM - prime time
        {"hour": 20, "minute": 0, "weight": 0.9},  # 8 PM - evening
    ],
}

# Best days for each platform (0=Monday, 6=Sunday)
PLATFORM_OPTIMAL_DAYS = {
    "tiktok": [1, 2, 3, 4],  # Tue, Wed, Thu, Fri - highest engagement
    "youtube": [3, 4, 5],  # Thu, Fri, Sat - weekend viewing prep
    "instagram": [0, 1, 2, 4],  # Mon, Tue, Wed, Fri - weekday engagement
    "facebook": [0, 1, 2, 3],  # Mon, Tue, Wed, Thu - business days
}

_EPS = 1e-9


def static_hour_prior_24(platform: str) -> List[float]:
    """Map PLATFORM_OPTIMAL_TIMES into a 24-slot UTC hour prior (normalized)."""
    optimal_times = PLATFORM_OPTIMAL_TIMES.get(platform, PLATFORM_OPTIMAL_TIMES["tiktok"])
    w = [0.02 + _EPS] * 24
    for t in optimal_times:
        h = int(t["hour"]) % 24
        w[h] += float(t["weight"])
    s = sum(w)
    return [x / s for x in w]


def _pick_weighted_hour(hour_weights: List[float]) -> int:
    total = sum(hour_weights)
    if total <= _EPS:
        return random.randint(0, 23)
    r = random.uniform(0.0, total)
    c = 0.0
    for h, wt in enumerate(hour_weights):
        c += wt
        if r <= c:
            return h
    return 23


def _apply_subsecond_jitter(anchor: datetime, now: datetime) -> datetime:
    """
    Spread within the chosen UTC hour, then apply a variable symmetric wall-clock
    offset in whole seconds (not a fixed ±30 minute band).
    """
    within_hour = random.randint(0, 3599)
    out = anchor + timedelta(seconds=within_hour)
    # Variable outer jitter: different cap every call (roughly 2 min … ~2.5 h half-span).
    outer_span = random.randint(120, 9000)
    out += timedelta(seconds=random.randint(-outer_span, outer_span))
    out = out.replace(microsecond=0)
    if out <= now:
        out += timedelta(days=1)
    return out


def _pick_day_offset(
    now: datetime,
    platform: str,
    num_days: int,
    used_days: set,
    blocked_days: Optional[set],
) -> int:
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
        if not pool:
            return random.randint(1, num_days)
        return random.choice(pool)

    available_days.sort(key=lambda x: (-x[1], random.random()))
    return available_days[0][0]


def calculate_smart_schedule(
    platforms: List[str],
    num_days: int = 7,
    user_timezone: str = "UTC",
    *,
    hour_weights_by_platform: Optional[Dict[str, List[float]]] = None,
    blocked_day_offsets: Optional[set] = None,
) -> Dict[str, datetime]:
    """
    Calculate smart upload times per platform (UTC).

    ``user_timezone`` is reserved for future locale-aware priors; DB path uses UTC
    publish hours from ``completed_at`` / ``created_at``.

    ``hour_weights_by_platform``: optional precomputed length-24 distributions (any
    positive values; normalized internally). When missing for a platform, static
    research priors are used.

    ``blocked_day_offsets``: 1-based day indices from ``_now_utc().date()`` to skip
    (e.g. days that already have scheduled rows for this user).
    """
    del user_timezone  # reserved — priors are UTC until locale pipeline exists
    now = _now_utc()
    schedule: Dict[str, datetime] = {}
    used_days: set = set()

    platforms_sorted = sorted(platforms)
    for platform in platforms_sorted:
        hour_weights = None
        if hour_weights_by_platform:
            hour_weights = hour_weights_by_platform.get(platform)
        if not hour_weights or len(hour_weights) != 24:
            hour_weights = static_hour_prior_24(platform)
        else:
            s = sum(max(0.0, float(x)) for x in hour_weights) + _EPS
            hour_weights = [max(0.0, float(x)) / s for x in hour_weights]

        day_offset = _pick_day_offset(now, platform, num_days, used_days, blocked_day_offsets)
        used_days.add(day_offset)

        chosen_hour = _pick_weighted_hour(hour_weights)
        target_date = now + timedelta(days=day_offset)
        anchor = datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            chosen_hour,
            0,
            0,
            tzinfo=timezone.utc,
        )
        schedule[platform] = _apply_subsecond_jitter(anchor, now)

    return schedule


async def get_existing_scheduled_days(conn, user_id: str, num_days: int = 7) -> set:
    """Calendar day offsets (from today) that already have a scheduled_time for this user."""
    now = _now_utc()
    end_date = now + timedelta(days=num_days)

    existing = await conn.fetch(
        """
        SELECT DISTINCT DATE(scheduled_time) as sched_date
        FROM uploads
        WHERE user_id = $1
        AND scheduled_time >= $2
        AND scheduled_time <= $3
        AND status IN ('pending', 'queued', 'scheduled', 'staged', 'ready_to_publish')
    """,
        user_id,
        now,
        end_date,
    )

    used_days: set = set()
    for row in existing:
        if row["sched_date"]:
            day_diff = (row["sched_date"] - now.date()).days
            if day_diff > 0:
                used_days.add(day_diff)

    return used_days
