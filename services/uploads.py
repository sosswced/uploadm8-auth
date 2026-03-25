from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List


# Platform-specific optimal posting times (in UTC)
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


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def calculate_smart_schedule(platforms: List[str], num_days: int = 7, user_timezone: str = "UTC") -> Dict[str, datetime]:
    now = _now_utc()
    schedule = {}
    used_days = set()

    platforms_sorted = sorted(platforms)
    for platform in platforms_sorted:
        optimal_times = PLATFORM_OPTIMAL_TIMES.get(platform, PLATFORM_OPTIMAL_TIMES["tiktok"])
        optimal_days = PLATFORM_OPTIMAL_DAYS.get(platform, [0, 1, 2, 3, 4])

        available_days = []
        for day_offset in range(1, num_days + 1):
            target_date = now + timedelta(days=day_offset)
            weekday = target_date.weekday()
            if day_offset not in used_days:
                priority = 2 if weekday in optimal_days else 1
                available_days.append((day_offset, priority, weekday))

        if not available_days:
            day_offset = random.randint(1, num_days)
        else:
            available_days.sort(key=lambda x: (-x[1], random.random()))
            day_offset = available_days[0][0]

        used_days.add(day_offset)

        weights = [t["weight"] for t in optimal_times]
        total_weight = sum(weights)
        rand_val = random.uniform(0, total_weight)

        cumulative = 0
        selected_time = optimal_times[0]
        for t in optimal_times:
            cumulative += t["weight"]
            if rand_val <= cumulative:
                selected_time = t
                break

        minute_offset = random.randint(-30, 30)
        target_date = now + timedelta(days=day_offset)
        scheduled_dt = target_date.replace(
            hour=selected_time["hour"],
            minute=max(0, min(59, selected_time["minute"] + minute_offset)),
            second=0,
            microsecond=0,
        )
        if scheduled_dt <= now:
            scheduled_dt += timedelta(days=1)
        schedule[platform] = scheduled_dt

    return schedule


async def get_existing_scheduled_days(conn, user_id: str, num_days: int = 7) -> set:
    now = _now_utc()
    end_date = now + timedelta(days=num_days)

    existing = await conn.fetch(
        """
        SELECT DISTINCT DATE(scheduled_time) as sched_date
        FROM uploads
        WHERE user_id = $1
        AND scheduled_time >= $2
        AND scheduled_time <= $3
        AND status IN ('pending', 'queued', 'scheduled')
    """,
        user_id,
        now,
        end_date,
    )

    used_days = set()
    for row in existing:
        if row["sched_date"]:
            day_diff = (row["sched_date"] - now.date()).days
            if day_diff > 0:
                used_days.add(day_diff)

    return used_days
