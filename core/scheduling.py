"""
UploadM8 smart scheduling — optimal upload time calculation per platform.
Extracted from app.py; pure logic plus one DB lookup.
"""

import random
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from core.helpers import _now_utc

logger = logging.getLogger("uploadm8-api")


# Platform-specific optimal posting times (in UTC)
# Based on general social media engagement research
PLATFORM_OPTIMAL_TIMES = {
    "tiktok": [
        {"hour": 7, "minute": 0, "weight": 0.8},   # 7 AM - morning scroll
        {"hour": 12, "minute": 0, "weight": 0.9},  # 12 PM - lunch break
        {"hour": 15, "minute": 0, "weight": 0.7},  # 3 PM - afternoon break
        {"hour": 19, "minute": 0, "weight": 1.0},  # 7 PM - evening prime time
        {"hour": 21, "minute": 0, "weight": 0.95}, # 9 PM - night engagement
        {"hour": 23, "minute": 0, "weight": 0.6},  # 11 PM - late night
    ],
    "youtube": [
        {"hour": 12, "minute": 0, "weight": 0.7},  # 12 PM - lunch views
        {"hour": 14, "minute": 0, "weight": 0.8},  # 2 PM - afternoon
        {"hour": 17, "minute": 0, "weight": 0.9},  # 5 PM - after work/school
        {"hour": 19, "minute": 0, "weight": 1.0},  # 7 PM - prime time
        {"hour": 21, "minute": 0, "weight": 0.95}, # 9 PM - evening viewing
    ],
    "instagram": [
        {"hour": 6, "minute": 0, "weight": 0.7},   # 6 AM - early morning
        {"hour": 11, "minute": 0, "weight": 0.85}, # 11 AM - mid-morning
        {"hour": 13, "minute": 0, "weight": 0.9},  # 1 PM - lunch
        {"hour": 17, "minute": 0, "weight": 0.8},  # 5 PM - commute
        {"hour": 19, "minute": 0, "weight": 1.0},  # 7 PM - prime time
        {"hour": 21, "minute": 0, "weight": 0.9},  # 9 PM - evening
    ],
    "facebook": [
        {"hour": 9, "minute": 0, "weight": 0.8},   # 9 AM - morning check
        {"hour": 11, "minute": 0, "weight": 0.7},  # 11 AM - mid-morning
        {"hour": 13, "minute": 0, "weight": 0.9},  # 1 PM - lunch break
        {"hour": 16, "minute": 0, "weight": 0.85}, # 4 PM - afternoon
        {"hour": 19, "minute": 0, "weight": 1.0},  # 7 PM - prime time
        {"hour": 20, "minute": 0, "weight": 0.9},  # 8 PM - evening
    ],
}

# Best days for each platform (0=Monday, 6=Sunday)
PLATFORM_OPTIMAL_DAYS = {
    "tiktok": [1, 2, 3, 4],      # Tue, Wed, Thu, Fri - highest engagement
    "youtube": [3, 4, 5],        # Thu, Fri, Sat - weekend viewing prep
    "instagram": [0, 1, 2, 4],   # Mon, Tue, Wed, Fri - weekday engagement
    "facebook": [0, 1, 2, 3],    # Mon, Tue, Wed, Thu - business days
}


def calculate_smart_schedule(platforms: List[str], num_days: int = 7, user_timezone: str = "UTC") -> Dict[str, datetime]:
    """
    Calculate optimal upload times for each platform.
    Ensures uploads are spread across different days.
    Returns a dict mapping platform -> scheduled datetime
    """
    now = _now_utc()
    schedule = {}
    used_days = set()

    # Sort platforms to ensure consistent ordering
    platforms_sorted = sorted(platforms)

    for platform in platforms_sorted:
        optimal_times = PLATFORM_OPTIMAL_TIMES.get(platform, PLATFORM_OPTIMAL_TIMES["tiktok"])
        optimal_days = PLATFORM_OPTIMAL_DAYS.get(platform, [0, 1, 2, 3, 4])

        # Find an available day that hasn't been used
        available_days = []
        for day_offset in range(1, num_days + 1):
            target_date = now + timedelta(days=day_offset)
            weekday = target_date.weekday()

            # Prefer optimal days for this platform, but allow any day if needed
            if day_offset not in used_days:
                priority = 2 if weekday in optimal_days else 1
                available_days.append((day_offset, priority, weekday))

        if not available_days:
            # All days used, pick a random future day
            day_offset = random.randint(1, num_days)
        else:
            # Sort by priority (optimal days first), then randomize within priority
            available_days.sort(key=lambda x: (-x[1], random.random()))
            day_offset = available_days[0][0]

        used_days.add(day_offset)

        # Pick an optimal time slot with weighted randomization
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

        # Add randomization to the time (+-30 minutes)
        minute_offset = random.randint(-30, 30)

        # Calculate the final datetime
        target_date = now + timedelta(days=day_offset)
        scheduled_dt = target_date.replace(
            hour=selected_time["hour"],
            minute=max(0, min(59, selected_time["minute"] + minute_offset)),
            second=0,
            microsecond=0
        )

        # Make sure it's in the future
        if scheduled_dt <= now:
            scheduled_dt += timedelta(days=1)

        schedule[platform] = scheduled_dt

    return schedule


async def get_existing_scheduled_days(conn, user_id: str, num_days: int = 7) -> set:
    """Get days that already have scheduled uploads for this user"""
    now = _now_utc()
    end_date = now + timedelta(days=num_days)

    existing = await conn.fetch("""
        SELECT DISTINCT DATE(scheduled_time) as sched_date
        FROM uploads
        WHERE user_id = $1
        AND scheduled_time >= $2
        AND scheduled_time <= $3
        AND status IN ('pending', 'queued', 'scheduled')
    """, user_id, now, end_date)

    used_days = set()
    for row in existing:
        if row["sched_date"]:
            day_diff = (row["sched_date"] - now.date()).days
            if day_diff > 0:
                used_days.add(day_diff)

    return used_days
