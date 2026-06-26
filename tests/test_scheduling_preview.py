"""Tests for smart_schedule_seed parity between preview and calculate."""

from __future__ import annotations

from core.scheduling import calculate_smart_schedule


def test_smart_schedule_seed_matches_preview():
    platforms = ["tiktok", "youtube"]
    seed = "client-preview-seed-abc"
    a = calculate_smart_schedule(
        platforms,
        num_days=14,
        user_timezone="America/Chicago",
        random_seed=seed,
    )
    b = calculate_smart_schedule(
        platforms,
        num_days=14,
        user_timezone="America/Chicago",
        random_seed=seed,
    )
    assert a == b
    assert len(a) == 2
