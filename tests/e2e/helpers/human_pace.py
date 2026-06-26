"""Human-like pacing to avoid API rate limits and bot heuristics."""

from __future__ import annotations

import os
import random
import time

CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def request_delay_ms() -> int:
    base = int(os.environ.get("E2E_REQUEST_DELAY_MS", "350"))
    jitter = int(os.environ.get("E2E_REQUEST_JITTER_MS", "150"))
    if jitter <= 0:
        return base
    return base + random.randint(0, jitter)


def pause_between_requests() -> None:
    ms = request_delay_ms()
    if ms > 0:
        time.sleep(ms / 1000.0)


def slow_mo_ms() -> int:
    if os.environ.get("E2E_HEADED", "").lower() not in ("1", "true", "yes"):
        return 0
    try:
        return int(os.environ.get("E2E_SLOW_MO_MS", "120"))
    except ValueError:
        return 120


def click_delay_ms() -> int:
    try:
        return int(os.environ.get("E2E_CLICK_DELAY_MS", "280"))
    except ValueError:
        return 280
