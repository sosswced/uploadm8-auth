from __future__ import annotations

import time
from typing import Optional

from playwright.sync_api import Page


def is_rate_limited(page: Page, response_status: Optional[int] = None) -> bool:
    if response_status == 429:
        return True
    try:
        text = (page.locator("body").inner_text(timeout=1200) or "").lower()
    except Exception:
        text = ""
    url = (page.url or "").lower()
    return (
        "rate limit" in text
        or "too many request" in text
        or "429" in text
        or "rate_limit" in url
    )


def goto_with_backoff(
    page: Page,
    url: str,
    *,
    wait_until: str = "domcontentloaded",
    timeout: int = 30000,
    retries: int = 3,
) -> None:
    delay = 1.2
    for attempt in range(retries + 1):
        resp = page.goto(url, wait_until=wait_until, timeout=timeout)
        status = resp.status if resp else None
        if not is_rate_limited(page, status):
            return
        if attempt >= retries:
            return
        page.wait_for_timeout(int(delay * 1000))
        time.sleep(0.05)
        delay *= 2.0
