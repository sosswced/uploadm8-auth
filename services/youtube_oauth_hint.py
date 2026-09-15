"""YouTube / Google OAuth helpers — login email for reconnect login_hint."""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

import httpx

logger = logging.getLogger("uploadm8-api")

# Loose but practical: must look like an email, not a handle or channel title.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize_oauth_login_hint(raw: Any) -> Optional[str]:
    """Return a lowercase email suitable for Google ``login_hint``, else None."""
    text = str(raw or "").strip().lstrip("@")
    if not text or " " in text:
        return None
    if not _EMAIL_RE.match(text):
        return None
    return text.lower()


async def fetch_google_oauth_email(
    client: httpx.AsyncClient,
    access_token: str,
) -> Optional[str]:
    """Best-effort Google account email after YouTube OAuth (openid/email scopes)."""
    token = str(access_token or "").strip()
    if not token:
        return None
    try:
        resp = await client.get(
            "https://openidconnect.googleapis.com/v1/userinfo",
            headers={"Authorization": f"Bearer {token}"},
        )
        if resp.status_code != 200:
            logger.debug(
                "Google userinfo for login_hint failed: status=%s",
                resp.status_code,
            )
            return None
        data = resp.json() if resp.content else {}
        return normalize_oauth_login_hint((data or {}).get("email"))
    except Exception as exc:
        logger.debug("Google userinfo for login_hint skipped: %s", exc)
        return None
