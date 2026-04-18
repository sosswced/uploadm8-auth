"""
HttpOnly auth cookies (access JWT + opaque refresh) for browser sessions.

Bearer ``Authorization`` remains supported for API clients and cross-host dev.
"""

from __future__ import annotations

from typing import Optional

from starlette.responses import Response

from core.config import (
    ACCESS_TOKEN_MINUTES,
    AUTH_ACCESS_COOKIE,
    AUTH_COOKIE_DOMAIN,
    AUTH_COOKIE_PATH,
    AUTH_COOKIE_SAMESITE,
    AUTH_COOKIE_SECURE,
    AUTH_REFRESH_COOKIE,
    REFRESH_TOKEN_DAYS,
)


def _same_site() -> str:
    s = (AUTH_COOKIE_SAMESITE or "lax").strip().lower()
    if s in ("lax", "strict", "none"):
        return s
    return "lax"


def set_auth_cookies(response: Response, access_token: str, refresh_token: str) -> None:
    """Attach HttpOnly cookies. JSON body may still include tokens for non-browser clients."""
    max_access = int(ACCESS_TOKEN_MINUTES * 60)
    max_refresh = int(REFRESH_TOKEN_DAYS * 86400)
    common = {
        "httponly": True,
        "secure": AUTH_COOKIE_SECURE,
        "samesite": _same_site(),
        "path": AUTH_COOKIE_PATH or "/",
    }
    if AUTH_COOKIE_DOMAIN:
        common["domain"] = AUTH_COOKIE_DOMAIN

    response.set_cookie(AUTH_ACCESS_COOKIE, access_token, max_age=max_access, **common)
    response.set_cookie(AUTH_REFRESH_COOKIE, refresh_token, max_age=max_refresh, **common)


def clear_auth_cookies(response: Response) -> None:
    common = {
        "httponly": True,
        "secure": AUTH_COOKIE_SECURE,
        "samesite": _same_site(),
        "path": AUTH_COOKIE_PATH or "/",
        "max_age": 0,
    }
    if AUTH_COOKIE_DOMAIN:
        common["domain"] = AUTH_COOKIE_DOMAIN
    response.set_cookie(AUTH_ACCESS_COOKIE, "", **common)
    response.set_cookie(AUTH_REFRESH_COOKIE, "", **common)


def access_token_from_cookie(request_cookies: dict) -> Optional[str]:
    v = request_cookies.get(AUTH_ACCESS_COOKIE)
    return v if v else None


def refresh_token_from_cookie(request_cookies: dict) -> Optional[str]:
    v = request_cookies.get(AUTH_REFRESH_COOKIE)
    return v if v else None
