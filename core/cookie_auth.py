"""
HttpOnly auth cookies (access JWT + opaque refresh) for browser sessions.

Bearer ``Authorization`` remains supported for API clients and cross-host dev.
"""

from __future__ import annotations

from typing import Optional

from starlette.requests import Request
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

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


def _same_site() -> str:
    s = (AUTH_COOKIE_SAMESITE or "lax").strip().lower()
    if s in ("lax", "strict", "none"):
        return s
    return "lax"


def _normalize_host(host: Optional[str]) -> Optional[str]:
    if not host:
        return None
    h = host.split(":")[0].strip().lower().strip("[]")
    return h or None


def _is_local_host(host: Optional[str]) -> bool:
    if not host:
        return False
    return host in _LOCAL_HOSTS or host.endswith(".localhost")


def host_from_request(request: Optional[Request]) -> Optional[str]:
    if request is None:
        return None
    return request.headers.get("host")


def _cookie_common(request: Optional[Request] = None) -> dict:
    """
    Cookie attributes for Set-Cookie / clear.

    When the request Host is localhost / 127.0.0.1, ignore production
    AUTH_COOKIE_DOMAIN and AUTH_COOKIE_SECURE from .env so local ``uvicorn``
    on http://127.0.0.1:8001 can set HttpOnly session cookies.
    """
    domain = AUTH_COOKIE_DOMAIN
    secure = AUTH_COOKIE_SECURE
    req_host = _normalize_host(host_from_request(request))
    if _is_local_host(req_host):
        domain = None
        secure = False

    common = {
        "httponly": True,
        "secure": secure,
        "samesite": _same_site(),
        "path": AUTH_COOKIE_PATH or "/",
    }
    if domain:
        common["domain"] = domain
    return common


def set_auth_cookies(
    response: Response,
    access_token: str,
    refresh_token: str,
    request: Optional[Request] = None,
) -> None:
    """Attach HttpOnly cookies. JSON body may still include tokens for non-browser clients."""
    max_access = int(ACCESS_TOKEN_MINUTES * 60)
    max_refresh = int(REFRESH_TOKEN_DAYS * 86400)
    common = _cookie_common(request)
    response.set_cookie(AUTH_ACCESS_COOKIE, access_token, max_age=max_access, **common)
    response.set_cookie(AUTH_REFRESH_COOKIE, refresh_token, max_age=max_refresh, **common)


def clear_auth_cookies(response: Response, request: Optional[Request] = None) -> None:
    common = {**_cookie_common(request), "max_age": 0}
    response.set_cookie(AUTH_ACCESS_COOKIE, "", **common)
    response.set_cookie(AUTH_REFRESH_COOKIE, "", **common)


def access_token_from_cookie(request_cookies: dict) -> Optional[str]:
    v = request_cookies.get(AUTH_ACCESS_COOKIE)
    return v if v else None


def refresh_token_from_cookie(request_cookies: dict) -> Optional[str]:
    v = request_cookies.get(AUTH_REFRESH_COOKIE)
    return v if v else None
