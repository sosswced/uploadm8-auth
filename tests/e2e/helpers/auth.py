"""Master-admin login for API (httpx) and browser (Playwright storage state)."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from tests.e2e.helpers.config import (
    auth_state_path,
    e2e_api_timeout_s,
    e2e_base_url,
    e2e_master_email,
    e2e_master_password,
)


class E2EAuthError(RuntimeError):
    pass


def require_master_credentials() -> tuple[str, str]:
    email, password = e2e_master_email(), e2e_master_password()
    if not email or not password:
        raise E2EAuthError(
            "Set E2E_MASTER_ADMIN_EMAIL + E2E_MASTER_ADMIN_PASSWORD in .env "
            "(or E2E_MASTER_ADMIN_EMAIL + password; email may fall back to BOOTSTRAP_ADMIN_EMAIL)."
        )
    return email, password


def api_login(*, attempts: int = 5) -> dict[str, Any]:
    """Return token payload from POST /api/auth/login (retries on transient errors)."""
    from tests.e2e.helpers.api_ready import wait_for_api_ready

    email, password = require_master_credentials()
    base = e2e_base_url()
    timeout = e2e_api_timeout_s()
    ready = wait_for_api_ready(base, timeout_s=min(180.0, max(60.0, timeout * 2)), require_db=True)
    if not ready.get("ok"):
        raise E2EAuthError(f"API not ready for login: {ready}")
    last: Exception | None = None
    for i in range(attempts):
        try:
            with httpx.Client(base_url=base, timeout=timeout) as client:
                r = client.post("/api/auth/login", json={"email": email, "password": password})
            if r.status_code != 200:
                raise E2EAuthError(f"Login failed ({r.status_code}): {r.text[:500]}")
            data = r.json()
            if not data.get("access_token"):
                raise E2EAuthError("Login response missing access_token")
            return data
        except (httpx.HTTPError, E2EAuthError) as e:
            last = e
            time.sleep(min(8.0, 0.5 * (2**i)))
    assert last is not None
    raise last


def api_login_with_cookies() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Login via httpx and return tokens + Playwright-compatible cookie dicts."""
    email, password = require_master_credentials()
    base = e2e_base_url()
    parsed = urlparse(base)
    host = parsed.hostname or "127.0.0.1"
    timeout = e2e_api_timeout_s()
    with httpx.Client(base_url=base, timeout=timeout) as client:
        r = client.post("/api/auth/login", json={"email": email, "password": password})
        if r.status_code != 200:
            raise E2EAuthError(f"Login failed ({r.status_code}): {r.text[:500]}")
        data = r.json()
        pw_cookies: list[dict[str, Any]] = []
        for c in client.cookies.jar:
            pw_cookies.append(
                {
                    "name": c.name,
                    "value": c.value,
                    "domain": c.domain or host,
                    "path": c.path or "/",
                    "httpOnly": True,
                    "secure": parsed.scheme == "https",
                    "sameSite": "Lax",
                }
            )
    if not data.get("access_token"):
        raise E2EAuthError("Login response missing access_token")
    return data, pw_cookies


def api_client(*, bearer: str | None = None) -> httpx.Client:
    token = bearer or api_login()["access_token"]
    return httpx.Client(
        base_url=e2e_base_url(),
        timeout=e2e_api_timeout_s(),
        headers={"Authorization": f"Bearer {token}"},
    )


_TRANSIENT_API_EXC = (httpx.HTTPError, ConnectionError, OSError)


def close_api_client(client: httpx.Client | None) -> None:
    if client is None:
        return
    try:
        client.close()
    except Exception:
        pass


def refresh_api_client(client: httpx.Client | None = None) -> httpx.Client:
    close_api_client(client)
    return api_client()


def api_request_with_retry(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    attempts: int = 5,
    **kwargs: Any,
) -> tuple[httpx.Response, httpx.Client]:
    """Retry and refresh the httpx client on stale connections / DB wake (503)."""
    last: Exception | None = None
    active = client
    for i in range(attempts):
        try:
            resp = active.request(method, path, **kwargs)
            if resp.status_code == 401 and i < attempts - 1:
                active = refresh_api_client(active)
                time.sleep(0.5 * (i + 1))
                continue
            # Neon wake / pool recovery — do not treat as hard E2E failure yet.
            if resp.status_code in (502, 503) and i < attempts - 1:
                time.sleep(min(8.0, 0.6 * (2**i)))
                active = refresh_api_client(active)
                continue
            return resp, active
        except _TRANSIENT_API_EXC as e:
            last = e
            active = refresh_api_client(active)
            time.sleep(min(8.0, 0.5 * (2**i)))
    assert last is not None
    raise last


def api_get_with_retry(client: httpx.Client, path: str, *, attempts: int = 3, **kwargs: Any) -> tuple[httpx.Response, httpx.Client]:
    return api_request_with_retry(client, "GET", path, attempts=attempts, **kwargs)


def fetch_me(client: httpx.Client) -> dict[str, Any]:
    r = client.get("/api/me")
    r.raise_for_status()
    return r.json()


def _human_login_preferred() -> bool:
    import os

    explicit = os.environ.get("E2E_HUMAN_LOGIN", "").lower()
    if explicit in ("0", "false", "no"):
        return False
    if explicit in ("1", "true", "yes"):
        return True
    return _env_flag("E2E_HEADED")


def _storage_state_has_bearer_injection(path) -> bool:
    """Older E2E runs stored bearer tokens — incompatible with cookie-primary same-origin auth."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return True
    bearer_keys = {"um8_send_bearer", "uploadm8_access_token", "uploadm8_refresh_token"}
    for origin in data.get("origins") or []:
        for entry in origin.get("localStorage") or []:
            if entry.get("name") in bearer_keys:
                return True
    return False


def ensure_playwright_storage_state(playwright) -> str:
    """
    Build browser session for same-origin E2E (cookie-primary, no bearer injection).

    Uses the login form when E2E_HUMAN_LOGIN=1 (default when E2E_HEADED=1), otherwise
    httpx login cookies only. Never sets um8_send_bearer — that mode is for cross-host dev.
    """
    from tests.e2e.helpers.browser_session import bootstrap_human_session, human_login_via_form
    from tests.e2e.helpers.human_pace import CHROME_UA

    path = auth_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and not _env_flag("E2E_FORCE_RELOGIN") and not _storage_state_has_bearer_injection(path):
        return str(path)

    base = e2e_base_url()
    # Headless cache bake — avoids a second visible Chrome window during pytest/live demo.
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(
        viewport={"width": 1440, "height": 900},
        ignore_https_errors=True,
        user_agent=CHROME_UA,
    )
    page = context.new_page()

    if _human_login_preferred():
        human_login_via_form(page, base)
    else:
        _tokens, cookie_jar = api_login_with_cookies()
        if cookie_jar:
            context.add_cookies(cookie_jar)
        bootstrap_human_session(page, base)

    context.storage_state(path=str(path))
    context.close()
    browser.close()
    return str(path)


def _env_flag(name: str) -> bool:
    import os

    return os.environ.get(name, "").lower() in ("1", "true", "yes")
