"""GET sweep including slow paths (pikzels, thumbnail-studio jobs) when opted in."""

from __future__ import annotations

import os
import time

import httpx
import pytest

from tests.e2e.helpers.openapi_catalog import (
    assert_acceptable_status,
    build_context,
    fetch_openapi,
    iter_read_smoke_cases,
)

pytestmark = [pytest.mark.e2e, pytest.mark.api_smoke, pytest.mark.overnight]


def _collect_slow_cases(api_session: httpx.Client):
    openapi = fetch_openapi(api_session)
    ctx = build_context(api_session)
    return list(iter_read_smoke_cases(openapi, ctx=ctx, safe_only=True))


def _smoke_timeout() -> httpx.Timeout:
    read_s = float(os.environ.get("E2E_SMOKE_READ_TIMEOUT_S", "45"))
    return httpx.Timeout(connect=20.0, read=read_s, write=20.0, pool=20.0)


def _get_with_retry(client: httpx.Client, path: str, *, attempts: int = 4) -> httpx.Response:
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            r = client.get(path, timeout=_smoke_timeout())
            if r.status_code == 503 and i < attempts - 1:
                time.sleep(0.6 * (i + 1))
                continue
            return r
        except httpx.HTTPError as e:
            last_exc = e
            time.sleep(0.6 * (i + 1))
    assert last_exc is not None
    raise last_exc


@pytest.mark.timeout(2400)
def test_slow_api_get_endpoints(api_session: httpx.Client, monkeypatch):
    """Covers pikzels + thumbnail-studio GET catalog when E2E_INCLUDE_SLOW_API=1."""
    if os.environ.get("E2E_SKIP_API_SMOKE", "").lower() in ("1", "true", "yes"):
        pytest.skip("E2E_SKIP_API_SMOKE set")
    monkeypatch.setenv("E2E_INCLUDE_SLOW_API", "1")
    cases = _collect_slow_cases(api_session)
    slow_cases = [c for c in cases if "pikzels" in c.path.lower() or "thumbnail-studio/jobs" in c.path.lower()]
    assert slow_cases, "No pikzels/thumbnail-studio slow GET cases — is OpenAPI current?"
    failures: list[str] = []
    for i, case in enumerate(slow_cases):
        try:
            r = _get_with_retry(api_session, case.path)
            assert_acceptable_status(r.status_code, case)
        except AssertionError as e:
            failures.append(str(e))
        except httpx.HTTPError as e:
            failures.append(f"{case.id}: transport error {e}")
        if i % 5 == 4:
            time.sleep(0.2)
    assert not failures, "Slow API GET smoke failures:\n" + "\n".join(failures[:40])
