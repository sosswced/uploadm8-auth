"""Wait for local API + DB readiness before Playwright / OpenAPI storms."""

from __future__ import annotations

import time
from typing import Any

import httpx

from tests.e2e.helpers.config import e2e_api_timeout_s, e2e_base_url


def wait_for_api_ready(
    base_url: str | None = None,
    *,
    timeout_s: float = 120.0,
    require_db: bool = True,
) -> dict[str, Any]:
    """
    Poll ``/api/health`` (DB readiness) then ``/ready`` until ready.

    Uses short per-request timeouts so a wedged pool cannot hang the waiter
    forever (``/health`` stays green without DB; ``/ready`` must fail fast).
    """
    base = (base_url or e2e_base_url()).rstrip("/")
    deadline = time.time() + timeout_s
    last: dict[str, Any] = {"ok": False, "base_url": base}
    attempt = 0
    # Keep probe timeouts short — /ready must not block for full e2e_api_timeout_s.
    probe_timeout = min(12.0, max(5.0, e2e_api_timeout_s() / 3))
    while time.time() < deadline:
        attempt += 1
        try:
            with httpx.Client(base_url=base, timeout=probe_timeout) as client:
                if require_db:
                    r = client.get("/api/health")
                    body: dict[str, Any] = {}
                    try:
                        body = r.json() if r.content else {}
                    except Exception:
                        body = {}
                    last = {
                        "ok": r.status_code == 200 and bool(body.get("db", True)),
                        "status_code": r.status_code,
                        "body": body,
                        "attempt": attempt,
                        "base_url": base,
                        "path": "/api/health",
                    }
                    if last["ok"]:
                        # Confirm /ready also answers (bounded) — catches pool wedge
                        # where /api/health raced a free connection but queue is stuck.
                        try:
                            ready = client.get("/ready")
                            if ready.status_code == 200:
                                last["path"] = "/ready"
                                return last
                            last["ok"] = False
                            last["ready_status"] = ready.status_code
                        except httpx.HTTPError as e:
                            last["ok"] = False
                            last["ready_error"] = str(e)[:200]
                else:
                    r = client.get("/health")
                    last = {
                        "ok": r.status_code == 200,
                        "status_code": r.status_code,
                        "attempt": attempt,
                        "base_url": base,
                        "path": "/health",
                    }
                    if last["ok"]:
                        return last
        except httpx.HTTPError as e:
            last = {
                "ok": False,
                "error": str(e)[:200],
                "attempt": attempt,
                "base_url": base,
            }
        time.sleep(min(8.0, 0.5 * attempt))
    last["timeout_s"] = timeout_s
    return last
