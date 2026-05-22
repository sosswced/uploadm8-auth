#!/usr/bin/env python3
"""
Fire test: real pytest + optional DB/Redis/App lifespan.

1) pytest (always)
2) Postgres ping on DATABASE_URL
3) Import app (full module)
4) TestClient /health + /ready if DB + Redis reachable

From a laptop, Render internal DB hostnames often fail DNS. Set DATABASE_URL_FIRE
to the External Database URL from Render to run step 4 locally.

Run:  python tools/fire_test.py

Production smoke after deploy:
  curl -sS https://YOUR_API/ready | jq .
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path


def _p(msg: str) -> None:
    print(msg, flush=True)


async def _db_ping(url: str, timeout_s: float = 8.0) -> bool:
    import asyncpg

    try:
        conn = await asyncio.wait_for(asyncpg.connect(url, timeout=timeout_s), timeout=timeout_s + 2)
        try:
            return (await conn.fetchval("SELECT 1")) == 1
        finally:
            await conn.close()
    except Exception:
        return False


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except ImportError:
        _p("FAIL: pip install python-dotenv")
        return 1

    if not os.environ.get("JWT_SECRET"):
        _p("FAIL: JWT_SECRET not set")
        return 1

    if not os.environ.get("DATABASE_URL"):
        _p("FAIL: DATABASE_URL not set")
        return 1

    if os.environ.get("DATABASE_URL_FIRE"):
        os.environ["DATABASE_URL"] = os.environ["DATABASE_URL_FIRE"]

    _p("== 1. pytest tests/ ==")
    if subprocess.run(
        [sys.executable, "-m", "pytest", str(root / "tests"), "-v", "--tb=short"],
        cwd=str(root),
    ).returncode:
        _p("FAIL: pytest")
        return 1
    _p("    OK")

    _p("== 2. Postgres SELECT 1 ==")
    db_ok = asyncio.run(_db_ping(os.environ["DATABASE_URL"]))
    if db_ok:
        _p("    OK")
    else:
        _p(
            "    SKIP (host not reachable from this machine - normal for Render internal URL).\n"
            "    Set DATABASE_URL_FIRE to external URL for full local run."
        )

    _p("== 3. Import app ==")
    import app as app_mod  # noqa: E402

    _p(f"    OK - {app_mod.app.title} v{app_mod.app.version}")

    if db_ok and os.environ.get("REDIS_URL"):
        _p("== 4. TestClient -> /health + /ready ==")
        try:
            from starlette.testclient import TestClient
        except ImportError:
            _p("FAIL: pip install httpx")
            return 1
        with TestClient(app_mod.app) as client:
            h = client.get("/health")
            if h.status_code != 200 or h.json().get("status") != "ok":
                _p(f"FAIL /health: {h.status_code} {h.text}")
                return 1
            r = client.get("/ready")
            data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            if r.status_code != 200:
                _p(f"FAIL /ready: {r.status_code} {data or r.text}")
                return 1
            _p(f"    OK - {data.get('checks', {})}")
    elif not db_ok:
        _p("== 4. SKIP TestClient (DB unreachable) ==")
    else:
        _p("== 4. SKIP TestClient (no REDIS_URL) ==")

    _p("== 5. Wallet conn.transaction() ==")
    import inspect
    from services import wallet as w

    for name in ("reserve_tokens", "spend_tokens", "credit_wallet"):
        if "conn.transaction()" not in inspect.getsource(getattr(w, name)):
            _p(f"FAIL: {name}")
            return 1
    _p("    OK")

    _p("== 6. publish_stage._publish_single ==")
    from stages import publish_stage as ps

    if not hasattr(ps, "_publish_single"):
        _p("FAIL")
        return 1
    _p("    OK")

    _p("\n>>> FIRE TEST PASSED <<<\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
