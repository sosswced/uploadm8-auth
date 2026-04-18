#!/usr/bin/env python3
"""
UploadM8 test runner — loads .env from project root, then invokes pytest or helpers.

One-shot full pipeline (unit → core e2e → live → admin → exhaustive UI; optional admin-full if env set):

  python run_tests.py mega

Per-suite commands (each is one pytest invocation unless noted):

  python run_tests.py                    # everything under tests/ (heavy suites skip without env flags)
  python run_tests.py verify             # Python + Playwright + env sanity
  python run_tests.py unit               # -m not e2e
  python run_tests.py e2e                # core browser tests only (excludes live, admin, admin_full, exhaustive)
  python run_tests.py login              # tests/test_login.py
  python run_tests.py flows              # flows + full_app_flow + site_matrix + ui_flows
  python run_tests.py matrix             # site_matrix + ui_flows
  python run_tests.py tiers              # entitlement + API smoke (no browser)
  python run_tests.py live               # test_live_journey.py (needs PLAYWRIGHT_RUN_LIVE=1)
  python run_tests.py admin              # test_admin_e2e.py (needs PLAYWRIGHT_RUN_ADMIN_E2E=1)
  python run_tests.py admin-full         # test_admin_e2e_full.py (needs PLAYWRIGHT_ADMIN_E2E_FULL=1)
  python run_tests.py exhaustive         # test_exhaustive_ui.py user+admin click sweep (needs PLAYWRIGHT_RUN_EXHAUSTIVE=1)
  python run_tests.py full               # test_full_app_flow.py
  python run_tests.py interactions
  python run_tests.py visual
  python run_tests.py consistency        # tools/check_consistency.py
  python run_tests.py router-lint        # tools/lint_routers.py (line count + DB-touch lines)
  python run_tests.py stress             # locust
  python run_tests.py enterprise         # locust with preset spawn/time

Mega prerequisites: API + static on PLAYWRIGHT_BASE_URL; PLAYWRIGHT_TEST_EMAIL/PASSWORD (user sweep;
falls back to ADMIN_*); PLAYWRIGHT_ADMIN_EMAIL/PASSWORD (admin sweep). Optional: PLAYWRIGHT_ADMIN_E2E_FULL=1
appends admin-full after exhaustive.

Extra pytest args pass through: python run_tests.py e2e -v --tb=short
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except Exception:
        pass


def run_pytest(extra: list[str], env: dict | None = None) -> int:
    import subprocess

    merged = {**os.environ, **(env or {})}
    cmd = [sys.executable, "-m", "pytest"] + extra
    r = subprocess.run(cmd, cwd=str(ROOT), env=merged)
    return r.returncode


def cmd_verify() -> int:
    load_env()
    print("1. Python OK:", sys.version.split()[0])
    try:
        import playwright  # noqa: F401

        print("2. playwright (Python) OK")
    except ImportError:
        print("2. FAIL: pip install playwright")
        return 1
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            b = p.chromium.launch(headless=True)
            b.close()
        print("3. Chromium launch OK")
    except Exception as e:
        print("3. FAIL: run: python -m playwright install chromium")
        print("   ", e)
        return 1
    base = os.environ.get("PLAYWRIGHT_BASE_URL", "http://127.0.0.1:8000")
    api = os.environ.get("PLAYWRIGHT_API_BASE", "http://127.0.0.1:8000")
    print(f"4. PLAYWRIGHT_BASE_URL={base}")
    print(f"   PLAYWRIGHT_API_BASE={api}")
    email = os.environ.get("PLAYWRIGHT_TEST_EMAIL") or os.environ.get("PLAYWRIGHT_ADMIN_EMAIL")
    pwd = os.environ.get("PLAYWRIGHT_TEST_PASSWORD") or os.environ.get("PLAYWRIGHT_ADMIN_PASSWORD")
    if email and pwd:
        print("5. Playwright credentials: set (email present)")
    else:
        print("5. WARN: PLAYWRIGHT_TEST_EMAIL/PASSWORD or PLAYWRIGHT_ADMIN_* not set — e2e will skip")
    print("6. Run API:  python -m uvicorn app:app --host 127.0.0.1 --port 8000")
    print("   E2E: without server, @e2e tests skip (set PLAYWRIGHT_REQUIRE_SERVER=1 to fail instead)")
    print("7. Serve frontend/ (static) on 8080 if needed:  cd frontend && python -m http.server 8080  →  PLAYWRIGHT_BASE_URL=http://127.0.0.1:8080")
    ex = os.environ.get("PLAYWRIGHT_RUN_EXHAUSTIVE", "")
    print(f"8. Exhaustive UI sweep: PLAYWRIGHT_RUN_EXHAUSTIVE={ex!r} (set 1 for python run_tests.py exhaustive or mega)")
    return 0


def cmd_consistency() -> int:
    script = ROOT / "tools" / "check_consistency.py"
    if not script.is_file():
        print("tools/check_consistency.py not found — skipping")
        return 0
    import subprocess

    return subprocess.run([sys.executable, str(script)], cwd=str(ROOT)).returncode


def cmd_router_lint() -> int:
    script = ROOT / "tools" / "lint_routers.py"
    if not script.is_file():
        print("tools/lint_routers.py not found — skipping")
        return 0
    import subprocess

    return subprocess.run([sys.executable, str(script)], cwd=str(ROOT)).returncode


def cmd_mega() -> int:
    """Run all suites in order: unit → core e2e → live → admin → exhaustive; optional admin-full."""
    import subprocess

    steps: list[tuple[list[str], dict[str, str]]] = [
        (["-m", "not e2e", "tests/"], {}),
        (
            ["-m", "e2e and not live and not admin and not admin_full and not exhaustive", "tests/"],
            {},
        ),
        (["tests/test_live_journey.py"], {"PLAYWRIGHT_RUN_LIVE": "1"}),
        (["tests/test_admin_e2e.py"], {"PLAYWRIGHT_RUN_ADMIN_E2E": "1"}),
        (["tests/test_exhaustive_ui.py"], {"PLAYWRIGHT_RUN_EXHAUSTIVE": "1"}),
    ]
    if os.environ.get("PLAYWRIGHT_ADMIN_E2E_FULL", "").strip().lower() in ("1", "true", "yes"):
        steps.append(
            (
                ["tests/test_admin_e2e_full.py"],
                {"PLAYWRIGHT_RUN_ADMIN_E2E": "1", "PLAYWRIGHT_ADMIN_E2E_FULL": "1"},
            )
        )
    for argv, env_extra in steps:
        merged = {**os.environ, **env_extra}
        r = subprocess.run([sys.executable, "-m", "pytest"] + argv, cwd=str(ROOT), env=merged)
        if r.returncode != 0:
            return r.returncode
    return 0


def cmd_stress() -> int:
    import subprocess

    host = os.environ.get("LOCUST_HOST", "http://127.0.0.1:8000")
    print(f"Starting Locust against {host} — open http://localhost:8089")
    return subprocess.run(
        [sys.executable, "-m", "locust", "-H", host],
        cwd=str(ROOT),
    ).returncode


def main() -> int:
    load_env()
    argv = sys.argv[1:]
    mode = "all"
    pytest_tail: list[str] = []
    if argv:
        if argv[0] in (
            "verify",
            "headless",
            "login",
            "flows",
            "full",
            "interactions",
            "visual",
            "consistency",
            "router-lint",
            "stress",
            "enterprise",
            "all",
            "e2e",
            "unit",
            "matrix",
            "tiers",
            "live",
            "admin",
            "admin-full",
            "exhaustive",
            "mega",
        ):
            mode = argv[0]
            pytest_tail = argv[1:]
        else:
            pytest_tail = argv

    env_extra: dict[str, str] = {}
    if mode == "headless":
        env_extra["PLAYWRIGHT_HEADLESS"] = "true"
    if mode == "live":
        env_extra["PLAYWRIGHT_RUN_LIVE"] = "1"
    if mode == "admin":
        env_extra["PLAYWRIGHT_RUN_ADMIN_E2E"] = "1"
    if mode == "admin-full":
        env_extra["PLAYWRIGHT_RUN_ADMIN_E2E"] = "1"
        env_extra["PLAYWRIGHT_ADMIN_E2E_FULL"] = "1"
    if mode == "exhaustive":
        env_extra["PLAYWRIGHT_RUN_EXHAUSTIVE"] = "1"

    if mode == "verify":
        return cmd_verify()

    if mode == "mega":
        return cmd_mega()

    if mode == "consistency":
        return cmd_consistency()

    if mode == "router-lint":
        return cmd_router_lint()

    if mode == "stress":
        return cmd_stress()

    if mode == "enterprise":
        env_extra.setdefault("LOCUST_USERS", os.environ.get("LOCUST_USERS", "10"))
        env_extra.setdefault("LOCUST_SPAWN_RATE", os.environ.get("LOCUST_SPAWN_RATE", "2"))
        env_extra.setdefault("LOCUST_RUN_TIME", os.environ.get("LOCUST_RUN_TIME", "60s"))
        return cmd_stress()

    # pytest modes
    tests_dir = ROOT / "tests"
    if mode == "login":
        files = ["tests/test_login.py"]
    elif mode == "matrix":
        files = ["tests/test_site_matrix.py", "tests/test_ui_flows.py"]
    elif mode == "tiers":
        files = ["tests/test_entitlement_flows.py", "tests/test_api_entitlements_smoke.py"]
    elif mode == "live":
        files = ["tests/test_live_journey.py"]
    elif mode == "admin":
        files = ["tests/test_admin_e2e.py"]
    elif mode == "admin-full":
        files = ["tests/test_admin_e2e_full.py"]
    elif mode == "exhaustive":
        files = ["tests/test_exhaustive_ui.py"]
    elif mode == "flows":
        files = [
            "tests/test_flows.py",
            "tests/test_full_app_flow.py",
            "tests/test_site_matrix.py",
            "tests/test_ui_flows.py",
        ]
    elif mode == "full":
        files = ["tests/test_full_app_flow.py"]
    elif mode == "interactions":
        files = ["tests/test_interactions.py"]
    elif mode == "visual":
        files = ["tests/test_visual.py"]
    elif mode == "e2e":
        files = ["tests/"]
        # Exclude heavy / gated suites (live, admin, admin_full, exhaustive)
        pytest_tail = ["-m", "e2e and not live and not admin and not admin_full and not exhaustive"] + pytest_tail
    elif mode == "unit":
        files = ["tests/"]
        pytest_tail = ["-m", "not e2e"] + pytest_tail
    else:
        files = ["tests/"]

    # If visual file missing, pytest will error — create minimal test_visual.py
    targets: list[str] = []
    for pattern in files:
        if pattern.endswith("/"):
            targets.append(str(tests_dir))
        else:
            p = ROOT / pattern
            if p.exists():
                targets.append(pattern)
            elif mode == "visual":
                continue

    if not targets:
        targets = ["tests/"]

    cmd = targets + pytest_tail
    # default: quiet for CI
    if not pytest_tail and mode in ("all", "unit", "e2e", "live", "admin-full", "exhaustive"):
        cmd = ["-q"] + cmd

    return run_pytest(cmd, env_extra)


if __name__ == "__main__":
    sys.exit(main())
