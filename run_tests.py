#!/usr/bin/env python3
"""
UploadM8 test runner: loads .env from the project root, then invokes pytest or
small helper commands.

Examples:
  python run_tests.py
  python run_tests.py unit
  python run_tests.py tiers
  python run_tests.py consistency
  python run_tests.py router-lint
  python run_tests.py frontend-lint
  python run_tests.py stress
  python run_tests.py enterprise
  python run_tests.py e2e
  python run_tests.py overnight

Extra pytest args pass through: python run_tests.py unit -v --tb=short
"""
from __future__ import annotations

import os
import subprocess
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
    merged = {**os.environ, **(env or {})}
    return subprocess.run([sys.executable, "-m", "pytest"] + extra, cwd=str(ROOT), env=merged).returncode


def cmd_verify() -> int:
    load_env()
    print("1. Python OK:", sys.version.split()[0])
    print("2. pytest command OK")
    print("3. API command: python -m uvicorn app:app --host 127.0.0.1 --port 8000")
    return 0


def cmd_consistency() -> int:
    script = ROOT / "tools" / "check_consistency.py"
    if not script.is_file():
        print("tools/check_consistency.py not found - skipping")
        return 0
    return subprocess.run([sys.executable, str(script)], cwd=str(ROOT)).returncode


def cmd_router_lint() -> int:
    script = ROOT / "tools" / "lint_routers.py"
    if not script.is_file():
        print("tools/lint_routers.py not found - skipping")
        return 0
    return subprocess.run([sys.executable, str(script)], cwd=str(ROOT)).returncode


def cmd_frontend_lint() -> int:
    script = ROOT / "scripts" / "lint_frontend_inline.py"
    if not script.is_file():
        print("scripts/lint_frontend_inline.py not found - skipping")
        return 0
    return subprocess.run([sys.executable, str(script)], cwd=str(ROOT)).returncode


def cmd_stress() -> int:
    host = os.environ.get("LOCUST_HOST", "http://127.0.0.1:8000")
    print(f"Starting Locust against {host} - open http://localhost:8089")
    return subprocess.run([sys.executable, "-m", "locust", "-H", host], cwd=str(ROOT)).returncode


def main() -> int:
    load_env()
    argv = sys.argv[1:]
    mode = "all"
    pytest_tail: list[str] = []
    known = {
        "verify",
        "unit",
        "tiers",
        "settings",
        "consistency",
        "router-lint",
        "frontend-lint",
        "stress",
        "enterprise",
        "e2e",
        "overnight",
        "all",
    }
    if argv:
        if argv[0] in known:
            mode = argv[0]
            pytest_tail = argv[1:]
        else:
            pytest_tail = argv

    if mode == "verify":
        return cmd_verify()
    if mode == "consistency":
        return cmd_consistency()
    if mode == "router-lint":
        return cmd_router_lint()
    if mode == "frontend-lint":
        return cmd_frontend_lint()
    if mode == "stress":
        return cmd_stress()
    if mode == "enterprise":
        os.environ.setdefault("LOCUST_USERS", os.environ.get("LOCUST_USERS", "10"))
        os.environ.setdefault("LOCUST_SPAWN_RATE", os.environ.get("LOCUST_SPAWN_RATE", "2"))
        os.environ.setdefault("LOCUST_RUN_TIME", os.environ.get("LOCUST_RUN_TIME", "60s"))
        return cmd_stress()

    if mode in ("e2e", "overnight"):
        targets = ["tests/e2e/"]
        if mode == "overnight":
            pytest_tail = ["-m", "overnight"] + pytest_tail
        return run_pytest(targets + pytest_tail)

    if mode == "tiers":
        targets = [
            "tests/test_entitlement_flows.py",
            "tests/test_api_entitlements_smoke.py",
            "tests/test_thumbnail_studio_personas_pikzels_api.py",
        ]
    elif mode == "settings":
        targets = [
            "tests/test_settings_combinatorics.py",
            "tests/test_upload_job_payload.py",
            "tests/test_upload_preferences_runtime.py",
        ]
        pytest_tail = ["-m", "not slow"] + pytest_tail
    else:
        targets = ["tests/"]

    if mode == "unit":
        pytest_tail = ["-m", "not slow and not e2e"] + pytest_tail

    cmd = targets + pytest_tail
    if not pytest_tail and mode in ("all", "unit"):
        cmd = ["-q"] + cmd
    return run_pytest(cmd)


if __name__ == "__main__":
    sys.exit(main())
