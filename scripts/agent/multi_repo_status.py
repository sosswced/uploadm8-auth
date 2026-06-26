#!/usr/bin/env python3
"""
Multi-repo orchestration status for backend + frontend (Level 8+).

Reports git state for uploadm8-auth (this repo) and uploadm8-frontend clone.

Examples:
  python scripts/agent/multi_repo_status.py --json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRONTEND_REPO = Path.home() / "Dev" / "uploadm8-frontend"
ALT_FRONTEND = ROOT.parent / "uploadm8-frontend"


def git_info(repo: Path) -> dict:
    if not (repo / ".git").exists():
        return {"exists": False, "path": str(repo)}

    def run(*args: str) -> str:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return (proc.stdout or "").strip()

    status = run("status", "--porcelain")
    return {
        "exists": True,
        "path": str(repo),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "changed_files": len(status.splitlines()) if status else 0,
        "remote": run("remote", "get-url", "origin") or "",
    }


def resolve_frontend() -> Path:
    if FRONTEND_REPO.is_dir():
        return FRONTEND_REPO
    if ALT_FRONTEND.is_dir():
        return ALT_FRONTEND
    return FRONTEND_REPO


def main() -> int:
    parser = argparse.ArgumentParser(description="UploadM8 multi-repo status")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    backend = git_info(ROOT)
    frontend_path = resolve_frontend()
    frontend = git_info(frontend_path)

    needs_backend = backend.get("dirty", False)
    needs_frontend = frontend.get("dirty", False)

    report = {
        "backend": backend,
        "frontend": frontend,
        "both_dirty": needs_backend and needs_frontend,
        "ship_order": [],
        "skills": [],
    }

    if needs_backend:
        report["ship_order"].append("uploadm8-backend-push")
        report["skills"].append("uploadm8-backend-push")
    if needs_frontend:
        report["ship_order"].append("uploadm8-frontend-push")
        report["skills"].append("uploadm8-frontend-push")
    if report["ship_order"]:
        report["skills"].append("uploadm8-ascended-ci")
        report["next"] = "Run ascended CI on each PR after push (user must request commit/push)"
    else:
        report["next"] = "No dirty repos — ready for feature work"

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
